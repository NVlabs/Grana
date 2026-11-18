# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initialize and run the local annotation playground in the foreground."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time

from .runner import KEY_NAME, ROUTES, SCHEMA, SERVICES, atomic_json, check_models, one_line, read_json

CHECKOUT = Path(__file__).resolve().parents[2]
DEFAULT_PORT, DEFAULT_REQUESTS = 8840, 3
MAX_MODELS_BYTES = 64 * 1024
GENERIC_RULES = (
    "Annotate the active objective using the declared stage labels and prompts. "
    "Do not invent stages, force monotonic order, or infer success from the instruction. "
    "Use visible evidence for boundaries, quality and outcomes; retain unknown when evidence is insufficient. "
    "Nominal includes necessary handling and sensible recovery. Suboptimal requires visible ineffective action. "
    "For this playground, failure means an action visibly destroys an already achieved task state; "
    "the explicit human label mapping is failure to regressed. Slow motion alone is not failure. "
    "High outcome requires visible completion of the objective; low requires visible failure or abandonment. "
    "These are generic demonstration criteria, not calibrated task-specific rules."
)


def ports(base):
    """The home page on ``base`` and the label pages on the ports after it."""
    return {"overview": base, **{mode: base + offset for offset, mode in enumerate(SERVICES, 1)}}


def port(text):
    try:
        value = int(text)
    except ValueError:
        value = 0
    if not 1024 <= value <= 65535 - len(SERVICES):
        raise argparse.ArgumentTypeError(f"{text!r} is not a base port in 1024..{65535 - len(SERVICES)}")
    return value


def requests(text):
    try:
        value = int(text)
    except ValueError:
        value = 0
    if not 1 <= value <= 10000:
        raise argparse.ArgumentTypeError(f"{text!r} is not a request count in 1..10000")
    return value


def episodes(text):
    """``all`` or episode indices such as 0,3,5-9."""
    if text.strip() == "all":
        return "all"
    values = []
    try:
        for part in text.split(","):
            low, _, high = part.strip().partition("-")
            values += range(int(low), int(high or low) + 1)
    except ValueError:
        values = []
    if not values or min(values) < 0 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{text!r} is not a list of distinct episode indices such as 0,3,5-9")
    return values


def configured_models(api_key_env, *, model=None, base_url=None):
    """Create a model only from explicit CLI/environment configuration."""
    from grana.annotation.api import provider_settings

    provider = provider_settings(model=model, base_url=base_url, required=False)
    if any(value is None for value in provider.values()):
        return []
    provider = provider_settings(**provider)
    return check_models([{"id": "default", "label": provider["model"].rsplit("/", 1)[-1], **provider,
                          "api_key_env": api_key_env, "reasoning_effort": "xhigh", "protocol": "responses"}])


def load_models(path):
    """A --models file: a JSON array of model entries (check_models); lines starting with // are comments."""
    path = Path(path).expanduser()
    if not path.is_file() or path.stat().st_size > MAX_MODELS_BYTES:
        raise ValueError(f"--models {path} is not a model list of at most 64 KiB")
    def unique(pairs):
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("duplicate key")
        return dict(pairs)
    try:
        # A JSON string cannot span lines, so a line starting with // is never inside one.
        text = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("//"))
        models = json.loads(text, object_pairs_hook=unique)
    except ValueError as error:
        raise ValueError(f"--models {path.name} is not a JSON model list ({one_line(error, 200)})") from None
    try:
        return check_models(models)
    except ValueError as error:
        raise ValueError(f"--models {path.name}: {error}") from None


def initialize(args):
    import yaml
    from grana.annotation.api import KEY_ENV
    from grana.annotation.source import LeRobotSource
    from grana.data.builder import load_build_config, validate_build_config

    cfg = load_build_config(args.config)
    root = args.workspace.expanduser().resolve()
    source = Path(cfg["source"]).resolve()
    if (root.is_relative_to(CHECKOUT) or root.is_relative_to(source)
            or any((parent / ".git").exists() for parent in (root, *root.parents))):
        raise ValueError("--workspace must be outside Git checkouts and the raw source dataset")
    if args.models and any((args.api_key_env, args.model, args.base_url)):
        raise ValueError("--models entries specify their own model, base_url and api_key_env; do not combine these options")
    if args.api_key_env is not None and not re.fullmatch(KEY_NAME, args.api_key_env):
        raise ValueError("--api-key-env must name an environment variable, not contain its value")
    models = None
    if args.models:
        models = load_models(args.models)
    elif any(value is not None for value in (args.api_key_env, args.model, args.base_url)):
        models = configured_models(args.api_key_env or KEY_ENV, model=args.model, base_url=args.base_url)
    registry = [{"id": row["label"], **row} for row in cfg["stages"]]
    expected = {"task": cfg["task"], "task_instruction": cfg["compressed_task_prompt"],
                "views": cfg["views"], "stage_semantics": "active_subtask", "registry": registry}
    chosen = {route: getattr(args, f"{route}_episodes") for route in ROUTES}
    state_file = root / "run-state.json"
    if state_file.exists():
        state = read_json(state_file)
        if state.get("schema_version") != SCHEMA:
            raise ValueError("This workspace was made by an earlier playground; choose a fresh --workspace. Nothing was overwritten.")
        settings = read_json(root / "settings.json")
        if state.get("source") != str(source) or any(settings.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing workspace uses different source/settings; choose a fresh --workspace. Nothing was overwritten.")
        saved_models = check_models(state.get("models"))
        if models is not None and saved_models != models:
            raise ValueError("Existing workspace uses other models (" + ", ".join(entry["id"] for entry in state.get("models") or [])
                             + "); omit provider options to reuse its saved model list. Nothing was overwritten.")
        if args.settings and read_json(args.settings) != settings:
            raise ValueError("Existing settings differ; choose a fresh --workspace")
        if args.criteria and read_json(args.criteria) != read_json(root / "criteria.json"):
            raise ValueError("Existing criteria differ; choose a fresh --workspace")
        if args.max_requests is not None and args.max_requests != state["max_requests"]:
            raise ValueError(f"This workspace's budget is {state['max_requests']} requests; it is fixed when the workspace is created")
        for route, value in chosen.items():
            if value is not None and value != "all" and sorted(value) != state["routes"][route]:
                raise ValueError(f"This workspace's {route} episodes are fixed when it is created: {state['routes'][route]}")
        # Ports are not data: a resumed workspace can move to other ports.
        if state["ports"] != ports(args.port) or state.get("recipe") != str(args.config.resolve()):
            atomic_json(state_file, {**state, "ports": ports(args.port), "recipe": str(args.config.resolve())})
        return root
    if root.exists() and any(root.iterdir()):
        raise ValueError("A new workspace must be empty; existing files were preserved")
    if models is None:
        models = configured_models(KEY_ENV)
    settings = (read_json(args.settings) if args.settings else
                {**expected, "task_policy": {"version": "generic-draft", "status": "draft", "rules": GENERIC_RULES}})
    if not isinstance(settings, dict) or set(settings) != {*expected, "task_policy"} or any(settings[key] != value for key, value in expected.items()):
        raise ValueError("Settings task, instruction, active_subtask semantics, registry and views must exactly match --config")
    policy = settings["task_policy"]
    if (not isinstance(policy, dict) or set(policy) != {"version", "status", "rules"}
            or policy["status"] != "draft" or any(not isinstance(policy[key], str) or not policy[key].strip() for key in ("version", "rules"))):
        raise ValueError("Settings need task_policy with nonempty version/rules and status draft")
    criteria = read_json(args.criteria) if args.criteria else {key: policy[key] for key in ("version", "rules")}
    if (not isinstance(criteria, dict) or set(criteria) != {"version", "rules"}
            or any(not isinstance(value, str) or not value.strip() for value in criteria.values())):
        raise ValueError("Criteria must contain exactly nonempty version and rules")
    if args.settings and criteria != {key: policy[key] for key in ("version", "rules")}:
        raise ValueError("--criteria must exactly match the settings task_policy version/rules")
    settings["task_policy"] = {**policy, **criteria}
    reader = LeRobotSource(source, cfg["views"], allow_shared_videos=True)
    available = sorted(reader.episodes) if cfg["episodes"] == "all" else sorted(cfg["episodes"])
    if set(available) - reader.episodes.keys():
        raise ValueError(f"Episodes absent from the dataset: {sorted(set(available) - reader.episodes.keys())}")
    # Each route takes the recipe's episodes unless told otherwise; routes may share episodes.
    routes = {}
    for route, value in chosen.items():
        routes[route] = available if value in (None, "all") else sorted(value)
        if set(routes[route]) - set(available):
            raise ValueError(f"--{route}-episodes lists episodes the recipe does not select: "
                             f"{sorted(set(routes[route]) - set(available))}")
    # Seed transfer needs separate seed and held-out episodes. That prerequisite
    # belongs to its route; human labeling and pure API work with one episode.
    episodes = []
    for index in sorted({episode for value in routes.values() for episode in value}):
        length = reader.episodes[index].get("length")
        if type(length) is not int or length < 2:
            raise ValueError("Source episodes need valid native frame counts")
        episodes.append({"episode": index, "frames": length})
    recipes = {}
    for mode in SERVICES:
        recipe = deepcopy(cfg)
        # The seed page holds the seeded route's human labels; its drafts get their own page.
        recipe.update(labels=str(root / mode / "labels"), output=str(root / mode / "cache"),
                      episodes=routes["seeded" if mode == "seed" else mode])
        recipes[mode] = validate_build_config(recipe)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    for mode, recipe in recipes.items():
        (root / mode / "labels").mkdir(mode=0o700, parents=True)
        (root / f"{mode}.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    atomic_json(root / "settings.json", settings)
    atomic_json(root / "criteria.json", criteria)
    atomic_json(state_file, {"schema_version": SCHEMA, "status": "ready", "root": str(root), "checkout": str(CHECKOUT),
                             "python": sys.executable, "recipe": str(args.config.resolve()), "source": str(source),
                             "ports": ports(args.port), "recipe_episodes": len(available), "routes": routes,
                             "episodes": episodes, "automatic_api_calls": False,
                             "max_requests": args.max_requests or DEFAULT_REQUESTS, "models": models})
    return root


def serve(root, *, transport=None):
    """Run the label pages and the home page until Ctrl+C or SIGTERM.

    ``transport`` (an httpx transport) runs the home page in this process with every API request sent through
    it, as tests and mocked end-to-end runs do; the command line never sets it.
    """
    root = Path(root).resolve()
    taken = read_json(root / "run-state.json")["ports"]
    children = []
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    with (root / ".launcher.lock").open("a+") as lease, ExitStack() as stack:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This workspace already has a foreground launcher") from None
        # Refuse occupied ports; never terminate an existing service.
        for number in taken.values():
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(("127.0.0.1", number))
                except OSError:
                    raise ValueError(f"Port {number} is already in use; stop its owner or choose another --port") from None
        home = None
        try:
            for mode in SERVICES:
                command = [sys.executable, "-m", "tools.annotation_playground.label_service", "--root", str(root), "--mode", mode]
                children.append((mode, subprocess.Popen(command, cwd=CHECKOUT)))
            if transport is None:
                command = [sys.executable, "-m", "tools.annotation_playground.server", "--root", str(root)]
                children.append(("overview", subprocess.Popen(command, cwd=CHECKOUT)))
            else:
                from .server import running
                home = stack.enter_context(running(root, transport=transport))
                threading.Thread(target=home.serve_forever, kwargs={"poll_interval": 0.5}, name="playground-home",
                                 daemon=True).start()
            print(f"Annotation home: http://127.0.0.1:{taken['overview']}/ (Ctrl+C stops all {len(SERVICES) + 1} local services)",
                  flush=True)
            while True:
                for mode, process in children:
                    if process.poll() is not None:
                        raise RuntimeError(f"The {mode} service exited ({process.returncode}); stopping the playground")
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        finally:
            if home is not None:
                home.shutdown()
            for _, process in children:
                if process.poll() is None:
                    process.terminate()
            deadline = time.monotonic() + 5
            for _, process in children:
                try:
                    process.wait(timeout=max(0.01, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Existing cache recipe; only its task/source/configuration is reused")
    parser.add_argument("--workspace", type=Path, required=True, help="Private output directory outside code and raw data; reuse it to resume")
    parser.add_argument("--settings", type=Path, help="Optional full API settings JSON matching the recipe; otherwise generic draft criteria are used")
    parser.add_argument("--criteria", type=Path, help="Optional {version,rules} JSON for the standalone API route")
    parser.add_argument("--models", type=Path,
                        help="Optional JSON list of the models the page may generate with, fixed when the workspace is created "
                             "(without provider configuration, human labeling remains available)")
    parser.add_argument("--model", help="Provider model name; otherwise GRANA_ANNOTATION_MODEL for a new workspace")
    parser.add_argument("--base-url", help="Responses API base URL; otherwise GRANA_ANNOTATION_BASE_URL for a new workspace")
    parser.add_argument("--api-key-env",
                        help="The configured model's credential variable name only (default: GRANA_ANNOTATION_API_KEY); "
                             "a --models list names one per model")
    parser.add_argument("--max-requests", type=requests,
                        help=f"HTTP requests all routes and models may send together, fixed when the workspace is created (default: {DEFAULT_REQUESTS})")
    for route in ROUTES:
        parser.add_argument(f"--{route}-episodes", type=episodes, metavar="EPISODES",
                            help=f"The {route} route's episodes, e.g. 0,3,5-9, fixed when the workspace is created (default: the recipe's)")
    parser.add_argument("--port", type=port, default=DEFAULT_PORT,
                        help=f"Home page port; the four label pages use the next four (default: {DEFAULT_PORT})")
    parser.add_argument("--init-only", action="store_true", help="Create/validate workspace without starting services or sending requests")
    return parser


def main(argv=None):
    options = parser()
    args = options.parse_args(argv)
    os.umask(0o077)
    try:
        root = initialize(args)
        if args.init_only:
            print(f"Initialized private playground workspace: {root}; no services or API calls started.")
        else:
            serve(root)
    except (OSError, ValueError, RuntimeError) as error:
        options.exit(1, f"Playground: {error}\n")


if __name__ == "__main__":
    main()
