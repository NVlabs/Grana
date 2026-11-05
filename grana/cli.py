# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a cache, validate, train through Ray, or run checkpoint inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .config import load_config, physical_batch_size, supervision_budget
from ._paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(prog="grana")
    commands = parser.add_subparsers(dest="command", required=True)
    label = commands.add_parser("label", help="Label LeRobot v2/v3 subtasks, outcomes and action quality in the browser")
    label.add_argument("--config", required=True, type=Path, help="Cache recipe with a separate labels directory")
    label.add_argument("--port", type=int, default=8790, help="Loopback HTTP port (default: 8790)")
    drafts = commands.add_parser("draft-labels", help="Import completed skill API results as drafts for grana label")
    drafts.add_argument("--config", required=True, type=Path, help="The labeling recipe, as for grana label")
    drafts.add_argument("--from", dest="sources", required=True, action="append", type=Path, metavar="PATH",
                        help="A skill API result folder or file, a jobs list, or a folder of results; repeatable")
    drafts.add_argument("--dry-run", action="store_true", help="Check everything and report; write nothing")
    drafts.add_argument("--failure-quality", choices=("regressed",),
                        help="Explicit task-policy mapping; required if candidate AQ contains failure")
    drafts.add_argument("--policy-dir", type=Path, help="Exact frozen annotation policy used by preparation and submission")
    drafts.add_argument("--json", action="store_true", help="Print the full report as JSON")
    cache = commands.add_parser("build-cache", help="Convert annotated LeRobot v2 episodes into a grana cache")
    cache.add_argument("--config", required=True, type=Path)
    annotation = commands.add_parser("annotate", help="Propose RGB-only LeRobot task stages for review")
    annotation.add_argument("--config", required=True, type=Path)
    annotation.add_argument("--plan", action="store_true", help="Validate sources and bounded request plan without API calls")
    annotation.add_argument("--preflight", action="store_true", help="Run only the small structured-output API probe")
    annotation.add_argument("--retry-uncertain", action="store_true", help="Explicitly retry uncertain transport attempts; earlier attempts may be billed")
    review = commands.add_parser("review-annotations", help="Serve local synchronized videos and editable stage candidates")
    review.add_argument("--run", required=True, type=Path)
    review.add_argument("--port", type=int, default=8788)
    export = commands.add_parser("export-stages", help="Export stage-only JSONL with source and review provenance")
    export.add_argument("--run", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--revision", action="append", default=[], metavar="EPISODE=PATH")
    for name in ("validate", "train"):
        recipe = commands.add_parser(name)
        recipe.add_argument("--config", required=True, type=Path)
        recipe.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        recipe.add_argument("--check-data", action="store_true", help="Read cache schemas and verify payload paths")
    from grana.inference.cli import add_arguments, run
    add_arguments(commands.add_parser("infer", help="Score eight RGB observations with a grana checkpoint"))
    args = parser.parse_args()
    if args.command == "label":
        from grana.annotation.label_server import serve_labels
        from grana.data.builder import load_build_config
        serve_labels(load_build_config(args.config), port=args.port, config_path=args.config)
        return
    if args.command == "annotate":
        from grana.annotation.workflow import load_annotation_config, plan_annotation, preflight, run_annotation
        config = load_annotation_config(args.config)
        if args.plan and args.preflight:
            parser.error("Choose either --plan or --preflight")
        if args.plan:
            result = plan_annotation(config)[1]
        elif args.preflight:
            result = preflight(config, args.retry_uncertain)
        else:
            result = run_annotation(config, args.retry_uncertain)
        print(json.dumps(result, indent=2))
        return
    if args.command == "review-annotations":
        from grana.annotation.review import serve_review
        serve_review(args.run, port=args.port)
        return
    if args.command == "export-stages":
        from grana.annotation.workflow import export_stages
        revisions = {}
        for item in args.revision:
            episode, path = item.split("=", 1)
            if int(episode) in revisions:
                parser.error("Duplicate revision episode")
            revisions[int(episode)] = Path(path)
        print(json.dumps(export_stages(args.run, args.output, revisions), indent=2))
        return
    if args.command == "draft-labels":
        import yaml
        from grana.annotation.drafts import draft_labels, exit_status, summary_lines
        say = lambda line: print(line, file=sys.stderr, flush=True)
        try:
            report = draft_labels(args.config, args.sources, dry_run=args.dry_run, failure_quality=args.failure_quality,
                                  policy_dir=args.policy_dir, log=say)
        except (OSError, ValueError, subprocess.SubprocessError, yaml.YAMLError) as error:
            # Unusable inputs, found before anything was written: one line, as for grana label.
            reason = str(error)
            drafts.exit(2, f"grana draft-labels: {' '.join(reason.split()) or type(error).__name__}\n")
        for line in summary_lines(report):
            say(line)
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        if exit_status(report):
            raise SystemExit(1)
        return
    if args.command == "build-cache":
        from grana.data.builder import build_cache, load_build_config
        print(json.dumps(build_cache(load_build_config(args.config)), indent=2))
        return
    if args.command == "infer":
        run(args)
        return
    if args.command == "train" and args.check_data:
        parser.error("--check-data applies to validate; run validate --check-data before train")
    config = load_config(args.config, args.set)
    if args.command == "validate":
        report = {"config": config, "physical_batch_size": physical_batch_size(config),
                  "supervision_budget": supervision_budget(config)}
        if args.check_data:
            from grana.data.cache import CachedTrajectories
            cache = CachedTrajectories(config["data"]["caches"], views=config["data"]["views"])
            report["data"] = cache.summary()
            if config["evaluation"]["enabled"]:
                from grana.training.evaluation import build_evaluation_plan
                heldout = CachedTrajectories(config["evaluation"]["caches"], views=config["data"]["views"])
                report["evaluation_data"] = heldout.summary()
                report["evaluation_plan"] = build_evaluation_plan(
                    heldout, config["evaluation"], config["data"], training_dataset=cache, roles=config["roles"],
                ).manifest()
        print(json.dumps(report, indent=2))
        return
    from .runtime import require_owned_environment
    require_owned_environment()
    root = PROJECT_ROOT
    command = [sys.executable, str(root / "scripts/ray_migration/run_specs_cli.py"), "--config", str(args.config.resolve())]
    for value in args.set:
        command.extend(["--set", value])
    subprocess.run(command, check=True, cwd=root)


if __name__ == "__main__":
    main()
