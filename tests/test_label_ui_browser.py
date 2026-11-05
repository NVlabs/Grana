# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Chrome regression: GRANA_BROWSER_SMOKE=1 python -m pytest -s this_file.

Needs existing Node.js 22+, Chrome/Chromium, ffmpeg and ffprobe; downloads nothing.
Every run generates its own 30 fps videos. No private recordings or external API.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GRANA_BROWSER_SMOKE") != "1", reason="Opt-in browser smoke test"
)


@pytest.fixture
def browser_workspace(tmp_path):
    node = shutil.which("node")
    chrome = os.environ.get("CHROME_BIN") or shutil.which("google-chrome") or shutil.which("chromium")
    if not node or not chrome or not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Node.js 22+, Chrome/Chromium, ffmpeg and ffprobe are required")
    version = int(subprocess.check_output([node, "-p", "process.versions.node.split('.')[0]"], text=True))
    if version < 22:
        pytest.skip("The browser test uses Node.js 22+ native WebSocket support")
    parent = Path(os.environ.get("GRANA_BROWSER_OUTPUT", str(tmp_path)))
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="browser-", dir=parent))
    print(f"Browser artifacts: {output}", flush=True)

    import pyarrow as pa
    import pyarrow.parquet as pq
    from grana.data.targets.composition import PROGRESS_METHOD

    source = output / "raw"
    (source / "meta").mkdir(parents=True)
    keys = {"top": "observation.images.top", "wrist": "observation.images.wrist"}
    info = dict(
        codebase_version="v2.1", fps=30, chunks_size=1000,
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        features={key: {"dtype": "video", "shape": [180, 320, 3]} for key in keys.values()},
    )
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/episodes.jsonl").write_text("".join(
        json.dumps(dict(episode_index=index, length=720)) + "\n" for index in (0, 1)
    ))
    for index in (0, 1):
        parquet = source / f"data/chunk-000/episode_{index:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"frame_index": list(range(720)), "episode_index": [index] * 720,
                                 "timestamp": [frame / 30 for frame in range(720)]}), parquet)
        for view, key in keys.items():
            video = source / f"videos/chunk-000/{key}/episode_{index:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            pattern = "testsrc2" if view == "top" else "testsrc"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"{pattern}=size=320x180:rate=30",
                "-frames:v", "720", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(video),
            ], check=True, timeout=45)
    recipe = dict(
        source=str(source), output=str(output / "cache"), labels=str(output / "labels"), episodes="all",
        task="token-and-lid", compressed_task_prompt="Place the token, then close the lid.",
        stages=[dict(label="1", prompt="Place the token in the tray"), dict(label="2", prompt="Close the tray lid")],
        views=keys, fps=5,
        progress=dict(method=PROGRESS_METHOD, parameter_status="example",
                      parameter_source="Synthetic browser test", stage_parameters=[
                          dict(rho=50, p_cap=0.95, t_min_seconds=1),
                          dict(rho=50, p_cap=0.95, t_min_seconds=1),
                      ]),
    )
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2))
    (output / "environment.json").write_text(json.dumps({
        "chrome": subprocess.check_output([chrome, "--version"], text=True).strip(),
        "node": subprocess.check_output([node, "--version"], text=True).strip(),
        "fixture": "24 s, 720 frames at 30 fps, two synthetic cameras; camera transport failure injected in Chrome",
    }, indent=2))
    return node, chrome, output, recipe


def test_label_page_in_chrome(browser_workspace):
    from datasets import load_from_disk
    from grana.annotation.label_server import create_label_server
    from grana.data.builder import build_cache, validate_build_config

    node, chrome, output, recipe = browser_workspace
    server = create_label_server(validate_build_config(recipe), port=0, config_path=output / "recipe.json")
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        with (output / "browser.log").open("w") as log:
            process = subprocess.Popen([
                node, str(Path(__file__).with_name("label_browser_smoke.cjs")),
                f"http://127.0.0.1:{server.server_port}", chrome, str(output),
            ], stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=360)
            except subprocess.TimeoutExpired:
                # The harness handles SIGTERM by closing its CDP connection and Chrome child.
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                pytest.fail(f"Browser timed out; see {output / 'browser.log'}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    cache = {"passed": False}
    try:
        report = build_cache(validate_build_config(recipe))
        row = load_from_disk(str(output / "cache/processed_dataset"))[0]
        cache.update(episodes=[episode["episode_index"] for episode in report["episodes"]],
                     reader_validation=report["reader_validation"], frames=row["num_frames"],
                     views=row["view_names"], stages=sorted(set(row["stage_ids"])),
                     quality=sorted(set(row["action_quality"])), final_progress=row["progress"][-1])
        assert cache["episodes"] == [0]  # The unfinished second episode never enters training.
        assert cache["reader_validation"] == "passed"
        assert cache["frames"] == 120 and cache["views"] == ["top", "wrist"]
        assert cache["stages"] == [0, 1] and 2 in cache["quality"]
        assert cache["final_progress"] == pytest.approx(0.5)
        cache["passed"] = True
    except Exception as error:
        cache["error"] = repr(error)
    (output / "build-cache-check.json").write_text(json.dumps(cache, indent=2))
    assert code == 0, (output / "browser.log").read_text()[-14000:]
    assert cache["passed"], cache
