# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The browser application must work from a source release or installed wheel."""
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path("grana/annotation/label_ui")


class AssetReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        value = attrs.get("src") if tag in ("script", "img") else attrs.get("href") if tag == "link" else None
        if value:
            assert "://" not in value and not value.startswith("//"), "Label UI assets must be available offline"
            self.paths.add(value)


def test_label_app_assets_are_present_in_source_archive_and_wheel_configuration(tmp_path):
    references = AssetReferences()
    references.feed((ROOT / ASSETS / "index.html").read_text())
    required = references.paths | {"index.html"}
    package_data = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]
    assert {f"label_ui/{name}" for name in required} <= set(package_data["grana.annotation"])
    output = tmp_path / "grana-source.tar.gz"
    process = subprocess.run([sys.executable, str(ROOT / "scripts/package_release.py"), "--output", str(output)],
                             cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    with tarfile.open(output) as archive:
        for name in required:
            packaged = archive.extractfile("grana/" + str(ASSETS / name))
            assert packaged is not None
            assert packaged.read() == (ROOT / ASSETS / name).read_bytes()
        # Runtime labels and cached video clips belong outside the source application.
        shipped = {name for name in archive.getnames() if "/label_ui/" in name}
        assert shipped == {"grana/" + str(ASSETS / name) for name in required}
        for name in ("test_label_ui.cjs", "label_browser_smoke.cjs"):
            packaged = archive.extractfile("grana/tests/" + name)
            assert packaged is not None
            assert packaged.read() == (ROOT / "tests" / name).read_bytes()
