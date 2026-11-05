# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source archives include only reviewed files and can reproduce themselves."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_checker", ROOT / "scripts/check_release.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def run_packager(root, output, *, epoch="0"):
    return subprocess.run(
        [sys.executable, str(root / "scripts/package_release.py"), "--output", str(output)],
        cwd=output.parent, capture_output=True, text=True, timeout=60,
        env={**os.environ, "SOURCE_DATE_EPOCH": epoch},
    )


def write_sources(root, files):
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    names = sorted({"release-files.txt", *files})
    (root / "release-files.txt").write_text("# Exact public files\n" + "\n".join(names) + "\n")


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "source"
    files = {
        name: (ROOT / name).read_bytes()
        for name in ("scripts/package_release.py", "scripts/check_release.py")
    }
    files.update({
        "README.md": b"[data](assets/data.json) [table][table]\n[view](assets/view.html)\n[table]: assets/table.csv\n",
        "assets/data.json": b'{"public": true}\n',
        "assets/table.csv": b"name,value\npublic,1\n",
        "assets/view.html": b"<!doctype html><p>Public visualization</p>\n",
        "scripts/public.sh": b"#!/bin/sh\nprintf public\n",
        "configs/public.yaml": b"public: true\n",
    })
    write_sources(root, files)
    return root


def test_allowlist_excludes_unlisted_files_of_every_previously_globbed_type(source_tree, tmp_path):
    private = ("docs/private.md", "configs/private.yaml", "scripts/private.py",
               "grana/private.py", "requirements/private.txt", "tests/private.py",
               "assets/private.json", "assets/private.csv", "assets/private.html")
    for name in private:
        path = source_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic private sentinel\n")
    output = tmp_path / "source.tar.gz"
    result = run_packager(source_tree, output)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(output.with_suffix(".gz.manifest.json").read_text())
    expected = checker.read_release(source_tree)
    assert receipt["file_count"] == len(expected)
    assert receipt["source_bytes"] == sum(map(len, expected.values()))
    assert receipt["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with tarfile.open(output) as archive:
        assert archive.getnames() == ["grana/" + name for name in sorted(expected)]
        for name, content in expected.items():
            assert archive.extractfile("grana/" + name).read() == content
            assert receipt["files"][name] == hashlib.sha256(content).hexdigest()
        assert not set(private) & receipt["files"].keys()


def test_reproducible_archive_ignores_host_metadata_and_rebuilds_without_git(source_tree, tmp_path):
    first, second = tmp_path / "first.tar.gz", tmp_path / "different-name.tar.gz"
    assert run_packager(source_tree, first, epoch="123456789").returncode == 0
    for name in checker.read_release(source_tree):
        os.utime(source_tree / name, (1700000000, 1700000000))
        (source_tree / name).chmod(0o777)
    assert run_packager(source_tree, second, epoch="123456789").returncode == 0
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first) as archive:
        for info in archive.getmembers():
            assert info.isfile()
            assert (info.uid, info.gid, info.uname, info.gname) == (0, 0, "", "")
            assert info.mtime == 123456789
            content = archive.extractfile(info).read()
            assert info.mode == (0o755 if content.startswith(b"#!") else 0o644)
            assert not info.pax_headers
        archive.extractall(tmp_path / "unpacked", filter="data")
    unpacked = tmp_path / "unpacked/grana"
    assert not (unpacked / ".git").exists()
    rebuilt = tmp_path / "rebuilt.tar.gz"
    result = run_packager(unpacked, rebuilt, epoch="123456789")
    assert result.returncode == 0, result.stderr
    assert rebuilt.read_bytes() == first.read_bytes()


@pytest.mark.parametrize("name", ["../outside", "/etc/passwd", "assets/../README.md", "./README.md",
                                  "assets//data.json", "assets\\data.json", "assets/*.json", "C:/file"])
def test_manifest_rejects_unsafe_paths(source_tree, name):
    with (source_tree / "release-files.txt").open("a") as manifest:
        manifest.write(name + "\n")
    with pytest.raises(ValueError, match="Unsafe release path"):
        checker.read_release(source_tree)


@pytest.mark.parametrize("target", ["assets/data.json", "assets", "release-files.txt"])
def test_manifest_rejects_symlink_files_and_parent_directories(source_tree, tmp_path, target):
    original = source_tree / target
    moved = tmp_path / (original.name + "-outside")
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(ValueError, match="rejects symlinks"):
        checker.read_release(source_tree)


@pytest.mark.parametrize("extra, message", [("README.md", "duplicate"), ("missing.json", "No such file"),
                                            ("assets", "regular file")])
def test_manifest_rejects_duplicates_missing_and_nonregular_files(source_tree, extra, message):
    with (source_tree / "release-files.txt").open("a") as manifest:
        manifest.write(extra + "\n")
    with pytest.raises((OSError, ValueError), match=message):
        checker.read_release(source_tree)


def test_rebuild_inputs_are_required(source_tree):
    path = source_tree / "release-files.txt"
    path.write_text(path.read_text().replace("scripts/check_release.py\n", ""))
    with pytest.raises(ValueError, match="rebuild inputs"):
        checker.read_release(source_tree)


@pytest.mark.parametrize("epoch", ["-1", "4294967296", "invalid"])
def test_invalid_timestamp_creates_no_archive(source_tree, tmp_path, epoch):
    output = tmp_path / "invalid.tar.gz"
    result = run_packager(source_tree, output, epoch=epoch)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize("existing", ["archive", "receipt"])
def test_existing_outputs_are_never_overwritten(source_tree, tmp_path, existing):
    output = tmp_path / "source.tar.gz"
    target = output if existing == "archive" else output.with_suffix(".gz.manifest.json")
    target.write_text("existing artifact")
    result = run_packager(source_tree, output)
    assert result.returncode != 0
    assert "fresh artifact path" in result.stderr
    assert target.read_text() == "existing artifact"


def test_reference_gate_checks_packaged_set_before_writing(source_tree, tmp_path):
    # A link cannot pass merely because its private target exists in the checkout.
    (source_tree / "private.json").write_text("{}")
    (source_tree / "README.md").write_text("[private][result]\n\n[result]: private.json\n")
    output = tmp_path / "invalid.tar.gz"
    result = run_packager(source_tree, output)
    assert result.returncode != 0
    assert "README.md:3:" in result.stderr
    assert "private.json" in result.stderr
    assert not output.exists()
    check = subprocess.run([sys.executable, str(ROOT / "scripts/check_release.py"), "--root", str(source_tree)],
                           capture_output=True, text=True, timeout=20)
    assert check.returncode != 0
    assert "private.json" in check.stderr


def test_markdown_reference_syntax_and_examples():
    content = r'''[inline](assets/a.json#section)
[image ![alt](assets/a.json)](assets/a.json)
[nested [label]](assets/a.json "title")
[space](<assets/with space.json> 'title')
[encoded](assets/with%20space.json?download=1#section)
[parens](assets/a(b).json)
[escaped](assets/a\(b\).json)
[multiline](
  assets/a.json
  "title"
)
[named][ Public DATA ] [public data][] [public data]
[public data]:
  <assets/a.json> "title"
<img src="assets/a.json"><a href="assets/a.json">HTML</a>
[external](https://example.invalid/missing) [mail](mailto:hello@example.invalid)
[fragment](#heading) [directory](assets/)
`[inline example](missing.json)`
``[example `nested`](missing.json)``
<!-- [hidden](missing.json) -->
```md
[fenced](missing.json)
```
~~~markdown
[tilde fenced][undefined]
~~~
    [nested list continuation](assets/a.json)
'''
    sources = {"README.md": content.encode(), "assets/a.json": b"{}",
               "assets/with space.json": b"{}", "assets/a(b).json": b"{}"}
    checker.validate_references(sources)
    links, undefined = checker.markdown_references(content)
    assert not undefined
    assert len(links) >= 20
    assert all(target != "missing.json" for _, target in links)


@pytest.mark.parametrize("content, expected", [
    ("[inline](gone.json)", "gone.json"),
    ("![image](gone.svg)", "gone.svg"),
    ("[full][asset]\n[asset]: gone.csv", "gone.csv"),
    ("[collapsed][]\n[collapsed]: gone.html", "gone.html"),
    ("[shortcut]\n[shortcut]: gone.json", "gone.json"),
    ("[missing][undefined]", "undefined reference"),
    ("[missing][]", "undefined reference"),
    ('<a href="gone.json">link</a>', "gone.json"),
    ("[escape](../outside.json)", "outside.json"),
    ("[absolute](/tmp/local.json)", "nonportable"),
])
def test_broken_markdown_links_are_reported(content, expected):
    with pytest.raises(ValueError, match=expected):
        checker.validate_references({"README.md": content.encode()})


def test_public_release_preserves_annotation_runtime_and_rebuilds(tmp_path):
    """Exercise the real release manifest; integrated documentation must pass too."""
    output = tmp_path / "public.tar.gz"
    result = run_packager(ROOT, output)
    assert result.returncode == 0, result.stderr
    with tarfile.open(output) as archive:
        archive.extractall(tmp_path / "public", filter="data")
    unpacked = tmp_path / "public/grana"
    for name in ("README.md", "LICENSE", "NOTICE", "CONTRIBUTING.md", "SECURITY.md",
                 "THIRD_PARTY_LICENSES.md", "THIRD_PARTY_NOTICES.md", "scripts/setup_env.sh",
                 "scripts/check_env.py", "requirements/annotation.txt", "requirements/labeling-cpu.txt",
                 "docs/installation.md", "docs/project-configuration.md",
                 "tests/fixtures/evaluation_legacy_a05e497.json"):
        assert (unpacked / name).read_bytes() == (ROOT / name).read_bytes()
    package_data = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]
    runtime_assets = ["grana/annotation/" + name for name in package_data["grana.annotation"]]
    runtime_assets.extend("skills/grana-robot-annotation/" + name for name in (
        "SKILL.md", "agents/openai.yaml", "references/annotation-policy.md", "references/api-workflow.md",
        "references/causal-aq-policy.json", "references/evidence-review.md", "references/learning-from-examples.md",
        "references/output-contract.md", "references/quality-improvement.md", "references/stage-annotation.md",
        "references/stage-task-cards.md", "references/synthetic-candidate.json", "references/synthetic-context.json",
        "references/synthetic-settings.json", "references/task-rule-hypotheses.md",
    ))
    for name in runtime_assets:
        assert (unpacked / name).read_bytes() == (ROOT / name).read_bytes()
    stage_bundle, quality_bundle = tmp_path / "stages.json", tmp_path / "quality.json"
    stage_candidate = tmp_path / "stage-candidate.json"
    fixture = json.loads((unpacked / "skills/grana-robot-annotation/references/synthetic-candidate.json").read_text())
    stage_candidate.write_text(json.dumps({"stages": fixture["stages"]}))
    for args in (
        ["prepare", "--context", str(unpacked / "skills/grana-robot-annotation/references/synthetic-context.json"),
         "--output", str(stage_bundle)],
        ["prepare-quality", "--stage-bundle", str(stage_bundle), "--stage-candidate", str(stage_candidate),
         "--output", str(quality_bundle)],
    ):
        check = subprocess.run([sys.executable, str(unpacked / "scripts/annotation_skill.py"), *args],
                               cwd=tmp_path, capture_output=True, text=True, timeout=20)
        assert check.returncode == 0, check.stderr
    assert json.loads(stage_bundle.read_text())["mode"] == "stages"
    assert json.loads(quality_bundle.read_text())["stage_lock"]["candidate"] == {"stages": fixture["stages"]}
    for name in ("docs/private.md", "configs/private.yaml", "scripts/private.py", "configs/private.json"):
        (unpacked / name).write_text("synthetic private sentinel\n")
    rebuilt = tmp_path / "public-rebuilt.tar.gz"
    result = run_packager(unpacked, rebuilt)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == rebuilt.read_bytes()
