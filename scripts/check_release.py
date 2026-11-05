#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the explicit source allowlist and local Markdown file references.

Usage: python scripts/check_release.py [--root extracted/grana]
No Git metadata, network access or optional dependencies are used. Links are
resolved against the packaged file set, not other files in the checkout.
Inline links/images, reference definitions and full/collapsed/shortcut reference
links, URL escapes, balanced parentheses and HTML href/src attributes are read.
Fenced/inline code and HTML comments are examples, not live references.
Fragments are not checked: renderer-specific heading IDs are outside this file
availability check. Local directory links must contain at least one public file.
"""
from __future__ import annotations

import argparse
from html import unescape
from html.parser import HTMLParser
import posixpath
from pathlib import Path, PurePosixPath
import re
import stat
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "release-files.txt"
REBUILD_FILES = {MANIFEST, "scripts/package_release.py", "scripts/check_release.py"}


def _source_file(root: Path, name: str) -> Path:
    """Reject noncanonical names and symlinks in every component, not just leaves."""
    if (not name or name.strip() != name or "\\" in name or ":" in name
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
            or any(char in name for char in "*?[]")
            or name.startswith("/") or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError(f"Unsafe release path: {name!r}")
    current = root
    for part in PurePosixPath(name).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Source release rejects symlinks: {name}")
    if not stat.S_ISREG(current.stat().st_mode):
        raise ValueError(f"Source release requires a regular file: {name}")
    return current


def read_release(root: Path) -> dict[str, bytes]:
    """Read only explicit, unique manifest entries into a stable source snapshot."""
    root = root.resolve()
    manifest = _source_file(root, MANIFEST).read_bytes()
    names: set[str] = set()
    for line_number, name in enumerate(manifest.decode("utf-8").splitlines(), 1):
        if not name or name.startswith("#"):
            continue
        if name in names:
            raise ValueError(f"{MANIFEST}:{line_number}: duplicate entry {name!r}")
        _source_file(root, name)
        names.add(name)
    missing = REBUILD_FILES - names
    if missing:
        raise ValueError(f"Allowlist must include rebuild inputs: {', '.join(sorted(missing))}")
    return {name: manifest if name == MANIFEST else _source_file(root, name).read_bytes()
            for name in sorted(names)}


def _blank(match: re.Match[str]) -> str:
    return "".join("\n" if char == "\n" else " " for char in match.group())


def _prose(text: str) -> str:
    """Mask examples while preserving line numbers for actionable diagnostics."""
    text = re.sub(r"<!--.*?-->", _blank, text, flags=re.S)
    lines = text.splitlines(keepends=True)
    fence: tuple[str, int] | None = None
    for index, line in enumerate(lines):
        marker = re.match(r" {0,3}(`{3,}|~{3,})(.*)", line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1] and not marker[2].strip():
                fence = None
            lines[index] = "\n" if line.endswith("\n") else ""
        elif marker and not (marker[1][0] == "`" and "`" in marker[2]):
            fence = (marker[1][0], len(marker[1]))
            lines[index] = "\n" if line.endswith("\n") else ""
    text = "".join(lines)
    return re.sub(r"(?<!`)(`+)(?!`)(.*?)(?<!`)\1(?!`)", _blank, text, flags=re.S)


def _unescape(value: str) -> str:
    return unescape(re.sub(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])", r"\1", value))


def _label(value: str) -> str:
    return " ".join(_unescape(value).split()).casefold()


def _destination(text: str, start: int) -> tuple[str, int] | None:
    """Read the Markdown destination, leaving a possible title to the caller."""
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        return None
    if text[index] == "<":
        end = index + 1
        while end < len(text):
            if text[end] == "\\":
                end += 2
                continue
            if text[end] == ">":
                return _unescape(text[index + 1:end]), end + 1
            if text[end] == "\n":
                return None
            end += 1
        return None
    beginning, depth = index, 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char.isspace() or (char == ")" and depth == 0):
            break
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        index += 1
    if depth:
        return None
    return _unescape(text[beginning:index]), index


class _HTMLLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.links.extend((self.getpos()[0], value) for key, value in attrs
                          if key in {"href", "src"} and value is not None)


def markdown_references(text: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return (line, destination) links and undefined explicit reference labels."""
    text = _prose(text)
    links: list[tuple[int, str]] = []
    undefined: list[tuple[int, str]] = []
    definitions: dict[str, str] = {}
    definition_pattern = re.compile(r"^ {0,3}\[((?:\\.|[^\]\\\n])+)\]:[ \t]*(?:\n[ \t]*)?", re.M)
    definition_spans: list[tuple[int, int]] = []
    for match in definition_pattern.finditer(text):
        parsed = _destination(text, match.end())
        if parsed is None:
            continue
        target, end = parsed
        definitions.setdefault(_label(match[1]), target)
        links.append((text.count("\n", 0, match.start()) + 1, target))
        definition_spans.append((match.start(), end))
    for start, end in reversed(definition_spans):
        text = text[:start] + "".join("\n" if c == "\n" else " " for c in text[start:end]) + text[end:]
    # Match bracket pairs with a stack, so linked images and nested labels work.
    stack: list[int] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "[":
            stack.append(index)
        elif char == "]" and stack:
            start = stack.pop()
            label = text[start + 1:index]
            line = text.count("\n", 0, start) + 1
            following = index + 1
            if following < len(text) and text[following] == "(":
                parsed = _destination(text, following + 1)
                if parsed:
                    target, end = parsed
                    # Accept an optional quoted/parenthesized link title.
                    tail = re.match(r'''\s*(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\((?:\\.|[^)\\])*\))?\s*\)''', text[end:])
                    if tail:
                        links.append((line, target))
                        index = end + tail.end() - 1
            else:
                explicit = re.match(r"[ \t]*(?:\n[ \t]*)?\[((?:\\.|[^\]\\\n])*)\]", text[following:])
                key = _label((explicit[1] or label) if explicit else label)
                if key in definitions:
                    links.append((line, definitions[key]))
                elif explicit:
                    undefined.append((line, key))
                if explicit:
                    index = following + explicit.end() - 1
        index += 1
    html = _HTMLLinks()
    html.feed(text)
    links.extend(html.links)
    return links, undefined


def validate_references(sources: dict[str, bytes]) -> None:
    """Fail on unavailable local targets, including files merely present on disk."""
    directories = {"."}
    for name in sources:
        directories.update(str(parent) for parent in PurePosixPath(name).parents)
    problems: list[str] = []
    for name, content in sources.items():
        if not name.lower().endswith(".md"):
            continue
        links, undefined = markdown_references(content.decode("utf-8"))
        problems.extend(f"{name}:{line}: undefined reference [{label}]" for line, label in undefined)
        for line, target in links:
            parsed = urlsplit(target)
            if parsed.scheme.lower() == "file":
                problems.append(f"{name}:{line}: nonportable local reference {target!r}")
                continue
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            local = unquote(parsed.path)
            if local.startswith("/") or "\\" in local:
                problems.append(f"{name}:{line}: nonportable local reference {target!r}")
                continue
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), local))
            if resolved not in sources and resolved not in directories:
                problems.append(f"{name}:{line}: reference {target!r} is not in {MANIFEST} ({resolved})")
    if problems:
        raise ValueError("Broken public Markdown references:\n" + "\n".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        sources = read_release(args.root)
        validate_references(sources)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Release check failed: {exc}\n")
    print(f"Release check passed: {len(sources)} explicit files; local Markdown targets are available.")


if __name__ == "__main__":
    main()
