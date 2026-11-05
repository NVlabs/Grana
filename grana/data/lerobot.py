# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read LeRobot v2.x and v3.0 datasets through one interface.

v2 stores one parquet file and one video per camera for each episode. v3 packs
many episodes into shared parquet and video files and records, for each
episode, the data file plus each camera's file and time range in
``meta/episodes/``. Both expose the same per-episode rows and video segments.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Integral, Real
from pathlib import Path


@dataclass(frozen=True)
class VideoSegment:
    """One camera's frames of one episode: the whole file (v2) or [start, end) seconds of a shared file (v3)."""
    path: Path
    start: float = 0.0
    end: float | None = None

    @property
    def shared(self) -> bool:
        return self.end is not None


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def template_path(info: dict, key: str, **fields) -> str:
    """meta/info.json's ``key`` path template, filled in with ``fields``."""
    template = info.get(key)
    if not isinstance(template, str):
        raise ValueError(f"meta/info.json needs a {key} path template")
    try:
        return template.format(**fields)
    except (LookupError, ValueError) as error:   # a field this LeRobot version does not define, or bad syntax
        names = ", ".join(f"{{{name}}}" for name in fields)
        raise ValueError(f"meta/info.json {key} may use only {names}: {template}") from error


class LeRobotDataset:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        version = str(self.info.get("codebase_version", ""))
        if version.startswith("v2."):
            self.version = 2
        elif version.startswith("v3."):
            self.version = 3
        else:
            raise ValueError(f"LeRobot dataset version {version or '(missing)'} is not supported; use v2.x or v3.0")
        fps = self.info.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, Real) or not fps > 0:
            raise ValueError("meta/info.json fps must be positive")
        self.fps = float(fps)
        self.chunks_size = _integer(self.info.get("chunks_size"), "chunks_size")
        if self.chunks_size < 1:
            raise ValueError("chunks_size must be positive")
        self.episodes = self._read_episodes()
        self._tasks = None

    # ---------- metadata ----------
    def _read_episodes(self) -> dict[int, dict]:
        rows = []
        if self.version == 2:
            for line in (self.root / "meta/episodes.jsonl").read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        else:
            import pyarrow.parquet as parquet

            for path in self._episode_files():
                # Skip the per-episode statistics: most of the bytes, never used here.
                keep = [name for name in parquet.read_schema(path).names if not name.startswith("stats/")]
                rows.extend(parquet.read_table(path, columns=keep).to_pylist())
        episodes = {}
        for row in rows:
            index = _integer(row.get("episode_index"), "episode_index")
            if index < 0 or index in episodes:
                raise ValueError("Episode metadata needs unique nonnegative episode_index values")
            episodes[index] = row
        return episodes

    def _episode_files(self) -> list[Path]:
        files = sorted((self.root / "meta/episodes").glob("chunk-*/file-*.parquet"))
        if not files:
            raise ValueError("LeRobot v3 metadata is missing meta/episodes/chunk-*/file-*.parquet")
        return files

    def metadata_files(self) -> list[Path]:
        if self.version == 2:
            names = [self.root / "meta" / name for name in ("info.json", "episodes.jsonl", "tasks.jsonl")]
            return [path for path in names if path.is_file()]
        return [self.root / "meta/info.json", self.root / "meta/tasks.parquet", *self._episode_files()]

    @property
    def tasks(self) -> dict[int, str]:
        """task_index -> task text (annotation columns store these indices)."""
        if self._tasks is None:
            tasks = {}
            if self.version == 2:
                for line in (self.root / "meta/tasks.jsonl").read_text().splitlines():
                    if line.strip():
                        row = json.loads(line)
                        index = _integer(row["task_index"], "task_index")
                        if index in tasks:
                            raise ValueError("Task indices must be unique")
                        tasks[index] = row["task"]
            else:
                import pyarrow.parquet as parquet

                for row in parquet.read_table(self.root / "meta/tasks.parquet").to_pylist():
                    text = row.get("task", row.get("__index_level_0__"))
                    index = _integer(row["task_index"], "task_index")
                    if index in tasks:
                        raise ValueError("Task indices must be unique")
                    tasks[index] = text
            self._tasks = tasks
        return self._tasks

    def length(self, index: int) -> int:
        length = _integer(self.episodes[index].get("length"), "episode length")
        if length < 1:
            raise ValueError("Episode length must be positive")
        return length

    def trajectory_id(self, index: int) -> str:
        identity = self.episodes[index].get("trajectory_id")
        return identity if isinstance(identity, str) and identity.strip() else f"{self.root.name}__episode_{index:06d}"

    # ---------- files ----------
    def _file(self, key: str, **fields) -> Path:
        path = (self.root / template_path(self.info, key, **fields)).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"LeRobot path must resolve to an existing file inside the dataset: {path}")
        return path

    def _location(self, index: int, column: str) -> int:
        row = self.episodes[index]
        if column not in row:
            raise ValueError(f"meta/episodes has no {column} for episode {index}")
        return _integer(row[column], column)

    def data_path(self, index: int) -> Path:
        if self.version == 2:
            return self._file("data_path", episode_index=index, episode_chunk=index // self.chunks_size)
        return self._file("data_path", chunk_index=self._location(index, "data/chunk_index"),
                          file_index=self._location(index, "data/file_index"))

    def column_names(self, index: int) -> list[str]:
        import pyarrow.parquet as parquet

        return parquet.read_schema(self.data_path(index)).names

    def rows(self, index: int, columns: list[str]) -> dict[str, list]:
        """This episode's rows of the given columns, in frame order."""
        import pyarrow.parquet as parquet

        path = self.data_path(index)
        if self.version == 2:
            return parquet.read_table(path, columns=columns).to_pydict()
        wanted = list(dict.fromkeys([*columns, "episode_index", "frame_index"]))
        table = parquet.read_table(path, columns=wanted, filters=[("episode_index", "=", index)])
        table = table.sort_by("frame_index")
        return table.select(columns).to_pydict()

    def video(self, index: int, video_key: str) -> VideoSegment:
        row = self.episodes[index]
        if self.version == 2:
            path = self._file("video_path", video_key=video_key, episode_index=index,
                              episode_chunk=index // self.chunks_size)
            return VideoSegment(path)
        prefix = f"videos/{video_key}/"
        try:
            chunk, file = (_integer(row[prefix + key], prefix + key) for key in ("chunk_index", "file_index"))
            start, end = (float(row[prefix + key]) for key in ("from_timestamp", "to_timestamp"))
        except (KeyError, TypeError) as error:   # absent, or null
            raise ValueError(f"Episode {index} has no video location for {video_key}") from error
        if not 0 <= start < end:
            raise ValueError(f"Episode {index} has an invalid time range for {video_key}")
        return VideoSegment(self._file("video_path", video_key=video_key, chunk_index=chunk, file_index=file), start, end)
