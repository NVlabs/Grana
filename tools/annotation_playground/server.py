# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded loopback home page; launch under an observable service manager."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .runner import Playground, Refused, note

WEB = Path(__file__).resolve().parent / "web"
# The label pages' logo, so the home page wears the same header.
LOGO = Path(__file__).resolve().parents[2] / "grana" / "annotation" / "label_ui" / "grana-mark.svg"
# Largest JSON body per route.
BODIES = {"/api/generate": 64 * 1024, "/api/quality": 4096, "/api/import": 4096, "/api/import-all": 16 * 1024, "/api/seeded/choose": 256 * 1024,
          "/api/seeded/example": 4096, "/api/seeded/compare": 4096}


def exactly(body, *names):
    if set(body) != set(names):
        raise ValueError("Expected exactly: " + ", ".join(names) if names else "Expected an empty JSON object")
    return body


class Handler(BaseHTTPRequestHandler):
    server_version = "AnnotationPlayground/2"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *_args):
        pass

    def _send(self, status, value, *, content_type=None):
        data = value if content_type else json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type or "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(data)

    def _allowed(self, *, write=False):
        """Loopback Host and Origin only; changes also only from this page itself (or a non-browser client)."""
        from grana._loopback import local_request
        site = self.headers.get("Sec-Fetch-Site", "none")
        if not local_request(self.headers):
            return False
        if not write:
            return site in ("none", "same-origin", "same-site")
        origin = self.headers.get("Origin")
        return site in ("none", "same-origin") and (origin is None or urlsplit(origin).port == self.server.server_address[1])

    def do_GET(self):
        if not self._allowed():
            return self._send(403, {"error": "Only local browser requests are allowed."})
        url = urlsplit(self.path)
        path, playground = url.path, self.server.playground
        if path == "/api/status":
            return self._send(200, playground.status())
        if path == "/api/seeded/compare-detail":
            query = parse_qs(url.query)
            try:
                if set(query) != {"compare", "episode"} or any(len(values) != 1 for values in query.values()):
                    raise ValueError("Expected compare and episode")
                episode = query["episode"][0]
                return self._send(200, playground.compare_detail(query["compare"][0], int(episode) if episode.isdigit() else None))
            except Refused as refused:
                return self._send(409, {"error": str(refused), "note": note(refused)})
            except (ValueError, KeyError, OSError) as error:
                return self._send(400, {"error": " ".join(str(error).split()) or "Bad request.", "note": note(error)})
        if path in ("/", "/index.html"):
            try:
                data = (WEB / "index.html").read_bytes()
            except OSError:
                return self._send(503, {"error": "Dashboard page is being prepared."})
            return self._send(200, data, content_type="text/html; charset=utf-8")
        if path == "/grana-mark.svg":
            return self._send(200, LOGO.read_bytes(), content_type="image/svg+xml")
        return self._send(404, {"error": "Not found."})

    def _body(self, limit):
        if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            raise ValueError("Send a bounded application/json body.")
        if len(self.headers.get_all("Content-Length", [])) != 1:
            raise ValueError("Send one Content-Length.")
        size = int(self.headers["Content-Length"])
        if not 1 <= size <= limit:
            raise ValueError("The request body is empty or too large.")
        data = self.rfile.read(size)
        if len(data) != size:
            raise ValueError("The request body is incomplete.")
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key.")
                result[key] = value
            return result
        body = json.loads(data, object_pairs_hook=unique)
        if not isinstance(body, dict):
            raise ValueError("Send a JSON object.")
        return body

    def do_POST(self):
        if not self._allowed(write=True):
            return self._send(403, {"error": "Only this local page may change the playground."})
        path = urlsplit(self.path).path
        if path not in BODIES:
            return self._send(404, {"error": "Not found."})
        playground = self.server.playground
        try:
            body = self._body(BODIES[path])
            fields = lambda *names: exactly(body, *names)
            if path == "/api/generate":
                with_quality = body.pop("with_quality", False)
                result = playground.start(**fields("route", "episodes", "model"), with_quality=with_quality)
            elif path == "/api/quality":
                result = playground.add_quality(**fields("route", "episode", "model"))
            elif path == "/api/import":
                result = playground.import_episode(**fields("route", "episode", "model"))
            elif path == "/api/import-all":
                if (body.get("action") not in ("preview", "confirm", "cancel")
                        or not set(body) <= {"route", "action", "model", "folder", "preview"}):
                    raise ValueError("Expected route, action preview, confirm or cancel, and optionally model, folder or preview")
                result = playground.import_all(body.get("route"), body["action"], model=body.get("model"),
                                               folder=body.get("folder"), preview=body.get("preview"))
            elif path == "/api/seeded/choose":
                result = playground.choose(**fields("seeds", "held_out"))
            elif path == "/api/seeded/example":
                fields()
                result = playground.build_example()
            else:
                result = playground.compare(**fields("model"))
        except Refused as refused:   # the page words the note in its language
            return self._send(409, {"accepted": False, "error": str(refused), "note": note(refused)})
        except (ValueError, TypeError, KeyError, UnicodeError, OSError) as error:
            return self._send(400, {"accepted": False, "error": " ".join(str(error).split()) or "Bad request.",
                                    "note": note(error)})
        # Accepted work runs in the background; its progress appears in /api/status.
        return self._send(202 if result.get("accepted") else 200, result)


@contextmanager
def running(root, *, transport=None, port=None):
    """The home page's HTTP server for one workspace (on its configured port unless ``port`` is given), holding its
    service lock; serve_forever runs it."""
    root = Path(root).resolve()
    with (root / ".dashboard-service.lock").open("a+") as lease:
        os.chmod(root / ".dashboard-service.lock", 0o600)
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("The playground dashboard is already running.")
        playground = Playground(root, recover=True, transport=transport)
        server = ThreadingHTTPServer(("127.0.0.1", playground.config["ports"]["overview"] if port is None else port), Handler)
        server.daemon_threads = True
        server.playground = playground
        try:
            yield server
        finally:
            server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with running(args.root) as server:
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
