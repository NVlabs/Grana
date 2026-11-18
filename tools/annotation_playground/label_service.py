# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the existing label editor, allowing navigation from this private hub."""
import argparse
from html import escape
import json
from pathlib import Path
import signal
from urllib.request import urlopen
from urllib.parse import urlsplit

from grana.annotation.label_server import LabelHandler, create_label_server, local_request
from grana.data.builder import load_build_config

from .runner import SERVICES, read_json

WEB = Path(__file__).resolve().parent / "web"


class PlaygroundLabelHandler(LabelHandler):
    def do_GET(self):
        if urlsplit(self.path).path == "/playground-status":
            if self._refuse_remote():
                return
            try:
                with urlopen(f"http://127.0.0.1:{self.server.playground_ports['overview']}/api/status", timeout=3) as response:
                    body = response.read(4 * 1024 * 1024)
                return self._send(200, body, "application/json")
            except OSError:
                return self._send(503, b'{"error":"Annotation home is unavailable."}', "application/json")
        return super().do_GET()

    def _send(self, status, body, content_type, extra=()):
        if status == 200 and content_type.startswith("text/html"):
            css = (WEB / "editor-navigation.css").read_bytes()
            script = (WEB / "editor-navigation.js").read_bytes()
            ports = escape(json.dumps(self.server.playground_ports))
            body = body.replace(b"</head>", b"<style>" + css + b"</style></head>", 1)
            body = body.replace(b"<body>", (f'<body data-playground-mode="{self.server.playground_mode}" '
                                            f'data-playground-ports="{ports}">').encode(), 1)
            body = body.replace(b"</body>", b"<script>" + script + b"</script></body>", 1)
        return super()._send(status, body, content_type, extra)

    def _refuse_remote(self):
        ref = urlsplit(self.headers.get("Referer", ""))
        # Only the editor document gets this local navigation allowance.
        # API mutations retain the repository's original origin checks.
        if (self.command == "GET" and urlsplit(self.path).path == "/"
                and local_request(self.headers)
                and self.headers.get("Sec-Fetch-Site") == "same-site"
                and self.headers.get("Sec-Fetch-Mode") == "navigate"
                and (self.headers.get("Sec-Fetch-Dest") == "document"
                     or (self.headers.get("Sec-Fetch-Dest") == "iframe"
                         and ref.scheme == "http" and ref.hostname in ("127.0.0.1", "localhost")
                         and ref.port == self.server.playground_ports["overview"]))):
            return False
        return super()._refuse_remote()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=SERVICES, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = root / (args.mode + ".yaml")
    ports = read_json(root / "run-state.json")["ports"]
    server = create_label_server(load_build_config(config), port=ports[args.mode], config_path=config)
    server.RequestHandlerClass = PlaygroundLabelHandler
    server.playground_mode, server.playground_ports = args.mode, ports

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        print(f"Annotation playground {args.mode}: http://127.0.0.1:{ports[args.mode]}/", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.project.close()
        server.server_close()


if __name__ == "__main__":
    main()
