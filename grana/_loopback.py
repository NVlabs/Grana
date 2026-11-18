# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host and Origin checks shared by the local HTTP servers.

Binding to 127.0.0.1 is not enough on its own: with DNS rebinding a web page on
another site can make the browser send same-origin requests to this port under
the attacker's host name. Requests are therefore accepted only when the Host
header names a loopback address, and a present Origin header must be a loopback
http origin. Any port is accepted, so SSH or editor port forwarding to another
local port keeps working.
"""
import re

_LOOPBACK = re.compile(r"(?:127\.0\.0\.1|localhost|\[::1\])(?::\d{1,5})?")


def allowed_host(header: str | None) -> bool:
    return bool(header) and _LOOPBACK.fullmatch(header.strip().lower()) is not None


def allowed_origin(header: str | None) -> bool:
    if header is None:
        return True
    value = header.strip().lower()
    return value.startswith("http://") and _LOOPBACK.fullmatch(value[len("http://"):]) is not None


def local_request(headers) -> bool:
    """True when Host and (if sent) Origin name a loopback server."""
    return allowed_host(headers.get("Host")) and allowed_origin(headers.get("Origin"))
