"""Run Python inside a live Blender via the Blender Lab MCP add-on socket.

Protocol (see blender_mcp/addon/blender_mcp_addon/mcp_to_blender_server.py):
  request : {"type": "execute", "code": str, "strict_json": bool} + b"\\0"
  response: JSON + b"\\0"
Note the null-byte terminator -- newline is NOT the delimiter. The executed
code must assign a JSON-serializable dict to a variable named ``result``;
Blender objects are not serializable, so send names and numbers, not objects.

Usage:
    python tools/blender_bridge.py [--port N] < code.py
    BLENDER_MCP_PORT=9877 python tools/blender_bridge.py < code.py

Each Blender process serves one port, so parallel agents should each own a
port and launch their own instance with tools/launch_blender.py. That keeps
scene contents, sys.path and registered classes fully separate -- sharing one
instance between agents caused crashes and false failures.
"""

import argparse
import json
import os
import socket
import sys

DEFAULT_PORT = int(os.environ.get("BLENDER_MCP_PORT", "9876"))


class NotASandboxInstance(RuntimeError):
    """Raised instead of sending code to a Blender we did not start."""


# One probe per (host, port) per process. The probe is a round trip and the
# answer cannot change for the life of an instance -- the marker lives in its
# environment, which is fixed at spawn.
_MARKER_CACHE = {}

_MARKER_PROBE = ("import os\n"
                 "result = {'marker': os.environ.get('FOAMCRAFT_TEST_INSTANCE')}\n")


def sandbox_marker(port=DEFAULT_PORT, host="localhost", refresh=False):
    """The FOAMCRAFT_TEST_INSTANCE of the instance on `port`, or None.

    Reading an environment variable cannot damage a scene, so this one snippet
    is sent unguarded; it is the handshake that decides whether anything else
    may be.
    """
    key = (host, port)
    if not refresh and key in _MARKER_CACHE:
        return _MARKER_CACHE[key]
    try:
        reply = _send(_MARKER_PROBE, port, host, False, 10)
    except Exception:
        reply = None
    marker = None
    if isinstance(reply, dict) and reply.get("status") == "ok":
        marker = (reply.get("result") or {}).get("marker")
    _MARKER_CACHE[key] = marker
    return marker


def _send(code, port, host, strict_json, timeout):
    """The raw round trip, with no ownership check. Internal."""
    request = json.dumps({"type": "execute", "code": code,
                          "strict_json": strict_json}).encode("utf-8") + b"\0"
    conn = socket.create_connection((host, port), timeout=timeout)
    conn.settimeout(timeout)
    try:
        conn.sendall(request)
        buffer = bytearray()
        while b"\0" not in buffer:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buffer.extend(chunk)
    finally:
        conn.close()
    if b"\0" in buffer:
        buffer = buffer[:buffer.index(b"\0")]
    return json.loads(buffer.decode("utf-8"))


def execute(code, port=DEFAULT_PORT, host="localhost", strict_json=False,
            timeout=300, require_sandbox=True):
    """Send `code` to Blender and return the decoded response dict.

    Refuses unless the instance proves it was started by
    tools/launch_blender.py. The suite's scripts clear scenes, bake over
    objects and quit Blender, and on 2026-11-03 they did all of that inside a
    user's own session because the port happened to answer -- see
    tools/sandbox_guard.py for the account.

    Checking here rather than only in the callers is deliberate: this is the
    one function every path goes through, so a new script cannot forget to ask
    permission. Pass require_sandbox=False only to read state you are certain
    cannot change anything.
    """
    if require_sandbox and sandbox_marker(port, host) is None:
        raise NotASandboxInstance(
            "REFUSING to send code to port %d.\n"
            "  That Blender was not started by tools/launch_blender.py, so it\n"
            "  is somebody's working session. This suite clears scenes and\n"
            "  quits Blender; running against it can destroy unsaved work.\n"
            "  Start your own instance:\n"
            "      python tools/launch_blender.py --port <free port>"
            % port)
    return _send(code, port, host, strict_json, timeout)


def _execute_unguarded(code, port=DEFAULT_PORT, host="localhost",
                       strict_json=False, timeout=300):
    """Escape hatch, named so it cannot be used by accident."""
    return _send(code, port, host, strict_json, timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()

    code = sys.stdin.read()
    try:
        data = execute(code, port=args.port, host=args.host)
    except NotASandboxInstance as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"CONNECT_ERROR on port {args.port}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2

    if data.get("stdout"):
        print(data["stdout"], end="")
    if data.get("stderr"):
        print("STDERR:", data["stderr"], file=sys.stderr)
    if data.get("status") == "ok":
        result = data.get("result")
        if result:
            print("RESULT:", json.dumps(result, indent=2))
        return 0
    print("BLENDER_ERROR:", data.get("message", json.dumps(data)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
