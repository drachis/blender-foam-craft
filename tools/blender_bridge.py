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


def execute(code, port=DEFAULT_PORT, host="localhost", strict_json=False,
            timeout=300):
    """Send `code` to Blender and return the decoded response dict."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()

    code = sys.stdin.read()
    try:
        data = execute(code, port=args.port, host=args.host)
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
