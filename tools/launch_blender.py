"""Start a Blender instance serving the MCP socket on a chosen port.

One Blender process per port. Give every parallel worker its own port so the
scene, sys.path and registered add-on classes stay separate -- sharing a single
instance between concurrent workers caused hard crashes and false test results.

    python tools/launch_blender.py --port 9877              # headless
    python tools/launch_blender.py --port 9878 --gui        # with a window
    python tools/launch_blender.py --port 9877 --stop

Headless suits anything that inspects data (mesh queries, exported files).
Draw handlers and screenshots need --gui, since background Blender has no
window to render or capture.
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time

# GUI mode has no --command hook, so the add-on's own start operator is called
# instead, after pointing its port preference at the port we want. Going
# through the operator matters: it also registers execute_interactive.run as an
# app timer, and without that timer the socket binds but never services a
# request -- the connection is accepted and then reset.
_GUI_START = """
import bpy
def _start():
    addon = next((k for k in bpy.context.preferences.addons.keys()
                  if k.endswith('mcp')), None)
    if addon is None:
        print('FoamCraft launcher: MCP add-on is not enabled')
        return None
    prefs = bpy.context.preferences.addons[addon].preferences
    # The splash covers the viewport and lands in every screenshot.
    #
    # bpy.ops.wm.splash_close exists as an attribute but is not callable --
    # "could not be found" -- so it was silently doing nothing here. Turning the
    # preference off only affects the next launch, since the splash is already
    # up by the time this runs. Reloading the home file clears it now, and
    # wiping the scene costs nothing in a throwaway test instance that builds
    # its own.
    bpy.context.preferences.view.show_splash = False
    try:
        bpy.ops.wm.read_homefile(use_empty=True)
    except RuntimeError:
        pass
    prefs.host, prefs.port = 'localhost', {port}
    if not getattr(prefs, 'use_online_access', True):
        try:
            prefs.use_online_access = True
        except AttributeError:
            pass
    bpy.ops.blmcp.server_stop()
    bpy.ops.blmcp.server_start()
    return None
# Defer to the first timer tick: preferences and operators are not reliably
# available while --python-expr runs during startup.
bpy.app.timers.register(_start, first_interval=0.5)
"""


def blender_executable():
    found = shutil.which("blender")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
    ):
        if os.path.exists(candidate):
            return candidate
    raise SystemExit("Could not find blender. Put it on PATH.")


def port_open(port, host="localhost"):
    with socket.socket() as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((host, port)) == 0


def wait_for_port(port, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(port):
            return True
        time.sleep(0.5)
    return False


def _pid_on_port(port):
    """PID listening on `port`, or None. Used because a background Blender sits
    in a blocking serve loop and ignores wm.quit_blender()."""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3].upper() == "LISTENING" \
                and parts[1].endswith(f":{port}"):
            return parts[4]
    return None


def stop(port):
    """Shut down the instance on `port`, window and process included."""
    if not port_open(port):
        print(f"nothing listening on {port}")
        return 0

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from blender_bridge import execute
    try:
        execute("import bpy\nbpy.ops.wm.quit_blender()\nresult = {}", port=port,
                timeout=10)
    except OSError:
        pass  # the socket dropping is the expected result of quitting
    time.sleep(1.5)

    if port_open(port):
        # Background instances block inside the serve loop and never process
        # the quit operator, so fall back to terminating the process.
        pid = _pid_on_port(port)
        if pid:
            subprocess.run(["taskkill", "/PID", pid, "/F"],
                           capture_output=True, timeout=20)
            time.sleep(1.0)

    print(f"stopped instance on {port}" if not port_open(port)
          else f"instance on {port} still listening")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gui", action="store_true",
                        help="Show a window. Required for draw handlers and screenshots.")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--blend", default=None, help="Optional .blend to open.")
    args = parser.parse_args()

    if args.stop:
        return stop(args.port)

    if port_open(args.port):
        print(f"port {args.port} already serving; reusing it")
        return 0

    exe = blender_executable()
    if args.gui:
        cmd = [exe]
        if args.blend:
            cmd.append(args.blend)
        cmd += ["--python-expr", _GUI_START.format(port=args.port)]
    else:
        cmd = [exe, "--background"]
        if args.blend:
            cmd.append(args.blend)
        cmd += ["--command", "blender_mcp", "--port", str(args.port)]

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if wait_for_port(args.port):
        print(f"blender serving on port {args.port} ({'gui' if args.gui else 'headless'})")
        return 0
    print(f"timed out waiting for port {args.port}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
