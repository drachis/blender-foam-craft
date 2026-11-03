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
import tempfile
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


# Kept in step with tools/sandbox_guard.ENV_MARKER by this import rather than
# by a second literal, so the two can never drift apart and silently disarm
# the guard.
try:
    from sandbox_guard import ENV_MARKER as SANDBOX_ENV
except ImportError:  # running from another directory
    SANDBOX_ENV = "FOAMCRAFT_TEST_INSTANCE"


def _pidfile(port):
    """Where the PID of the instance on `port` is recorded.

    In the system temp directory rather than the repo, because it is runtime
    state and must not be committed or shipped.
    """
    return os.path.join(tempfile.gettempdir(), "foamcraft_port_%d.pid" % port)


def _record_pid(port, pid):
    try:
        with open(_pidfile(port), "w", encoding="utf-8") as f:
            f.write(str(pid))
    except OSError:
        pass  # netstat fallback still applies


def _recorded_pid(port):
    """The PID this launcher recorded for `port`, if it is still a blender."""
    try:
        with open(_pidfile(port), encoding="utf-8") as f:
            pid = f.read().strip()
    except OSError:
        return None
    if not pid.isdigit():
        return None
    return pid if _is_blender(pid) else None


def _is_blender(pid):
    """True when `pid` is a live blender.exe.

    Checked before anything is killed, so a recycled PID now belonging to
    something else is never terminated.
    """
    try:
        listing = subprocess.run(
            ["tasklist", "/FI", "PID eq %s" % pid, "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "blender.exe" in listing.lower()


def _forget_pid(port):
    try:
        os.remove(_pidfile(port))
    except OSError:
        pass


def port_open(port, host="localhost"):
    with socket.socket() as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((host, port)) == 0


def _marker_on_port(port):
    """The FOAMCRAFT_TEST_INSTANCE value of the Blender on `port`, or None.

    Asks over the MCP bridge, because that is the only channel that reaches
    *inside* the process -- and it is the process's own environment that says
    who started it. Anything that fails (no bridge, a different protocol on
    the port, a timeout) returns None and is therefore treated as somebody
    else's, which is the safe direction.
    """
    try:
        from blender_bridge import execute
    except Exception:
        return None
    code = ("import os\n"
            "result = {'marker': os.environ.get('FOAMCRAFT_TEST_INSTANCE')}\n")
    try:
        # Unguarded on purpose: this *is* the ownership handshake, and
        # reading an environment variable cannot change anything.
        out = execute(code, port=port, timeout=10, require_sandbox=False)
    except Exception:
        return None
    # The bridge returns the whole envelope: {"status", "result", "stdout",
    # "stderr"}. Anything other than a clean ok counts as "cannot prove it",
    # which is the safe answer.
    if isinstance(out, dict) and out.get("status") == "ok":
        return (out.get("result") or {}).get("marker")
    return None


def wait_for_port(port, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(port):
            return True
        time.sleep(0.5)
    return False


def _pid_on_port(port):
    """PID owning `port`, or None.

    Needed because a background Blender sits in a blocking serve loop and
    ignores wm.quit_blender(), so the process has to be terminated.

    The LISTENING row is not a reliable place to read the PID. Measured
    2026-11-03 on a live instance:

        TCP  127.0.0.1:9920  0.0.0.0:0        CLOSED       35088
        TCP  127.0.0.1:9920  127.0.0.1:0      LISTENING    0

    The listener reported PID 0 while the real owner appeared on a CLOSED
    row. Taking the LISTENING row therefore returned "0", `taskkill /PID 0`
    failed quietly, and stop() reported "still listening" and gave up -- which
    is why a full run left 25 Blender windows open.

    So: gather every PID mentioned for this local port, discard 0, and prefer
    one that is actually a blender.exe. Checking the image name is what makes
    this safe to hand to taskkill.
    """
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    candidates = []
    recorded = _recorded_pid(port)
    if recorded:
        candidates.append(recorded)
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[1].endswith(f":{port}"):
            pid = parts[-1]
            if pid.isdigit() and pid != "0" and pid not in candidates:
                candidates.append(pid)
    if not candidates:
        return None

    # Prefer a blender.exe, so a recycled PID belonging to something else is
    # never handed to taskkill.
    try:
        listing = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq blender.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        listing = ""
    blender_pids = set()
    for row in listing.splitlines():
        cells = [c.strip('" ') for c in row.split('","')]
        if len(cells) >= 2 and cells[1].isdigit():
            blender_pids.add(cells[1])

    for pid in candidates:
        if pid in blender_pids:
            return pid
    return None


def stop(port):
    """Shut down the instance on `port`, window and process included.

    Only ours. This function quits Blender and then force-kills the process,
    so pointed at the wrong port it closes somebody's session and discards
    whatever was unsaved in it. run_tests.py calls it on every port it is
    about to use, which is how a shared port turns into lost work.
    """
    if not port_open(port):
        print(f"nothing listening on {port}")
        return 0

    if _marker_on_port(port) is None:
        print(f"REFUSING to stop port {port}: the instance there was not "
              f"started by this launcher, so quitting it could discard "
              f"somebody's unsaved work.", file=sys.stderr)
        return 2

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

    # The port can linger in TIME_WAIT after the process is gone, so judge by
    # the process, not the socket -- otherwise a clean stop reports failure
    # and a failed one can hide behind a draining socket.
    pid_now = _recorded_pid(port) or _pid_on_port(port)
    if pid_now is None:
        _forget_pid(port)
        print(f"stopped instance on {port}")
        return 0

    print(f"FAILED to stop instance on {port}: process {pid_now} is still "
          f"running. It will keep a Blender window open; close it by hand or "
          f"run: taskkill /PID {pid_now} /F", file=sys.stderr)
    return 1


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
        # Something is listening. port_open is a bare TCP connect and cannot
        # tell this suite's Blender from anyone else's, so ask the instance to
        # prove it was started by us before handing tests to it.
        #
        # On 2026-11-03 this branch printed "reusing it" for a user's own
        # Blender, which had the MCP add-on on the same port. The tests then
        # cleared their scene -- see tools/sandbox_guard.py for the full
        # account. Adopting an unknown process was never safe; it only looked
        # safe because the ports usually happened to be ours.
        owner = _marker_on_port(args.port)
        if owner is None:
            print(
                f"REFUSING to use port {args.port}: something is listening "
                f"there, but it was not started by this launcher.\n"
                f"  It is most likely a Blender someone is working in. The "
                f"tests clear the scene, so using it could destroy their "
                f"work.\n"
                f"  Pick a free port, or close that instance first:\n"
                f"      python tools/launch_blender.py --port {args.port + 1}",
                file=sys.stderr)
            return 2
        print(f"port {args.port} already serving this suite; reusing it")
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

    # Show the window without activating it (SW_SHOWNOACTIVATE). The suite
    # starts a fresh Blender per test file, and each one stealing the keyboard
    # made the machine unusable while tests ran. Not minimised and not
    # off-screen, deliberately: a window that is not composited does not draw,
    # and the screenshot tests read what the window drew.
    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 4          # SW_SHOWNOACTIVATE
    # The child's own environment is what proves it is disposable. An
    # environment variable rather than a scene property or a driver-namespace
    # entry because it belongs to the process: it survives read_homefile, file
    # loads and add-on reloads, and a user's Blender cannot pick it up by
    # accident. tools/sandbox_guard.py reads it before anything destructive.
    env = dict(os.environ)
    env[SANDBOX_ENV] = str(args.port)
    child = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             startupinfo=startupinfo, env=env)
    # Written down rather than rediscovered later: netstat's LISTENING row
    # intermittently reports PID 0, and a stop that cannot find the process
    # leaves a window open.
    _record_pid(args.port, child.pid)
    if wait_for_port(args.port):
        print(f"blender serving on port {args.port} ({'gui' if args.gui else 'headless'})")
        return 0
    print(f"timed out waiting for port {args.port}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
