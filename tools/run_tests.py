"""Run the FoamCraft test suite against a live Blender and print a summary.

    python tools/run_tests.py                  # a fresh Blender per test file
    python tools/run_tests.py --only theta     # one file
    python tools/run_tests.py --headless       # no window; skips what needs one
    python tools/run_tests.py --share          # one instance for the lot

Blender is started **with a window** by default. This is a tool for drawing on
a sculpt and looking at the result, so the parts that only exist on screen --
the overlay, the captures -- are not an optional extra to be checked when
someone remembers a flag. Running headless silently skipped them, and a check
that quietly does not run is the kind this project has been bitten by.

Each test file assigns a ``result`` dict of {test_name: {..., "checks": {...},
"PASS": bool}} plus an ``ALL_PASS``. This runner sends each file through the
bridge and collects those, so the files stay the specification and this stays
plumbing.

EVERY FILE GETS ITS OWN BLENDER

Sharing one instance made the suite lie. Tests passed alone and failed after
others, and the same test flipped both ways in a session with no change to the
code under test -- so a green run was not evidence, which is worse than a red
one. Three causes, none of them fixable from inside a test:

  Leftover datablocks rename what comes next. A stale object called Sculpt
  makes the next one Sculpt.001, and anything looking for a name finds the
  wrong thing or nothing.

  Test files purge foam_craft from sys.modules to pick up edits, which orphans
  the add-on's registration -- the draw handler keeps running from a module
  object nothing can reach any more.

  Scene settings persist. One file's choice of stroke object is the next
  file's starting state.

A process boundary fixes all three at once and cannot be forgotten. It costs a
Blender launch per file, which is the price of the result meaning something.
--share restores the old behaviour for a quick loop while writing a test.

Exits non-zero if anything failed, so it can gate a commit.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import blender_bridge  # noqa: E402

# Explicit rather than globbed: a file that fails to import should be a visible
# failure, not a silently skipped test.
#
# The list is the weak point in that argument, and it has already failed once:
# test_cut_accuracy, test_gp_junctions and test_parallel_waves existed for
# weeks without ever being added here, so they never ran and the suite
# reported green without them. A globbed list cannot forget a file; an
# explicit one can, and did. Keeping it explicit for the stated reason means
# accepting that adding a test file is a two-step job -- write it, then list
# it -- so check this tuple whenever tests/ gains a file.
TEST_FILES = (
    # First, because it is the one that says whether the rest can be believed.
    # First of all, because it is what stops this suite destroying
    # someone's work. If the guard is broken, nothing else should run.
    "tests/test_sandbox_guard.py",
    "tests/test_determinism.py",
    # The property every other result rests on: one mesh, one
    # pattern, every time.
    "tests/test_reproducible_export.py",
    # Second, because everything else is built on this primitive.
    "tests/test_edgenet_cut.py",
    "tests/test_gp_pattern.py",
    "tests/test_gp_junctions.py",
    "tests/test_cut_accuracy.py",
    "tests/test_parallel_waves.py",
    "tests/test_curvature_darts.py",
    "tests/test_bake_pattern.py",
    "tests/test_region_pattern.py",
    "tests/test_shape_sweep.py",
    "tests/test_export_svg.py",
    # The model the SVG and raster backends both draw from.
    "tests/test_marks_model.py",
    # Bevel labels slid, shrunk or led out so none overlap (foam_layout).
    "tests/test_label_layout.py",
    "tests/test_print_panel.py",
    # Pure arithmetic, no scene -- it also runs standalone with
    #     python tests/test_tile_svg.py
    "tests/test_tile_svg.py",
    "tests/test_export_raster.py",
    # Several objects on one sheet (Export UV Layout's rule), piece names
    # and seam numbers continuing across them.
    "tests/test_multi_object.py",
    # The screen shows the registration ticks the sheet prints.
    "tests/test_overlay_ticks.py",
    "tests/test_overlay_gui.py",
    # Every setting rebuilds the overlay when changed (the Convex/Concave
    # Colour toggle did not).
    "tests/test_settings_refresh.py",
)


# Files that genuinely need a window, and why.
#
# A background Blender has no window, so draw handlers never run and
# screenshots come back blank -- which is why the suite used to start a
# windowed Blender for every file. But only these need one, and starting
# fifteen windows to serve one is both slower and, far worse, fifteen
# chances to steal the keyboard out from under whoever is working on the
# machine. The rest report numbers and run headless, where they are also
# quicker to start.
#
# Each of the others guards itself: they check `bpy.app.background` and skip
# their captures rather than failing, so running the whole suite windowed
# with --gui still works and still writes every screenshot.
GUI_FILES = frozenset({
    # Compares a capture of the UV editor with the overlay on against one
    # with it off; there is nothing to compare without a window.
    "tests/test_overlay_gui.py",
    # Toggles Convex/Concave Colour and runs the overlay's timer tick, which
    # needs the live window's UV editor to rebuild for. Its other two cases
    # are data-only and run either way.
    "tests/test_settings_refresh.py",
    # NOT tests/test_export_raster.py, though it draws into a GPUOffScreen.
    # A background Blender refuses that at first -- and names the cure in the
    # refusal, gpu.init(). The file probes both and all of it passes headless
    # (measured 2026-11-16), so it does not earn a window.
})


def _unlisted_test_files():
    """Test files on disk that TEST_FILES does not mention.

    The gap above went unnoticed because nothing ever compared the list to the
    directory. This does, and the runner prints the difference rather than
    silently running a subset -- which keeps the explicit list honest without
    giving up the visible-import-failure property that motivates it.
    """
    listed = {f.split("/")[-1] for f in TEST_FILES}
    on_disk = {p.name for p in (ROOT / "tests").glob("test_*.py")}
    return sorted(on_disk - listed)

LAUNCHER = ROOT / "tools" / "launch_blender.py"


def _port_free(port):
    """True when nothing at all is listening on `port`."""
    import socket
    with socket.socket() as probe:
        probe.settimeout(0.3)
        return probe.connect_ex(("localhost", port)) != 0


def free_ports(start, count):
    """`count` ports nobody is listening on, searching upward from `start`.

    The suite used to take start, start+1, start+2 ... unconditionally. When
    one of those happened to be a Blender somebody was working in, the runner
    stopped it and started its own there -- which on 2026-11-03 discarded a
    user's unsaved sculpt. Ports in use belong to someone; step over them.
    """
    found, port, ceiling = [], start, start + 400
    while len(found) < count and port < ceiling:
        if _port_free(port):
            found.append(port)
        port += 1
    if len(found) < count:
        raise SystemExit(
            "could not find %d free ports from %d upward" % (count, start))
    return found


def serving(port):
    try:
        blender_bridge.execute("result = {'ok': True}", port=port)
        return True
    except Exception:                                  # noqa: BLE001
        return False


def start(port, gui):
    cmd = [sys.executable, str(LAUNCHER), "--port", str(port)]
    if gui:
        # Off every monitor. A window that takes the keyboard out from under
        # whoever is at the machine is not an acceptable cost of running the
        # suite, and an off-screen one draws identically -- measured, see
        # launch_blender.py's --offscreen.
        cmd += ["--gui", "--offscreen"]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop(port):
    """Stop the instance on `port`, and say so when it did not work.

    stderr used to go to DEVNULL. A run that failed to stop all 13 of its
    instances therefore looked exactly like a clean one, and the windows piled
    up until someone noticed the taskbar.
    """
    proc = subprocess.run([sys.executable, str(LAUNCHER), "--port", str(port),
                           "--stop"], check=False,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or
                         "failed to stop instance on %d\n" % port)
        _LEAKED.append(port)
    return proc.returncode


_LEAKED = []


def run_file(path, port):
    code = (ROOT / path).read_text(encoding="utf-8")
    try:
        response = blender_bridge.execute(code, port=port)
    except Exception as exc:                           # noqa: BLE001
        return {"__error__": "%s: %s" % (type(exc).__name__, exc)}
    if not isinstance(response, dict):
        return {"__error__": "unexpected response: %r" % (response,)}
    if "result" in response:
        return response["result"]
    return {"__error__": response.get("message") or str(response)[:400]}


def report_on(path, report):
    """Print one file's results. Returns (tests, failures)."""
    if "__error__" in report:
        print("  ERROR: %s" % report["__error__"])
        return 0, 1
    total = failures = 0
    for name, entry in report.items():
        if name == "ALL_PASS" or not isinstance(entry, dict):
            continue
        total += 1
        passed = entry.get("PASS")
        print("  %-46s %s" % (name, "pass" if passed else "FAIL"))
        if not passed:
            failures += 1
            for check, ok in (entry.get("checks") or {}).items():
                if not ok:
                    print("      failed check: %s" % check)
            for line in entry.get("ERROR") or []:
                print("      %s" % line)
    return total, failures


def main():
    # Labels carry U+2192/U+2190 arrows. Redirected to a file on Windows,
    # stdout is the ANSI code page, which cannot encode them, and printing a
    # report crashed the run. Replace rather than fail: a mangled arrow in
    # a log line is harmless, a dead test run is not.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9885)
    parser.add_argument("--only", help="run just one file by substring")
    parser.add_argument("--gui-all", dest="force_gui", action="store_true",
                        help="open a window for every file, not just the ones "
                             "that need one. Writes every screenshot, and "
                             "takes the keyboard fifteen times")
    parser.add_argument("--headless", action="store_true",
                        help="start Blender without a window. Faster, but the "
                             "overlay and screenshot checks cannot run and will "
                             "say so rather than passing")
    parser.add_argument("--share", action="store_true",
                        help="one Blender for every file, the old behaviour. "
                             "Quicker while writing a test, but results depend "
                             "on what ran before them")
    parser.add_argument("--keep", action="store_true",
                        help="with --share, leave Blender running afterwards")
    args = parser.parse_args()
    args.gui = not args.headless

    files = [f for f in TEST_FILES if not args.only or args.only in f]
    if not files:
        print("no test files match %r" % args.only)
        return 1

    total = failures = 0
    started_at = time.time()

    if args.share:
        started = False
        if not serving(args.port):
            print("starting blender on port %d ..." % args.port)
            start(args.port, args.gui and (
                args.force_gui or any(f in GUI_FILES for f in files)))
            started = True
        for path in files:
            print("\n=== %s" % path)
            counted, failed = report_on(path, run_file(path, args.port))
            total += counted
            failures += failed
        if started and not args.keep:
            stop(args.port)
    else:
        # Chosen up front, skipping anything already listening, so the suite
        # never stops or starts on a port that belongs to someone else.
        ports = free_ports(args.port, len(files))
        for index, path in enumerate(files):
            port = ports[index]
            # Windowed only where the file needs it, unless --gui forces it
            # for all of them.
            want_gui = args.gui and (args.force_gui or path in GUI_FILES)
            print("\n=== %s   (fresh blender on %d%s)"
                  % (path, port, ", windowed" if want_gui else ""))
            start(port, want_gui)
            try:
                counted, failed = report_on(path, run_file(path, port))
            finally:
                stop(port)
            total += counted
            failures += failed

    print("\n%d test(s), %d failure(s) in %.0fs"
          % (total, failures, time.time() - started_at))

    if _LEAKED:
        print("\nWARNING: %d Blender instance(s) could not be stopped and are "
              "still running: %s" % (_LEAKED.__len__(),
                                     ", ".join(str(p) for p in _LEAKED)),
              file=sys.stderr)

    # Say so loudly rather than quietly running a subset. A suite that reports
    # green while three of its files were never listed is worse than a red one.
    unlisted = _unlisted_test_files()
    if unlisted and not args.only:
        print("\nWARNING: %d test file(s) on disk are not in TEST_FILES and "
              "did not run:" % len(unlisted))
        for name in unlisted:
            print("    tests/%s" % name)
        print("Add them to TEST_FILES in this file, or delete them.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
