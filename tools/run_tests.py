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
    "tests/test_determinism.py",
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
    "tests/test_overlay_gui.py",
)


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


def serving(port):
    try:
        blender_bridge.execute("result = {'ok': True}", port=port)
        return True
    except Exception:                                  # noqa: BLE001
        return False


def start(port, gui):
    cmd = [sys.executable, str(LAUNCHER), "--port", str(port)]
    if gui:
        cmd.append("--gui")
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop(port):
    subprocess.run([sys.executable, str(LAUNCHER), "--port", str(port),
                    "--stop"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9885)
    parser.add_argument("--only", help="run just one file by substring")
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
            start(args.port, args.gui)
            started = True
        for path in files:
            print("\n=== %s" % path)
            counted, failed = report_on(path, run_file(path, args.port))
            total += counted
            failures += failed
        if started and not args.keep:
            stop(args.port)
    else:
        for index, path in enumerate(files):
            port = args.port + index
            print("\n=== %s   (fresh blender on %d)" % (path, port))
            stop(port)                 # in case one was left behind
            start(port, args.gui)
            try:
                counted, failed = report_on(path, run_file(path, port))
            finally:
                stop(port)
            total += counted
            failures += failed

    print("\n%d test(s), %d failure(s) in %.0fs"
          % (total, failures, time.time() - started_at))

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
