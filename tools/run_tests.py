"""Run the FoamCraft test suite against a live Blender and print a summary.

    python tools/run_tests.py                  # start Blender, run, stop it
    python tools/run_tests.py --port 9885      # reuse one already serving
    python tools/run_tests.py --keep           # leave Blender running after

Each test file assigns a ``result`` dict of {test_name: {..., "checks": {...},
"PASS": bool}} plus an ``ALL_PASS``. This runner sends each file through the
bridge and collects those, so the files stay the specification and this stays
plumbing.

Exits non-zero if anything failed, so it can gate a commit.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import blender_bridge  # noqa: E402

# Explicit rather than globbed: a file that fails to import should be a visible
# failure, not a silently skipped test.
TEST_FILES = (
    "tests/test_gp_pattern.py",
    "tests/test_curvature_darts.py",
    "tests/test_bake_pattern.py",
)


def run_file(path, port):
    code = (ROOT / path).read_text(encoding="utf-8")
    try:
        response = blender_bridge.execute(code, port=port)
    except Exception as exc:                       # noqa: BLE001
        return {"__error__": "%s: %s" % (type(exc).__name__, exc)}
    if not isinstance(response, dict):
        return {"__error__": "unexpected response: %r" % (response,)}
    if "result" in response:
        return response["result"]
    return {"__error__": response.get("message") or str(response)[:400]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9885)
    parser.add_argument("--keep", action="store_true",
                        help="leave Blender running when finished")
    parser.add_argument("--only", help="run just one file by substring")
    parser.add_argument("--gui", action="store_true",
                        help="start Blender with a window, so tests that can "
                             "capture screenshots do. Headless runs skip them")
    args = parser.parse_args()

    launcher = ROOT / "tools" / "launch_blender.py"
    started = False
    try:
        blender_bridge.execute("result = {'ok': True}", port=args.port)
    except Exception:                              # noqa: BLE001
        print("starting blender on port %d ..." % args.port)
        cmd = [sys.executable, str(launcher), "--port", str(args.port)]
        if args.gui:
            cmd.append("--gui")
        subprocess.run(cmd, check=True)
        started = True

    files = [f for f in TEST_FILES
             if not args.only or args.only in f]
    failures = 0
    total = 0
    for path in files:
        print("\n=== %s" % path)
        report = run_file(path, args.port)
        if "__error__" in report:
            print("  ERROR: %s" % report["__error__"])
            failures += 1
            continue
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

    print("\n%d test(s), %d failure(s)" % (total, failures))
    if started and not args.keep:
        subprocess.run([sys.executable, str(launcher), "--port",
                        str(args.port), "--stop"], check=False)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
