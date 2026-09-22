"""Cut a release branch with the development files stripped out, and build it.

    python tools/make_release.py            # next number after the last one
    python tools/make_release.py --number 3 # force 0.03
    python tools/make_release.py --dry-run  # say what it would do

Why a branch per release, rather than one long-lived release branch: a release
is a thing that shipped, and people run it for months after it stops being the
newest. Keeping 0.01 alive means a bug report against 0.01 can be reproduced
against 0.01, and a fix can be cherry-picked back to main without the branch
having drifted underneath it in the meantime. A single rolling branch loses
that the first time it is updated.

The branches are named 0.01, 0.02, ... -- zero-padded so they sort correctly
in every tool that sorts them as text, which is most of them.

WHAT THIS SCRIPT WILL NOT DO

It does not push. main lives in a private repo and releases go to a public
one, and pushing to the public repo is the step that makes the work visible to
the world -- that is a decision, not a build step, so it prints the command
and stops. Run it yourself when you are ready.

WHAT GETS STRIPPED

Everything that exists to develop the add-on rather than to run it: the agent
and process notes, the diagnosis write-ups, the test suite and its tooling,
the Blender source checkout, and the superseded top-level scripts. What
survives is the package, the README, the licence and the images the README
references -- which is what someone installing the add-on has any use for.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = "foamcraft_drachis"
PUBLIC_REMOTE_URL = "https://github.com/drachis/blender-foam-craft.git"

# Paths removed from the release branch. Directories need the trailing slash
# only for readability here; both are handled.
DEV_PATHS = (
    "CLAUDE.md",
    "WORKFLOW.md",
    "tests",
    "tools",
    "blender",
    "blender_mcp",
    "blender_python_reference_5_2",
    "TestModel",
    "screenshots",
    ".claude",
    ".gitmodules",
    # Superseded by the copies inside the package; kept on main as history.
    "uv_fold_angle_overlay.py",
    "export_paper_model_angles.py",
)

# Anything matching these is a dev document too, discovered rather than listed
# so a new investigation does not have to remember to update this file.
DEV_PATTERNS = (re.compile(r"^RESEARCH_.*\.md$"),
                re.compile(r"^DESIGN_.*\.md$"))


def run(*args, **kw):
    """Run a command, raising with its output if it fails."""
    kw.setdefault("cwd", ROOT)
    kw.setdefault("text", True)
    kw.setdefault("capture_output", True)
    proc = subprocess.run(args, **kw)
    if proc.returncode != 0:
        raise SystemExit("command failed: %s\n%s%s"
                         % (" ".join(args), proc.stdout, proc.stderr))
    return proc.stdout.strip()


def git(*args):
    return run("git", *args)


def current_branch():
    return git("rev-parse", "--abbrev-ref", "HEAD")


def tree_is_clean():
    return git("status", "--porcelain") == ""


def existing_release_numbers():
    """Every release branch that exists, local or on either remote."""
    out = git("branch", "--all", "--format=%(refname:short)")
    found = set()
    for line in out.splitlines():
        name = line.strip().split("/")[-1]
        m = re.fullmatch(r"0\.(\d{2})", name)
        if m:
            found.add(int(m.group(1)))
    return found


def next_number():
    used = existing_release_numbers()
    return (max(used) + 1) if used else 1


def tracked_paths():
    """Every path git has under version control, as a set of top-level names.

    Existence on disk is the wrong question. screenshots/ is gitignored, so it
    is present in the working tree and absent from the index -- and `git rm`
    on it fails the whole cut. What can be removed from a branch is what the
    branch contains.
    """
    out = git("ls-files")
    return {line.split("/")[0] for line in out.splitlines() if line}


def dev_paths_present():
    """DEV_PATHS that git tracks, plus whatever matches DEV_PATTERNS.

    Untracked dev clutter needs no action: it is not in the branch, so it
    cannot reach anyone who clones or downloads it.
    """
    tracked = tracked_paths()
    present = [p for p in DEV_PATHS if p in tracked]
    for name in sorted(os.listdir(ROOT)):
        if name in tracked and any(p.match(name) for p in DEV_PATTERNS):
            present.append(name)
    return present


def set_manifest_version(number):
    """Keep the manifest's version in step with the branch name.

    Branch 0.07 ships version 0.7.0. One number, written the two ways their
    respective tools expect, rather than two numbers to keep in step by hand.
    """
    path = os.path.join(ROOT, PACKAGE, "blender_manifest.toml")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # Count the substitutions, not whether the text moved. Cutting 0.01 from a
    # manifest that already reads 0.1.0 is the normal case for a first release,
    # and an unchanged file there means "already correct", not "failed".
    new, count = re.subn(r'^version = "[^"]*"$', 'version = "0.%d.0"' % number,
                         text, count=1, flags=re.M)
    if count != 1:
        raise SystemExit("no version line to rewrite in %s" % path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    return "0.%d.0" % number


def blender_executable():
    for candidate in (
            r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender\blender.exe"):
        if os.path.exists(candidate):
            return candidate
    return shutil.which("blender")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int,
                        help="release number to cut; default is the next unused")
    parser.add_argument("--from-branch", default="main",
                        help="branch to cut from (default: main)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would happen and change nothing")
    args = parser.parse_args()

    number = args.number or next_number()
    if not 1 <= number <= 99:
        raise SystemExit("release numbers run 1..99; got %d" % number)
    branch = "0.%02d" % number

    if number in existing_release_numbers():
        raise SystemExit("branch %s already exists. Releases are permanent -- "
                         "cut the next number instead of reusing one." % branch)

    starting_branch = current_branch()
    if starting_branch != args.from_branch:
        raise SystemExit("on %r, expected %r. Switch first, so the release is "
                         "cut from what you think it is."
                         % (starting_branch, args.from_branch))
    if not tree_is_clean():
        raise SystemExit("working tree is not clean. A release cut from "
                         "uncommitted work cannot be reproduced from the "
                         "branch afterwards.")

    stripping = dev_paths_present()
    untracked_dev = [p for p in DEV_PATHS
                     if p not in stripping
                     and os.path.exists(os.path.join(ROOT, p))]
    print("release branch : %s" % branch)
    print("cut from       : %s (%s)" % (args.from_branch, git("rev-parse", "--short", "HEAD")))
    print("manifest version: 0.%d.0" % number)
    print("stripping      : %s" % ", ".join(stripping))
    if untracked_dev:
        print("already untracked (nothing to do): %s" % ", ".join(untracked_dev))
    if args.dry_run:
        print("\n--dry-run: nothing changed.")
        return 0

    git("checkout", "-b", branch)
    try:
        for path in stripping:
            # --force because removing the submodule directories stages
            # a .gitmodules change, which then blocks .gitmodules' own
            # removal. This is a throwaway branch built from a verified
            # clean tree, so there is nothing here to lose.
            git("rm", "-r", "--quiet", "--force", "--", path)
        version = set_manifest_version(number)
        git("add", "--", "%s/blender_manifest.toml" % PACKAGE)
        git("commit", "-m",
            "Release %s\n\n"
            "Strip the development files: this branch is what gets installed, "
            "so it carries the package, the README, the licence and the images "
            "the README uses, and nothing that only matters while building it."
            "\n\nManifest version %s." % (branch, version))
    except BaseException:
        git("checkout", "--force", starting_branch)
        git("branch", "-D", branch)
        raise

    exe = blender_executable()
    if exe is None:
        print("\nBlender not found; skipping the zip build.")
    else:
        out_dir = os.path.join(ROOT, "dist")
        os.makedirs(out_dir, exist_ok=True)
        run(exe, "--command", "extension", "build",
            "--source-dir", PACKAGE, "--output-dir", out_dir)
        built = os.path.join(out_dir, "%s-0.%d.0.zip" % (PACKAGE, number))
        print("\nbuilt: %s" % built)

    print("\nOn branch %s. main is untouched." % branch)
    print("\nTo publish, add the public remote once:")
    print("    git remote add public %s" % PUBLIC_REMOTE_URL)
    print("then:")
    print("    git push public %s" % branch)
    print("\nNot done automatically: pushing to the public repo is what makes "
          "this visible to everyone, which is your call and not a build step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
