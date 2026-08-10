"""Sequenced screenshot paths, so captures sort in the order they were taken.

Alphabetical filenames jump around as topics change, which makes it hard to
follow progress. Every capture gets a zero-padded counter prefix instead:

    001_overlay_torus.png
    002_preview_flat_cube.png

Use from inside Blender:

    import sys; sys.path.insert(0, r"O:\\Work\\BlenderPlugin\\FoamCraft\\tools")
    from shot import next_path
    bpy.ops.screen.screenshot_area(filepath=next_path("overlay_torus"))

The counter is derived by scanning the directory, so there is no state file to
keep in sync and concurrent workers cannot clobber each other's numbering
beyond sharing a slot, which sorting still handles sensibly.
"""

import os
import re

SHOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "screenshots")

_PREFIX = re.compile(r"^(\d{3,})_")


def next_index(directory=SHOTS_DIR):
    """One past the highest existing sequence number."""
    os.makedirs(directory, exist_ok=True)
    highest = 0
    for name in os.listdir(directory):
        match = _PREFIX.match(name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def next_path(label, ext=".png", directory=SHOTS_DIR):
    """Absolute path for the next capture, e.g. .../007_overlay_torus.png."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return os.path.join(directory, f"{next_index(directory):03d}_{safe}{ext}")


def renumber(directory=SHOTS_DIR):
    """Prefix any unnumbered files by modification time, oldest first.

    Files that already carry a sequence number keep it, so re-running this is
    safe and does not churn names that have already been referenced.
    """
    os.makedirs(directory, exist_ok=True)
    loose = [n for n in os.listdir(directory)
             if not _PREFIX.match(n)
             and os.path.isfile(os.path.join(directory, n))]
    loose.sort(key=lambda n: os.path.getmtime(os.path.join(directory, n)))
    index = next_index(directory)
    renamed = []
    for name in loose:
        target = f"{index:03d}_{name}"
        os.rename(os.path.join(directory, name), os.path.join(directory, target))
        renamed.append((name, target))
        index += 1
    return renamed


if __name__ == "__main__":
    for before, after in renumber():
        print(f"{before} -> {after}")
