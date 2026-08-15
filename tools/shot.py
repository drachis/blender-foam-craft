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


# --- Capturing a cut so the cut is actually in frame ---------------------

def seam_frame(obj):
    """Where the seams are and which way they face, in world space.

    Returns ``(centre, normal, radius)`` or None when the mesh has no seams.

    A capture framed on the whole object shows the cut edge-on as often as not,
    or hides it round the back -- which produced pictures that looked like a
    failed cut when the cut was fine. Averaging the normals of the faces beside
    the seams gives a direction to look from, and their spread gives a distance
    that fills the frame with the part that matters.
    """
    import bmesh
    from mathutils import Vector

    mesh = obj.data
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        seams = [e for e in bm.edges if e.seam]
        if not seams:
            return None
        matrix = obj.matrix_world
        normal_matrix = matrix.to_3x3().inverted().transposed()

        points = []
        normal = Vector((0.0, 0.0, 0.0))
        for edge in seams:
            for vert in edge.verts:
                points.append(matrix @ vert.co)
            for face in edge.link_faces:
                normal += (normal_matrix @ face.normal).normalized()
        if not points:
            return None
        centre = sum(points, Vector()) / len(points)
        if normal.length < 1e-9:
            # Seams wrapping a closed shape cancel out. Looking from anywhere
            # is then as good as anywhere else.
            normal = Vector((0.0, -1.0, 0.0))
        radius = max((p - centre).length for p in points) or 1.0
        return centre, normal.normalized(), radius
    finally:
        bm.free()


def capture_cut(obj, label, directory=None):
    """Photograph an object's seams, looking straight down them.

    Handles the three things that made earlier captures lie: seams are only
    drawn in edit mode, so a solid object-mode view of a cut sphere is identical
    to an uncut one; other objects in the scene sit in front of the subject and
    read as the result; and a view framed on the whole object shows the cut
    edge-on or hides it behind.

    Returns the path written, or None when there is nothing to show -- silence
    being much better than a picture of the wrong thing.
    """
    import bpy

    if bpy.app.background:
        return None
    screen = bpy.context.screen
    if screen is None:
        return None
    frame = seam_frame(obj)
    if frame is None:
        return None
    centre, normal, radius = frame

    area = next((a for a in screen.areas if a.type == 'VIEW_3D'), None)
    if area is None:
        return None
    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
    space = area.spaces.active
    if region is None or not hasattr(space, "region_3d"):
        return None

    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    hidden = []
    for other in bpy.context.view_layer.objects:
        if other is not obj and not other.hide_get():
            other.hide_set(True)
            hidden.append(other)
    for other in list(bpy.context.selected_objects):
        other.select_set(False)
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')

    view = space.region_3d
    view.view_perspective = 'ORTHO'
    # Look *along* the inward direction, so the surface carrying the seams
    # faces the camera rather than being seen edge-on.
    view.view_rotation = (-normal).to_track_quat('-Z', 'Y')
    view.view_location = centre
    view.view_distance = radius * 2.6

    path = next_path(label, directory=directory) if directory \
        else next_path(label)
    space.shading.type = 'SOLID'
    if hasattr(space, "show_region_ui"):
        space.show_region_ui = False
    if hasattr(space, "show_region_toolbar"):
        space.show_region_toolbar = False
    with bpy.context.temp_override(area=area, region=region):
        bpy.ops.screen.screenshot_area(filepath=path)

    bpy.ops.object.mode_set(mode='OBJECT')
    for other in hidden:
        other.hide_set(False)
    return path
