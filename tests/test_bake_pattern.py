"""Checks for the Bake Pattern operator.

Run against a live Blender with the add-on installed::

    python tools/run_tests.py --only bake

The bake is what turns the cutter into something a user can press. Its contract
is narrow and each part of it is a way the tool could quietly ruin a day's work:

    the sculpt is never modified          it is the asset; the pattern is not
    the drawing is never consumed         re-baking after adjusting a stroke is
                                          the entire working loop
    re-baking replaces, never accumulates otherwise draw-bake-look-adjust turns
                                          into a cleanup chore
    the copy carries its inputs           so a re-bake needs no restating

Every check states both bounds.
"""

import math
import sys
import importlib
import traceback

import bpy
import bmesh
from mathutils import Vector

_MOD = "bl_ext.user_default.foam_craft"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foam_craft.foam_gp_pattern as gp     # noqa: E402
import bl_ext.user_default.foam_craft.foam_angle_core as core   # noqa: E402

importlib.reload(gp)
importlib.reload(core)


def _clear():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.objects):
        if block.users == 0:
            bpy.data.objects.remove(block)


def _build_scene(ring_z=0.3137, n=160, offset=(0.0, 0.0, 0.0)):
    """A sphere with a closed ring drawn round it on a Grease Pencil object.

    The Grease Pencil is deliberately given a different transform from the
    sculpt: the cutter works in the target's local space, so a bake that forgets
    to convert between them lands the pattern somewhere else entirely, and with
    both objects at the origin that mistake is invisible.
    """
    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=48, ring_count=24)
    sculpt = bpy.context.object
    sculpt.name = "Sculpt"
    sculpt.location = offset

    bpy.ops.object.grease_pencil_add(type='EMPTY', location=offset)
    gp_obj = bpy.context.object
    gp_obj.name = "PatternStrokes"
    data = gp_obj.data
    layer = data.layers.new("Pattern") if not data.layers else data.layers[0]
    frame = layer.frames.new(0) if not layer.frames else layer.frames[0]
    drawing = frame.drawing
    drawing.add_strokes([n])
    radius = math.sqrt(1.0 - ring_z * ring_z)
    positions = drawing.attributes["position"].data
    for i in range(n):
        angle = 2 * math.pi * i / n
        positions[i].vector = (radius * math.cos(angle),
                               radius * math.sin(angle), ring_z)
    # Required after writing positions directly; without it the change is not
    # guaranteed to be seen by anything reading the drawing back.
    drawing.tag_positions_changed()

    # A closed stroke. "cyclic" is a curve-domain attribute rather than a
    # property, and it does not exist until something creates it -- so a ring
    # drawn without it is read as an open line and leaves the piece uncut.
    cyclic = drawing.attributes.get("cyclic")
    if cyclic is None:
        cyclic = drawing.attributes.new("cyclic", 'BOOLEAN', 'CURVE')
    cyclic.data[0].value = True

    settings = bpy.context.scene.foam_craft
    settings.gp_object = gp_obj
    bpy.context.view_layer.objects.active = sculpt
    return sculpt, gp_obj


def _stroke_count(gp_obj):
    layer = gp_obj.data.layers[0]
    return len(layer.frames[0].drawing.strokes)


def _mesh_counts(obj):
    return (len(obj.data.vertices), len(obj.data.edges),
            len(obj.data.polygons))


def _island_count(obj):
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.active
    count = len(core.uv_islands(bm, uv_layer)) if uv_layer else 0
    seams = sum(1 for e in bm.edges if e.seam)
    bpy.ops.object.mode_set(mode='OBJECT')
    return count, seams


# --- 1. Reading strokes back in the target's space ----------------------
def test_strokes_convert_between_object_spaces():
    """A stroke drawn round a sculpt must land on it wherever both objects are.

    Checked with the two objects offset from the origin *and* from each other,
    because at the origin an identity transform hides a missing conversion.
    """
    sculpt, gp_obj = _build_scene(offset=(3.0, -2.0, 1.5))
    gp_obj.location = (3.5, -2.0, 1.5)      # deliberately not the sculpt's

    points = gp.strokes_from_grease_pencil(gp_obj, sculpt)
    flat = [p for pts, _cyclic in points for p in pts]
    # In the sculpt's local space the ring should sit on the unit sphere.
    radii = [p.length for p in flat]

    return {
        "strokes": len(points),
        "points": len(flat),
        "min_radius": round(min(radii), 4),
        "max_radius": round(max(radii), 4),
        "checks": {
            "one_stroke_read": len(points) == 1,
            "all_points_read": len(flat) == 160,
            # The offset between the two objects is 0.5, so a missed conversion
            # would push the radius well outside this band.
            "landed_in_sculpt_space":
                0.95 < min(radii) and max(radii) < 1.05,
        },
    }


# --- 2. The bake produces pieces and leaves the inputs alone ------------
def test_bake_makes_pieces_without_touching_inputs():
    sculpt, gp_obj = _build_scene()
    before = _mesh_counts(sculpt)
    strokes_before = _stroke_count(gp_obj)

    result = bpy.ops.object.foam_bake_pattern(snap_distance=0.002)
    pattern = bpy.data.objects.get("Sculpt_Pattern")
    islands, seams = _island_count(pattern) if pattern else (0, 0)

    return {
        "operator": list(result),
        "pattern_exists": pattern is not None,
        "islands": islands,
        "seam_edges": seams,
        "source_before": before,
        "source_after": _mesh_counts(sculpt),
        "strokes_before": strokes_before,
        "strokes_after": _stroke_count(gp_obj),
        "records_source": str(pattern.get("foam_pattern_source")) if pattern else None,
        "records_gp": str(pattern.get("foam_pattern_gp")) if pattern else None,
        "source_name": sculpt.name,
        "gp_name": gp_obj.name,
        "checks": {
            "operator_finished": 'FINISHED' in result,
            "pattern_was_made": pattern is not None,
            # A closed ring divides a sphere in two. Not "> 1", which an
            # all-seamed mesh also satisfies.
            "exactly_two_pieces": islands == 2,
            "seams_neither_empty_nor_all":
                pattern is not None
                and 0 < seams < len(pattern.data.edges),
            "sculpt_untouched": _mesh_counts(sculpt) == before,
            "drawing_untouched": _stroke_count(gp_obj) == strokes_before,
            # Recorded so a re-bake finds its inputs without the user
            # restating them. Compared against the objects actually built, not
            # against literals, since Blender renames on a name clash.
            "inputs_recorded":
                pattern is not None
                and str(pattern.get("foam_pattern_source")) == sculpt.name
                and str(pattern.get("foam_pattern_gp")) == gp_obj.name,
        },
    }


# --- 3. Re-baking replaces rather than piling up ------------------------
def test_rebake_replaces_the_copy():
    """Draw, bake, look, adjust, bake again is the working loop. If each bake
    left another copy behind, the loop would come with a cleanup chore."""
    sculpt, gp_obj = _build_scene()
    for _ in range(3):
        bpy.context.view_layer.objects.active = sculpt
        bpy.ops.object.foam_bake_pattern(snap_distance=0.002)

    patterns = [o for o in bpy.data.objects if o.get("foam_pattern_source")]
    pattern = bpy.data.objects.get("Sculpt_Pattern")
    islands, _seams = _island_count(pattern) if pattern else (0, 0)

    # Baking with the pattern copy selected must be refused: it is an output,
    # and cutting it again would compound the pattern into itself.
    bpy.context.view_layer.objects.active = pattern
    try:
        # Reports an error, which bpy.ops turns into an exception rather than a
        # CANCELLED return, so refusal has to be caught rather than inspected.
        bpy.ops.object.foam_bake_pattern(snap_distance=0.002)
        second = ['FINISHED']
    except RuntimeError:
        second = ['CANCELLED']

    return {
        "pattern_copies": len(patterns),
        "islands_after_three": islands,
        "bake_on_copy": list(second),
        "checks": {
            "exactly_one_copy": len(patterns) == 1,
            "still_two_pieces": islands == 2,
            "refuses_to_bake_a_copy": 'CANCELLED' in second,
        },
    }



# --- Drawing arbitrary strokes on a sphere ------------------------------

def _sphere_point(theta, phi):
    """A point on the unit sphere from polar and azimuthal angles."""
    return Vector((math.sin(theta) * math.cos(phi),
                   math.sin(theta) * math.sin(phi),
                   math.cos(theta)))


def _sphere_circle(centre_theta, centre_phi, angular_radius, n=64):
    """A closed circle drawn on the sphere's surface, of a given angular size."""
    centre = _sphere_point(centre_theta, centre_phi)
    # Any two directions perpendicular to the centre span the circle's plane.
    helper = Vector((0.0, 0.0, 1.0))
    if abs(centre.dot(helper)) > 0.9:
        helper = Vector((1.0, 0.0, 0.0))
    e1 = centre.cross(helper).normalized()
    e2 = centre.cross(e1).normalized()
    points = []
    for i in range(n):
        a = 2 * math.pi * i / n
        points.append(centre * math.cos(angular_radius)
                      + (e1 * math.cos(a) + e2 * math.sin(a))
                      * math.sin(angular_radius))
    return points


def _gp_from_strokes(strokes, name="PatternStrokes"):
    """A Grease Pencil object carrying `strokes` as [(points, cyclic)]."""
    bpy.ops.object.grease_pencil_add(type='EMPTY', location=(0, 0, 0))
    gp_obj = bpy.context.object
    gp_obj.name = name
    data = gp_obj.data
    layer = data.layers.new("Pattern") if not data.layers else data.layers[0]
    frame = layer.frames.new(0) if not layer.frames else layer.frames[0]
    drawing = frame.drawing
    drawing.add_strokes([len(pts) for pts, _c in strokes])
    positions = drawing.attributes["position"].data
    index = 0
    for pts, _cyclic in strokes:
        for point in pts:
            positions[index].vector = point
            index += 1
    drawing.tag_positions_changed()
    if any(c for _p, c in strokes):
        cyclic = drawing.attributes.get("cyclic")
        if cyclic is None:
            cyclic = drawing.attributes.new("cyclic", 'BOOLEAN', 'CURVE')
        for i, (_p, c) in enumerate(strokes):
            cyclic.data[i].value = bool(c)
    return gp_obj


def _bake_strokes_on_sphere(strokes, segments=64, rings=32):
    """Build a sphere, draw `strokes` on it, bake, and report the pieces."""
    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=segments,
                                         ring_count=rings)
    sculpt = bpy.context.object
    sculpt.name = "Sculpt"
    gp_obj = _gp_from_strokes(strokes)
    bpy.context.scene.foam_craft.gp_object = gp_obj
    bpy.context.view_layer.objects.active = sculpt

    outcome = bpy.ops.object.foam_bake_pattern(snap_distance=0.004)
    pattern = bpy.data.objects.get("Sculpt_Pattern")
    if pattern is None:
        return {"failed": list(outcome)}, None

    bpy.context.view_layer.objects.active = pattern
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(pattern.data)

    # Regions the cut encloses: groups of faces reachable without crossing a
    # seam. This is the cutter's own job. UV island count answers a different
    # question -- a region with a hole in it cannot be flattened without a
    # further cut, so it unwraps to more than one island while still being one
    # piece of surface.
    seen, regions, sizes = set(), 0, []
    for face in bm.faces:
        if face in seen:
            continue
        regions += 1
        stack, size = [face], 0
        seen.add(face)
        while stack:
            current = stack.pop()
            size += 1
            for loop in current.loops:
                if loop.edge.seam:
                    continue
                for other in loop.edge.link_faces:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
        sizes.append(size)
    seams = sum(1 for e in bm.edges if e.seam)
    faces = len(bm.faces)
    bpy.ops.object.mode_set(mode='OBJECT')
    return {
        "regions": regions,
        "region_sizes": sorted(sizes, reverse=True),
        "seam_edges": seams,
        "faces": faces,
    }, pattern


# --- 4. A hash of crossing strokes encloses one square ------------------
def test_hash_grid_encloses_one_square():
    """Two strokes each way across a patch of sphere: one square, one hull.

    The interesting part is what the protruding ends do. A `#` has eight of
    them, each stopping in open surface, and none of them encloses anything --
    so a cutter that let a dangling end split the surface would report more than
    two regions here. It is the cheapest test for that mistake.
    """
    strokes = []
    for theta in (1.35, 1.79):
        strokes.append(([_sphere_point(theta, -0.55 + 1.1 * i / 47.0)
                         for i in range(48)], False))
    for phi in (-0.22, 0.22):
        strokes.append(([_sphere_point(1.15 + 0.84 * i / 47.0, phi)
                         for i in range(48)], False))

    report, _pattern = _bake_strokes_on_sphere(strokes)
    sizes = report.get("region_sizes", [])
    return {
        **report,
        "checks": {
            # The square, and everything else.
            "exactly_two_regions": report.get("regions") == 2,
            # ...and the square must be the small one, not a stray sliver.
            "one_small_one_large":
                len(sizes) == 2 and sizes[0] > sizes[1] > 20,
            "seams_neither_empty_nor_all":
                0 < report.get("seam_edges", 0) < report.get("faces", 1) * 2,
        },
    }


# --- 5. A figure of eight encloses two discs ----------------------------
def test_figure_eight_encloses_two_discs():
    """Two closed circles drawn on the front of a sphere: two discs and a hull.

    Only the front is drawn on, so the back must come through uncut -- a cutter
    that reached through the surface would split the far side as well and report
    more than three regions.
    """
    strokes = [
        (_sphere_circle(1.30, 0.0, 0.20), True),
        (_sphere_circle(1.84, 0.0, 0.20), True),
    ]
    report, _pattern = _bake_strokes_on_sphere(strokes)
    sizes = report.get("region_sizes", [])
    return {
        **report,
        "checks": {
            # Two discs and the rest of the hull.
            "exactly_three_regions": report.get("regions") == 3,
            "two_discs_and_a_hull":
                len(sizes) == 3 and sizes[0] > sizes[1] > 5 and sizes[2] > 5,
            # The discs are drawn the same size, so they must come out alike.
            # A lopsided pair would mean one circle failed to close.
            "discs_are_similar":
                len(sizes) == 3 and abs(sizes[1] - sizes[2]) <= max(
                    4, 0.25 * sizes[1]),
        },
    }


# --- 6. A theta: a circle with a bar across it --------------------------
def test_theta_splits_its_circle_in_two():
    """A circle crossed by a bar: two half-discs and the hull.

    This is the case the hash and the figure of eight both miss. The bar's ends
    have to meet the circle, so the two strokes must intersect properly -- and
    since strokes are cut one after another, by the time the bar is cut the
    circle is already real topology and the crossing is an ordinary one.

    Get that wrong and the count says so directly: an uncrossed bar leaves the
    disc whole and reports two regions instead of three.
    """
    circle = _sphere_circle(1.5708, 0.0, 0.28)
    # The bar runs past the circle on both sides, so the crossings are
    # unambiguous. The overhanging ends enclose nothing and must not count.
    bar = [_sphere_point(1.5708, -0.34 + 0.68 * i / 39.0) for i in range(40)]
    report, _pattern = _bake_strokes_on_sphere([(circle, True), (bar, False)])
    sizes = report.get("region_sizes", [])
    return {
        **report,
        "checks": {
            # Two halves of the disc, and the rest of the sphere.
            "exactly_three_regions": report.get("regions") == 3,
            # The bar cuts through the middle, so the halves should be alike.
            # A lopsided pair means the bar missed the centre or one crossing
            # failed to take.
            "halves_are_even":
                len(sizes) == 3 and sizes[1] > 5
                and abs(sizes[1] - sizes[2]) <= max(4, 0.3 * sizes[1]),
            "hull_is_the_largest":
                len(sizes) == 3 and sizes[0] > sizes[1] + sizes[2],
        },
    }


TESTS = (
    test_strokes_convert_between_object_spaces,
    test_bake_makes_pieces_without_touching_inputs,
    test_rebake_replaces_the_copy,
    test_hash_grid_encloses_one_square,
    test_figure_eight_encloses_two_discs,
    test_theta_splits_its_circle_in_two,
)

result = {}
_all_pass = True
for _test in TESTS:
    try:
        _report = _test()
        _report["PASS"] = all(_report.get("checks", {}).values())
    except Exception:
        _report = {"ERROR": traceback.format_exc().splitlines()[-5:],
                   "PASS": False}
    result[_test.__name__] = _report
    _all_pass = _all_pass and _report["PASS"]
result["ALL_PASS"] = _all_pass
