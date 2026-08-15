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
    selection is irrelevant               the sculpt and strokes are named in
                                          the panel, so having the Grease Pencil
                                          active -- the normal state after
                                          drawing -- must not disable the bake

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


_TOOLS = "O:/Work/BlenderPlugin/FoamCraft/tools"


def _shot(label, show_seams_on=None):
    """Capture the viewport, so a run can be checked by eye as well as by count.

    A test reporting "3 regions" proves nothing if the scene it built was not
    the one intended -- and several cases here were measuring a shape that had
    projected onto a rim rather than the surface, while still returning a
    confident number. The picture is what catches that.

    Headless runs skip silently: there is no window to capture and the counts
    are still worth having. Use ``tools/run_tests.py --gui`` to collect them.
    """
    if bpy.app.background:
        return None
    if _TOOLS not in sys.path:
        sys.path.insert(0, _TOOLS)
    try:
        from shot import next_path
    except ImportError:
        # Loud, not silent. This failed for a whole run because the path to
        # tools/ had a stray tab in it, and because the capture is optional the
        # tests went on passing while producing no pictures at all.
        print("FoamCraft tests: cannot import shot from %r" % _TOOLS)
        return None

    # The splash sits over the viewport and would be most of the picture.
    try:
        bpy.ops.wm.splash_close()
    except Exception:                                  # noqa: BLE001
        pass
    screen = bpy.context.screen
    if screen is None:
        return None
    hidden = []
    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        # view_all polls for a view3d *region*, not merely the area, so the
        # override has to name both. Overriding the area alone raised
        # "Expected a view3d region" -- and only once framing was added, which
        # is why a bare screenshot had worked.
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if region is None:
            continue
        path = next_path("test_%s" % label)
        # Seams are only drawn in edit mode. A solid object-mode view of a cut
        # sphere is identical to an uncut one, which makes the picture worse
        # than useless -- it looks like confirmation while showing nothing.
        entered = False
        if show_seams_on is not None:
            bpy.ops.object.mode_set(mode='OBJECT')
            for obj in list(bpy.context.selected_objects):
                obj.select_set(False)
            show_seams_on.select_set(True)
            bpy.context.view_layer.objects.active = show_seams_on
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='DESELECT')
            entered = True
        with bpy.context.temp_override(area=area, region=region):
            area.spaces.active.shading.type = 'SOLID'
            bpy.ops.view3d.view_all()
            bpy.ops.screen.screenshot_area(filepath=path)
        if entered:
            bpy.ops.object.mode_set(mode='OBJECT')
        for other in hidden:
            other.hide_set(False)
        return path
    return None


def _clear():
    """Empty the scene *and* the orphaned data behind it.

    Deleting objects is not enough. Their meshes and Grease Pencil data linger
    with no users, so the next `Sculpt` becomes `Sculpt.001` and the bake's
    lookup for `Sculpt_Pattern` finds a copy left by an earlier test. That made
    these tests pass alone and fail when run after the others -- inherited state
    has produced wrong numbers in this project more than once.
    """
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    settings = getattr(bpy.context.scene, "foam_craft", None)
    if settings is not None:
        settings.gp_object = None
        settings.pattern_source = None
    for collection in (bpy.data.objects, bpy.data.meshes,
                       bpy.data.grease_pencils):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


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
    settings.pattern_source = sculpt
    # Left active on purpose: right after drawing, the Grease Pencil is what is
    # selected, and the bake must not care.
    bpy.context.view_layer.objects.active = gp_obj
    return sculpt, gp_obj


def _pattern_of(sculpt):
    """The pattern copy made from `sculpt`, found by what it records.

    Not by name. Blender renames on a clash, so a leftover object called Sculpt
    makes the next one Sculpt.001 and its copy Sculpt.001_Pattern -- and a test
    looking up a hardcoded name reports "no pattern was made" about a bake that
    worked and said so. The copy stores foam_pattern_source for exactly this,
    so using it also checks that the record is good for something.
    """
    return next((o for o in bpy.data.objects
                 if o.get("foam_pattern_source") == sculpt.name), None)


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
    pattern = _pattern_of(sculpt)
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
        bpy.ops.object.foam_bake_pattern(snap_distance=0.002)

    patterns = [o for o in bpy.data.objects if o.get("foam_pattern_source")]
    pattern = _pattern_of(sculpt)
    islands, _seams = _island_count(pattern) if pattern else (0, 0)

    # Naming the pattern copy as the sculpt must be refused: it is an output,
    # and cutting it again would compound the pattern into itself.
    bpy.context.scene.foam_craft.pattern_source = pattern
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


_LABEL = ["shape", ""]


def _bake_strokes_on_sphere(strokes, segments=64, rings=32):
    """Build a sphere, draw `strokes` on it, bake, and report the pieces."""
    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=segments,
                                         ring_count=rings)
    sculpt = bpy.context.object
    # Whatever Blender called it. Renaming to a fixed "Sculpt" only invites the
    # clash that _pattern_of already had to work around, and nothing here needs
    # a particular name.
    _LABEL[1] = sculpt.name
    gp_obj = _gp_from_strokes(strokes)
    bpy.context.scene.foam_craft.gp_object = gp_obj
    bpy.context.scene.foam_craft.pattern_source = sculpt
    bpy.context.view_layer.objects.active = gp_obj

    shots = [_shot("%s_drawn" % _LABEL[0])]
    outcome = bpy.ops.object.foam_bake_pattern(snap_distance=0.004)
    pattern = _pattern_of(sculpt)
    if pattern is None:
        return {"failed": list(outcome), "shots": shots}, None
    shots.append(_shot("%s_cut" % _LABEL[0], show_seams_on=pattern))

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
    # Cutting must not leave rubbish behind. A vertex placed on top of one that
    # is already there gives a zero-length edge and collapses the faces around
    # it -- visible in the viewport as a pinched black wedge, and in a pattern
    # as a piece with a spike no saw can follow.
    degenerate = sum(1 for f in bm.faces if f.calc_area() < 1e-8)
    zero_edges = sum(1 for e in bm.edges if e.calc_length() < 1e-6)
    bpy.ops.object.mode_set(mode='OBJECT')
    return {
        "regions": regions,
        "region_sizes": sorted(sizes, reverse=True),
        "seam_edges": seams,
        "faces": faces,
        "degenerate_faces": degenerate,
        "zero_length_edges": zero_edges,
        "shots": [s for s in shots if s],
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

    _LABEL[0] = "hash"
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
            "no_degenerate_geometry":
                report.get("degenerate_faces") == 0
                and report.get("zero_length_edges") == 0,
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
    _LABEL[0] = "eight"
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
            "no_degenerate_geometry":
                report.get("degenerate_faces") == 0
                and report.get("zero_length_edges") == 0,
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
    _LABEL[0] = "theta"
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
            "no_degenerate_geometry":
                report.get("degenerate_faces") == 0
                and report.get("zero_length_edges") == 0,
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
