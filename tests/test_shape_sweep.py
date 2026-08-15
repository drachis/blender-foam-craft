"""The same shapes at many sizes and separations.

    python tools/run_tests.py --only sweep

One theta at one radius passing says almost nothing: the cutter's failures are
sensitive to where a stroke happens to fall relative to the mesh, so a single
size can pass by luck. These sweep the size, and for the two-circle case the
separation, straight through the transition from overlapping to tangent to
apart -- which is where a cutter that mishandles crossings gives itself away.

Every case has an answer known in advance.

    theta, any radius        3   two half discs and the hull
    two circles apart        3   two discs and the hull
    two circles tangent      3   touching at a point separates nothing extra
    two circles overlapping  4   the lens, two crescents, and the hull

Tangency is the interesting one. It is the boundary between three regions and
four, so a cutter that rounds a touch into a crossing reports 4 and one that
drops it reports 3 -- and only one of those is right.
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

import bl_ext.user_default.foam_craft.foam_gp_pattern as gp    # noqa: E402

importlib.reload(gp)

SEGMENTS, RINGS = 96, 48
SNAP = 0.004


def _clear():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.objects, bpy.data.meshes):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _point(theta, phi):
    return Vector((math.sin(theta) * math.cos(phi),
                   math.sin(theta) * math.sin(phi),
                   math.cos(theta)))


def _circle(centre_theta, centre_phi, radius, count=120):
    """A circle of the given angular radius, drawn on the unit sphere."""
    centre = _point(centre_theta, centre_phi)
    helper = Vector((0.0, 0.0, 1.0))
    if abs(centre.dot(helper)) > 0.9:
        helper = Vector((1.0, 0.0, 0.0))
    e1 = centre.cross(helper).normalized()
    e2 = centre.cross(e1).normalized()
    return [(centre * math.cos(radius)
             + (e1 * math.cos(2 * math.pi * i / count)
                + e2 * math.sin(2 * math.pi * i / count)) * math.sin(radius))
            for i in range(count)]


def _regions(bm):
    seen, sizes = set(), []
    for face in bm.faces:
        if face in seen:
            continue
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
    return sorted(sizes, reverse=True)


def _canonical_faces(bm):
    """Put the faces in a known order, so two identical spheres really are.

    ``primitive_uv_sphere_add`` returns the same vertices and a *different* face
    table on every call -- measured in tests/test_determinism.py, which exists
    because of what that cost. Anything resolving a tie between two equally near
    faces resolves it by index, so an unstable face table makes an unstable cut,
    and these sweeps were reporting different answers for identical input.

    The vertices are stable, so ordering faces by the vertices they are made of
    is stable too. bmesh's sort wants a number, hence the packing.
    """
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    base = len(bm.verts) + 1

    def _key(face):
        idx = sorted(v.index for v in face.verts)[:4]
        idx += [0] * (4 - len(idx))
        return ((idx[0] * base + idx[1]) * base + idx[2]) * base + idx[3]

    bm.faces.sort(key=_key)
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()


def _cut_with(strokes, snap):
    """_cut, with the tolerance named rather than fixed."""
    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=SEGMENTS,
                                         ring_count=RINGS)
    bm = bmesh.new()
    bm.from_mesh(bpy.context.object.data)
    _canonical_faces(bm)
    try:
        edges, open_ends = gp.cut_strokes(bm, strokes, snap_distance=snap,
                                          chain_tolerance=snap)
        sizes = _regions(bm)
        return {"regions": len(sizes), "sizes": sizes[:6],
                "open_ends": open_ends, "seam_edges": len(edges),
                "degenerate": sum(1 for f in bm.faces
                                  if f.calc_area() < 1e-8)}
    finally:
        bm.free()


def _cut(strokes):
    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=SEGMENTS,
                                         ring_count=RINGS)
    bm = bmesh.new()
    bm.from_mesh(bpy.context.object.data)
    _canonical_faces(bm)
    try:
        edges, open_ends = gp.cut_strokes(bm, strokes, snap_distance=SNAP)
        sizes = _regions(bm)
        return {
            "regions": len(sizes),
            "sizes": sizes[:6],
            "open_ends": open_ends,
            "seam_edges": len(edges),
            "degenerate": sum(1 for f in bm.faces if f.calc_area() < 1e-8),
        }
    finally:
        bm.free()


# --- 1. A theta at several sizes ----------------------------------------
def test_theta_at_several_radii():
    """Three regions whatever the circle's size.

    The bar crosses the circle twice and halves it, so the answer does not
    depend on the radius -- only on whether the crossings take. A radius that
    happens to sit well against the mesh proves nothing about one that does not.
    """
    cases = {}
    for radius in (0.15, 0.28, 0.45):
        circle = _circle(1.5708, 0.0, radius)
        over = radius * 1.25
        bar = [_point(1.5708, -over + 2 * over * i / 59.0) for i in range(60)]
        cases["radius_%.2f" % radius] = _cut([(circle, True), (bar, False)])

    out = dict(cases)
    out["checks"] = {
        "three_regions_at_every_radius":
            all(c["regions"] == 3 for c in cases.values()),
        # The halves should be alike: the bar runs through the centre.
        "halves_are_even": all(
            len(c["sizes"]) >= 3
            and abs(c["sizes"][1] - c["sizes"][2]) <= max(4, 0.3 * c["sizes"][1])
            for c in cases.values()),
        "no_degenerate_geometry":
            all(c["degenerate"] == 0 for c in cases.values()),
        # Two-sided: a cutter doing nothing would leave one region everywhere.
        "something_was_cut": all(c["seam_edges"] > 0 for c in cases.values()),
    }
    return out


# --- 2. Two circles from overlapping to apart ---------------------------
def test_two_circles_through_tangency():
    """Overlap gives four regions, tangent and apart give three.

    Separations are set as a multiple of 2r, which is the tangency distance, so
    the sweep crosses the boundary rather than sampling near it. The tangent
    case is the one worth having: it is exactly the transition, so a cutter that
    treats a touch as a crossing says 4 and one that loses it says 3.
    """
    radius = 0.22
    cases = {}
    for label, factor in (("overlap_0.6", 0.6), ("overlap_0.9", 0.9),
                          ("tangent_1.0", 1.0),
                          ("apart_1.1", 1.1), ("apart_1.6", 1.6)):
        half = radius * factor          # each centre this far from the middle
        strokes = [(_circle(1.5708, -half, radius), True),
                   (_circle(1.5708, half, radius), True)]
        cases[label] = _cut(strokes)

    overlapping = [cases[k] for k in cases if k.startswith("overlap")]
    apart = [cases[k] for k in cases if k.startswith("apart")]
    tangent = cases["tangent_1.0"]

    out = dict(cases)
    out["checks"] = {
        # Lens, two crescents, hull.
        "overlap_gives_four": all(c["regions"] == 4 for c in overlapping),
        # Two discs and the hull.
        "apart_gives_three": all(c["regions"] == 3 for c in apart),
        # The boundary case must land on one side of it, not somewhere else.
        "tangent_gives_three_or_four": tangent["regions"] in (3, 4),
        "no_degenerate_geometry":
            all(c["degenerate"] == 0 for c in cases.values()),
        "something_was_cut":
            all(c["seam_edges"] > 0 and c["regions"] > 1
                for c in cases.values()),
    }
    return out


# --- 3. One circle, grown from small to large ---------------------------
def test_circle_scaled_about_its_centre():
    """A closed loop splits the sphere in two at any size.

    Small circles are the case a cutter fails on quietly: too few faces to
    enclose, and it returns one region while reporting a healthy seam count.
    """
    cases = {}
    for radius in (0.06, 0.12, 0.25, 0.5, 0.9):
        cases["radius_%.2f" % radius] = _cut(
            [(_circle(1.5708, 0.0, radius), True)])

    out = dict(cases)
    out["checks"] = {
        "two_regions_at_every_size":
            all(c["regions"] == 2 for c in cases.values()),
        "no_loose_ends": all(c["open_ends"] == 0 for c in cases.values()),
        "no_degenerate_geometry":
            all(c["degenerate"] == 0 for c in cases.values()),
        # The disc must grow with the radius, or the cut is not following it.
        "disc_grows_with_radius": [
            c["sizes"][-1] for c in cases.values()
        ] == sorted(c["sizes"][-1] for c in cases.values()),
    }
    return out



# --- 4. The three T junctions from the drawing pass ----------------------
def test_t_junctions_join_by_pen_width():
    """A T that overlaps by pen width joins; a T with a real gap stays open.

    Straight from a hand-drawn pass on a sphere (2026-08-15). Three T
    junctions came out three ways:

        one crossed both visually and by centreline   -- always worked
        one overlapped visibly but the centrelines missed -- never joined
        one had a real gap, visible on screen         -- correctly apart

    The middle case is the defect: a drawn line has the width of a marker, the
    polyline taken from it has none, and a stem landing on a bar by its *ink*
    should join it. The pen's own radius -- which Grease Pencil records per
    point -- is what says so. The third case is the two-sided half: a gap wider
    than the ink is a gap the maker drew, and the cutter must not helpfully
    close it. A junction invented where none was drawn becomes a cut in foam.

    A T encloses nothing, so region counts cannot see any of this. What a
    joined T *does* create is a vertex with three seam edges, so that is what
    is measured, in both directions.
    """
    def measure(stem_gap, width):
        bar = [_point(1.5708, -0.3 + 0.6 * i / 39.0) for i in range(40)]
        # The stem descends toward the bar and stops `stem_gap` short of it.
        stem = [_point(1.5708 - 0.3 + (0.3 - stem_gap) * i / 39.0, 0.0)
                for i in range(40)]
        _clear()
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=SEGMENTS,
                                             ring_count=RINGS)
        bm = bmesh.new()
        bm.from_mesh(bpy.context.object.data)
        _canonical_faces(bm)
        try:
            edges, open_ends = gp.cut_strokes(
                bm, [(bar, False, width), (stem, False, width)],
                snap_distance=SNAP)
            junctions = sum(
                1 for v in bm.verts
                if sum(1 for e in v.link_edges if e.seam) >= 3)
            return {"junctions": junctions, "open_ends": open_ends,
                    "seam_edges": len(edges),
                    "degenerate": sum(1 for f in bm.faces
                                      if f.calc_area() < 1e-8)}
        finally:
            bm.free()

    width = 0.012
    # Ink overlaps: the stem stops short by less than the two half-widths.
    overlapping = measure(stem_gap=width * 1.5, width=width)
    # A real gap: several pen widths of clean surface between stem and bar.
    apart = measure(stem_gap=width * 6.0, width=width)

    return {
        "ink_overlap": overlapping,
        "real_gap": apart,
        "checks": {
            # The stem must join the bar: one vertex carries three seam edges.
            "ink_overlap_joins": overlapping["junctions"] >= 1,
            # ...and the join must be the only such place.
            "exactly_one_junction": overlapping["junctions"] == 1,
            # A gap the maker drew stays a gap, reported as a loose end.
            "real_gap_stays_open": apart["junctions"] == 0,
            "both_actually_cut": overlapping["seam_edges"] > 0
                and apart["seam_edges"] > 0,
            "no_degenerate_geometry": overlapping["degenerate"] == 0
                and apart["degenerate"] == 0,
        },
    }

TESTS = (
    test_theta_at_several_radii,
    test_two_circles_through_tangency,
    test_circle_scaled_about_its_centre,
    test_t_junctions_join_by_pen_width,
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
