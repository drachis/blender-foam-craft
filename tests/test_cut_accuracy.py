"""How faithfully the cut follows the line that was drawn.

    python tools/launch_blender.py --port 9885
    python tools/blender_bridge.py --port 9885 < tests/test_cut_accuracy.py

This is the module's own success criterion, in the user's words: every point
of the grease pencil curve should have seam within a small world-space
tolerance of it. The junction tests in test_gp_junctions.py ask whether the
right things *joined*; this file asks whether the seam went where the pen
went.

Each check here started life as a throwaway probe written against the live
workshop file while chasing a defect, and is kept because the next change to
the cutter needs the same question asked again -- the rule this project
already applies to every other measurement (CLAUDE.md, "Every check goes in
tests/, never in a scratch file").

Two measurement traps, both hit for real during that chase and both worth
not falling into again:

- **Compare against the drawn points projected onto the surface, not the raw
  ones.** Strokes drawn with a Shrinkwrap modifier hover a fixed distance
  above the mesh -- 0.01 on the workshop file -- so measuring from the raw
  points reports that hover as if it were cutting error, and it is not. It
  put a constant 0.01 "error" in an earlier report that was entirely
  fictional.
- **Say which distance is being measured.** Distance to the nearest seam
  *edge* and to the nearest cut *vertex* are different numbers; a dense cut
  makes them nearly equal, a coarse one does not.
"""

import math
import sys
import importlib
import traceback

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

_MOD = "bl_ext.user_default.foam_craft"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foam_craft.foam_gp_pattern as gp        # noqa: E402

importlib.reload(gp)


def _clear():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def _sphere(segments=48, rings=24):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=segments,
                                         ring_count=rings)
    bm = bmesh.new()
    bm.from_mesh(bpy.context.object.data)
    return bm


def _project(bm, points):
    """The drawn points as they lie on the surface -- see the module note."""
    bvh = BVHTree.FromBMesh(bm)
    return [bvh.find_nearest(p)[0].copy() for p in points]


def _gap_to_seam(points, bm):
    """Distance from each point to the nearest seam *edge*, sorted."""
    seam = [e for e in bm.edges if e.seam]
    out = []
    for p in points:
        best = None
        for e in seam:
            a, b = e.verts[0].co, e.verts[1].co
            span = b - a
            denom = span.length_squared
            t = 0.0 if denom < 1e-18 \
                else max(0.0, min(1.0, (p - a).dot(span) / denom))
            d = (p - (a + span * t)).length
            if best is None or d < best:
                best = d
        if best is not None:
            out.append(best)
    return sorted(out)


def _stats(values):
    if not values:
        return None
    n = len(values)
    return {"median": round(values[n // 2], 6),
            "p90": round(values[min(n - 1, int(n * 0.9))], 6),
            "worst": round(values[-1], 6)}


def _spiral(turns=2.5, count=220, tilt=0.55):
    """A stroke that crosses edges at every angle, including near corners.

    A great circle or a latitude ring is a special case -- it runs parallel
    to one family of edges the whole way -- and a cutter can do well on those
    while mishandling an ordinary curve. A spiral sweeps through every
    crossing angle there is, so it exercises the awkward near-corner cases
    the workshop file kept landing on rather than avoiding them.
    """
    points = []
    for i in range(count):
        f = i / (count - 1.0)
        theta = 0.9 + tilt * f
        phi = 2.0 * math.pi * turns * f
        points.append(Vector((math.sin(theta) * math.cos(phi),
                              math.sin(theta) * math.sin(phi),
                              math.cos(theta))))
    return points


# --- 1. The cut lands on the line ---------------------------------------
def test_cut_lands_on_the_drawn_line():
    """With a snap distance small relative to the mesh, the seam should sit
    essentially exactly on the drawn curve -- this is the whole promise.

    Both bounds: asserting closeness alone would also pass if the cutter had
    produced almost nothing, so the seam must also be long enough to be the
    curve rather than a fragment of it.
    """
    _clear()
    bm = _sphere()
    stroke = _spiral()
    projected = _project(bm, stroke)
    edges, open_ends = gp.cut_strokes(bm, [(stroke, False)],
                                      snap_distance=0.001)
    for e in edges:
        e.seam = True
    gaps = _gap_to_seam(projected, bm)
    seam_len = sum(e.calc_length() for e in bm.edges if e.seam)
    bm.free()

    # The stroke's own length is the yardstick for "the seam is the curve".
    drawn_len = sum((b - a).length for a, b in zip(projected, projected[1:]))
    return {
        "gap_to_seam": _stats(gaps),
        "open_ends": open_ends,
        "seam_length": round(seam_len, 4),
        "drawn_length": round(drawn_len, 4),
        "checks": {
            # A tenth of a millimetre on a 1m sphere.
            "every_point_on_the_seam": gaps[-1] <= 1e-4,
            # ...and the seam really is the whole curve, not a stub. Allowed
            # to exceed the drawn length: the seam follows mesh edges through
            # each face, which is a slightly longer path than the chord.
            "seam_covers_the_curve": seam_len >= drawn_len * 0.95,
            "seam_is_not_wild": seam_len <= drawn_len * 2.0,
            "no_open_ends": open_ends == 0,
        },
    }


# --- 2. What the snap distance costs ------------------------------------
def test_snap_distance_trades_fidelity_for_welding():
    """`snap_distance` is a real trade, and this pins which way it runs.

    It feeds `bmesh.ops.remove_doubles` at the end of the cut, which exists
    to stop the cut leaving zero-length edges and the collapsed faces around
    them. But it welds by distance, so a generous setting also drags the cut
    off the line it was following. Measured on the workshop file, whose
    setting was 0.022 on a 1m sphere: the drawn points' p90 offset fell from
    0.0045 to 0.0 as the setting was tightened to 0.001.

    That is worth a check rather than a comment, because the direction is
    not obvious from the code and the setting is exposed in the UI: a future
    change that made a large snap harmless would mean the weld had stopped
    doing its job, and one that made a small snap inaccurate would mean the
    cut had stopped following the line.
    """
    stroke = _spiral()
    rows = {}
    for snap in (0.001, 0.02):
        _clear()
        bm = _sphere()
        projected = _project(bm, stroke)
        edges, _ends = gp.cut_strokes(bm, [(stroke, False)],
                                      snap_distance=snap)
        for e in edges:
            e.seam = True
        gaps = _gap_to_seam(projected, bm)
        rows["snap_%g" % snap] = {
            "stats": _stats(gaps),
            "over_2mm": sum(1 for g in gaps if g > 0.002),
        }
        bm.free()

    tight = rows["snap_0.001"]
    loose = rows["snap_0.02"]
    out = dict(rows)
    out["checks"] = {
        # Tight: the cut is on the line.
        "tight_snap_is_accurate": tight["stats"]["p90"] <= 1e-4,
        # Loose: measurably worse. If this ever passes trivially -- both
        # equal -- the weld has stopped affecting the cut and the setting is
        # doing nothing, which is worth failing over rather than celebrating.
        "loose_snap_is_measurably_worse":
            loose["stats"]["p90"] > tight["stats"]["p90"],
        # And the damage is bounded by the setting itself: welding moves a
        # vertex onto a neighbour within `snap`, so it cannot move further.
        "loose_snap_stays_within_its_own_tolerance":
            loose["stats"]["worst"] <= 0.02 + 1e-6,
    }
    return out


# --- 3. Which route each segment took -----------------------------------
def test_bridge_routes_are_sane_on_an_ordinary_curve():
    """`_bridge` decides how the cut gets from one drawn point to the next,
    and the route it picks is the difference between a seam on the line and
    a seam detoured through a face corner.

    This was a probe, written when the workshop file showed a notch in the
    seam: it counted the routes and found 5 segments being sent through a
    shared face corner, dragging the cut as much as 0.075 off the drawn
    line. `corner_fallback` is that route, now taken only when the proper
    walk cannot find a way across, and it is the number to watch.

    The same counters go to the log on every bake (`cut_routes`), so this
    check and a real bake report the same thing in the same words.
    """
    _clear()
    bm = _sphere()
    stroke = _spiral()
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    bvh = BVHTree.FromBMesh(bm)

    stats = {}
    hits = gp._classify(bm, bvh, stroke)
    bridged, missed = gp._bridge(hits, stats)
    bm.free()

    segments = len(stroke) - 1
    routed = sum(stats.get(k, 0) for k in
                 ("same_face", "shared_edge", "through_corner", "walked",
                  "corner_fallback", "no_route"))
    fallbacks = stats.get("corner_fallback", 0)
    worst_detour = stats.get("corner_fallback_worst", 0.0)
    # The sphere's own edge length, the natural yardstick for "how far off
    # the line is too far": 48 segments around the equator.
    edge_len = 2.0 * math.pi / 48.0
    return {
        "routes": stats,
        "segments": segments,
        "points_after_bridging": len(bridged),
        "missed": missed,
        "sphere_edge_length": round(edge_len, 5),
        "checks": {
            # Every segment took exactly one route, so the tally is complete
            # -- otherwise the numbers below describe only part of the cut.
            "every_segment_accounted_for": routed == segments,
            # No segment left uncrossed.
            "no_segment_without_a_route": stats.get("no_route", 0) == 0,
            # Corner fallbacks are a known, deliberate compromise: when the
            # face walk cannot find the way across, bending the cut to the
            # shared corner keeps the seam connected, which beats a hole in
            # the piece boundary. So this is not asserted at zero -- it is
            # asserted *rare*, because the failure that matters is the route
            # becoming common again (it was the default before, and put a
            # visible notch in the workshop file's seam).
            "corner_fallbacks_are_rare": fallbacks <= max(2, segments * 0.02),
            # ...and bounded. A fallback moves the cut to a corner of the
            # face it was crossing, so it can never be off by more than the
            # face is wide; well inside that is the check worth having.
            "corner_fallback_detour_is_small":
                worst_detour <= edge_len * 0.25,
            # Bridging inserts crossings, so it can only add points.
            "bridging_added_crossings": len(bridged) > len(stroke),
        },
    }


TESTS = (
    test_cut_lands_on_the_drawn_line,
    test_snap_distance_trades_fidelity_for_welding,
    test_bridge_routes_are_sane_on_an_ordinary_curve,
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
