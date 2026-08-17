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
from mathutils import Vector, kdtree
from mathutils.bvhtree import BVHTree

_MOD = "bl_ext.user_default.foam_craft"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foam_craft.foam_gp_pattern as gp        # noqa: E402

importlib.reload(gp)


# The tolerances the check is swept over, in millimetres on a metre-scale
# model. The loosest is the bar: 1 mm is finer than a hand cutting foam can
# hold, so a drawn point with no cut vertex within it was not cut at all,
# as against merely cut a little off. Sweeping rather than picking one
# number is what separates those two failures -- an imprecise cut sheds
# points gradually as the tolerance tightens, a missing one does not move.
TOLERANCES_MM = (0.1, 0.25, 0.5, 1.0)


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


def _vertex_distances(drawn, bm, edges):
    """For each drawn point, how far the nearest vertex of the cut is.

    Distance to the nearest cut *vertex*, not to the nearest seam edge. A
    point can sit exactly on a seam edge while no vertex was ever made for
    it -- the edge simply passes by -- and that is the cut skipping detail
    the maker drew.

    Note this asks whether *a* vertex is near each point, not whether each
    point got one of its own. At a 1 mm tolerance several drawn points may
    legitimately share a single vertex; a hand-drawn stroke is sampled far
    finer than any saw can follow, and collapsing that detail is the cut
    working, not failing. What must never happen is a stretch of the drawn
    line with no cut vertex anywhere near it.
    """
    cut_verts = {v.index: v for e in edges if e.is_valid for v in e.verts}
    if not cut_verts:
        return None
    tree = kdtree.KDTree(len(cut_verts))
    for i, v in enumerate(cut_verts.values()):
        tree.insert(v.co, i)
    tree.balance()

    surface = BVHTree.FromBMesh(bm)
    out = []
    for stroke_index, point_index, point in drawn:
        # The drawn line is cut where it lies *on* the surface: a stroke
        # hovering above the mesh (a Shrinkwrap offset, or the sagitta of a
        # faceted sphere under an analytic curve) is not the cutter missing.
        landed = surface.find_nearest(point)[0]
        if landed is None:
            continue
        _co, _i, distance = tree.find(landed)
        out.append((distance, stroke_index, point_index, point, landed))
    return out


def _misses_at(distances, tolerance_mm):
    """The drawn points with no cut vertex within `tolerance_mm`, described
    well enough to go and find them on the curve."""
    limit = tolerance_mm / 1000.0
    return [{"stroke": si, "point": pi,
             "drawn_at": tuple(round(c, 5) for c in point),
             "on_surface_at": tuple(round(c, 5) for c in landed),
             "nearest_vertex_mm": round(d * 1000.0, 4)}
            for d, si, pi, point, landed in distances if d > limit]


# --- 0. Every drawn point becomes a vertex ------------------------------
def test_every_drawn_point_becomes_a_cut_vertex():
    """The whole promise of this module, stated as plainly as it can be:
    cut the pattern, and every point the maker drew has a vertex of the cut
    within a hair of it. Anything less means the seam is not the line that
    was drawn, and in foam that is a piece cut to the wrong shape.

    Run with `snap_distance=0`, so nothing is snapped, welded or otherwise
    tidied afterwards -- the question is what the cut itself produced, not
    what survived the clean-up. (The cutter still keeps its own internal
    epsilon, a thousandth of the shortest edge of the face being cut, to
    avoid making zero-length edges and the collapsed faces around them.
    That is the tolerance below, and it is a property of the geometry rather
    than a setting.)

    Three arrangements, because the failures this suite has actually seen
    all involved more than one line:

      plane_single   -- one curve on flat ground, where the drawn points lie
                        exactly on the mesh and there is no projection to
                        argue about. If this fails, nothing else matters.
      plane_crossing -- two curves that cross, the shape a pattern is made
                        of, and where a face's net can be refused.
      sphere_spiral  -- a curve over a curved, irregularly-cut surface,
                        sweeping every crossing angle.

    On failure this reports each point that was missed and where it is, in
    both the drawn frame and on the surface, so it can be found rather than
    hunted for.
    """
    cases = {}
    for label in ("plane_single", "plane_crossing", "sphere_spiral"):
        _clear()
        if label.startswith("plane"):
            bpy.ops.mesh.primitive_plane_add(size=2.0)
            me = bpy.context.object.data
            bm0 = bmesh.new()
            bm0.from_mesh(me)
            bmesh.ops.subdivide_edges(bm0, edges=bm0.edges[:], cuts=15,
                                      use_grid_fill=True)
            bm0.to_mesh(me)
            bm0.free()
            bm = bmesh.new()
            bm.from_mesh(me)
            shortest = min(e.calc_length() for e in bm.edges)
            wiggle = [Vector((-0.94 + 1.88 * i / 59.0,
                              0.37 * math.sin(i / 59.0 * math.pi * 3.0), 0.0))
                      for i in range(60)]
            strokes = [(wiggle, False)]
            if label == "plane_crossing":
                strokes.append(
                    ([Vector((0.21 + 0.3 * math.sin(i / 49.0 * math.pi),
                              -0.96 + 1.92 * i / 49.0, 0.0))
                      for i in range(50)], False))
        else:
            bm = _sphere()
            shortest = min(e.calc_length() for e in bm.edges)
            strokes = [(_spiral(), False)]

        edges, open_ends = gp.cut_strokes(bm, strokes, snap_distance=0.0)
        for e in edges:
            e.seam = True

        drawn = [(si, pi, p)
                 for si, (points, _c) in enumerate(strokes)
                 for pi, p in enumerate(points)]
        distances = _vertex_distances(drawn, bm, edges)
        bm.free()
        if distances is None:
            cases[label] = {"points": len(drawn),
                            "error": "the cut produced no vertices at all"}
            continue

        worst = max(d for d, _s, _p, _pt, _l in distances)
        cases[label] = {
            "points": len(drawn),
            "worst_mm": round(worst * 1000.0, 4),
            # The sweep: how many points are left uncut as the tolerance
            # tightens. A cut that is merely imprecise loses points slowly
            # as this narrows; one that missed a stretch of the line
            # entirely shows the same count at every tolerance, which is the
            # signal worth telling apart.
            "missed_at_mm": {str(mm): len(_misses_at(distances, mm))
                             for mm in TOLERANCES_MM},
            # Named, not just counted, at the loosest tolerance -- anything
            # failing there is genuinely uncut rather than slightly off.
            "uncut_at_%gmm" % max(TOLERANCES_MM):
                _misses_at(distances, max(TOLERANCES_MM))[:8],
            "open_ends": open_ends,
        }

    # The bar. A millimetre on a metre-scale sculpt is finer than the thing
    # will be cut by hand, so a point with no vertex within it was not cut.
    bar = max(TOLERANCES_MM)
    out = dict(cases)
    out["tolerances_mm"] = TOLERANCES_MM
    out["checks"] = {
        "every_point_cut_on_a_plane":
            cases["plane_single"]["missed_at_mm"][str(bar)] == 0,
        "every_point_cut_where_curves_cross":
            cases["plane_crossing"]["missed_at_mm"][str(bar)] == 0,
        "every_point_cut_on_a_sphere":
            cases["sphere_spiral"]["missed_at_mm"][str(bar)] == 0,
        # Two-sided: zero misses would also be reported by a run that
        # measured nothing, so the points have to have been there.
        "the_points_were_really_checked": all(
            c.get("points", 0) > 40 for c in cases.values()),
    }
    return out


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
    them. But it welds by distance, so a generous setting also pulls the cut
    off the line it was following.

    Read the direction of this check carefully, because the obvious reading
    is the wrong one. `snap_distance` is the smallest feature the saw can
    actually cut (see CLAUDE.md, "A tolerance that stands for a physical
    limit is not a tuning knob") -- it is a measurement of the workshop, not
    a knob to turn until the numbers look good. So this does **not** say
    "prefer a small snap". It says the setting has a real and bounded cost,
    which is what lets a coarse result be read correctly when it happens: a
    cut that cannot follow the drawing at the blade's own width means the
    pattern has detail finer than the process can make, and that belongs in
    front of the maker rather than being tuned away.

    Worth a check rather than a comment because both directions are
    failures. A large snap that stopped costing anything would mean the weld
    had quietly stopped welding; a small snap that was inaccurate would mean
    the cut had stopped following the line.
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
    # Every route a segment can take. `crossings` is deliberately absent --
    # it counts hits, not segments, and adding it here would break the tally.
    routed = sum(stats.get(k, 0) for k in
                 ("same_face", "shared_edge", "through_corner", "crossed",
                  "walked", "corner_fallback", "no_route"))
    return {
        "routes": stats,
        "segments": segments,
        "points_after_bridging": len(bridged),
        "missed": missed,
        "checks": {
            # Every segment took exactly one route, so the tally is complete
            # -- otherwise the numbers below describe only part of the cut.
            "every_segment_accounted_for": routed == segments,
            # No segment left uncrossed.
            "no_segment_without_a_route": stats.get("no_route", 0) == 0,
            # And none bent to a corner. This was once asserted merely
            # "rare", because the face walk it fell back from could not
            # always find its way across -- collecting the crossings and
            # sorting them, the way the knife does, removed the need for the
            # fallback rather than making it rarer, so zero is the honest
            # bar now. A seam bent to a corner is a pattern that does not
            # match what the maker drew.
            "no_corner_fallbacks": stats.get("corner_fallback", 0) == 0,
            # The old face walk should likewise never be needed.
            "no_face_walks_needed": stats.get("walked", 0) == 0,
            # Bridging inserts crossings, so it can only add points.
            "bridging_added_crossings": len(bridged) > len(stroke),
        },
    }


TESTS = (
    test_every_drawn_point_becomes_a_cut_vertex,
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
