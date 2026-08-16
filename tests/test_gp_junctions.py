"""Checks for the junction/self-crossing defect in foam_craft.foam_gp_pattern.

Root cause and reproduction are written up in RESEARCH_gp_pattern_seam_gaps.md
-- read that first. Short version: ``bmesh.utils.face_split_edgenet`` silently
drops a *whole face's* split when any one edge in that face's net doesn't form
a valid planar graph (reaches the boundary at only one end, or crosses another
net edge without sharing a vertex). Two shapes in this module currently
produce exactly that: a stroke that crosses its own earlier path, and a
stroke's end snapped onto another stroke that hasn't been cut yet. These tests
pin down the expected behaviour with known-answer synthetic shapes so the fix
(building a shared cut-graph, mirroring editmesh_knife.cc's KnifeVert/
KnifeEdge, per RESEARCH_gp_pattern_seam_gaps.md's implementation plan) can be
checked against something concrete rather than eyeballed against a screenshot.

Run against a live Blender with the add-on installed::

    python tools/launch_blender.py --port 9885
    python tools/blender_bridge.py --port 9885 < tests/test_gp_junctions.py

Same rule as tests/test_gp_pattern.py: every check states both bounds.
"""

import sys
import importlib
import traceback

import bpy
import bmesh
from mathutils import Vector

_MOD = "bl_ext.user_default.foam_craft"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foam_craft.foam_gp_pattern as gp        # noqa: E402
import bl_ext.user_default.foam_craft.foam_angle_core as core      # noqa: E402

importlib.reload(gp)
importlib.reload(core)


def _clear():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def _grid_plane(cuts=15):
    """A 2x2 plane subdivided into a (cuts+1)^2 grid, edge length 2/(cuts+1)."""
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    me = bpy.context.object.data
    bm0 = bmesh.new()
    bm0.from_mesh(me)
    bmesh.ops.subdivide_edges(bm0, edges=bm0.edges[:], cuts=cuts,
                              use_grid_fill=True)
    bm0.to_mesh(me)
    bm0.free()
    return me, 2.0 / (cuts + 1)


def _densify(points, per_segment=12):
    """Sample a polyline finely -- the user was explicit that a dense, near
    1:1 cut from the curve is fine and expected; the defect is about
    degenerate faces/edges, not density."""
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        for i in range(1, per_segment + 1):
            t = i / per_segment
            out.append(a + (b - a) * t)
    return out


def _bake(bm, strokes, snap, chain_tolerance=0.0):
    edges, open_ends = gp.cut_strokes(bm, strokes, snap_distance=snap,
                                      chain_tolerance=chain_tolerance)
    return edges, open_ends


def _seam_regions(bm):
    """Connected groups of faces reachable without crossing a seam edge --
    same measure as test_gp_pattern.py's _seam_report, inlined here so this
    file can work directly off a live bmesh without a mesh round-trip."""
    seen = set()
    regions = 0
    for face in bm.faces:
        if face in seen:
            continue
        regions += 1
        stack = [face]
        seen.add(face)
        while stack:
            current = stack.pop()
            for loop in current.loops:
                if loop.edge.seam:
                    continue
                for other in loop.edge.link_faces:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
    return regions


def _max_gap_to_seam(points, seam_edges):
    """Worst-case distance from a drawn point to the nearest seam edge --
    the module's own success criterion, stated by the user: every point in
    the grease pencil curve should have a seam point within a reasonable
    distance."""
    worst = 0.0
    for p in points:
        best = None
        for e in seam_edges:
            a, b = e.verts[0].co, e.verts[1].co
            span = b - a
            denom = span.length_squared
            t = 0.0 if denom < 1e-18 else max(0.0, min(1.0, (p - a).dot(span) / denom))
            d = (p - (a + span * t)).length
            if best is None or d < best:
                best = d
        if best is not None:
            worst = max(worst, best)
    return worst


# --- 1. A stroke that crosses its own earlier path ----------------------
def test_self_crossing_stroke_keeps_both_arms():
    """One open stroke, four points, drawn like an X: A->B is one diagonal,
    B->C is an edge, C->D is the other diagonal -- which crosses A->B's
    diagonal at the centre, well after B was already visited. This is the
    "doubling back" shape named directly by the user and reproduced from the
    live file in RESEARCH_gp_pattern_seam_gaps.md (chain 1's 8-edge dangling
    run). A and D sit exactly on the plane's own boundary edge, so -- unlike
    an arbitrary interior tip -- there is no legitimate reason for either end
    to be left unconnected; every point on the drawn curve, tip to tip,
    should have real seam nearby.

    The real test, over the topological region-counting this file used
    before: for every point sampled along the drawn curve, is there a
    seam-tagged edge within a small, fixed distance of it? That is the
    module's actual job -- the curve drawn should be the seam cut, however
    the stroke happened to be shaped -- and it does not depend on correctly
    predicting the crossing's topology by hand. Regions/wire counts are kept
    as a secondary sanity check, not the primary criterion.
    """
    _clear()
    me, edge = _grid_plane(cuts=15)
    a = Vector((-1.0, -0.8, 0.0))    # on the plane's left boundary
    b = Vector((0.8, 0.8, 0.0))
    c = Vector((0.8, -0.8, 0.0))
    d = Vector((-1.0, 0.8, 0.0))     # on the plane's left boundary
    points = _densify([a, b, c, d], per_segment=14)

    bm = bmesh.new()
    bm.from_mesh(me)
    edges, open_ends = _bake(bm, [(points, False)], snap=0.01)
    for e in edges:
        e.seam = True

    seam_edges = [e for e in bm.edges if e.seam]
    regions = _seam_regions(bm)
    gap = _max_gap_to_seam(points, seam_edges)
    n_wire = sum(1 for e in edges if e.is_valid and e.is_wire)
    bm.free()

    # A small, fixed tolerance -- not scaled to the mesh's own edge length --
    # because the claim being tested is "the cutter followed the curve
    # closely," and a bound that grows with the mesh would let a coarser mesh
    # hide the same defect a finer one would catch.
    tolerance = 0.01
    return {
        "regions": regions,
        "seam_edges": len(seam_edges),
        "open_ends": open_ends,
        "max_gap_to_seam": round(gap, 5),
        "wire_edges_remaining": n_wire,
        "edge_length": round(edge, 5),
        "tolerance": tolerance,
        "checks": {
            "every_drawn_point_has_nearby_seam": gap <= tolerance,
            "no_open_ends": open_ends == 0,
            "no_wire_left_behind": n_wire == 0,
            # Secondary: three regions -- the closed triangle B-C-X at the
            # crossing, and the two halves the boundary-to-boundary path
            # A-X-D splits the rest of the sheet into. (A and D anchor to
            # the boundary specifically so there is no ambiguous "legitimate
            # open tip" case, which also means this number is no longer the
            # 2 it would be with A/D floating -- the path itself now also
            # bisects the outer region.)
            "three_regions": regions == 3,
        },
    }


# --- 2. Two strokes forming a T, cut in the "wrong" order ----------------
def test_t_junction_is_order_independent():
    """A horizontal stroke spans the sheet boundary-to-boundary; a vertical
    stroke runs from the bottom boundary up to a point that lies *exactly* on
    the horizontal stroke's drawn path (no gap -- this isolates the
    order-dependency bug from the separate near-miss-extension feature).
    Tried both as (through-line first, T second) and (T first, through-line
    second): a human drawing this with the interactive knife gets the same
    result regardless of which line they drew first, which is the behaviour
    the user confirmed directly against their own Blender session.

    Known answer: the through-line alone splits the sheet in two (top/
    bottom). The T-stroke further splits the bottom half in two (left/
    right), so the whole thing is exactly 3 regions -- in both cut orders.
    """
    edge_ref = [None]
    results = {}
    for label, order in (("through_then_t", (0, 1)), ("t_then_through", (1, 0))):
        _clear()
        me, edge = _grid_plane(cuts=15)
        edge_ref[0] = edge
        through = _densify(
            [Vector((-1.0, 0.1, 0.0)), Vector((1.0, 0.1, 0.0))], per_segment=30)
        # The T's tip sits exactly on the through-line's drawn path.
        t_stroke = _densify(
            [Vector((0.3, -1.0, 0.0)), Vector((0.3, 0.1, 0.0))], per_segment=20)
        strokes = [(through, False), (t_stroke, False)]
        ordered = [strokes[i] for i in order]

        bm = bmesh.new()
        bm.from_mesh(me)
        edges, open_ends = _bake(bm, ordered, snap=0.01)
        for e in edges:
            e.seam = True
        seam_edges = [e for e in bm.edges if e.seam]
        regions = _seam_regions(bm)
        gap = _max_gap_to_seam(through + t_stroke, seam_edges)
        n_wire = sum(1 for e in edges if e.is_valid and e.is_wire)
        bm.free()

        results[label] = {
            "regions": regions,
            "open_ends": open_ends,
            "max_gap_to_seam": round(gap, 5),
            "wire_edges_remaining": n_wire,
        }

    tolerance = 0.01
    out = dict(results)
    out["tolerance"] = tolerance
    out["checks"] = {
        "three_regions_through_then_t": results["through_then_t"]["regions"] == 3,
        "three_regions_t_then_through": results["t_then_through"]["regions"] == 3,
        "same_result_either_order":
            results["through_then_t"]["regions"] ==
            results["t_then_through"]["regions"],
        "no_open_ends_either_order":
            results["through_then_t"]["open_ends"] == 0 and
            results["t_then_through"]["open_ends"] == 0,
        # The real test: every point on both drawn curves has a seam edge
        # within a small, fixed tolerance nearby -- regardless of cut order.
        "every_drawn_point_has_nearby_seam_either_order":
            results["through_then_t"]["max_gap_to_seam"] <= tolerance and
            results["t_then_through"]["max_gap_to_seam"] <= tolerance,
        "no_wire_left_behind_either_order":
            results["through_then_t"]["wire_edges_remaining"] == 0 and
            results["t_then_through"]["wire_edges_remaining"] == 0,
    }
    return out


TESTS = (
    test_self_crossing_stroke_keeps_both_arms,
    test_t_junction_is_order_independent,
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
