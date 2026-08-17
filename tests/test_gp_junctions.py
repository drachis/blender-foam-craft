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


# --- 3. A T whose stem stops a whisker short of the crossbar ------------
def test_near_miss_t_junction_extends_to_meet():
    """The junction a hand actually leaves: the stem stops just short of the
    crossbar, so the two curves never touch, but the drawn ink plainly
    overlaps and the maker meant a junction. Tested at both ends of the stem
    (a stroke drawn upward toward the bar, and one drawn downward away from
    it) because a stroke's head and tail are equally likely to be the end
    that meets something.

    The gap is 0.012 -- larger than the 0.01 snap tolerance, so a
    nearest-point test *cannot* close it, which is the point: this only
    passes if the stem is carried on in its own direction of travel. The
    pen half-widths (0.02 each) put the crossbar well inside that reach.

    Known answer: 3 regions, same as a T that actually touches -- the
    crossbar splits the sheet top/bottom, and the stem splits the bottom
    half left/right. A stem that fails to reach leaves 2.

    Also checks the crossbar was not dragged: its own drawn points must
    still have seam within tolerance along their original line, since the
    user's rule is that the T's top stays as drawn and the incoming line is
    what grows.
    """
    results = {}
    for label, flip in (("stem_tail_meets", False), ("stem_head_meets", True)):
        _clear()
        me, edge = _grid_plane(cuts=15)
        crossbar = _densify(
            [Vector((-1.0, 0.1, 0.0)), Vector((1.0, 0.1, 0.0))], per_segment=30)
        # Stops at y=0.088, a 0.012 gap below the crossbar at y=0.1.
        stem = _densify(
            [Vector((0.3, -1.0, 0.0)), Vector((0.3, 0.088, 0.0))],
            per_segment=20)
        if flip:
            stem = list(reversed(stem))
        strokes = [(crossbar, False, 0.02), (stem, False, 0.02)]

        bm = bmesh.new()
        bm.from_mesh(me)
        edges, open_ends = gp.cut_strokes(bm, strokes, snap_distance=0.01)
        for e in edges:
            e.seam = True
        seam_edges = [e for e in bm.edges if e.seam]
        regions = _seam_regions(bm)
        gap_all = _max_gap_to_seam(crossbar + stem, seam_edges)
        gap_crossbar = _max_gap_to_seam(crossbar, seam_edges)
        n_wire = sum(1 for e in edges if e.is_valid and e.is_wire)
        bm.free()

        results[label] = {
            "regions": regions,
            "open_ends": open_ends,
            "max_gap_to_seam": round(gap_all, 5),
            "max_gap_crossbar": round(gap_crossbar, 5),
            "wire_edges_remaining": n_wire,
        }

    tolerance = 0.01
    out = dict(results)
    out["tolerance"] = tolerance
    out["gap_drawn"] = 0.012
    out["checks"] = {
        "stem_tail_reaches_crossbar": results["stem_tail_meets"]["regions"] == 3,
        "stem_head_reaches_crossbar": results["stem_head_meets"]["regions"] == 3,
        "every_drawn_point_has_nearby_seam":
            results["stem_tail_meets"]["max_gap_to_seam"] <= tolerance and
            results["stem_head_meets"]["max_gap_to_seam"] <= tolerance,
        # The crossbar keeps its own line -- it is the stem that grows.
        "crossbar_not_dragged":
            results["stem_tail_meets"]["max_gap_crossbar"] <= tolerance and
            results["stem_head_meets"]["max_gap_crossbar"] <= tolerance,
        "no_open_ends":
            results["stem_tail_meets"]["open_ends"] == 0 and
            results["stem_head_meets"]["open_ends"] == 0,
        "no_wire_left_behind":
            results["stem_tail_meets"]["wire_edges_remaining"] == 0 and
            results["stem_head_meets"]["wire_edges_remaining"] == 0,
    }
    return out


# --- 4. Only a line aimed at another may reach it -----------------------
def test_near_miss_only_extends_along_direction_of_travel():
    """The negative cases, and the reason the rule is directional rather
    than just a wider tolerance: "if it does, yay cut, if not, don't keep
    those verts and keep going else where."

    Tested against `snap_stroke_ends` directly rather than through a full
    cut. That function is pure -- point lists in, point lists out -- so what
    it decided can be read exactly, where a region count downstream also
    reflects whether each stroke happened to reach the sheet's boundary and
    would confound "did not connect" with "connected but enclosed nothing".

    The stems below end beyond the 0.01 snap tolerance from the crossbar, so
    none of them can reach it sideways; only being carried on in their own
    direction can close the gap, and only if they were pointing at it:

      aimed      -- 0.012 short, runs at the crossbar. Must grow onto it.
      parallel   -- 0.012 away, runs alongside it, never converging. Must
                    not: passing near is not meeting.
      off_end    -- 0.012 short and aimed squarely at where the crossbar's
                    *line* lies, but past the end of the crossbar as
                    actually drawn. Must not -- nothing is there to meet.
      too_far    -- aimed at it, but 0.08 short, well past the 0.04 the two
                    pens' widths justify. Must not: this is meant to close
                    the gap a hand leaves at a junction, not to hunt for
                    distant lines to attach to.

    Note there is no "tip near it but pointing away" case: for a straight
    stroke that is not constructible, since whichever end lies nearest the
    crossbar necessarily has its outward direction pointing toward it. The
    realistic ways to not-be-a-junction are the three above.
    """
    crossbar = _densify(
        [Vector((-1.0, 0.1, 0.0)), Vector((0.0, 0.1, 0.0))], per_segment=30)
    cases = {
        "aimed": [Vector((-0.5, -0.6, 0.0)), Vector((-0.5, 0.088, 0.0))],
        "parallel": [Vector((-0.7, 0.088, 0.0)), Vector((-0.2, 0.088, 0.0))],
        # Aimed at y=0.1, but at x=+0.5 -- the crossbar stops at x=0.0.
        "off_end": [Vector((0.5, -0.6, 0.0)), Vector((0.5, 0.088, 0.0))],
        "too_far": [Vector((-0.5, -0.6, 0.0)), Vector((-0.5, 0.02, 0.0))],
    }

    results = {}
    for label, ends in cases.items():
        stem = _densify(ends, per_segment=20)
        strokes = [(list(crossbar), False, 0.02), (stem, False, 0.02)]
        out = gp.snap_stroke_ends(strokes, 0.01)
        stem_out = out[1][0]
        crossbar_out = out[0][0]
        # Did the stem gain a point, and does anything in it now reach the
        # crossbar's own line (y=0.1) within the pen's reach?
        grew = len(stem_out) - len(stem)
        touches = min(abs(p.y - 0.1) for p in stem_out)
        results[label] = {
            "points_before": len(stem),
            "points_after": len(stem_out),
            "grew_by": grew,
            "closest_approach_to_crossbar_line": round(touches, 5),
            "crossbar_points_before": len(crossbar),
            "crossbar_points_after": len(crossbar_out),
        }

    out = dict(results)
    out["checks"] = {
        # The one genuine junction: the stem grows to meet the crossbar.
        "aimed_reaches": results["aimed"]["grew_by"] > 0 and
                         results["aimed"]["closest_approach_to_crossbar_line"] <= 1e-6,
        # And the crossbar gains the meeting point too, so both lines pass
        # through it whichever is cut first.
        "crossbar_gains_the_junction":
            results["aimed"]["crossbar_points_after"] >
            results["aimed"]["crossbar_points_before"],
        "parallel_left_alone": results["parallel"]["grew_by"] == 0,
        "off_end_left_alone": results["off_end"]["grew_by"] == 0,
        "too_far_left_alone": results["too_far"]["grew_by"] == 0,
        # Two-sided: the three negatives passing would be trivially
        # satisfied by an extension that never fires at all, so the
        # crossbar must be untouched by them specifically.
        "negatives_did_not_touch_crossbar": all(
            results[label]["crossbar_points_after"] ==
            results[label]["crossbar_points_before"]
            for label in ("parallel", "off_end", "too_far")),
    }
    return out


# --- 5. Two strokes crossing exactly at an existing face corner ---------
def test_crossing_at_a_face_corner_still_joins():
    """Reported from the live file: two strokes that visibly cross leave a
    gap in the seam when the crossing lands on a face *corner* -- an
    existing mesh vertex -- rather than in open face.

    A corner is the awkward case because a point landing there is snapped
    onto real geometry that already exists, so it is no longer a free point
    in a face the way an ordinary crossing is, and any shortcut taken for
    "this point is already on real topology" risks skipping the very test
    that would notice the other stroke.

    Two diagonals of the sheet, corner to corner, which cross at the middle
    -- and on this grid the middle is exactly a vertex. Known answer: an X
    across a square divides it into 4 regions. A crossing that fails to
    join gives 2 (each diagonal cutting the sheet but not each other).

    Deliberately diagonal, so neither stroke runs along an existing edge:
    that isolates "crossing at a corner" from "running down an edge loop",
    which is a different case with its own handling.
    """
    _clear()
    me, edge = _grid_plane(cuts=15)
    # cuts=15 on a 2x2 plane puts grid lines every 0.125, so (0, 0) is a
    # vertex and both diagonals pass exactly through it.
    diag_a = _densify(
        [Vector((-1.0, -1.0, 0.0)), Vector((1.0, 1.0, 0.0))], per_segment=40)
    diag_b = _densify(
        [Vector((-1.0, 1.0, 0.0)), Vector((1.0, -1.0, 0.0))], per_segment=40)

    bm = bmesh.new()
    bm.from_mesh(me)
    edges, open_ends = gp.cut_strokes(bm, [(diag_a, False), (diag_b, False)],
                                      snap_distance=0.01)
    for e in edges:
        e.seam = True
    seam_edges = [e for e in bm.edges if e.seam]
    regions = _seam_regions(bm)
    gap = _max_gap_to_seam(diag_a + diag_b, seam_edges)
    n_wire = sum(1 for e in edges if e.is_valid and e.is_wire)

    # The crossing itself: is there a seam vertex at the sheet's centre, and
    # does it carry all four arms? A junction that "joined" by laying two
    # separate vertices a hair apart would still leave the piece open.
    centre = Vector((0.0, 0.0, 0.0))
    near_centre = [v for v in bm.verts if (v.co - centre).length < 1e-4]
    arms = 0
    if near_centre:
        arms = sum(1 for e in near_centre[0].link_edges if e.seam)
    bm.free()

    tolerance = 0.01
    return {
        "regions": regions,
        "open_ends": open_ends,
        "max_gap_to_seam": round(gap, 5),
        "wire_edges_remaining": n_wire,
        "verts_at_centre": len(near_centre),
        "seam_arms_at_centre": arms,
        "checks": {
            "four_regions": regions == 4,
            "every_drawn_point_has_nearby_seam": gap <= tolerance,
            # One vertex at the crossing, not two laid a hair apart.
            "single_vertex_at_the_crossing": len(near_centre) == 1,
            # All four arms of the X meet there.
            "all_four_arms_meet": arms == 4,
            "no_open_ends": open_ends == 0,
            "no_wire_left_behind": n_wire == 0,
        },
    }


# --- 6. One stroke through a corner, another crossing just beside it ----
def test_crossing_beside_a_face_corner_still_joins():
    """The harder half of the corner case, and the one the live file hit.

    When both strokes pass exactly through the same mesh vertex they join
    for free -- both snap to it, so they are the same point by construction
    (that is `test_crossing_at_a_face_corner_still_joins`). The failure is
    when only *one* of them lands on the corner and the other crosses a
    whisker beside it: the first is now anchored to real geometry while the
    second is still a free point in a face, so nothing makes them the same
    point, and any shortcut that treats "already on real topology" as
    "nothing more to check here" skips the test that would notice the
    crossing at all.

    `diag_a` runs corner to corner through the vertex at the origin.
    `diag_b` spans the sheet too but crosses `diag_a` at about
    (0.0067, 0.0067) -- roughly 0.0095 off the vertex, far enough not to
    snap to it and close enough to share its faces.

    Known answer: two lines that each span the sheet and cross once divide
    it into 4. A crossing that is missed leaves the seam open where they
    meet and the count drops.
    """
    _clear()
    me, edge = _grid_plane(cuts=15)
    diag_a = _densify(
        [Vector((-1.0, -1.0, 0.0)), Vector((1.0, 1.0, 0.0))], per_segment=40)
    diag_b = _densify(
        [Vector((-1.0, 0.5, 0.0)), Vector((1.0, -0.48, 0.0))], per_segment=40)

    bm = bmesh.new()
    bm.from_mesh(me)
    edges, open_ends = gp.cut_strokes(bm, [(diag_a, False), (diag_b, False)],
                                      snap_distance=0.01)
    for e in edges:
        e.seam = True
    seam_edges = [e for e in bm.edges if e.seam]
    regions = _seam_regions(bm)
    gap = _max_gap_to_seam(diag_a + diag_b, seam_edges)
    n_wire = sum(1 for e in edges if e.is_valid and e.is_wire)

    # Where the two lines actually meet, solved by hand above.
    crossing = Vector((0.0067, 0.0067, 0.0))
    junction_arms = 0
    for v in bm.verts:
        if (v.co - crossing).length > 0.03:
            continue
        junction_arms = max(junction_arms,
                            sum(1 for e in v.link_edges if e.seam))
    bm.free()

    tolerance = 0.01
    return {
        "regions": regions,
        "open_ends": open_ends,
        "max_gap_to_seam": round(gap, 5),
        "wire_edges_remaining": n_wire,
        "best_seam_degree_near_crossing": junction_arms,
        "checks": {
            "four_regions": regions == 4,
            "every_drawn_point_has_nearby_seam": gap <= tolerance,
            # A real crossing leaves a vertex with four seam arms. Two lines
            # that merely pass by each other leave only degree-2 vertices.
            "four_arms_meet_at_the_crossing": junction_arms >= 4,
            "no_open_ends": open_ends == 0,
            "no_wire_left_behind": n_wire == 0,
        },
    }


# --- 7. A line passing near a corner must not be dragged onto it --------
def test_line_passing_near_a_corner_is_not_snapped_to_it():
    """Reported from the live file as a notch in the seam, and traced to
    `_bridge` routing the cut *through* a shared face corner whenever two
    consecutive points landed on faces that touch only at that corner --
    with no tolerance, however far the drawn line actually passed from it.

    That is an assumption inherited from the old `face_split` cutter, where
    a cut had to run between vertices the face already owned. A knife does
    not do it: a vertex is hit only when the line passes within a hair of
    one, and otherwise the ordinary edge crossings either side are taken
    (editmesh_knife.cc:2883-2935). Measured on the workshop file before the
    fix: 5 segments routed through a corner, dragging the seam as much as
    0.075 off the line drawn.

    A diagonal stroke offset just enough to pass *beside* the grid vertices
    rather than through them. Every point of it must have seam within a
    tight tolerance -- a seam detoured to a corner and back cannot.

    Two-sided: the offset is 0.02, comfortably larger than the tolerance
    asserted, so a cutter that did snap to corners fails this rather than
    scraping through.
    """
    _clear()
    me, edge = _grid_plane(cuts=15)
    # Grid lines every 0.125. A 45-degree line through (0.02, 0) passes
    # 0.014 from each vertex it goes by -- beside them, not through.
    stroke = _densify(
        [Vector((-0.98, -1.0, 0.0)), Vector((1.0, 0.98, 0.0))],
        per_segment=40)

    bm = bmesh.new()
    bm.from_mesh(me)
    edges, open_ends = gp.cut_strokes(bm, [(stroke, False)],
                                      snap_distance=0.002)
    for e in edges:
        e.seam = True
    seam_edges = [e for e in bm.edges if e.seam]
    gap = _max_gap_to_seam(stroke, seam_edges)
    regions = _seam_regions(bm)
    bm.free()

    tolerance = 0.004
    return {
        "regions": regions,
        "open_ends": open_ends,
        "max_gap_to_seam": round(gap, 5),
        "tolerance": tolerance,
        "checks": {
            # The cut follows the drawn line rather than detouring via the
            # corners it passes.
            "cut_follows_the_drawn_line": gap <= tolerance,
            # And it still actually cut: a line corner to corner divides
            # the sheet in two.
            "two_regions": regions == 2,
            "no_open_ends": open_ends == 0,
        },
    }


TESTS = (
    test_self_crossing_stroke_keeps_both_arms,
    test_t_junction_is_order_independent,
    test_crossing_at_a_face_corner_still_joins,
    test_crossing_beside_a_face_corner_still_joins,
    test_line_passing_near_a_corner_is_not_snapped_to_it,
    test_near_miss_t_junction_extends_to_meet,
    test_near_miss_only_extends_along_direction_of_travel,
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
