"""Checks for the stroke cutter in foam_craft.foam_gp_pattern.

Run against a live Blender with the add-on installed::

    python tools/launch_blender.py --port 9885
    python tools/blender_bridge.py --port 9885 < tests/test_gp_pattern.py

Every check states **both** bounds. A check that can only fail in one direction
is not a check: an earlier version of this work reported "2256 of 2256 edges
selected" as a pass, when in fact the cutter had produced nothing and the
measurement had fallen through to the whole mesh. So "two islands" is asserted
as ``== 2``, never ``> 1`` (which an all-edges-seamed mesh also satisfies) and
never ``< n_faces`` (which anything satisfies).
"""

import math
import sys
import importlib
import traceback
from collections import Counter

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

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


def _ring(n=200, z=0.3137):
    """A closed stroke around a unit sphere, deliberately off the equator so it
    does not coincide with an existing edge loop."""
    r = math.sqrt(1.0 - z * z)
    return [Vector((r * math.cos(2 * math.pi * i / n),
                    r * math.sin(2 * math.pi * i / n), z)) for i in range(n)]


def _seam_report(me, projected):
    """Island count, seam shape and cut accuracy, read in edit mode."""
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='MINIMUM_STRETCH', margin=0.02)
    bm = bmesh.from_edit_mesh(me)
    uv_layer = bm.loops.layers.uv.active
    islands = core.uv_islands(bm, uv_layer)
    seam_edges = [e for e in bm.edges if e.seam]

    degrees = Counter()
    for edge in seam_edges:
        for vert in edge.verts:
            degrees[vert.index] += 1

    seam_verts = {v for e in seam_edges for v in e.verts}
    worst = 0.0
    for point in projected:
        worst = max(worst, min((point - v.co).length for v in seam_verts))

    out = {
        "islands": len(islands),
        "seam_edges": len(seam_edges),
        "total_edges": len(bm.edges),
        "faces": len(bm.faces),
        # A single closed loop means every seam vertex has exactly two seam
        # edges. A break shows up as a vertex of degree 1, a T-junction as 3.
        "one_closed_loop": (set(degrees.values()) == {2}),
        "cut_vs_stroke": round(worst, 6),
    }
    bpy.ops.object.mode_set(mode='OBJECT')
    return out


def _project(me, points):
    """The stroke as it lies on this surface.

    The cut must follow the mesh, so comparing it against an ideal analytic
    curve measures how coarse the mesh is, not how good the cutter is. On an 8x4
    sphere the facets sit 0.136 from the true sphere; that error belongs to the
    model. A hand-drawn stroke is already on the surface, so this is what a real
    one looks like.
    """
    bm = bmesh.new()
    bm.from_mesh(me)
    bvh = BVHTree.FromBMesh(bm)
    out = [bvh.find_nearest(p)[0].copy() for p in points]
    bm.free()
    return out


def _cut_into(me, points, snap=0.002, cyclic=True):
    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        new_edges, open_ends = gp.cut_stroke(bm, points, snap_distance=snap,
                                             cyclic=cyclic)
        for edge in new_edges:
            edge.seam = True
        bm.to_mesh(me)
        me.update()
        return len(new_edges), open_ends
    finally:
        bm.free()


# --- 1. The primitive itself -------------------------------------------
def test_face_split_follows_a_polyline():
    """One quad, one S-curve. The smallest shape that can break the method.

    A cutter that laid down straight chords instead of following the path would
    leave more than two faces here, and a no-op would leave one.
    """
    _clear()
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    me = bpy.context.object.data

    n_interior = 7
    stroke = []
    for i in range(n_interior + 2):
        t = i / (n_interior + 1)
        stroke.append(Vector((-1.0 + 2.0 * t,
                              0.62 * math.sin(t * math.pi * 2.0), 0.0)))
    interior = stroke[1:-1]

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()

    def edge_at_x(x):
        return next(e for e in bm.edges
                    if abs(e.verts[0].co.x - x) < 1e-6
                    and abs(e.verts[1].co.x - x) < 1e-6)

    def split_at(edge, target):
        a, b = edge.verts[0], edge.verts[1]
        span = b.co - a.co
        fac = (target - a.co).dot(span) / span.length_squared
        return bmesh.utils.edge_split(edge, a, fac)[1]

    v_entry = split_at(edge_at_x(-1.0), stroke[0])
    v_exit = split_at(edge_at_x(1.0), stroke[-1])
    bm.faces.ensure_lookup_table()
    bmesh.utils.face_split(bm.faces[0], v_entry, v_exit,
                           coords=[tuple(c) for c in interior])

    corners = {(round(x, 4), round(y, 4))
               for x in (-1.0, 1.0) for y in (-1.0, 1.0)}
    known = corners | {(round(v_entry.co.x, 4), round(v_entry.co.y, 4)),
                       (round(v_exit.co.x, 4), round(v_exit.co.y, 4))}
    new_pts = [v.co.copy() for v in bm.verts
               if (round(v.co.x, 4), round(v.co.y, 4)) not in known]
    present = {(round(v.co.x, 4), round(v.co.y, 4)) for v in bm.verts}
    worst = max(min((c - p).length for p in new_pts) for c in interior) \
        if new_pts else None
    faces = len(bm.faces)
    bm.free()

    return {
        "faces": faces,
        "interior_verts": len(new_pts),
        "max_deviation": round(worst, 8) if worst is not None else None,
        "checks": {
            "exactly_two_faces": faces == 2,
            "one_vert_per_coord": len(new_pts) == len(interior),
            "verts_on_the_path": worst is not None and worst < 1e-5,
            "corners_untouched": corners.issubset(present),
        },
    }


# --- 2. A closed stroke separates the surface ---------------------------
def test_ring_splits_a_sphere_in_two():
    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=48,
                                         ring_count=24)
    me = bpy.context.object.data
    points = _ring()
    projected = _project(me, points)
    n_new, open_ends = _cut_into(me, points)
    rep = _seam_report(me, projected)
    rep["new_edges"] = n_new
    rep["open_ends"] = open_ends
    rep["checks"] = {
        "exactly_two_islands": rep["islands"] == 2,
        "seams_neither_empty_nor_all":
            0 < rep["seam_edges"] < rep["total_edges"],
        "one_closed_loop": rep["one_closed_loop"],
        "no_open_ends": open_ends == 0,
        "cut_lands_on_the_stroke": rep["cut_vs_stroke"] < 1e-4,
    }
    return rep


# --- 3. The result does not depend on how dense the mesh is -------------
def test_density_does_not_change_the_line():
    """The claim that justifies this approach over the alternatives.

    Reports two separate numbers per density. The faceting gap is the mesh's
    own error and shrinks as it gets denser; no cutter can avoid it, because the
    seam has to lie on the surface. The cut error is the cutter's own, measured
    against the stroke as projected onto that same surface.
    """
    cases = {}
    points = _ring()
    for segments, rings in ((8, 4), (16, 8), (48, 24), (96, 48)):
        _clear()
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=segments,
                                             ring_count=rings)
        me = bpy.context.object.data
        projected = _project(me, points)
        faceting = max((p - q).length for p, q in zip(points, projected))
        _n, open_ends = _cut_into(me, points)
        rep = _seam_report(me, projected)
        cases["sphere_%dx%d" % (segments, rings)] = {
            "faces": rep["faces"],
            "mesh_faceting_vs_ideal": round(faceting, 5),
            "cut_vs_stroke": rep["cut_vs_stroke"],
            "islands": rep["islands"],
            "one_closed_loop": rep["one_closed_loop"],
            "open_ends": open_ends,
        }

    errs = [c["cut_vs_stroke"] for c in cases.values()]
    out = dict(cases)
    out["checks"] = {
        "two_islands_at_every_density":
            all(c["islands"] == 2 for c in cases.values()),
        "one_loop_at_every_density":
            all(c["one_closed_loop"] for c in cases.values()),
        "no_open_ends": all(c["open_ends"] == 0 for c in cases.values()),
        "cut_exact_everywhere": max(errs) < 1e-4,
        "accuracy_independent_of_density": (max(errs) - min(errs)) < 1e-4,
    }
    return out


TESTS = (
    test_face_split_follows_a_polyline,
    test_ring_splits_a_sphere_in_two,
    test_density_does_not_change_the_line,
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
