"""A spherical Voronoi cut into a sphere: the junction stress test.

    python tools/run_tests.py --only voronoi

WHY THIS SHAPE

A Voronoi is almost nothing but junctions. Every vertex in it is where three
cell boundaries meet, which is precisely where this cutter is weakest: a stroke
that *ends* at a junction has a loose end sitting mid-face, and a face can only
be split between two of its own boundary vertices, so that end cannot be cut.

It also has an answer known in advance. N seed points give N cells, so the
region count is not a matter of opinion.

The pattern is built from Blender's own convex hull. The hull of points on a
sphere is their Delaunay triangulation, and each triangle's plane normal meets
the sphere at that triangle's circumcentre -- a Voronoi vertex. Every hull edge
is shared by two triangles and so yields one Voronoi edge between their
circumcentres.

KNOWN LIMITATION, TRACKED HERE RATHER THAN HIDDEN

This does not pass yet, and the numbers below are the record of where it stands.
Chaining strokes that share an end into longer paths took 24 seeds from 1 region
to 6 and cut loose ends from 20 to 5, but a full set of cells needs every
boundary to close and some still do not. The check asserts the real answer, so
it stays red until the cutter earns it.

The same limitation was predicted from the designed Suzanne UVs, whose seams
decompose into 135 open paths meeting at 88 junctions with only one closed loop.
This is that problem in a shape with an exact answer.
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


def _clear():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.objects, bpy.data.meshes):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _fibonacci_sphere(count):
    """Seeds spread evenly by golden angle, so no cell is a degenerate sliver."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for i in range(count):
        z = 1.0 - 2.0 * i / (count - 1.0)
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        angle = golden * i
        points.append(Vector((radius * math.cos(angle),
                              radius * math.sin(angle), z)))
    return points


def _voronoi_edges(seeds):
    """Cell boundaries, as pairs of circumcentres on the unit sphere."""
    bm = bmesh.new()
    for point in seeds:
        bm.verts.new(point)
    bm.verts.ensure_lookup_table()
    bmesh.ops.convex_hull(bm, input=bm.verts[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    centres = {}
    for face in bm.faces:
        co = [v.co for v in face.verts]
        normal = (co[1] - co[0]).cross(co[2] - co[0])
        if normal.length < 1e-12:
            continue
        normal.normalize()
        if normal.dot(co[0]) < 0:
            normal = -normal
        centres[face] = normal

    out = []
    for edge in bm.edges:
        faces = [f for f in edge.link_faces if f in centres]
        if len(faces) == 2:
            out.append((centres[faces[0]].copy(), centres[faces[1]].copy()))
    bm.free()
    return out


def _arc(a, b, count=40):
    """Great-circle arc, sampled finely enough not to skip faces.

    Ten points per arc was too coarse on a 128x64 sphere: consecutive points
    landed on faces sharing neither an edge nor a corner, and those strokes were
    refused outright.
    """
    dot = max(-1.0, min(1.0, a.dot(b)))
    angle = math.acos(dot)
    if angle < 1e-9:
        return [a.copy(), b.copy()]
    sin_a = math.sin(angle)
    return [(a * (math.sin((1 - t) * angle) / sin_a)
             + b * (math.sin(t * angle) / sin_a)).normalized()
            for t in (i / (count - 1.0) for i in range(count))]


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


def _run(seed_count, segments=128, rings=64):
    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=segments,
                                         ring_count=rings)
    mesh = bpy.context.object.data
    edges = _voronoi_edges(_fibonacci_sphere(seed_count))
    strokes = [(_arc(a, b), False) for a, b in edges]

    bm = bmesh.new()
    bm.from_mesh(mesh)
    try:
        cut, open_ends = gp.cut_strokes(bm, strokes, snap_distance=0.006,
                                        chain_tolerance=0.02)
        sizes = _regions(bm)
        return {
            "seeds": seed_count,
            "voronoi_edges": len(edges),
            "regions": len(sizes),
            "largest": sizes[0] if sizes else 0,
            "cells_found": len([s for s in sizes[1:] if s > 10]),
            "open_ends": open_ends,
            "degenerate_faces": sum(1 for f in bm.faces
                                    if f.calc_area() < 1e-8),
            "seam_edges": len(cut),
        }
    finally:
        bm.free()


def test_voronoi_cells_become_pieces():
    """N seeds must give N cells plus nothing else.

    Asserted as ``== seeds`` deliberately. "More than one" would pass on a
    pattern that closed a single cell and merged the other twenty-three, which
    is roughly where this stands.
    """
    cases = {"seeds_%d" % n: _run(n) for n in (12, 24)}
    out = dict(cases)
    out["checks"] = {
        # Every cell closed, and no strays.
        "all_cells_closed":
            all(c["regions"] == c["seeds"] for c in cases.values()),
        # Every stroke end found something to land on.
        "no_loose_ends": all(c["open_ends"] == 0 for c in cases.values()),
        "no_degenerate_geometry":
            all(c["degenerate_faces"] == 0 for c in cases.values()),
        # Two-sided: something must actually have been cut. A cutter that did
        # nothing would leave one region and pass a "not too many" test.
        "something_was_cut":
            all(c["seam_edges"] > 0 and c["regions"] > 1
                for c in cases.values()),
    }
    return out


TESTS = (test_voronoi_cells_become_pieces,)

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
