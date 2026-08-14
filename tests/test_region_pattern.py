"""A pattern of regions tiling a whole shape -- the real job, in miniature.

    python tools/run_tests.py --only region

The sculpt this project exists for is covered in scale panels, so the case that
matters is not one drawn line but a whole pattern of regions covering a surface.
This is that, in the simplest shape with an exact answer: bands around a sphere
crossed by columns, which is also close to how such a pattern is really drawn.

    3 latitude rings                          closed loops
    6 meridian arcs spanning ring to ring     open, ending on the rings

Which gives 14 regions: a cap at each pole, and two bands of six.

Meridians deliberately stop at the outer rings instead of running through the
poles. A pole would be a six-way junction, and the point here is the ordinary
four-way crossing that a scale pattern is made of, not the worst case.

KNOWN LIMITATION, TRACKED RATHER THAN HIDDEN

This does not pass yet. A stroke that *ends* at a junction leaves a loose end
mid-face, and a face can only be split between two of its own boundary vertices,
so that end cannot be cut. Every meridian here ends on a ring, so every one of
them tests exactly that.

chain_strokes joins strokes sharing an end into longer paths before cutting,
which turns many junctions into interior points of a path, and it is what took
this from unusable to close. The remainder is the documented next step: split
two branches of a junction as one through-path first, so the junction becomes a
real vertex, then attach the rest.

The same limitation shows in the designed Suzanne UVs, whose seams decompose
into 135 open paths meeting at 88 junctions with only one closed loop.
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

RINGS = (1.05, 1.5708, 2.09)      # polar angles of the latitude rings
MERIDIANS = 6
# Two caps, plus one band between each neighbouring pair of rings, each band
# divided by the meridians.
EXPECTED = 2 + (len(RINGS) - 1) * MERIDIANS


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


def _pattern():
    """Rings and meridians, sampled finely enough not to skip faces.

    Coarse sampling was a real failure and not only a test artefact: on a dense
    sphere, successive points landed on faces sharing neither an edge nor a
    corner and the stroke was refused outright.
    """
    strokes = []
    for theta in RINGS:
        strokes.append(([_point(theta, 2 * math.pi * i / 96.0)
                         for i in range(96)], True))
    for m in range(MERIDIANS):
        phi = 2 * math.pi * m / MERIDIANS
        span = RINGS[-1] - RINGS[0]
        strokes.append(([_point(RINGS[0] + span * i / 47.0, phi)
                         for i in range(48)], False))
    return strokes


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


def test_bands_and_columns_tile_a_sphere():
    """Three rings and six meridians must leave 14 pieces, no more, no fewer.

    Asserted as ``== EXPECTED``. "Several" would pass on a pattern that closed
    two bands and merged the rest, and "not too many" would pass on a cutter
    that did nothing at all -- so the count is exact and a second check insists
    something was actually cut.
    """
    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=96, ring_count=48)
    mesh = bpy.context.object.data

    bm = bmesh.new()
    bm.from_mesh(mesh)
    try:
        cut, open_ends = gp.cut_strokes(bm, _pattern(), snap_distance=0.006,
                                        chain_tolerance=0.02)
        sizes = _regions(bm)
        report = {
            "expected_regions": EXPECTED,
            "regions": len(sizes),
            "sizes": sizes[:16],
            "open_ends": open_ends,
            "seam_edges": len(cut),
            "degenerate_faces": sum(1 for f in bm.faces
                                    if f.calc_area() < 1e-8),
            "zero_length_edges": sum(1 for e in bm.edges
                                     if e.calc_length() < 1e-6),
        }
    finally:
        bm.free()

    report["checks"] = {
        "every_region_closed": report["regions"] == EXPECTED,
        "no_loose_ends": report["open_ends"] == 0,
        "no_degenerate_geometry":
            report["degenerate_faces"] == 0
            and report["zero_length_edges"] == 0,
        # Two-sided. Without this, a cutter that produced nothing would satisfy
        # "not too many regions" and look fine.
        "something_was_cut": report["seam_edges"] > 0 and report["regions"] > 1,
    }
    return report


TESTS = (test_bands_and_columns_tile_a_sphere,)

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
