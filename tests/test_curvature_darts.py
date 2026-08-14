"""Checks for curvature measurement and relief dart placement.

Run against a live Blender with the add-on installed::

    python tools/launch_blender.py --port 9885
    python tools/blender_bridge.py --port 9885 < tests/test_curvature_darts.py

WHY THESE SHAPES

Every case here has an answer known in advance from differential geometry, so a
wrong result is wrong against arithmetic rather than against an impression. That
matters more than usual for darts, because "looks about right" is exactly how
the dart parameter came to be wrong three times running -- first a spacing in
millimetres, then a bend-per-dart, before landing on curvature.

    cylinder    developable. Bends as hard as you like and needs no relief at
                all, so it is the control: any method keyed to bending carpets
                it with darts. Must stay at zero however small the wedge.
    cone        developable everywhere except the tip, where the whole
                requirement concentrates. Total is 2*pi*(1 - r/slant), the pie
                slice cut from a paper disc to make one.
    sphere      total must be 4*pi exactly, by Gauss-Bonnet.
    torus       total must be 0, since its Euler characteristic is 0. Its outer
                half comes to +4*pi and its inner half to -4*pi.
    torus half  the trap. Seamed at the inner and outer equators it gives an
                annulus whose curvature cancels to zero while still needing
                more relief than anything else here. Reported by a user from a
                real session, and it is why the summary is split by sign.

Every check states both bounds. A check that can only fail in one direction is
not a check.
"""

import math
import sys
import importlib
import traceback

import bpy
import bmesh

_MOD = "bl_ext.user_default.foam_craft"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foam_craft.foam_angle_core as core   # noqa: E402

importlib.reload(core)

R_MAJOR, R_MINOR = 1.0, 0.4


def _clear():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def _bm_of_active():
    bm = bmesh.new()
    bm.from_mesh(bpy.context.object.data)
    return bm


def _deg(radians_value):
    return round(math.degrees(radians_value), 2)


# --- 1. Developable surfaces need no relief -----------------------------
def test_developable_surfaces_have_no_defect():
    """A flat sheet wraps a cylinder perfectly. A cone is flat but for its tip.

    The cone's whole requirement sits at one vertex and equals the pie slice you
    remove from a paper disc to roll one: 2*pi*(1 - r/slant).
    """
    _clear()
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.5, depth=2.0,
                                        end_fill_type='NOTHING')
    bm = _bm_of_active()
    cyl, _ = core.island_angle_defect(bm.faces)
    bm.free()

    _clear()
    radius, depth = 0.5, 1.0
    bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=radius, depth=depth,
                                    end_fill_type='NOTHING')
    bm = _bm_of_active()
    cone, per_vertex = core.island_angle_defect(bm.faces)
    apex = [d for v, d in per_vertex.items() if v.co.z > depth * 0.4]
    bm.free()

    slant = math.hypot(radius, depth)
    expected = 2 * math.pi * (1.0 - radius / slant)

    return {
        "cylinder_total_deg": _deg(cyl.total),
        "cylinder_absolute_deg": _deg(cyl.absolute),
        "cone_total_deg": _deg(cone.total),
        "cone_expected_deg": _deg(expected),
        "cone_apex_vertices": len(apex),
        "checks": {
            "cylinder_needs_nothing": abs(cyl.absolute) < 1e-4,
            "cone_matches_paper_disc":
                abs(cone.total - expected) < math.radians(0.5),
            # All of it at the tip, not spread around: exactly one vertex.
            "cone_curvature_is_at_the_apex": len(apex) == 1,
        },
    }


# --- 2. Closed surfaces match Gauss-Bonnet ------------------------------
def test_closed_surfaces_match_gauss_bonnet():
    """Sphere totals 4*pi; torus totals 0. Checked across densities, because a
    result that drifts with subdivision is measuring the mesh, not the shape."""
    spheres = {}
    for segments, rings in ((16, 8), (48, 24), (96, 48)):
        _clear()
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=segments,
                                             ring_count=rings)
        bm = _bm_of_active()
        summary, _ = core.island_angle_defect(bm.faces)
        bm.free()
        spheres["sphere_%dx%d" % (segments, rings)] = {
            "faces": segments * rings,
            "total": round(summary.total, 5),
            "ratio_to_4pi": round(summary.total / (4 * math.pi), 5),
        }

    _clear()
    bpy.ops.mesh.primitive_torus_add(major_radius=R_MAJOR,
                                     minor_radius=R_MINOR,
                                     major_segments=96, minor_segments=48)
    bm = _bm_of_active()
    torus, _ = core.island_angle_defect(bm.faces)

    # Outer half is dome-like (+4*pi), inner half saddle-like (-4*pi). Faces
    # straddling the boundary circle fall in neither set, which is why the
    # tolerance below is 3% rather than exact.
    outer = {f for f in bm.faces
             if all(math.hypot(v.co.x, v.co.y) >= R_MAJOR for v in f.verts)}
    inner = {f for f in bm.faces
             if all(math.hypot(v.co.x, v.co.y) <= R_MAJOR for v in f.verts)}
    out_sum, _ = core.island_angle_defect(outer)
    in_sum, _ = core.island_angle_defect(inner)
    bm.free()

    return {
        "spheres": spheres,
        "torus_total": round(torus.total, 5),
        "torus_outer_half": round(out_sum.total, 5),
        "torus_inner_half": round(in_sum.total, 5),
        "expected_half": round(4 * math.pi, 5),
        "checks": {
            "spheres_all_4pi": all(
                abs(s["ratio_to_4pi"] - 1.0) < 1e-3 for s in spheres.values()),
            "torus_totals_zero": abs(torus.total) < 1e-2,
            "outer_half_positive":
                abs(out_sum.total - 4 * math.pi) < 0.03 * 4 * math.pi,
            "inner_half_negative":
                abs(in_sum.total + 4 * math.pi) < 0.03 * 4 * math.pi,
            # Not merely "nonzero" -- the halves must be opposite.
            "halves_cancel": abs(out_sum.total + in_sum.total) < 1e-1,
        },
    }


# --- 3. A piece can total zero and still not lie flat -------------------
def test_mixed_curvature_is_not_flat():
    """The case a user hit in a real session, and the reason the summary is
    split by sign.

    Seaming a torus at its inner and outer equators gives annular pieces. Each
    carries dome-like curvature on its outer band and saddle-like on its inner,
    which cancel to nothing. Judged on the total, such a piece looks flat and
    is in fact the most work of any shape in this file.
    """
    _clear()
    bpy.ops.mesh.primitive_torus_add(major_radius=R_MAJOR,
                                     minor_radius=R_MINOR,
                                     major_segments=96, minor_segments=48)
    bm = _bm_of_active()
    top = {f for f in bm.faces if all(v.co.z >= -1e-9 for v in f.verts)}
    summary, _ = core.island_angle_defect(top)
    bm.free()

    return {
        "total_deg": _deg(summary.total),
        "positive_deg": _deg(summary.positive),
        "negative_deg": _deg(summary.negative),
        "absolute_deg": _deg(summary.absolute),
        "checks": {
            "total_is_misleadingly_zero": abs(summary.total) < 1e-2,
            "but_there_is_real_work": summary.absolute > math.radians(300.0),
            "both_signs_present":
                summary.positive > 0.0 and summary.negative < 0.0,
            # They must very nearly cancel; if they did not, the total would
            # have been an adequate summary and this test would be pointless.
            "signs_cancel":
                abs(summary.positive + summary.negative) < math.radians(1.0),
        },
    }


# --- 4. Darts follow curvature, not bending -----------------------------
def test_darts_follow_curvature():
    """Dart count scales with the wedge setting; total relief does not.

    Curvature is a property of the shape, so the setting only decides how finely
    a fixed total is divided. The cylinder is the control: any method keyed to
    how much the outline bends would cover it in darts.
    """
    def measure(wedge_deg):
        bpy.ops.object.mode_set(mode='EDIT')
        obj = bpy.context.object
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        runs = list(core.iter_boundary_dart_segments(
            bm, uv_layer, 0.0, 10.0, matrix=obj.matrix_world,
            dart_wedge_deg=wedge_deg, max_darts_per_run=200))
        loads = [l for r in runs for l in r.dart_loads]
        bpy.ops.object.mode_set(mode='OBJECT')
        return {
            "runs": len(runs),
            "darts": sum(len(r.dart_indices) for r in runs),
            "convex": sum(1 for l in loads if l > 0),
            "concave": sum(1 for l in loads if l < 0),
            "signed_load_deg": _deg(sum(loads)),
            "total_relief_deg": _deg(sum(abs(l) for l in loads)),
        }

    _clear()
    bpy.ops.mesh.primitive_torus_add(major_radius=R_MAJOR,
                                     minor_radius=R_MINOR,
                                     major_segments=48, minor_segments=24)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for edge in bm.edges:
        for ring in (R_MAJOR + R_MINOR, R_MAJOR - R_MINOR):
            if all(abs(math.hypot(v.co.x, v.co.y) - ring) < 1e-4
                   for v in edge.verts):
                edge.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')

    torus = {"wedge_%g" % w: measure(w) for w in (30.0, 15.0, 8.0)}

    _clear()
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.5, depth=2.0,
                                        end_fill_type='NOTHING')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    cylinder = measure(8.0)

    counts = [c["darts"] for c in torus.values()]
    reliefs = [c["total_relief_deg"] for c in torus.values()]

    out = dict(torus)
    out["cylinder_wedge_8"] = cylinder
    out["checks"] = {
        "darts_are_emitted": min(counts) > 0,
        # A smaller wedge must divide the same relief into more cuts.
        "smaller_wedge_gives_more_darts": counts == sorted(counts),
        # ...while the total relief stays put, because the shape has not changed.
        "total_relief_is_invariant":
            (max(reliefs) - min(reliefs)) < 0.05 * max(reliefs),
        # The control. Zero, not merely "fewer".
        "cylinder_needs_no_darts": cylinder["darts"] == 0,
        # A torus's curvature cancels, so its darts must balance.
        "torus_convex_and_concave_balance": all(
            c["convex"] == c["concave"] for c in torus.values()),
        "torus_signed_load_cancels": all(
            abs(c["signed_load_deg"]) < 1.0 for c in torus.values()),
    }
    return out


# --- 5. Convex and concave relief are told apart ------------------------
def test_relief_sign_is_carried_and_flagged():
    """A dart's sign has to survive to whatever draws or exports it.

    Positive relief cuts a wedge out. Negative relief needs the sheet to spread
    -- cutting a wedge there removes material the piece was already short of, so
    the two must never reach the maker as the same mark. The torus is the case
    that carries both along one boundary.
    """
    _clear()
    bpy.ops.mesh.primitive_torus_add(major_radius=R_MAJOR,
                                     minor_radius=R_MINOR,
                                     major_segments=48, minor_segments=24)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for edge in bm.edges:
        for ring in (R_MAJOR + R_MINOR, R_MAJOR - R_MINOR):
            if all(abs(math.hypot(v.co.x, v.co.y) - ring) < 1e-4
                   for v in edge.verts):
                edge.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)

    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.active
    runs = list(core.iter_boundary_dart_segments(
        bm, uv_layer, 0.0, 10.0, matrix=obj.matrix_world,
        dart_wedge_deg=15.0, max_darts_per_run=200))
    loads = [l for r in runs for l in r.dart_loads]

    # Every dart must carry a load; a zero would be a mark with no meaning.
    aligned = all(len(r.dart_loads) == len(r.dart_indices) for r in runs)

    islands = core.uv_islands(bm, uv_layer)
    saddled = 0
    for faces in islands:
        summary, _ = core.island_angle_defect(faces)
        if summary.negative < -math.radians(30.0):
            saddled += 1
    bpy.ops.object.mode_set(mode='OBJECT')

    return {
        "darts": len(loads),
        "convex": sum(1 for l in loads if l > 0),
        "concave": sum(1 for l in loads if l < 0),
        "zero_load": sum(1 for l in loads if l == 0),
        "islands": len(islands),
        "saddled_islands": saddled,
        "checks": {
            "loads_match_darts": aligned,
            "both_signs_reach_the_overlay":
                any(l > 0 for l in loads) and any(l < 0 for l in loads),
            # A mark with no load would be an instruction with no meaning.
            "no_meaningless_marks": all(l != 0 for l in loads),
            # The advisor must raise it, not leave it to the drawing alone.
            "advisor_flags_the_saddle": saddled > 0,
        },
    }


TESTS = (
    test_developable_surfaces_have_no_defect,
    test_closed_surfaces_match_gauss_bonnet,
    test_mixed_curvature_is_not_flat,
    test_darts_follow_curvature,
    test_relief_sign_is_carried_and_flagged,
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
