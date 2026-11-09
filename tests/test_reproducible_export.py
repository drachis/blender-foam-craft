"""The same mesh must produce the same pattern, every time.

    python tools/launch_blender.py --port 9885
    python tools/blender_bridge.py --port 9885 < tests/test_reproducible_export.py

This is the foundation the rest of the add-on stands on, and it was broken in
three separate places before 2026-11-09. A maker exporting one pattern twice
could get their registration marks and their relief cuts in different places,
because the code read things Blender does not promise:

  * `island_boundary_cycles` seeded its boundary walk from a set of BMLoop,
    twice over -- so an outline started at a different corner and the
    alignment ticks walked from it moved;
  * `boundary_defect_load` seeded a multi-source breadth-first search from a
    set of BMVert, and a multi-source search decides ties by seeding order,
    so the same curvature drained to a different edge and the relief darts
    landed elsewhere -- 18 darts every time, at drifting positions;
  * `_seam_chains` began a closed seam at `edge.verts[0]`, so a ring seam was
    walked one way round or the other and every By Distance mark shifted.

All three share one cause. **Set iteration and edge orientation over BMesh
elements follow identity hashing and build order, neither of which survives
rebuilding the same mesh.** The project had already learned this twice --
the convex/concave sign test, and the cut factor in foam_gp_pattern -- and
learned it again here.

So this file does not test any one of those fixes. It tests the property they
all broke: build two BMeshes from one mesh, and everything downstream must
agree exactly. A fourth instance of the same mistake fails here without
anyone having to think of it.
"""

import hashlib
import math
import sys
import traceback

import bpy
import bmesh

# Scene clearing goes through the guard, which refuses in a Blender this
# suite did not start.
_GUARD_TOOLS = r"O:\Work\BlenderPlugin\FoamCraft\tools"
if _GUARD_TOOLS not in sys.path:
    sys.path.insert(0, _GUARD_TOOLS)
import sandbox_guard  # noqa: E402

_MOD = "bl_ext.user_default.foamcraft_drachis"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foamcraft_drachis.foam_angle_core as core   # noqa: E402
import bl_ext.user_default.foamcraft_drachis.foam_export as fx         # noqa: E402


def _clear():
    """Empty the scene -- but only in a Blender this suite started."""
    sandbox_guard.clear_scene()


def _h(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _fresh(obj):
    """A new BMesh of `obj`, as every caller in the add-on builds one."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.edges.index_update()
    return bm


def _settings():
    return bpy.context.scene.foamcraft_drachis


def _gore_sphere(gores=6):
    """A sphere peeled into gores: poles, tapering pieces, curvature to drain.

    Chosen because it exercises all three broken paths at once -- closed-ish
    seams, a boundary walk through pole junctions, and enough Gaussian
    curvature that relief darts are actually placed.
    """
    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, segments=24,
                                         ring_count=16)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    step = 2.0 * math.pi / gores

    def phi(v):
        if abs(v.co.x) < 1e-9 and abs(v.co.y) < 1e-9:
            return None
        return math.atan2(v.co.y, v.co.x) % (2.0 * math.pi)

    for e in bm.edges:
        a, b = e.verts
        if abs(a.co.z - b.co.z) < 1e-9:
            continue
        pa, pb = phi(a), phi(b)
        p = pa if pa is not None else pb
        if p is None:
            continue
        if pa is not None and pb is not None:
            if min(abs(pa - pb), 2 * math.pi - abs(pa - pb)) > 1e-6:
                continue
        if abs(p - round(p / step) * step) < 1e-6:
            e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.04)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def _banded_sphere():
    """Latitude bands: annulus islands, and seams that are closed rings.

    The closed ring is what caught `_seam_chains` -- a chain with no ends has
    no natural place to start, so the start has to be chosen rather than
    found.
    """
    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, segments=16, ring_count=8)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        if abs(e.verts[0].co.z - e.verts[1].co.z) < 1e-9:
            e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.03)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def _probe(obj, settings):
    """Everything the add-on derives from one mesh, hashed.

    Taken through a *fresh* BMesh, because that is what exporting twice
    actually does and it is where identity hashing differs.
    """
    bm = _fresh(obj)
    try:
        uv = bm.loops.layers.uv.active
        islands = core.uv_islands(bm, uv, only_selected=False)

        # Island membership, order-independent, so this reports a changed
        # *partition* rather than a changed listing order.
        partition = _h("|".join(sorted(
            ",".join(sorted(str(f.index) for f in isl)) for isl in islands)))

        # The boundary walk: which loop each cycle starts at, and its length.
        cycles = []
        for isl in islands:
            for cyc in core.island_boundary_cycles(isl, uv):
                cycles.append("%d:%s" % (len(cyc), cyc[0].vert.index))
        cycle_sig = _h("|".join(cycles))

        # Where curvature drains to, which decides where relief is cut.
        loads = []
        for isl in islands:
            load = core.boundary_defect_load(isl)
            loads.append(",".join(sorted(
                "%d=%.9f" % (v.index, d) for v, d in load.items())))
        load_sig = _h("|".join(loads))

        runs = list(core.iter_boundary_dart_segments(
            bm, uv, settings.max_chord_deviation,
            settings.max_angle_variation_deg, islands=islands,
            dart_wedge_deg=settings.dart_wedge_deg,
            max_darts_per_run=settings.max_darts_per_run))
        dart_sig = _h("|".join(
            "%.9f,%.9f:%s" % (r.uv_start.x, r.uv_start.y,
                              ",".join(str(d) for d in r.dart_indices))
            for r in runs))
        dart_total = sum(len(r.dart_indices) for r in runs)

        svg = fx.generate_svg(bm, uv, settings, 1.0)
        return {
            "partition": partition,
            "cycles": cycle_sig,
            "defect_load": load_sig,
            "darts": dart_sig,
            "dart_total": dart_total,
            "runs": len(runs),
            "svg": _h(svg),
            "svg_bytes": len(svg),
        }
    finally:
        bm.free()


def _repeat(builder, settings_overrides, times=4):
    obj = builder()
    s = _settings()
    saved = {k: getattr(s, k) for k in settings_overrides}
    try:
        for k, v in settings_overrides.items():
            setattr(s, k, v)
        return [_probe(obj, s) for _ in range(times)]
    finally:
        for k, v in saved.items():
            setattr(s, k, v)


def _agreement(probes):
    """Which recorded fields are not the same in every probe."""
    return sorted(k for k in probes[0]
                  if len({p[k] for p in probes}) != 1)


def test_a_gore_sphere_exports_identically_every_time():
    """Four exports of one mesh, through four fresh BMeshes.

    The fixture is built once and never touched again, so anything that
    differs between the four is the add-on disagreeing with itself.

    Both bounds: every field identical, and the pattern must be substantial
    enough for that to mean something -- a mesh with no runs and no darts
    would agree trivially.
    """
    probes = _repeat(_gore_sphere, {
        "show_align_ticks": True,
        "tick_mode": 'PER_RUN',
        "ticks_per_run": 2,
        "show_darts": True,
    })
    differing = _agreement(probes)
    return {
        "probes": len(probes),
        "fields_that_differ": differing,
        "runs": probes[0]["runs"],
        "dart_total": probes[0]["dart_total"],
        "svg_bytes": probes[0]["svg_bytes"],
        "checks": {
            "every_derived_value_agrees": not differing,
            # Both bounds: an empty pattern agrees with itself.
            "the_pattern_has_runs": probes[0]["runs"] > 10,
            "relief_darts_were_placed": probes[0]["dart_total"] > 0,
            "the_sheet_is_not_empty": probes[0]["svg_bytes"] > 2000,
        },
    }


def test_closed_ring_seams_export_identically_every_time():
    """The By Distance case, on seams that are closed loops.

    A closed chain has no end to start from, so `_seam_chains` picks one --
    and picked it from `edge.verts[0]`, an orientation Blender does not
    promise. Walking the ring the other way moved every mark on it.
    """
    probes = _repeat(_banded_sphere, {
        "show_align_ticks": True,
        "tick_mode": 'BY_DISTANCE',
        "tick_spacing": 0.03,
    })
    differing = _agreement(probes)
    return {
        "probes": len(probes),
        "fields_that_differ": differing,
        "runs": probes[0]["runs"],
        "svg_bytes": probes[0]["svg_bytes"],
        "checks": {
            "every_derived_value_agrees": not differing,
            "the_pattern_has_runs": probes[0]["runs"] > 5,
            "the_sheet_is_not_empty": probes[0]["svg_bytes"] > 2000,
        },
    }


def test_curvature_drains_to_the_same_edges_every_time():
    """The defect routing on its own, which is the one that moves saw cuts.

    Separated from the export because it is the most consequential of the
    three: an alignment mark in the wrong place is a nuisance, a relief cut
    in the wrong place is a ruined panel.

    A multi-source breadth-first search decides ties by seeding order. The
    check is that the same interior curvature reaches the same boundary
    vertices, to nine decimal places, across four fresh BMeshes.
    """
    obj = _gore_sphere()
    signatures, totals = [], []
    for _ in range(4):
        bm = _fresh(obj)
        try:
            uv = bm.loops.layers.uv.active
            parts, total = [], 0.0
            for isl in core.uv_islands(bm, uv, only_selected=False):
                load = core.boundary_defect_load(isl)
                total += sum(abs(d) for d in load.values())
                parts.append(",".join(sorted(
                    "%.9f,%.9f,%.9f=%.9f" % (v.co.x, v.co.y, v.co.z, d)
                    for v, d in load.items())))
            signatures.append(_h("|".join(parts)))
            totals.append(round(total, 9))
        finally:
            bm.free()
    return {
        "signatures": signatures,
        "total_defect_routed": totals,
        "checks": {
            "the_same_edges_receive_the_same_curvature":
                len(set(signatures)) == 1,
            "the_totals_agree": len(set(totals)) == 1,
            # Both bounds: routing nothing would agree with itself. A sphere
            # cut into gores has real curvature to drain.
            "there_was_curvature_to_route": totals[0] > 0.1,
        },
    }


def test_islands_are_ordered_lists_in_face_table_order():
    """The root of the whole class: what `uv_islands` hands every consumer.

    It returned a list of *sets* of BMFace until 2026-11-09, and every
    consumer iterated that set -- to seed walks, to sum areas and centroids,
    to pick a representative face. Sets of BMesh elements iterate by
    Blender-side address, which a fresh BMesh changes (CLAUDE.md, "What
    survives rebuilding a BMesh"). Returning lists in face-table order is
    Blender's own idiom: `bm.faces` order was measured identical across five
    fresh BMeshes of one mesh.

    Expected values: six gores give exactly six islands (`== 6`, both
    bounds -- an all-seamed mesh would give one island per face, an
    unseamed one a single island). Every face appears exactly once.
    """
    obj = _gore_sphere()
    bm = _fresh(obj)
    try:
        uv = bm.loops.layers.uv.active
        islands = core.uv_islands(bm, uv, only_selected=False)
        table = list(bm.faces)
        position = {f: i for i, f in enumerate(table)}
        flat = [f for isl in islands for f in isl]
        inner_ordered = all(
            [position[f] for f in isl] == sorted(position[f] for f in isl)
            for isl in islands)
        firsts = [position[isl[0]] for isl in islands]
        return {
            "islands": len(islands),
            "checks": {
                "six_gores_make_six_islands": len(islands) == 6,
                "islands_are_lists": all(type(isl) is list for isl in islands),
                "every_face_exactly_once":
                    len(flat) == len(table) and len(set(flat)) == len(table),
                "faces_in_table_order_within_an_island": inner_ordered,
                "islands_in_order_of_their_first_face": firsts == sorted(firsts),
            },
        }
    finally:
        bm.free()


TESTS = (
    test_islands_are_ordered_lists_in_face_table_order,
    test_a_gore_sphere_exports_identically_every_time,
    test_closed_ring_seams_export_identically_every_time,
    test_curvature_drains_to_the_same_edges_every_time,
)

result = {}
_all_pass = True
for _test in TESTS:
    try:
        _report = _test()
        _report["PASS"] = all(_report.get("checks", {}).values())
    except Exception:
        _report = {"ERROR": traceback.format_exc().splitlines()[-6:],
                   "PASS": False}
    result[_test.__name__] = _report
    _all_pass = _all_pass and _report["PASS"]
result["ALL_PASS"] = _all_pass
