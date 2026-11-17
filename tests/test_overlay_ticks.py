"""The overlay shows the alignment ticks the export prints -- the same ones.

    python tools/launch_blender.py --port 9885
    python tools/blender_bridge.py --port 9885 < tests/test_overlay_ticks.py

The maker's standing requirement is that the exported sheet is "the UV layout
with the overlay". For the registration ticks that was simply false: the
sheet printed them and the UV editor showed none, so there was no way to see
on screen what would land on the foam.

The overlay now caches them in UV space (`foam_overlay._cache.align_ticks`),
obtained by asking `foam_marks._alignment_ticks` -- the function the export
itself uses -- and mapping its millimetres back through the sheet transform.
So the check here is parity, not a count someone chose: push the cached UV
endpoints forward through `foam_marks._Sheet.to_sheet` and they must be the
'align' strokes `foam_marks.build_marks` returns for the same mesh and
settings. Where a tick goes is foam_marks' business and is deliberately not
asserted here; that it is the *same place* on screen and paper is.

Headless: this reads the cache the rebuild fills, never a pixel. Whether the
marks reach the screen is tests/test_overlay_gui.py.
"""

import math
import sys
import traceback

import bpy
import bmesh
from mathutils import Vector

sys.path.insert(0, r"O:\Work\BlenderPlugin\FoamCraft\tools")

import sandbox_guard  # noqa: E402  (guards every _clear below)

_MOD = "bl_ext.user_default.foamcraft_drachis"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foamcraft_drachis.foam_marks as fm         # noqa: E402
import bl_ext.user_default.foamcraft_drachis.foam_angle_core as core  # noqa: E402
import bl_ext.user_default.foamcraft_drachis.foam_overlay as ov       # noqa: E402

# Two marks are the same mark when every coordinate agrees to within this
# fraction of the sheet's size.
#
# It was first written as an absolute 1e-4 mm, and that failed for a reason
# worth keeping: mathutils Vectors are 32-bit floats, and the sheet's
# millimetres are computed in them. The 2 m cube unwraps to a sheet about
# 8 m across, where adjacent float32 values are 5e-4 mm apart -- so a tenth
# of a micron is finer than the numbers being compared can be written, and
# one tick in 120 missed by a last bit. Seven significant digits is what
# there is; one part in a million of the sheet leaves room for a few
# operations' worth of rounding and is 8 microns on that 8 m sheet, still
# far below anything a pen can draw.
TOL_REL = 1e-6

BASE = {
    "show_align_ticks": True,
    "tick_mode": 'PER_RUN',
    "ticks_per_run": 2,
    "tick_spacing": 0.025,
    "tick_length": 0.006,
    "tick_numbers": False,
    "only_selected": False,
    "show_seam_numbers": True,
    "pattern_surface": 'OUTER',
}


def _clear():
    """Empty the scene, but only in a Blender this suite started."""
    sandbox_guard.clear_scene()


def _seamed_cube():
    """A 2 m cube, every edge a seam: six one-quad islands (CLAUDE.md).

    Left in edit mode, which is where the overlay works.
    """
    _clear()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    return obj


def _gore_sphere(gores=6):
    """A sphere peeled into gores, as tests/test_reproducible_export.py
    builds it: tapering pieces whose tips have little room, which is where a
    tick gets shortened -- so the length handling is exercised, not only the
    position. Left in edit mode.
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
    return obj


def _unmatched(ours, theirs, tol):
    """How many of `ours` have no partner in `theirs`, and the reverse.

    A multiset comparison with a tolerance: each record may be used once.
    Records are tuples whose leading entries are floats (compared within
    `tol` millimetres) and whose trailing entries, if any, are strings (compared
    exactly). Rounding both sides and comparing the rounded values would be
    shorter and wrong -- two numbers a billionth apart can straddle a
    rounding boundary.
    """
    def same(a, b):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if isinstance(x, str) or isinstance(y, str):
                if x != y:
                    return False
            elif abs(x - y) > tol:
                return False
        return True

    left = list(theirs)
    missing = 0
    for rec in ours:
        for i, other in enumerate(left):
            if same(rec, other):
                del left[i]
                break
        else:
            missing += 1
    return missing, len(left)


def _measure(obj, **overrides):
    """Rebuild the overlay and build the export's marks, for one mesh.

    Both are taken from the same edit BMesh in the same state, so the only
    thing that can differ between them is the code under test.
    """
    s = bpy.context.scene.foamcraft_drachis
    wanted = dict(BASE)
    wanted.update(overrides)
    saved = {k: getattr(s, k) for k in wanted}
    try:
        for k, v in wanted.items():
            setattr(s, k, v)
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        scale_length = bpy.context.scene.unit_settings.scale_length

        # As tools/golden_marks.py does: _rebuild returns nothing and fills
        # the module's _cache.
        ov._rebuild([(obj, bm, uv_layer)], s)
        cached_ticks = list(ov._cache.align_ticks)
        cached_labels = list(ov._cache.align_labels)

        # The cache must be plain numbers: it outlives the BMesh.
        plain = all(isinstance(c, float) for t in cached_ticks for c in t) \
            and all(isinstance(l[0], float) and isinstance(l[1], float)
                    and isinstance(l[2], str) for l in cached_labels)

        sheet = fm._Sheet(core.uv_islands(bm, uv_layer, only_selected=False),
                          uv_layer, scale_length, obj.matrix_world)

        def mm(u, v):
            p = sheet.to_sheet(Vector((u, v)))
            return (p.x, p.y)

        overlay_ticks = [mm(u0, v0) + mm(u1, v1)
                         for u0, v0, u1, v1 in cached_ticks]
        overlay_labels = [mm(u, v) + (text,) for u, v, text in cached_labels]

        marks = fm.build_marks(bm, uv_layer, s, scale_length,
                               obj.matrix_world)
        export_ticks = [st.points[0] + st.points[-1]
                        for st in marks.strokes if st.kind == 'align']
        export_labels = [(l.x, l.y, l.text)
                         for l in marks.labels if l.kind == 'align']
        lengths = [math.hypot(t[2] - t[0], t[3] - t[1])
                   for t in overlay_ticks]
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
    return {
        "overlay_ticks": overlay_ticks,
        "export_ticks": export_ticks,
        "overlay_labels": overlay_labels,
        "export_labels": export_labels,
        "plain": plain,
        "longest_mm": max(lengths) if lengths else 0.0,
        "tol_mm": TOL_REL * max(marks.width_mm, marks.height_mm),
        "tick_length_mm": wanted["tick_length"] * scale_length * 1000.0,
    }


def _parity(fixture, **overrides):
    obj = fixture()
    m = _measure(obj, **overrides)
    bpy.ops.object.mode_set(mode='OBJECT')
    tol = m["tol_mm"]
    only_overlay, only_export = _unmatched(m["overlay_ticks"],
                                           m["export_ticks"], tol)
    lab_overlay, lab_export = _unmatched(m["overlay_labels"],
                                         m["export_labels"], tol)
    # How far the worst-placed tick is from its nearest printed stroke, so
    # the log says how much of the tolerance is actually being used.
    worst = max((min(max(abs(a - b) for a, b in zip(t, e))
                     for e in m["export_ticks"])
                 for t in m["overlay_ticks"]), default=0.0)         if m["export_ticks"] else 0.0
    n = len(m["overlay_ticks"])
    return {
        "overlay_ticks": n,
        "export_ticks": len(m["export_ticks"]),
        "ticks_only_on_screen": only_overlay,
        "ticks_only_on_paper": only_export,
        "overlay_labels": len(m["overlay_labels"]),
        "export_labels": len(m["export_labels"]),
        "longest_mm": round(m["longest_mm"], 4),
        "tolerance_mm": tol,
        "worst_mismatch_mm": worst,
        "checks": {
            # Lower bound: an empty cache agrees with an empty sheet, and
            # would prove nothing.
            "there_are_ticks": n > 0,
            # Upper bound and identity together: the same number, and every
            # one of them paired off against a printed stroke.
            "same_count_as_the_sheet": n == len(m["export_ticks"]),
            "none_only_on_screen": only_overlay == 0,
            "none_only_on_paper": only_export == 0,
            "labels_match_the_sheet":
                len(m["overlay_labels"]) == len(m["export_labels"])
                and lab_overlay == 0 and lab_export == 0,
            # The sheet draws a tick `tick_length` long, from the cut line
            # inward, shortened where a piece is narrow. (Until 2026-11-17
            # it also had a tail a quarter as long across the line, and the
            # upper bound here was 1.25x; the maker asked for marks that
            # only go inward.) So none may exceed 1x, and the longest must
            # be a real fraction of it -- a tick mapped back at the wrong
            # scale fails one bound or the other.
            "length_is_the_physical_one":
                0.2 * m["tick_length_mm"] < m["longest_mm"]
                <= m["tick_length_mm"] + 2 * tol,
            "cache_holds_plain_numbers": m["plain"],
        },
    }


def test_cube_per_run_ticks_match_the_sheet():
    return _parity(_seamed_cube, tick_mode='PER_RUN', ticks_per_run=2)


def test_cube_by_distance_ticks_match_the_sheet():
    # A 2 m cube edge at 0.4 m spacing: several marks per seam.
    return _parity(_seamed_cube, tick_mode='BY_DISTANCE', tick_spacing=0.4)


def test_gore_sphere_per_run_ticks_match_the_sheet():
    return _parity(_gore_sphere, tick_mode='PER_RUN', ticks_per_run=2)


def test_gore_sphere_by_distance_ticks_match_the_sheet():
    return _parity(_gore_sphere, tick_mode='BY_DISTANCE', tick_spacing=0.03)


def test_tick_numbers_match_the_sheet():
    """With numbering on, each index sits where the sheet prints it.

    Expected: one label per tick, on both -- `_alignment_ticks` emits a
    Label with every Stroke when `tick_numbers` is set.
    """
    report = _parity(_gore_sphere, tick_mode='PER_RUN', ticks_per_run=2,
                     tick_numbers=True)
    report["checks"]["one_label_per_tick"] = \
        report["overlay_labels"] == report["overlay_ticks"] > 0
    return report


def test_ticks_off_caches_none():
    """Alignment Ticks off: nothing cached, and nothing printed either.

    The 'on' half is in the same test so the zero has something to be
    smaller than -- a cache that was never filled also holds none.
    """
    obj = _seamed_cube()
    on = _measure(obj, show_align_ticks=True)
    off = _measure(obj, show_align_ticks=False, tick_numbers=True)
    bpy.ops.object.mode_set(mode='OBJECT')
    return {
        "on": len(on["overlay_ticks"]),
        "off": len(off["overlay_ticks"]),
        "checks": {
            "on_has_ticks": len(on["overlay_ticks"]) > 0,
            "off_has_no_ticks": len(off["overlay_ticks"]) == 0,
            "off_has_no_labels": len(off["overlay_labels"]) == 0,
            "sheet_agrees_it_is_off": len(off["export_ticks"]) == 0,
        },
    }


def test_only_selected_hides_ticks_without_moving_them():
    """Only Selected shows the selected pieces' ticks, where they print.

    The decision is made on the whole mesh whatever is selected (the export
    ignores selection); selection only hides. On the all-seamed cube with
    one face selected the answer is arithmetic: six congruent one-face
    pieces carry the same number of ticks each, so the selected piece shows
    exactly a sixth of them -- and every one it shows is a printed stroke.
    """
    obj = _seamed_cube()
    full = _measure(obj, only_selected=False)

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    for f in bm.faces:
        f.select_set(False)
    bm.faces[0].select_set(True)
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)
    selected_faces = sum(1 for f in bm.faces if f.select)

    part = _measure(obj, only_selected=True)
    bpy.ops.object.mode_set(mode='OBJECT')

    n_full = len(full["overlay_ticks"])
    n_part = len(part["overlay_ticks"])
    strays, _ = _unmatched(part["overlay_ticks"], part["export_ticks"],
                           part["tol_mm"])
    return {
        "full": n_full,
        "selected_piece": n_part,
        "selected_faces": selected_faces,
        "shown_but_not_printed": strays,
        "checks": {
            "fixture_has_one_face_selected": selected_faces == 1,
            "full_has_ticks": n_full > 0,
            "one_sixth_of_them_shown": n_part * 6 == n_full,
            "every_shown_tick_is_printed": strays == 0,
            # The sheet itself is unchanged by selection.
            "sheet_ignores_selection":
                len(part["export_ticks"]) == len(full["export_ticks"]),
        },
    }


def test_tick_settings_are_in_the_cache_signature():
    """Changing any tick setting must read as a different signature.

    The ticks are cached; a setting missing from the signature leaves the
    old marks on screen until something unrelated forces a rebuild. Each of
    the settings is flipped alone and compared against the unchanged baseline,
    and the baseline is taken twice so "differs" is not satisfied by a
    signature that never repeats.
    """
    obj = _seamed_cube()
    s = bpy.context.scene.foamcraft_drachis
    saved = {k: getattr(s, k) for k in BASE}
    changes = {
        "show_align_ticks": False,
        "tick_mode": 'BY_DISTANCE',
        "ticks_per_run": 3,
        "tick_spacing": 0.05,
        "tick_length": 0.009,
        "tick_numbers": True,
        # The bevel labels' two settings (2026-11-17): what a label says,
        # and which way its arrow points.
        "show_seam_numbers": False,
        "pattern_surface": 'INNER',
    }
    try:
        for k, v in BASE.items():
            setattr(s, k, v)
        bm = bmesh.from_edit_mesh(obj.data)
        pairs = [(obj, bm, bm.loops.layers.uv.active)]
        base = ov._signature(pairs, s)
        again = ov._signature(pairs, s)
        differs = {}
        for k, v in changes.items():
            setattr(s, k, v)
            differs[k] = ov._signature(pairs, s) != base
            setattr(s, k, BASE[k])
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
    bpy.ops.object.mode_set(mode='OBJECT')
    checks = {"signature_is_repeatable": base == again}
    for k, changed in differs.items():
        checks["%s_changes_the_signature" % k] = changed
    return {"settings_checked": len(differs), "checks": checks}


def _bevel_measure(obj, **overrides):
    """The overlay's cached bevel labels, arrows and run-end marks, pushed
    forward to sheet mm, beside the sheet's own -- one edit BMesh, one
    state, as `_measure` does for the ticks."""
    s = bpy.context.scene.foamcraft_drachis
    wanted = dict(BASE)
    wanted.update(show_brackets=True, show_heat_form=True,
                  show_seam_numbers=True, pattern_surface='OUTER')
    wanted.update(overrides)
    saved = {k: getattr(s, k) for k in wanted}
    try:
        for k, v in wanted.items():
            setattr(s, k, v)
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        scale_length = bpy.context.scene.unit_settings.scale_length
        ov._rebuild([(obj, bm, uv_layer)], s)
        cached = list(ov._cache.bevel_marks)
        cached_ends = list(ov._cache.run_end_marks)
        sheet = fm._Sheet(core.uv_islands(bm, uv_layer, only_selected=False),
                          uv_layer, scale_length, obj.matrix_world)

        def mm(u, v):
            p = sheet.to_sheet(Vector((u, v)))
            return (p.x, p.y)

        screen_labels = [mm(m.u, m.v) + (m.text,) for m in cached]
        # The blade arrow is a glyph in the label since 2026-11-17 (the
        # drawn 'bevel_arrow' stroke is retired), so "arrows" now compares
        # every glyph-bearing label's place and turn, screen against paper.
        glyphs = ("\u2192", "\u2190")
        screen_arrows = [mm(m.u, m.v) + (m.rotation_deg,) for m in cached
                         if any(g in m.text for g in glyphs)]
        screen_ends = [mm(u0, v0) + mm(u1, v1)
                       for u0, v0, u1, v1 in cached_ends]
        marks = fm.build_marks(bm, uv_layer, s, scale_length,
                               obj.matrix_world)
        paper_labels = [(l.x, l.y, l.text) for l in marks.labels
                        if l.kind in ('angle', 'heat')]
        paper_arrows = [(l.x, l.y, l.rotation_deg) for l in marks.labels
                        if l.kind == 'angle'
                        and any(g in l.text for g in glyphs)]
        paper_ends = [st.points[0] + st.points[-1] for st in marks.strokes
                      if st.kind == 'bracket']
        plain = all(
            isinstance(m.u, float) and isinstance(m.v, float)
            and isinstance(m.text, str)
            and all(isinstance(x, float) for pt in m.arrow for x in pt)
            for m in cached) and all(isinstance(c, float)
                                     for t in cached_ends for c in t)
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
    tol = TOL_REL * max(marks.width_mm, marks.height_mm)
    return {
        "labels": (screen_labels, paper_labels),
        "arrows": (screen_arrows, paper_arrows),
        "ends": (screen_ends, paper_ends),
        "plain": plain, "tol": tol,
    }


def _bevel_parity(fixture, want_ends):
    """Cached on-screen bevel labels, arrows and run-end marks equal the
    sheet's as multisets. `want_ends`: whether this fixture has run ends
    part way along a seam (the gore sphere does; on a cube every run end
    is a corner, and nothing may be cached there)."""
    obj = fixture()
    m = _bevel_measure(obj)
    bpy.ops.object.mode_set(mode='OBJECT')
    tol = m["tol"]
    out, checks = {}, {}
    for key in ("labels", "arrows", "ends"):
        screen, paper = m[key]
        only_screen, only_paper = _unmatched(screen, paper, tol)
        out[key] = {"screen": len(screen), "paper": len(paper),
                    "only_on_screen": only_screen,
                    "only_on_paper": only_paper}
        checks[key + "_match_the_sheet"] = (
            len(screen) == len(paper) and only_screen == 0
            and only_paper == 0)
    checks["there_are_labels"] = out["labels"]["screen"] > 0
    checks["there_are_arrows"] = out["arrows"]["screen"] > 0
    if want_ends:
        checks["there_are_mid_seam_run_end_marks"] = out["ends"]["screen"] > 0
    else:
        checks["no_run_end_marks_at_corners"] = out["ends"]["screen"] == 0
    checks["cache_holds_plain_values"] = m["plain"]
    out["tolerance_mm"] = tol
    out["checks"] = checks
    return out


def test_cube_bevel_labels_match_the_sheet():
    """Cube: 24 labels "<n>, 45\u00b0" and 24 arrows on screen, exactly where and
    as the sheet prints them, and no run-end mark at all -- every run end
    on a cube is a corner, where the maker wants nothing drawn. (The screen
    used to draw a T-bar at both ends of every run's bracket.) Since
    2026-11-17 the 24 arrows are the glyphs in the 24 turned labels, and
    the screen's turn must equal the sheet's."""
    report = _bevel_parity(_seamed_cube, want_ends=False)
    report["checks"]["twenty_four_of_each"] = \
        report["labels"]["screen"] == 24 and report["arrows"]["screen"] == 24
    return report


def test_gore_sphere_bevel_labels_match_the_sheet():
    """Gore sphere: labels, arrows and the mid-seam run-end marks on
    screen are the sheet's, as multisets, and there are some of each."""
    return _bevel_parity(_gore_sphere, want_ends=True)


TESTS = (
    test_cube_per_run_ticks_match_the_sheet,
    test_cube_by_distance_ticks_match_the_sheet,
    test_gore_sphere_per_run_ticks_match_the_sheet,
    test_gore_sphere_by_distance_ticks_match_the_sheet,
    test_tick_numbers_match_the_sheet,
    test_ticks_off_caches_none,
    test_only_selected_hides_ticks_without_moving_them,
    test_tick_settings_are_in_the_cache_signature,
    test_cube_bevel_labels_match_the_sheet,
    test_gore_sphere_bevel_labels_match_the_sheet,
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
