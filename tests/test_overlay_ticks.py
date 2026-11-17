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
import bl_ext.user_default.foamcraft_drachis.foam_layout as fl        # noqa: E402

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

        # Place, turn, size and text (2026-11-17): the labels are laid out
        # by foam_layout before they print -- slid, shrunk one size per
        # piece, or put outside on a leader -- and the screen must show the
        # laid-out label, not the one `_bevel_marks` first proposed. Size is
        # in mm on both sides (`_BevelMark.size_mm` is the printed size).
        screen_labels = [mm(m.u, m.v) + (m.rotation_deg, m.size_mm, m.text)
                         for m in cached]
        screen_leaders = [mm(*m.leader[0]) + mm(*m.leader[1])
                          for m in cached if m.leader]
        # The blade arrow is a glyph in the label since 2026-11-17 (the
        # drawn 'bevel_arrow' stroke is retired), so "arrows" now compares
        # every glyph-bearing label's place and turn, screen against paper.
        glyphs = ("\u2192", "\u2190")
        screen_arrows = [mm(m.u, m.v) + (m.rotation_deg,) for m in cached
                         if any(g in m.text for g in glyphs)]
        screen_ends = [mm(u0, v0) + mm(u1, v1)
                       for u0, v0, u1, v1 in cached_ends]
        # What every export prints: build_marks lays the labels out by
        # default.
        marks = fm.build_marks(bm, uv_layer, s, scale_length,
                               obj.matrix_world)
        paper_labels = [(l.x, l.y, l.rotation_deg, l.size_mm, l.text)
                        for l in marks.labels
                        if l.kind in ('angle', 'heat')]
        paper_leaders = [st.points[0] + st.points[-1] for st in marks.strokes
                         if st.kind == 'leader']
        # The layout's own account of the same sheet, from the raw marks,
        # so a fixture can say whether it actually slid or led anything.
        layout_report = {}
        laid = fl.layout_labels(
            fm.build_marks(bm, uv_layer, s, scale_length, obj.matrix_world,
                           layout=False),
            float(s.min_label_mm), layout_report)
        default_is_laid_out = laid == marks
        full_mm = ov._cache.angle_font_mm
        paper_arrows = [(l.x, l.y, l.rotation_deg) for l in marks.labels
                        if l.kind == 'angle'
                        and any(g in l.text for g in glyphs)]
        paper_ends = [st.points[0] + st.points[-1] for st in marks.strokes
                      if st.kind == 'bracket']
        plain = all(
            isinstance(m.u, float) and isinstance(m.v, float)
            and isinstance(m.text, str)
            and all(isinstance(x, float) for pt in m.arrow for x in pt)
            and isinstance(m.size_mm, float)
            and all(isinstance(x, float) for pt in m.leader for x in pt)
            for m in cached) and all(isinstance(c, float)
                                     for t in cached_ends for c in t)
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
    tol = TOL_REL * max(marks.width_mm, marks.height_mm)
    hows = layout_report.get("how", [])
    return {
        "labels": (screen_labels, paper_labels),
        "arrows": (screen_arrows, paper_arrows),
        "ends": (screen_ends, paper_ends),
        "leaders": (screen_leaders, paper_leaders),
        "plain": plain, "tol": tol,
        "how": {h: hows.count(h) for h in sorted(set(hows))},
        "default_is_laid_out": default_is_laid_out,
        "full_mm": full_mm,
        "sizes_mm": sorted({round(m.size_mm, 4) for m in cached}),
    }


def _bevel_parity(fixture, want_ends, **overrides):
    """Cached on-screen bevel labels (place, turn, size, text), arrows,
    run-end marks and leaders equal the sheet's as multisets. `want_ends`:
    whether this fixture has run ends part way along a seam (the gore
    sphere does; on a cube every run end is a corner, and nothing may be
    cached there)."""
    obj = fixture()
    m = _bevel_measure(obj, **overrides)
    bpy.ops.object.mode_set(mode='OBJECT')
    tol = m["tol"]
    out, checks = {}, {}
    out["how"] = m["how"]
    out["sizes_mm"] = m["sizes_mm"]
    out["full_mm"] = m["full_mm"]
    # The exports print the laid-out sheet: build_marks' default is exactly
    # layout_labels applied to its raw marks, not some other pass.
    checks["build_marks_default_is_the_layout"] = m["default_is_laid_out"]
    for key in ("labels", "arrows", "ends", "leaders"):
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


def _crowded_box():
    """The crowded fixture, from tests/test_label_layout.py's two known
    cases at once: a 2 x 2 x 0.2 m box, every edge a seam -- its side strips
    are 2000 x 200 mm, and the labels of a strip's two long edges each run
    right across it and meet, so the layout must SLIDE them (measured there:
    they collide before) -- with the top square shrunk to 0.5% in UV, about
    10 mm on the sheet, narrower than one label at the 2.5 mm floor, so its
    four labels must go out on LEADERS. Left in edit mode."""
    _clear()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.object
    obj.scale = (1.0, 1.0, 0.1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    top = max(obj.data.polygons, key=lambda p: p.normal.z)
    uv = obj.data.uv_layers.active.data
    idx = list(top.loop_indices)
    cu = sum(uv[i].uv.x for i in idx) / len(idx)
    cv = sum(uv[i].uv.y for i in idx) / len(idx)
    for i in idx:
        uv[i].uv = (cu + (uv[i].uv.x - cu) * 0.005,
                    cv + (uv[i].uv.y - cv) * 0.005)
    bpy.ops.object.mode_set(mode='EDIT')
    return obj


def test_crowded_box_laid_out_labels_and_leaders_match_the_sheet():
    """The crowded box: the screen shows the LAID-OUT labels -- where the
    layout slid them, at the size it shrank them to, turned as printed --
    and the leaders of the four that went outside, all equal to the sheet's
    as multisets.

    Piece numbers are off: on a 10 mm piece a number covers the whole
    piece and every leader would cross it, so the tiny piece's labels come
    back 'unplaced' instead (pinned in test_label_layout.py). Expected from
    that file's measurements: some labels 'slid', exactly 4 'leader', 0
    'unplaced', and the screen holds the same 4 leaders.
    """
    report = _bevel_parity(_crowded_box, want_ends=False,
                           show_island_ids=False, show_align_ticks=True)
    how = report["how"]
    c = report["checks"]
    c["layout_slid_some_labels"] = how.get("slid", 0) > 0
    c["layout_led_exactly_four"] = how.get("leader", 0) == 4
    c["nothing_unplaced"] = how.get("unplaced", 0) == 0
    c["four_leaders_on_screen"] = report["leaders"]["screen"] == 4
    # Some piece was shrunk, and the screen carries the shrunk size: at
    # least two sizes cached, the largest the sheet's full angle size.
    c["a_shrunk_size_is_cached"] = len(report["sizes_mm"]) >= 2 and \
        abs(max(report["sizes_mm"]) - report["full_mm"]) < 1e-3 * max(
            report["full_mm"], 1e-9)
    return report


def test_label_px_scales_with_printed_size():
    """`_label_px`: the full angle size draws at angle_px, half the size at
    half (rounded), and the 6..400 px clamp holds both ways. Arithmetic,
    so a wrong ratio cannot pass."""
    f = ov._label_px
    return {
        "checks": {
            "full_size_is_angle_px": f(40, 8.0, 8.0) == 40,
            "half_size_is_half": f(40, 4.0, 8.0) == 20,
            "unknown_size_is_angle_px": f(40, 0.0, 8.0) == 40
            and f(40, 4.0, 0.0) == 40,
            "floor_is_six": f(40, 0.1, 8.0) == 6,
            "ceiling_is_four_hundred": f(300, 16.0, 8.0) == 400,
        },
    }


SNAPSHOT = r"O:\Work\BlenderPlugin\FoamCraft\tests\fixtures\beast_skull_snapshot.blend"
HEADS = ("head_simplified", "head_simplified.001")


def _copy_settings(src, dst):
    """Every writable non-pointer setting, src -> dst (as
    tests/test_multi_object.py does for the same snapshot)."""
    for prop in src.bl_rna.properties:
        name = prop.identifier
        if name in ("rna_type", "name") or prop.is_readonly \
                or prop.type == 'POINTER':
            continue
        try:
            setattr(dst, name, getattr(src, name))
        except (AttributeError, TypeError, ValueError):
            pass


def _world_height(obj):
    """World-space bounding-box extent in z, from the vertices through
    matrix_world, in Blender units."""
    mw = obj.matrix_world
    zs = [(mw @ v.co).z for v in obj.data.vertices]
    return (max(zs) - min(zs)) if zs else 0.0


def _append_heads():
    """Both heads and the snapshot's first scene, appended from a temp copy
    (opening a file inside a bridge request replaces the session; the
    git-ignored original is never touched, and nothing is saved), linked
    into the current scene. (objs, snapshot scene, temp path), or
    (None, None, reason) when the fixture is absent."""
    import os
    import shutil
    import tempfile
    if not os.path.exists(SNAPSHOT):
        return None, None, ("fixture %s is absent (git-ignored, local only)"
                            % SNAPSHOT)
    _clear()
    # An earlier test's appended heads, unlinked but still in bpy.data,
    # would make these arrive renamed ".002" and the test skip (measured:
    # the second snapshot test in this file skipped until this purge). The
    # appended snapshot scene is what still uses them, so it goes first.
    for sc in [sc for sc in bpy.data.scenes if sc != bpy.context.scene]:
        bpy.data.scenes.remove(sc)
    # Something else still holds them (users never reached 0, measured), so
    # the old heads are removed by name, unlinked from wherever they are.
    for name in HEADS:
        old = bpy.data.objects.get(name)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
    tmp = os.path.join(tempfile.mkdtemp(prefix="foam_overlay_"), "copy.blend")
    shutil.copyfile(SNAPSHOT, tmp)
    with bpy.data.libraries.load(tmp, link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n in HEADS]
        dst.scenes = list(src.scenes[:1])
    objs = {o.name: o for o in dst.objects if o is not None}
    if set(objs) != set(HEADS):
        return None, None, "snapshot lacks %r; appended %r" % (
            HEADS, sorted(objs))
    for o in objs.values():
        bpy.context.scene.collection.objects.link(o)
    snap_scene = next((sc for sc in dst.scenes if sc is not None), None)
    return objs, snap_scene, tmp


def _operator_marks(path):
    """The `Marks` the image export operator hands its renderer, captured
    by wrapping the `render_marks` its execute() calls (looked up in the
    module the registered class came from, which is not the freshly
    imported one above). (status, marks or None, render info)."""
    op_cls = bpy.types.FOAMCRAFT_DRACHIS_OT_export_pattern_image
    g = op_cls.execute.__globals__
    original = g["render_marks"]
    seen = {}

    def spy(marks, path, **kw):
        seen["marks"] = marks
        info = original(marks, path, **kw)
        seen["info"] = info
        return info
    g["render_marks"] = spy
    try:
        status = bpy.ops.foamcraft_drachis.export_pattern_image(
            filepath=path, file_format='PNG', resolution_mode='DPI', dpi=10)
    finally:
        g["render_marks"] = original
    return status, seen.get("marks"), seen.get("info") or {}


def test_snapshot_heads_overlay_is_the_printed_layout():
    """The maker's two heads in multi-object edit mode: the overlay's cached
    bevel labels and leaders equal, as multisets, the laid-out labels the
    IMAGE OPERATOR hands its renderer -- captured by wrapping the
    operator's own `render_marks` while it runs, so this is the real export
    path, not a re-derivation. And that laid-out sheet has 0 label/label
    overlaps (foam_layout's own oriented boxes).

    From a temp copy of the snapshot, appended (opening a file inside a
    bridge request replaces the session), with the snapshot scene's
    settings and unit scale copied on; never saved. Only Selected is
    forced off (every piece is then on screen) and the image is drawn at
    10 dpi, small, since only its marks are compared.

    Also reports -- measures, asserts nothing about -- the scale question
    for the maker: the head's world-space height, its parent's scale, the
    scene's unit scale, and what size the sheet prints at.
    """
    import os
    objs, snap_scene, tmp = _append_heads()
    if objs is None:
        return {"skipped": tmp, "checks": {}}
    s = bpy.context.scene.foamcraft_drachis
    file_unit_scale = None
    if snap_scene is not None:
        file_unit_scale = snap_scene.unit_settings.scale_length
        if getattr(snap_scene, "foamcraft_drachis", None):
            _copy_settings(snap_scene.foamcraft_drachis, s)
        bpy.context.scene.unit_settings.scale_length = file_unit_scale
    saved_only = s.only_selected
    s.only_selected = False
    a, b = objs[HEADS[0]], objs[HEADS[1]]
    # What a reader who skips the update sees (test_label_layout.py's
    # snapshot path appends and reads matrix_world without one).
    before_update_scale = [round(x, 4) for x in a.matrix_world.to_scale()]
    # The parent ("retopo") comes along unlinked; evaluate before reading
    # matrix_world (tests/test_multi_object.py found it stale otherwise).
    bpy.context.view_layer.update()
    parent = a.parent
    mw = a.matrix_world
    world = [mw @ v.co for v in a.data.vertices]
    local = [v.co for v in a.data.vertices]
    scale_report = {
        "head_world_scale_before_view_layer_update": before_update_scale,
        "head_world_extent_xyz_bu": [
            round(max(p[i] for p in world) - min(p[i] for p in world), 4)
            for i in range(3)],
        "head_mesh_extent_xyz_bu": [
            round(max(p[i] for p in local) - min(p[i] for p in local), 3)
            for i in range(3)],
        "head_world_height_bu": round(_world_height(a), 4),
        "head_dimensions_bu": [round(x, 4) for x in a.dimensions],
        "head_local_scale": [round(x, 4) for x in a.scale],
        "parent": parent.name if parent else None,
        "parent_scale": [round(x, 4) for x in parent.scale] if parent else None,
        "parent_world_scale": [round(x, 4)
                               for x in parent.matrix_world.to_scale()]
        if parent else None,
        "parent_rotation_deg": [round(math.degrees(x), 2) for x in
                                parent.matrix_world.to_euler()]
        if parent else None,
        "head_world_scale": [round(x, 4) for x in a.matrix_world.to_scale()],
        "scene_unit_scale_length": file_unit_scale,
        "scene_length_unit": snap_scene.unit_settings.length_unit
        if snap_scene else None,
    }

    bpy.ops.object.select_all(action='DESELECT')
    a.select_set(True)
    b.select_set(True)
    bpy.context.view_layer.objects.active = a
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    pairs = ov._edit_meshes(bpy.context)
    ov._rebuild(pairs, s)
    cached = list(ov._cache.bevel_marks)

    # The image operator's own marks.
    status, marks, info = _operator_marks(
        os.path.join(os.path.dirname(tmp), "heads.png"))
    sheet, _parts = fm._layout_sheet(
        [fm.SheetItem(bm, uv, o.matrix_world.copy(),
                      fm.object_prefix(o.name, o.foamcraft_drachis_prefix),
                      o.name) for o, bm, uv in pairs],
        s, bpy.context.scene.unit_settings.scale_length)
    del _parts
    bpy.ops.object.mode_set(mode='OBJECT')
    s.only_selected = saved_only

    if marks is None:
        return {"status": list(status), "checks": {"operator_rendered": False}}

    def mm(u, v):
        p = sheet.to_sheet(Vector((u, v)))
        return (p.x, p.y)
    screen = [mm(m.u, m.v) + (m.rotation_deg, m.size_mm, m.text)
              for m in cached if m.tone != 'heat']
    paper = [(l.x, l.y, l.rotation_deg, l.size_mm, l.text)
             for l in marks.labels if l.kind == 'angle']
    screen_leaders = [mm(*m.leader[0]) + mm(*m.leader[1])
                      for m in cached if m.leader]
    paper_leaders = [st.points[0] + st.points[-1] for st in marks.strokes
                     if st.kind == 'leader']
    tol = TOL_REL * max(marks.width_mm, marks.height_mm)
    only_s, only_p = _unmatched(screen, paper, tol)
    lead_s, lead_p = _unmatched(screen_leaders, paper_leaders, tol)

    quads = [fl._label_quad(l) for l in marks.labels if l.kind == 'angle']
    overlaps = sum(1 for i in range(len(quads))
                   for j in range(i + 1, len(quads))
                   if fl._quads_overlap(quads[i], quads[j]))
    scale_report.update({
        "sheet_mm": [round(marks.width_mm, 1), round(marks.height_mm, 1)],
        # World z extent at the file's unit scale, in mm. (The head lies
        # along world y: its longest side is head_world_extent_xyz_bu[1].)
        "head_world_z_height_mm": round(
            _world_height(a) * file_unit_scale * 1000.0, 1)
        if file_unit_scale else None,
        "image_dpi": info.get("dpi"), "image_pixels": info.get("pixels"),
        "angle_full_mm": round(ov._cache.angle_font_mm, 3),
        "min_label_mm": float(s.min_label_mm),
    })
    return {
        "scale": scale_report,
        "labels": {"screen": len(screen), "paper": len(paper),
                   "only_on_screen": only_s, "only_on_paper": only_p},
        "leaders": {"screen": len(screen_leaders),
                    "paper": len(paper_leaders),
                    "only_on_screen": lead_s, "only_on_paper": lead_p},
        "label_label_overlaps": overlaps,
        "sizes_mm": sorted({round(l.size_mm, 3) for l in marks.labels
                            if l.kind == 'angle'}),
        "tolerance_mm": tol,
        "checks": {
            "operator_finished": status == {'FINISHED'},
            "there_are_labels": len(paper) > 0,
            "labels_match_the_printed_layout":
                len(screen) == len(paper) and only_s == 0 and only_p == 0,
            "leaders_match_the_printed_layout":
                len(screen_leaders) == len(paper_leaders)
                and lead_s == 0 and lead_p == 0,
            "no_label_label_overlap": overlaps == 0,
        },
    }


# --- piece names and leaders on the maker's heads ----------------------------
#
# An independent model of a label's box, written out again rather than
# imported from foam_layout so a wrong box there cannot agree with itself:
# `len(text) * size * GLYPH_ADVANCE_EM` wide by `size` tall, centred on
# (x, y) for a 'middle' label, turned `rotation_deg` counter-clockwise on
# the page (y down). Piece names do not turn.

# A piece name is never shrunk below the maker's Smallest Label
# (`min_label_mm`), the legible floor the bevel labels already keep, nor
# above the set Number Size. (It was a quarter of the set size: 5.9 mm on
# this 1.35 m sheet, which left 12 of the 52 names -- "HE P" one of them --
# still too big for their pieces and over the ticks, measured here.)
#
# "Clear" means clear by the layout's own margin, 0.15 of the text height
# around the box (foam_layout.CLEAR_EM), so a name the model says fits and
# one this audit passes are the same claim; with no margin, two names the
# model put at the floor for touching a tick by a hair measured clear.
NAME_CLEAR_EM = 0.15
# The shortest leader that reads as a line rather than a dot, in text
# heights of its own label -- the maker saw a ~1 mm stub at "32, 17°" on
# piece HE P and took it for a stray mark (2026-11-17).
MIN_LEADER_TEXT_HEIGHTS = 1.5


def _box(x, y, size, text, rotation_deg=0.0, valign='middle', pad=0.0):
    a = math.radians(rotation_deg or 0.0)
    rx, ry = math.cos(a), -math.sin(a)
    ux, uy = -math.sin(a), -math.cos(a)
    hw = len(text) * size * fm.GLYPH_ADVANCE_EM / 2
    hh = size / 2
    if valign == 'baseline':
        x, y = x + ux * hh, y + uy * hh
    hw, hh = hw + pad, hh + pad
    return tuple((x + sx * rx * hw + sy * ux * hh,
                  y + sx * ry * hw + sy * uy * hh)
                 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)))


def _lbox(label):
    return _box(label.x, label.y, label.size_mm, label.text,
                label.rotation_deg, label.valign)


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_cross(p, q, r, s, eps=1e-9):
    d1, d2 = _orient(r, s, p), _orient(r, s, q)
    d3, d4 = _orient(p, q, r), _orient(p, q, s)
    return (((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps))
            and ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)))


def _in_box(p, quad):
    sg = [_orient(quad[i], quad[(i + 1) % 4], p) for i in range(4)]
    return all(v > 1e-12 for v in sg) or all(v < -1e-12 for v in sg)


def _seg_hits_box(p, q, quad):
    return (_in_box(p, quad) or _in_box(q, quad)
            or any(_seg_cross(p, q, quad[i], quad[(i + 1) % 4])
                   for i in range(4)))


def _boxes_overlap(a, b, eps=1e-6):
    for quad in (a, b):
        for i in range(4):
            p, q = quad[i], quad[(i + 1) % 4]
            ax, ay = q[1] - p[1], p[0] - q[0]
            n = math.hypot(ax, ay) or 1.0
            pa = [(ax * c[0] + ay * c[1]) / n for c in a]
            pb = [(ax * c[0] + ay * c[1]) / n for c in b]
            if max(pa) <= min(pb) + eps or max(pb) <= min(pa) + eps:
                return False
    return True


def _pt_in_ring(pt, ring):
    x, y = pt
    inside = False
    for i in range(len(ring)):
        (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % len(ring)]
        if (y0 > y) != (y1 > y) and \
                x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _pt_in_rings(pt, rings):
    return sum(_pt_in_ring(pt, r) for r in rings) % 2 == 1


def _segs_of(stroke):
    pts = stroke.points
    out = list(zip(pts, pts[1:]))
    if stroke.closed and len(pts) > 2:
        out.append((pts[-1], pts[0]))
    return out


def _seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx
                                               + (p[1] - a[1]) * dy) / l2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _name_audit(marks, set_mm, floor_mm):
    """Every piece name on a printed sheet, against its own piece and every
    other mark. Returns {text: [faults]} for the faulty ones, plus counts."""
    rings_of = {}
    for l in marks.labels:
        if l.kind == 'angle' and l.anchor is not None:
            rings_of.setdefault(l.anchor.piece, l.anchor.rings)
    # A piece with no bevel label (every run under the minimum angle) has
    # no anchor to carry its rings: its own are then the outline rings
    # around the name, even-odd. (Two such names on the heads, measured.)
    outline_rings = [tuple(st.points) for st in marks.strokes
                     if st.kind == 'outline' and st.closed]
    outline_segs = [sg for st in marks.strokes if st.kind == 'outline'
                    for sg in _segs_of(st)]
    tick_segs = [sg for st in marks.strokes if st.kind in ('align', 'bracket')
                 for sg in _segs_of(st)]
    other_segs = [sg for st in marks.strokes
                  if st.kind not in ('align', 'bracket', 'outline', 'piece')
                  for sg in _segs_of(st)]
    names = [l for l in marks.labels if l.kind == 'number']
    others = [l for l in marks.labels if l.kind != 'number' and l.text]
    faults, owners = {}, {}
    for n in names:
        box = _lbox(n)
        f = []
        own = [p for p, r in rings_of.items() if _pt_in_rings((n.x, n.y), r)]
        if len(own) > 1:
            f.append("in_more_than_one_piece")
        rings = rings_of[own[0]] if own else [
            r for r in outline_rings if _pt_in_ring((n.x, n.y), r)]
        if own:
            owners[n.text] = own[0]
        if not rings or not _pt_in_rings((n.x, n.y), rings):
            f.append("in_no_piece")
        elif not all(_pt_in_rings(c, rings) for c in box):
            f.append("corner_outside_own_piece")
        box = _box(n.x, n.y, n.size_mm, n.text,
                   pad=NAME_CLEAR_EM * n.size_mm)
        if any(_seg_hits_box(p, q, box) for p, q in outline_segs):
            f.append("crosses_an_outline")
        if any(_seg_hits_box(p, q, box) for p, q in tick_segs):
            f.append("on_a_tick")
        if any(_seg_hits_box(p, q, box) for p, q in other_segs):
            f.append("on_another_stroke")
        if any(_boxes_overlap(box, _lbox(o)) for o in others):
            f.append("on_a_label")
        if any(m is not n and _boxes_overlap(box, _lbox(m)) for m in names):
            f.append("on_another_name")
        if f:
            faults[n.text] = f
    return {
        "names": len(names),
        "pieces": len(rings_of),
        "faults": faults,
        "at_floor": sorted(n.text for n in names
                           if n.size_mm <= floor_mm * (1 + 1e-6)),
        "below_floor": [n.text for n in names
                        if n.size_mm < floor_mm * (1 - 1e-6)],
        "above_set_size": [n.text for n in names
                           if n.size_mm > set_mm * (1 + 1e-6)],
        "shrunk": sum(1 for n in names if n.size_mm < set_mm * (1 - 1e-6)),
        "sizes_mm": sorted(round(n.size_mm, 2) for n in names),
        "set_mm": set_mm, "floor_mm": floor_mm,
    }


def _leader_audit(marks):
    """Every leader: at least MIN_LEADER_TEXT_HEIGHTS of its label's size
    long, ending on its own run; and its label clear of every other mark
    and of every piece."""
    label_of = {l.anchor: l for l in marks.labels
                if l.kind == 'angle' and l.anchor is not None}
    rings = [st.points for st in marks.strokes
             if st.kind == 'outline' and st.closed]
    segs = [(st, sg) for st in marks.strokes if st.kind != 'piece'
            for sg in _segs_of(st)]
    short, off_run, label_hits = [], [], []
    lengths = []
    for st in marks.strokes:
        if st.kind != 'leader':
            continue
        l = label_of.get(st.anchor)
        if l is None:
            off_run.append("orphan")
            continue
        start, end = st.points[0], st.points[-1]
        length = math.dist(start, end)
        lengths.append(round(length / l.size_mm, 2))
        if length < MIN_LEADER_TEXT_HEIGHTS * l.size_mm - 1e-6:
            short.append((l.text, round(length, 2), round(l.size_mm, 2)))
        run = st.anchor.run
        if min(_seg_dist(end, a, b) for a, b in zip(run, run[1:])) > 1e-6:
            off_run.append(l.text)
        box = _lbox(l)
        hit = (any(other is not st and _seg_hits_box(p, q, box)
                   for other, (p, q) in segs)
               or any(o is not l and o.text
                      and _boxes_overlap(box, _lbox(o))
                      for o in marks.labels)
               or any(_pt_in_rings(c, [r]) for r in rings for c in box))
        if hit:
            label_hits.append(l.text)
    return {"leaders": len(lengths), "short": short, "off_run": off_run,
            "label_not_clear": label_hits,
            "length_in_text_heights": sorted(lengths)}


def test_snapshot_heads_piece_names_and_leaders():
    """The maker's two heads, as the image operator prints them and as the
    overlay caches them (2026-11-17, three defects in the captures):

    1. Piece names. Each is sized so its box fits inside its own piece,
       clear of the outline, every alignment / run-end tick and every other
       label -- never below a quarter of the set size. A name that does
       not fit even there is kept at the floor and REPORTED: the faulty
       names measured here must be exactly the ones the model reported
       (`names_out`, how == 'floor'), and those must be at the floor.
       On screen, the overlay's cached names equal the printed ones in
       place, text and size.
    2. (Margin brackets: tests/test_overlay_gui.py, which sees the draw.)
    3. Leaders. Each is at least 1.5 text heights of its own label long,
       ends on its own run, and its label is clear of every other mark and
       off every piece.
    """
    import os
    objs, snap_scene, tmp = _append_heads()
    if objs is None:
        return {"skipped": tmp, "checks": {}}
    s = bpy.context.scene.foamcraft_drachis
    if snap_scene is not None:
        if getattr(snap_scene, "foamcraft_drachis", None):
            _copy_settings(snap_scene.foamcraft_drachis, s)
        bpy.context.scene.unit_settings.scale_length = \
            snap_scene.unit_settings.scale_length
    saved_only = s.only_selected
    s.only_selected = False
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs.values():
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[HEADS[0]]
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    scale_length = bpy.context.scene.unit_settings.scale_length
    try:
        pairs = ov._edit_meshes(bpy.context)
        ov._rebuild(pairs, s)
        cached = list(ov._cache.island_labels)
        cached_full = float(getattr(ov._cache, "number_font_mm", 0.0))
        status, marks, _info = _operator_marks(
            os.path.join(os.path.dirname(tmp), "heads.png"))
        sheet, parts = fm._layout_sheet(
            [fm.SheetItem(bm, uv, o.matrix_world.copy(),
                          fm.object_prefix(o.name,
                                           o.foamcraft_drachis_prefix),
                          o.name) for o, bm, uv in pairs], s, scale_length)
        names_out = []
        try:
            fm.marks_for_sheet(sheet, parts, s, scale_length,
                               names_out=names_out)
        except TypeError:
            names_out = None          # the model cannot report yet
        n_islands = sum(len(part.islands) for part in parts)
        del parts
        set_mm = float(sheet.permille(s.number_size, s.overlay_scale))
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
        s.only_selected = saved_only
    if marks is None:
        return {"status": list(status), "checks": {"operator_rendered": False}}

    floor_mm = min(float(s.min_label_mm), set_mm)
    names = _name_audit(marks, set_mm, floor_mm)
    leaders = _leader_audit(marks)
    reported_floor = sorted(lab.text for _o, lab, how in (names_out or [])
                            if how == 'floor')
    faulty = sorted(names["faults"])

    def mm(u, v):
        p = sheet.to_sheet(Vector((u, v)))
        return (p.x, p.y)
    screen = [mm(c[0], c[1]) + (float(c[4]) if len(c) > 4 else -1.0, c[2])
              for c in cached]
    paper = [(l.x, l.y, l.size_mm, l.text) for l in marks.labels
             if l.kind == 'number']
    tol = TOL_REL * max(marks.width_mm, marks.height_mm)
    only_s, only_p = _unmatched(screen, paper, tol)

    return {
        "names": {k: v for k, v in names.items() if k != "faults"},
        "faulty_names": names["faults"],
        "names_reported_at_floor": reported_floor,
        "leaders": leaders,
        "screen_names": {"screen": len(screen), "paper": len(paper),
                         "only_on_screen": only_s, "only_on_paper": only_p,
                         "cached_set_mm": cached_full},
        "checks": {
            "operator_finished": status == {'FINISHED'},
            # Both bounds on the count: one name per piece, none dropped.
            "one_name_per_piece": names["names"] == n_islands > 0,
            "none_below_the_floor": names["below_floor"] == [],
            "none_above_the_set_size": names["above_set_size"] == [],
            "model_reports_its_floor_names": names_out is not None,
            # Every name that crosses or covers anything is one the model
            # said could not fit, and every one it said so about does.
            "faulty_names_are_exactly_the_reported_ones":
                faulty == reported_floor,
            "reported_names_are_at_the_floor":
                set(reported_floor) <= set(names["at_floor"]),
            "screen_names_equal_the_printed":
                len(screen) == len(paper) and only_s == 0 and only_p == 0,
            "screen_knows_the_set_size":
                abs(cached_full - set_mm) <= 1e-6 * max(set_mm, 1e-9),
            "there_are_leaders": leaders["leaders"] > 0,
            "no_short_leader": leaders["short"] == [],
            "every_leader_ends_on_its_run": leaders["off_run"] == [],
            "every_leader_label_clear": leaders["label_not_clear"] == [],
        },
    }


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
    test_crowded_box_laid_out_labels_and_leaders_match_the_sheet,
    test_label_px_scales_with_printed_size,
    test_snapshot_heads_overlay_is_the_printed_layout,
    test_snapshot_heads_piece_names_and_leaders,
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
