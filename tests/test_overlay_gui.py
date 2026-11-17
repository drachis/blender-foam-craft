"""Checks that need a window: does the overlay actually draw?

    python tools/run_tests.py --gui --only overlay

Everything else in this suite reads numbers out of Blender. None of it can tell
you whether a single pixel reached the screen -- and this project has already
shipped an overlay that could never have drawn at all, because it asked for a
shader name removed in 4.0 and the failure only surfaced when someone looked.

So these compare pictures. Capture the UV editor with the overlay off, capture
it again with the overlay on, and count the pixels that changed. A draw handler
that throws, draws nothing, or draws off-screen all leave the two frames
identical, and none of those can be caught any other way.

Headless runs skip, reporting why rather than passing silently: a check that
cannot fail is worse than no check.
"""

import math
import sys
import traceback

import bpy
import bmesh

# The add-on is re-enabled first, and this file must not reload the module
# afterwards.
#
# The draw handler and the cache belong to whichever module instance was live
# when the add-on registered. The other test files purge the whole package from
# sys.modules to pick up edits, which orphans that registration: the handler
# still runs from a module object nothing can reach any more, so this test would
# inspect a fresh empty copy and report "nothing to draw" about an overlay that
# is working. Measured that way once -- draw handle unset, timer unregistered,
# cache empty -- with the add-on apparently loaded.
_ADDON = "bl_ext.user_default.foamcraft_drachis"
if bpy.context.object and bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
try:
    bpy.ops.preferences.addon_disable(module=_ADDON)
except Exception:                                        # noqa: BLE001
    pass
for _name in [n for n in list(sys.modules) if n.startswith(_ADDON)]:
    del sys.modules[_name]
bpy.ops.preferences.addon_enable(module=_ADDON)

import bl_ext.user_default.foamcraft_drachis.foam_overlay as overlay   # noqa: E402

_TOOLS = "O:/Work/BlenderPlugin/FoamCraft/tools"
# Every Nth pixel. Comparing several million floats in Python is slow and adds
# nothing -- a mark that changes only one pixel in a hundred is invisible anyway.
_SAMPLE = 37


# Scene clearing goes through the guard, which refuses in a Blender this
# suite did not start. Imported here, self-contained, because these files set
# sys.path several different ways and the protection must not depend on which.
_GUARD_TOOLS = r"O:\Work\BlenderPlugin\FoamCraft\tools"
if _GUARD_TOOLS not in sys.path:
    sys.path.insert(0, _GUARD_TOOLS)
import sandbox_guard  # noqa: E402


def _clear():
    """Empty the scene -- but only in a Blender this suite started.

    Delegates to tools/sandbox_guard, which refuses when the session cannot
    prove it is disposable. On 2026-11-03 this function, called directly,
    deleted a user's unsaved sculpt after the launcher adopted their own
    Blender off a shared port. The guard is what stops that repeating.
    """
    sandbox_guard.clear_scene()
    for collection in (bpy.data.objects, bpy.data.meshes):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _uv_area():
    """An area switched to the UV editor, with its window region."""
    screen = bpy.context.screen
    if screen is None:
        return None, None
    biggest = None
    for area in screen.areas:
        if area.type in {'PROPERTIES', 'OUTLINER'}:
            continue
        if biggest is None or area.width * area.height > \
                biggest.width * biggest.height:
            biggest = area
    if biggest is None:
        return None, None
    biggest.type = 'IMAGE_EDITOR'
    biggest.ui_type = 'UV'
    # Hide the sidebar. Toggling the overlay also changes what the panel shows,
    # so with it visible the two captures differ by repainted UI whether or not
    # a single mark was drawn -- 85 changed pixels, measured, from chrome alone.
    space = biggest.spaces.active
    if hasattr(space, "show_region_ui"):
        space.show_region_ui = False
    if hasattr(space, "show_region_toolbar"):
        space.show_region_toolbar = False
    region = next((r for r in biggest.regions if r.type == 'WINDOW'), None)
    return biggest, region


def _capture(area, region, label):
    if _TOOLS not in sys.path:
        sys.path.insert(0, _TOOLS)
    from shot import next_path
    path = next_path("gui_%s" % label)
    area.tag_redraw()
    with bpy.context.temp_override(area=area, region=region):
        bpy.ops.screen.screenshot_area(filepath=path)
    return path


def _changed_pixels(path_a, path_b, sample=_SAMPLE):
    """How many sampled pixels differ between two captures."""
    img_a = bpy.data.images.load(path_a)
    img_b = bpy.data.images.load(path_b)
    try:
        a = img_a.pixels[:]
        b = img_b.pixels[:]
        if len(a) != len(b):
            return None
        changed = 0
        total = 0
        for i in range(0, len(a), 4 * sample):
            total += 1
            if abs(a[i] - b[i]) > 0.02 or abs(a[i + 1] - b[i + 1]) > 0.02 \
                    or abs(a[i + 2] - b[i + 2]) > 0.02:
                changed += 1
        return changed, total
    finally:
        bpy.data.images.remove(img_a)
        bpy.data.images.remove(img_b)


def test_overlay_draws_into_the_uv_editor():
    """The overlay must change what is on screen. Off, then on, then compare.

    The failure this exists for: an overlay that raises inside its draw handler
    is silent -- Blender prints to the console and carries on rendering the
    frame without it -- so every data check passes while nothing is drawn.
    """
    if bpy.app.background:
        return {"skipped": "needs a window; run with --gui",
                "checks": {"ran_with_a_window": False}}

    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=24, ring_count=12)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.04)

    area, region = _uv_area()
    if area is None or region is None:
        return {"skipped": "no usable area", "checks": {"found_uv_area": False}}

    settings = bpy.context.scene.foamcraft_drachis
    settings.show_overlay = False
    off = _capture(area, region, "overlay_off")

    settings.show_overlay = True
    # The overlay rebuilds on a timer and refuses to draw while dirty, so a
    # capture taken immediately can legitimately show nothing. The operator
    # forces the rebuild that the timer would otherwise do.
    # Force the rebuild rather than waiting for it. The overlay rebuilds on a
    # timer and refuses to draw while dirty, and a timer never fires part-way
    # through a script -- so a capture taken here would legitimately be blank
    # and the test would blame the draw handler for it.
    try:
        with bpy.context.temp_override(area=area, region=region):
            bpy.ops.foamcraft_drachis.refresh_overlay()
        if hasattr(overlay, "_timer_tick"):
            overlay._timer_tick()
    except RuntimeError:
        pass
    on = _capture(area, region, "overlay_on")

    result = _changed_pixels(off, on)
    handler_live = overlay._draw_handle is not None
    runs = len(overlay._cache.runs)
    bpy.ops.object.mode_set(mode='OBJECT')
    if result is None:
        return {"error": "captures differ in size",
                "checks": {"comparable": False}}
    changed, total = result

    return {
        "off": off,
        "on": on,
        "sampled_pixels": total,
        "changed_pixels": changed,
        "cached_runs": runs,
        "draw_handler_registered": handler_live,
        "checks": {
            # If this is false the rest is meaningless: nothing was ever going
            # to draw, and a blank frame says nothing about the overlay.
            "draw_handler_registered": handler_live,
            # Something reached the screen.
            "overlay_changed_the_frame": changed > 0,
            # ...and it is marks, not the whole view. A changed background or a
            # different zoom would repaint nearly everything, which would mean
            # the two captures are not comparable rather than that it drew.
            "changed_a_minority_of_the_frame": 0 < changed < total * 0.5,
            # The data behind it exists, so a blank frame would be a draw fault
            # rather than an empty mesh.
            "there_was_something_to_draw": runs > 0,
        },
    }


def _gore_sphere(gores=6):
    """A 0.3 m sphere peeled into six gores, left in edit mode.

    The fixture tests/test_overlay_ticks.py checks the tick *positions* on;
    here it is only something realistic to look at. Real-world scale
    matters: a tick is 6 mm of foam, so on a 2 m test shape it would be a
    speck, and on this one it is a visible fraction of a gore.
    """
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


def _force_rebuild(area, region):
    """Make the overlay's cache current now, as the timer would later.

    A timer never fires part-way through a script, and the overlay refuses
    to draw while dirty -- so without this a capture is legitimately blank.
    """
    overlay.mark_dirty()
    with bpy.context.temp_override(area=area, region=region):
        bpy.ops.foamcraft_drachis.refresh_overlay()
    overlay._timer_tick()


def test_alignment_ticks_draw_into_the_uv_editor():
    """The registration ticks must reach the screen, and only they change.

    The overlay stays on for both captures; only Alignment Ticks is
    toggled. So whatever differs between the two frames is the ticks -- the
    run-end marks, bevel labels and piece names are identical in both.

    Taken twice: once with the whole layout in view, and once zoomed in
    (view ratio 4.0, which measured about 1.4x the fitted view), because
    a tick is 6 mm of a pattern roughly a metre across and at
    whole-layout framing it is a handful of pixels -- real, countable, and
    not something an eye can judge. The zoomed pair is the one to look at.
    """
    if bpy.app.background:
        return {"skipped": "needs a window; run with --gui",
                "checks": {"ran_with_a_window": False}}

    _clear()
    _gore_sphere()

    area, region = _uv_area()
    if area is None or region is None:
        return {"skipped": "no usable area", "checks": {"found_uv_area": False}}

    s = bpy.context.scene.foamcraft_drachis
    names = ("show_overlay", "show_align_ticks", "tick_mode", "ticks_per_run",
             "tick_numbers", "only_selected")
    saved = {k: getattr(s, k) for k in names}
    # Every 5th pixel rather than every 37th: a tick is a few pixels long,
    # and the coarse sampling the whole-overlay test uses could step over
    # most of them.
    sample = 5
    try:
        s.show_overlay = True
        s.only_selected = False
        s.tick_mode = 'PER_RUN'
        s.ticks_per_run = 2
        s.tick_numbers = False

        with bpy.context.temp_override(area=area, region=region):
            bpy.ops.image.view_all(fit_view=True)

        s.show_align_ticks = False
        _force_rebuild(area, region)
        ticks_off = len(overlay._cache.align_ticks)
        off = _capture(area, region, "ticks_off")

        s.show_align_ticks = True
        _force_rebuild(area, region)
        ticks_on = len(overlay._cache.align_ticks)
        on = _capture(area, region, "ticks_on")

        with bpy.context.temp_override(area=area, region=region):
            bpy.ops.image.view_zoom_ratio(ratio=4.0)
        on_zoom = _capture(area, region, "ticks_on_zoom")

        s.show_align_ticks = False
        _force_rebuild(area, region)
        off_zoom = _capture(area, region, "ticks_off_zoom")
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
        bpy.ops.object.mode_set(mode='OBJECT')

    whole = _changed_pixels(off, on, sample)
    zoomed = _changed_pixels(off_zoom, on_zoom, sample)
    if whole is None or zoomed is None:
        return {"error": "captures differ in size",
                "checks": {"comparable": False}}
    changed, total = whole
    zchanged, ztotal = zoomed

    return {
        "off": off, "on": on, "off_zoom": off_zoom, "on_zoom": on_zoom,
        "sampled_pixels": total,
        "changed_pixels": changed,
        "zoomed_sampled_pixels": ztotal,
        "zoomed_changed_pixels": zchanged,
        "cached_ticks_on": ticks_on,
        "cached_ticks_off": ticks_off,
        "checks": {
            "draw_handler_registered": overlay._draw_handle is not None,
            # The data: there when on, gone when off.
            "ticks_cached_when_on": ticks_on > 0,
            "none_cached_when_off": ticks_off == 0,
            # The picture: something changed, and it is thin marks -- not
            # the whole overlay repainting, which would mean the toggle had
            # disturbed something other than the ticks.
            "ticks_changed_the_frame": 0 < changed < total * 0.05,
            "ticks_changed_the_zoomed_frame": 0 < zchanged < ztotal * 0.05,
        },
    }


SNAPSHOT = r"O:\Work\BlenderPlugin\FoamCraft\tests\fixtures\beast_skull_snapshot.blend"
HEADS = ("head_simplified", "head_simplified.001")
# The maker's two dense examples, pieces 13 and 16, printed as letters now
# (foam_marks.piece_letters): M and P.
DENSE = (13, 16)


def _load_heads():
    """Both heads appended from a temp copy of the snapshot, with the
    snapshot scene's settings and unit scale, evaluated, in multi-object
    edit mode with every face selected. Never saved. (None, reason) when
    the fixture is absent. Appended rather than opened: opening a file
    inside a bridge request replaces the session running it."""
    import os
    import shutil
    import tempfile
    if not os.path.exists(SNAPSHOT):
        return None, "fixture %s is absent (git-ignored, local only)" % SNAPSHOT
    _clear()
    tmp = os.path.join(tempfile.mkdtemp(prefix="foam_gui_"), "copy.blend")
    shutil.copyfile(SNAPSHOT, tmp)
    with bpy.data.libraries.load(tmp, link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n in HEADS]
        dst.scenes = list(src.scenes[:1])
    objs = {o.name: o for o in dst.objects if o is not None}
    if set(objs) != set(HEADS):
        return None, "snapshot lacks %r" % (HEADS,)
    for o in objs.values():
        bpy.context.scene.collection.objects.link(o)
    snap = next((sc for sc in dst.scenes if sc is not None), None)
    s = bpy.context.scene.foamcraft_drachis
    if snap is not None:
        for prop in snap.foamcraft_drachis.bl_rna.properties:
            name = prop.identifier
            if name in ("rna_type", "name") or prop.is_readonly \
                    or prop.type == 'POINTER':
                continue
            try:
                setattr(s, name, getattr(snap.foamcraft_drachis, name))
            except (AttributeError, TypeError, ValueError):
                pass
        bpy.context.scene.unit_settings.scale_length = \
            snap.unit_settings.scale_length
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs.values():
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[HEADS[0]]
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    return objs, tmp


def _frame_uv_rect(area, region, rect, fill=0.9):
    """Zoom and pan the UV editor so `rect` (u0, v0, u1, v1) fills `fill`
    of the region, using the editor's own operators: zoom ratio from a
    measured pixels-per-UV at ratio 1, centre from the 2D cursor."""
    space = area.spaces.active
    view = region.view2d

    def redraw():
        # view2d is only brought up to date by a redraw; read it before
        # one and it describes the previous view (measured: the first
        # version of this framed the whole layout, at the old zoom).
        with bpy.context.temp_override(area=area, region=region):
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    with bpy.context.temp_override(area=area, region=region):
        bpy.ops.image.view_zoom_ratio(ratio=1.0)
    redraw()
    x0, _y = view.view_to_region(0.0, 0.0, clip=False)
    x1, _y = view.view_to_region(1.0, 0.0, clip=False)
    px_per_uv = abs(x1 - x0) or 1.0
    w = max(rect[2] - rect[0], 1e-6)
    h = max(rect[3] - rect[1], 1e-6)
    want = fill * min(region.width / w, region.height / h)
    centre = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
    space.cursor_location = centre
    with bpy.context.temp_override(area=area, region=region):
        bpy.ops.image.view_zoom_ratio(ratio=want / px_per_uv)
        bpy.ops.image.view_center_cursor()
    redraw()
    cx, cy = view.view_to_region(*centre, clip=False)
    ax, ay = view.view_to_region(rect[0], rect[1], clip=False)
    bx, by = view.view_to_region(rect[2], rect[3], clip=False)
    return {"px_per_uv_at_1": round(px_per_uv, 2),
            "ratio": round(want / px_per_uv, 3),
            "centre_px": [round(cx, 1), round(cy, 1)],
            "rect_px": [round(ax), round(ay), round(bx), round(by)],
            "region_px": [region.width, region.height]}


def test_snapshot_dense_pieces_on_screen_and_on_paper():
    """Look at it: the maker's dense pieces 13 and 16 (M, P), overlay on
    the UV editor beside the same pieces cut from the image export.

    The judgement -- do screen and paper match, does it read -- is made by
    looking at the two captures; the checks only make sure there is
    something to look at: both pieces found, laid-out labels cached for
    them (the overlay draws the layout's labels, which
    tests/test_overlay_ticks.py checks by number), the overlay changed the
    framed view, and the paper crop was written.
    """
    if bpy.app.background:
        return {"skipped": "needs a window; run with --gui",
                "checks": {"ran_with_a_window": False}}
    objs, tmp = _load_heads()
    if objs is None:
        return {"skipped": tmp, "checks": {}}
    import os
    import numpy as np
    if _TOOLS not in sys.path:
        sys.path.insert(0, _TOOLS)
    from shot import next_path
    from bl_ext.user_default.foamcraft_drachis import foam_marks as fm
    from bl_ext.user_default.foamcraft_drachis import foam_export as fx

    area, region = _uv_area()
    if area is None or region is None:
        return {"skipped": "no usable area", "checks": {"found_uv_area": False}}
    s = bpy.context.scene.foamcraft_drachis
    saved = {k: getattr(s, k) for k in ("show_overlay", "only_selected",
                                        "show_darts")}
    out = {}
    try:
        s.show_overlay = True
        s.only_selected = False
        # Relief marks are drawn in the fold colours too; off, so every
        # fold-coloured line on screen is a leader (see below). The sheet
        # follows the same toggle, so screen and paper still agree.
        s.show_darts = False

        # The sheet the export prints, and its pieces M and P: their rings
        # and every label (and leader) belonging to them, in mm, then UV.
        with fx.SheetSource(fx.export_objects(bpy.context)) as items:
            sheet, parts = fm._layout_sheet(
                items, s, bpy.context.scene.unit_settings.scale_length)
            marks = fm.marks_for_sheet(
                sheet, parts, s, bpy.context.scene.unit_settings.scale_length)
            to_uv = overlay._sheet_to_uv(sheet)
            del parts
        letters = {n: fm.piece_letters(n) for n in DENSE}
        numbers = [l for l in marks.labels if l.kind == 'number']
        boxes = {}
        for n, let in letters.items():
            hit = [l for l in numbers if l.text == let
                   or l.text.endswith(" " + let)]
            if not hit:
                continue
            name = hit[0]
            owner = None
            for l in marks.labels:
                if l.kind == 'angle' and l.anchor and any(
                        _inside((name.x, name.y), r)
                        for r in l.anchor.rings[:1]):
                    owner = l.anchor.piece
                    break
            if owner is None:
                continue
            pts = []
            for l in marks.labels:
                if l.kind == 'angle' and l.anchor and l.anchor.piece == owner:
                    pts += list(l.anchor.rings[0]) + [(l.x, l.y)]
            for st in marks.strokes:
                if st.kind == 'leader' and st.anchor \
                        and st.anchor.piece == owner:
                    pts += list(st.points)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            pad = 12.0
            boxes[name.text] = (owner, (min(xs) - pad, min(ys) - pad,
                                        max(xs) + pad, max(ys) + pad))
        out["pieces"] = {k: v[0] for k, v in boxes.items()}
        if len(boxes) != 2:
            return {**out, "checks": {"found_both_pieces": False}}

        # The image export of the same sheet, through the operator.
        sheet_png = next_path("dense_MP_export_sheet")
        status = bpy.ops.foamcraft_drachis.export_pattern_image(
            filepath=sheet_png, file_format='PNG', resolution_mode='DPI',
            dpi=200)
        img = bpy.data.images.load(sheet_png, check_existing=False)
        w, h = img.size
        px_per_mm = w / marks.width_mm
        pix = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(pix)
        pix = pix.reshape(h, w, 4)
        out["paper_sheet"] = sheet_png

        shots = {}
        for label, (owner, box_mm) in boxes.items():
            tag = label.replace(" ", "_")
            uv_a = to_uv(box_mm[0], box_mm[3])
            uv_b = to_uv(box_mm[2], box_mm[1])
            rect = (min(uv_a[0], uv_b[0]), min(uv_a[1], uv_b[1]),
                    max(uv_a[0], uv_b[0]), max(uv_a[1], uv_b[1]))
            row = {"box_mm": [round(v, 1) for v in box_mm],
                   "framing": _frame_uv_rect(area, region, rect)}
            s.show_overlay = False
            off = _capture(area, region, "dense_%s_overlay_off" % tag)
            s.show_overlay = True
            _force_rebuild(area, region)
            cached = [m for m in overlay._cache.bevel_marks
                      if m.piece == owner]
            # What the draw handler hands the GPU, by colour, for the last
            # frame drawn: the screen's own account of its lines.
            frames = []
            real_draw_lines = overlay._draw_lines

            def spy(segments_by_color, width):
                frames.append({tuple(round(c, 4) for c in k): len(v)
                               for k, v in segments_by_color.items()})
                return real_draw_lines(segments_by_color, width)
            overlay._draw_lines = spy
            try:
                on = _capture(area, region, "dense_%s_overlay_on" % tag)
            finally:
                overlay._draw_lines = real_draw_lines
            row.update(_fold_coloured_lines(frames, s, region))
            row.update({
                "screen": on, "screen_overlay_off": off,
                "cached_labels": len(cached),
                "cached_leaders": sum(1 for m in cached if m.leader),
                "cached_sizes_mm": sorted({round(m.size_mm, 3)
                                           for m in cached}),
                "cached_texts": sorted(m.text for m in cached),
                "changed_pixels": _changed_pixels(off, on, 5),
            })
            # The paper, cropped to the same millimetre box.
            x0 = max(int(box_mm[0] * px_per_mm), 0)
            x1 = min(int(box_mm[2] * px_per_mm), w)
            y0 = max(int(box_mm[1] * px_per_mm), 0)
            y1 = min(int(box_mm[3] * px_per_mm), h)
            crop = pix[h - y1:h - y0, x0:x1, :]
            new = bpy.data.images.new("crop_%s" % tag, x1 - x0, y1 - y0,
                                      alpha=False)
            new.pixels.foreach_set(crop.reshape(-1))
            paper = next_path("dense_%s_export_crop" % tag)
            new.filepath_raw = paper
            new.file_format = 'PNG'
            new.save()
            bpy.data.images.remove(new)
            row["paper"] = paper
            row["paper_px"] = [x1 - x0, y1 - y0]
            row["paper_labels"] = sorted(
                l.text for l in marks.labels
                if l.kind == 'angle' and l.anchor and l.anchor.piece == owner)
            shots[label] = row
        bpy.data.images.remove(img)
        out["shots"] = shots
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
        bpy.ops.object.mode_set(mode='OBJECT')
    rows = list(out["shots"].values())
    out["checks"] = {
        "found_both_pieces": len(rows) == 2,
        "labels_cached_for_them": all(r["cached_labels"] > 0 for r in rows),
        # The same strings on screen as on paper for each piece.
        "same_texts_as_paper": all(r["cached_texts"] == r["paper_labels"]
                                   for r in rows),
        # Framed on the piece: its box's centre lands in the middle tenth
        # of the region each way.
        "framed_on_the_piece": all(
            abs(r["framing"]["centre_px"][0]
                - r["framing"]["region_px"][0] / 2)
            < r["framing"]["region_px"][0] * 0.1
            and abs(r["framing"]["centre_px"][1]
                    - r["framing"]["region_px"][1] / 2)
            < r["framing"]["region_px"][1] * 0.1 for r in rows),
        "overlay_changed_the_framed_view": all(
            r["changed_pixels"] is not None
            and 0 < r["changed_pixels"][0] < r["changed_pixels"][1]
            for r in rows),
        # The overlay used to draw a fold-coloured "bracket" in the margin
        # beside every run -- the paper prints nothing there, and it ran
        # through outside labels. Now the only fold-coloured lines on
        # screen are the leaders of drawn outside labels: exactly two
        # points each, no more (a bracket) and no fewer (a lost leader).
        "fold_colours_distinct": all(r["fold_colours_distinct"]
                                     for r in rows),
        "drew_a_frame": all(r["frames_drawn"] > 0 for r in rows),
        "only_leaders_in_fold_colours": all(
            r["fold_points_drawn"] == r["leader_points_expected"]
            for r in rows),
        "export_finished": status == {'FINISHED'},
        "paper_crops_written": all(os.path.exists(r["paper"]) for r in rows),
    }
    return out


def _fold_coloured_lines(frames, s, region):
    """Of the last frame's lines, how many points were drawn in the convex
    or concave colour, and how many the drawn leaders account for.

    A leader takes its label's colour (convex, concave, or the neutral ink
    when fold colours are off) and is drawn only with its label, which the
    overlay hides where the run is shorter on screen than
    `min_label_run_px` -- mirrored here, over the whole cache (a frame
    draws every cached mark, not only the framed piece's).
    """
    convex = tuple(round(c, 4) for c in s.convex_color)
    concave = tuple(round(c, 4) for c in s.concave_color)
    ink = tuple(round(c, 4) for c in s.island_id_color)
    neutral = convex if not s.show_dart_colors else ink
    tone_colour = {'convex': convex, 'concave': concave, 'neutral': neutral}
    view = region.view2d
    expected = 0
    for m in overlay._cache.bevel_marks:
        if not m.leader or tone_colour.get(m.tone) not in (convex, concave):
            continue
        ax, ay = view.view_to_region(m.au, m.av, clip=False)
        bx, by = view.view_to_region(m.bu, m.bv, clip=False)
        if math.hypot(bx - ax, by - ay) < s.min_label_run_px:
            continue
        expected += 2
    last = frames[-1] if frames else {}
    drawn = sum(n for k, n in last.items() if k in (convex, concave))
    return {"frames_drawn": len(frames),
            "fold_points_drawn": drawn,
            "leader_points_expected": expected,
            "fold_colours_distinct": len({convex, concave, ink}) == 3}


def _inside(pt, ring):
    x, y = pt
    inside = False
    n = len(ring)
    for i in range(n):
        (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            if x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
    return inside


TESTS = (test_overlay_draws_into_the_uv_editor,
         test_alignment_ticks_draw_into_the_uv_editor,
         test_snapshot_dense_pieces_on_screen_and_on_paper)

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
