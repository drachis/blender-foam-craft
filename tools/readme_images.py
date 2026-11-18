"""Rebuild the README's pictures from the beast skull snapshot.

    python tools/launch_blender.py --port 9890 --gui
    python tools/blender_bridge.py --port 9890 < tools/readme_images.py

Kept rather than run once (CLAUDE.md: no code is one-off): every release
wants the README to show the add-on as it now is, on the model it was built
for. Writes, into docs/images/:

  overlay-skull.png  the UV editor on two skull pieces, overlay on, with the
                     FoamCraft sidebar open beside it
  print-skull.png    the same pieces cut from the image export -- what the
                     printer gets

The skull is tests/fixtures/beast_skull_snapshot.blend, git-ignored and local
only: the pictures are published, the model is not. It is appended from a
temp copy and never saved. The loader and framing are the GUI test's own
(tests/test_overlay_gui.py), taken from that file so the two cannot drift.
"""

import os
import sys

import bpy

ROOT = r"O:\Work\BlenderPlugin\FoamCraft"
OUT = os.path.join(ROOT, "docs", "images")

# The GUI test's helpers: everything above its TESTS tuple, which is the
# loader, the framing and the capture, without running the tests.
_src = open(os.path.join(ROOT, "tests", "test_overlay_gui.py"),
            encoding="utf-8").read()
_ns = {"__name__": "readme_images_helpers"}
exec(compile(_src[:_src.index("\nTESTS = (")], "test_overlay_gui", "exec"),
     _ns)

from bl_ext.user_default.foamcraft_drachis import foam_export as fx  # noqa: E402
from bl_ext.user_default.foamcraft_drachis import foam_marks as fm  # noqa: E402

# Framed on piece AW and its neighbours (W, M, F, B): bevel labels of both
# fold colours, leaders, ticks and run-end circles all in one view. SPAN is
# the view's height in multiples of AW's.
CENTRE = "AW"
SPAN = 2.6


def _redraw(n=2):
    for _ in range(n):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)


def main():
    if bpy.app.background:
        return {"skipped": "needs a window: launch with --gui"}
    objs, tmp = _ns["_load_heads"]()
    if objs is None:
        return {"skipped": tmp}
    import numpy as np
    s = bpy.context.scene.foamcraft_drachis
    s.show_overlay = True
    s.only_selected = False
    scale_length = bpy.context.scene.unit_settings.scale_length

    # The sheet the export prints, and the millimetre box around M and P.
    with fx.SheetSource(fx.export_objects(bpy.context)) as items:
        sheet, parts = fm._layout_sheet(items, s, scale_length)
        marks = fm.marks_for_sheet(sheet, parts, s, scale_length)
        to_uv = _ns["overlay"]._sheet_to_uv(sheet)
        del parts
    name = next(l for l in marks.labels if l.kind == 'number'
                and l.text.split(" ")[-1] == CENTRE)
    ring = None
    for l in marks.labels:
        if l.kind == 'angle' and l.anchor and _ns["_inside"](
                (name.x, name.y), l.anchor.rings[0]):
            ring = l.anchor.rings[0]
            break
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]

    # Screen: UV editor with the sidebar open on FoamCraft, which also
    # fixes the shape of the window region the box has to match.
    area, region = _ns["_uv_area"]()
    space = area.spaces.active
    space.show_region_ui = True
    ui = next(r for r in area.regions if r.type == 'UI')
    _redraw()
    try:
        ui.active_panel_category = "FoamCraft"
    except (AttributeError, TypeError):
        pass
    _redraw()
    # One box for both pictures: the centre piece and its neighbours,
    # SPAN times its height, in the visible part of the region's shape
    # (the sidebar overlaps the window region's right side).
    visible_w = region.width - ui.width
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_h = (max(ys) - min(ys)) * SPAN / 2
    half_w = half_h * visible_w / region.height
    box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    owners = {CENTRE}
    # Framed over the whole region, the box sitting in the part left of the
    # sidebar: extended right by the sidebar's share of the width.
    frame = (box[0], box[1],
             box[0] + 2 * half_h * region.width / region.height, box[3])
    a, b = to_uv(frame[0], frame[3]), to_uv(frame[2], frame[1])
    rect = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]),
            max(a[1], b[1]))
    _ns["_frame_uv_rect"](area, region, rect, fill=1.0)
    _ns["_force_rebuild"](area, region)
    _redraw(3)
    out = {}
    path = os.path.join(OUT, "overlay-skull.png")
    with bpy.context.temp_override(area=area):
        bpy.ops.screen.screenshot_area(filepath=path)
    out["overlay"] = path
    # (No separate sidebar picture: screenshot_area captures the whole area
    # whatever region is overridden -- measured, a byte-identical file --
    # and the overlay picture already shows the sidebar.)

    # Paper: the image export of the same sheet, cropped to the same box.
    sheet_png = os.path.join(os.path.dirname(tmp), "sheet.png")
    bpy.ops.foamcraft_drachis.export_pattern_image(
        filepath=sheet_png, file_format='PNG', resolution_mode='DPI', dpi=150)
    img = bpy.data.images.load(sheet_png, check_existing=False)
    w, h = img.size
    px_per_mm = w / marks.width_mm
    pix = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(pix)
    pix = pix.reshape(h, w, 4)
    x0, x1 = max(int(box[0] * px_per_mm), 0), min(int(box[2] * px_per_mm), w)
    y0, y1 = max(int(box[1] * px_per_mm), 0), min(int(box[3] * px_per_mm), h)
    crop = pix[h - y1:h - y0, x0:x1, :]
    new = bpy.data.images.new("readme_crop", x1 - x0, y1 - y0, alpha=False)
    new.pixels.foreach_set(crop.reshape(-1))
    path = os.path.join(OUT, "print-skull.png")
    new.filepath_raw = path
    new.file_format = 'PNG'
    new.save()
    bpy.data.images.remove(new)
    bpy.data.images.remove(img)
    out["print"] = path
    out["pieces"] = sorted(owners)
    bpy.ops.object.mode_set(mode='OBJECT')
    return out


result = main()
