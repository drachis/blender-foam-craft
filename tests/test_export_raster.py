"""What the raster exporter actually puts in the image file.

    python tools/launch_blender.py --port 9970
    python tools/blender_bridge.py --port 9970 < tests/test_export_raster.py

`foam_raster.render_marks` draws a `foam_marks.Marks` -- plain data in sheet
millimetres, origin top-left, y down -- and must not know anything else. So
almost every case here hands it a hand-built `Marks` whose picture can be
worked out with arithmetic, then reads pixels back *from the written file*.

Why from the file, and why with `bpy.data.images` rather than the library
that wrote it: the previous attempt at this exporter had mirrored text and a
writer type error, and neither fix was ever run. A check that inspects the
framebuffer, or reads back through the same OIIO call that wrote, can agree
with itself while the file on disk is upside down. Blender's image loader is
an independent reader with the opposite row convention, so a flip cannot
cancel out.

The convention used throughout: 100 mm at 254 dpi is exactly 1000 px, so
1 mm is 10 px and every expected pixel position is the millimetre figure
times ten.

DOES NOT NEED A WINDOW, which was assumed and turned out false. GPUOffScreen
needs a GPU context and a background Blender starts without one, but
`gpu.init()` makes one: every case here passes headless as well as windowed
(measured 2026-11-16, Blender 5.2). `test_gpu_context_probe` asks Blender
both questions and records the answers; if a build ever has no context even
after that, the drawing cases report themselves skipped instead of failing.
"""

import os
import sys
import time
import tempfile
import importlib
import traceback

import bpy
import bmesh
import numpy as np

sys.path.insert(0, r"O:\Work\BlenderPlugin\FoamCraft\tools")

import sandbox_guard  # noqa: E402  (guards every _clear below)
import shot           # noqa: E402

_MOD = "bl_ext.user_default.foamcraft_drachis"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foamcraft_drachis.foam_marks as fm     # noqa: E402
import bl_ext.user_default.foamcraft_drachis.foam_raster as fr    # noqa: E402

importlib.reload(fm)
importlib.reload(fr)

_TMP = tempfile.mkdtemp(prefix="foamcraft_raster_")

RED = (1.0, 0.0, 0.0)
BLACK = (0.0, 0.0, 0.0)
DPI = 254          # 10 px per mm exactly


def _gpu_probe():
    """Can this Blender make an offscreen buffer at all? (ok, message)"""
    import gpu
    try:
        off = gpu.types.GPUOffScreen(8, 8)
    except Exception as exc:                           # noqa: BLE001
        return False, "%s: %s" % (type(exc).__name__, exc)
    off.free()
    return True, "ok"


_GPU_OK, _GPU_MSG = _gpu_probe()


def _gpu_init_probe():
    """Does `gpu.init()` rescue a Blender that had no GPU context?

    A background Blender refuses with "requires the gpu module to be
    initialized. See gpu.init." -- which names a function, so rather than
    conclude "background cannot render" from the first refusal, call it and
    ask again. Only attempted when the first probe failed; a windowed
    instance has nothing to rescue.
    """
    import gpu
    if _GPU_OK:
        return {"attempted": False}
    if not hasattr(gpu, "init"):
        return {"attempted": False, "has_gpu_init": False}
    try:
        gpu.init()
    except Exception as exc:                           # noqa: BLE001
        return {"attempted": True, "init_error":
                "%s: %s" % (type(exc).__name__, exc)}
    ok, message = _gpu_probe()
    return {"attempted": True, "offscreen_after_init": ok, "message": message}


_GPU_INIT = _gpu_init_probe()
if _GPU_INIT.get("offscreen_after_init"):
    _GPU_OK, _GPU_MSG = True, "ok after gpu.init()"


def _needs_gpu(test):
    def wrapped():
        if not _GPU_OK:
            return {"skipped": "no GPU context (%s), even after gpu.init()"
                               % _GPU_MSG, "checks": {}}
        return test()
    wrapped.__name__ = test.__name__
    return wrapped


def _path(name):
    return os.path.join(_TMP, name)


def _load(path):
    """The file's pixels as (height, width, 3) uint8, TOP row first.

    Through Blender's loader, which hands rows back bottom first -- hence the
    reversal, and hence its independence from the OIIO writer.
    """
    image = bpy.data.images.load(path, check_existing=False)
    try:
        width, height = image.size
        flat = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(flat)
    finally:
        bpy.data.images.remove(image)
    rows = flat.reshape(height, width, 4)[::-1, :, :3]
    return (rows * 255.0 + 0.5).astype(np.uint8)


def _marks(width, height, fills=(), strokes=(), labels=()):
    return fm.Marks(float(width), float(height),
                    tuple(fills), tuple(strokes), tuple(labels))


def _rect(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _is(pixel, rgb, tol=2):
    return all(abs(int(pixel[i]) - rgb[i]) <= tol for i in range(3))


def _ink(rows, limit=128):
    """Boolean mask of dark pixels."""
    return rows.astype(np.int32).sum(axis=2) < limit * 3


def _ink_box(mask):
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ---------------------------------------------------------------------------

def test_gpu_context_probe():
    """Record whether this Blender can draw offscreen, instead of assuming.

    launch_blender.py says a background Blender cannot draw. That is a claim
    about the application, and CLAUDE.md wants those tested -- and it is only
    half true: the first GPUOffScreen is refused, and after `gpu.init()` it
    is not. Both answers are recorded. The assertions are the ones that must
    hold: a windowed instance can draw, and so can a background one once
    initialised, since `foam_raster` relies on exactly that.
    """
    return {
        "background": bool(bpy.app.background),
        "offscreen_available": _GPU_OK,
        "message": _GPU_MSG,
        "gpu_init": _GPU_INIT,
        "oiio_available": fr.oiio is not None,
        "checks": {
            "a_windowed_blender_can_draw": bpy.app.background or _GPU_OK,
            "a_background_blender_can_after_gpu_init":
                (not bpy.app.background) or _GPU_OK,
        },
    }


def test_pixel_size_is_arithmetic():
    """Sheet millimetres to pixels, with no GPU involved.

    100 x 50 mm at 254 dpi: 254 px per 25.4 mm is 10 px/mm, so 1000 x 500.
    By size: longest edge is the size, the other follows the aspect.
    Clamp: 2000 mm at 300 dpi would be 23622 px; held to 16384, which over
    2000/25.4 inches is 208.08 dpi -- and that, not 300, must be reported.
    """
    by_dpi = fr.pixel_size(100.0, 50.0, size=2048, dpi=254)
    by_size = fr.pixel_size(100.0, 50.0, size=2048)
    tall = fr.pixel_size(50.0, 100.0, size=2048)
    clamp = fr.pixel_size(2000.0, 1000.0, dpi=300)
    return {
        "by_dpi": list(by_dpi), "by_size": list(by_size),
        "tall": list(tall), "clamp": list(clamp),
        "checks": {
            "dpi_254_is_1000_by_500": by_dpi[:2] == (1000, 500),
            "dpi_reported_254": abs(by_dpi[2] - 254.0) < 1e-6,
            "dpi_not_clamped": by_dpi[3] is False,
            "size_sets_longest_edge": by_size[:2] == (2048, 1024),
            "size_dpi_follows": abs(by_size[2] - 2048 / (100 / 25.4)) < 1e-6,
            "longest_edge_may_be_height": tall[:2] == (1024, 2048),
            "clamped_to_max": clamp[:2] == (16384, 8192) and clamp[3] is True,
            "clamped_dpi_is_the_real_one": abs(clamp[2] - 208.0768) < 0.001,
        },
    }


@_needs_gpu
def test_sheet_at_254_dpi_is_1000_by_500():
    """The file itself has the size the arithmetic says."""
    path = _path("size.png")
    info = fr.render_marks(_marks(100, 50), path, dpi=DPI)
    rows = _load(path)
    return {
        "info": {k: v for k, v in info.items() if k != "path"},
        "file_shape": list(rows.shape),
        "checks": {
            "reported_1000_by_500": info["pixels"] == [1000, 500],
            "file_is_1000_by_500": rows.shape == (500, 1000, 3),
            "reported_dpi_254": info["dpi"] == 254.0,
            # An empty sheet is white paper, all of it.
            "empty_sheet_is_all_white": int(rows.min()) == 255,
        },
    }


@_needs_gpu
def test_a_fill_lands_where_its_millimetres_say():
    """A red rectangle near the top-left corner, and nowhere else.

    Rectangle x 5..25 mm, y 5..15 mm on a 100 x 50 mm sheet -> pixels
    x 50..250, y 50..150 counted from the TOP-left. Its centre is (150, 100).

    The three other candidates for that centre are exactly where it would be
    under each wrong orientation: (850, 100) if mirrored left-right,
    (150, 400) if flipped top-bottom, (850, 400) if both. All must be white.

    Edges are checked 5 px either side, so the position has both bounds and a
    rectangle of the wrong size or offset cannot pass on its centre alone.

    The grey fill is 0.5 -> 128 (127.5 rounds either way): a colour-space
    conversion on the way to the file would move it to 188 or 55.
    """
    path = _path("fill.png")
    fr.render_marks(_marks(100, 50, fills=[
        fm.Fill('piece', (_rect(5, 5, 25, 15),), RED),
        fm.Fill('piece', (_rect(60, 30, 80, 40),), (0.5, 0.5, 0.5)),
    ]), path, dpi=DPI)
    rows = _load(path)
    red, white = (255, 0, 0), (255, 255, 255)
    return {
        "centre": rows[100, 150].tolist(),
        "mirrored": rows[100, 850].tolist(),
        "flipped": rows[400, 150].tolist(),
        "grey": rows[350, 700].tolist(),
        "red_pixels": int((_is_mask(rows, red)).sum()),
        "checks": {
            "centre_is_red": _is(rows[100, 150], red),
            "not_mirrored": _is(rows[100, 850], white),
            "not_flipped": _is(rows[400, 150], white),
            "not_rotated": _is(rows[400, 850], white),
            "left_edge_at_50": _is(rows[100, 45], white) and _is(rows[100, 55], red),
            "right_edge_at_250": _is(rows[100, 245], red) and _is(rows[100, 255], white),
            "top_edge_at_50": _is(rows[45, 150], white) and _is(rows[55, 150], red),
            "bottom_edge_at_150": _is(rows[145, 150], red) and _is(rows[155, 150], white),
            # 200 x 100 px = 20000, within a pixel row/column of rounding.
            "area_is_20000_px": 19700 <= int(_is_mask(rows, red).sum()) <= 20300,
            "grey_is_not_colour_managed": _is(rows[350, 700], (128, 128, 128)),
        },
    }


def _is_mask(rows, rgb, tol=2):
    diff = np.abs(rows.astype(np.int32) - np.array(rgb, dtype=np.int32))
    return (diff <= tol).all(axis=2)


@_needs_gpu
def test_an_annulus_keeps_its_hole():
    """Two rings fill even-odd: the hole stays paper.

    Outer 10..40 mm square, hole 20..30 mm. Area 900 - 100 = 800 mm^2 =
    80000 px. A fan from the centre, or a tessellator that ignored the second
    ring, gives 90000 and a red centre.

    Done twice, with the hole wound the same way as the outer ring and the
    opposite way, because even-odd must not care and `build_marks` promises
    no winding.
    """
    outer = _rect(10, 10, 40, 40)
    hole = _rect(20, 20, 30, 30)
    out, checks = {}, {}
    for label, inner in (("same_winding", hole),
                         ("opposite_winding", tuple(reversed(hole)))):
        path = _path("annulus_%s.png" % label)
        fr.render_marks(_marks(50, 50, fills=[
            fm.Fill('piece', (outer, inner), RED)]), path, dpi=DPI)
        rows = _load(path)
        area = int(_is_mask(rows, (255, 0, 0)).sum())
        out[label] = {"hole_centre": rows[250, 250].tolist(), "area": area}
        checks[label + "_hole_is_white"] = _is(rows[250, 250], (255, 255, 255))
        checks[label + "_ring_is_red"] = (
            _is(rows[250, 150], (255, 0, 0)) and _is(rows[250, 350], (255, 0, 0))
            and _is(rows[150, 250], (255, 0, 0)) and _is(rows[350, 250], (255, 0, 0)))
        checks[label + "_area_is_80000_px"] = 78800 <= area <= 81200
    out["checks"] = checks
    return out


@_needs_gpu
def test_stroke_width_is_millimetres():
    """A horizontal line W mm wide covers round(W * dpi / 25.4) rows, +-1.

    1.0 mm at 254 dpi is 10 rows; 0.35 mm (the usual cut line) is 3.5 -> 3
    or 4. And a line far finer than a pixel is still drawn, as 1 to 2 rows:
    a cut line that vanishes at low resolution is worse than a thick one.

    Rows are counted down one column through the middle of the line, and the
    line's centre must sit on its y, so a line of the right width in the
    wrong place fails too.
    """
    out, checks = {}, {}
    for label, width_mm, expected in (("1mm", 1.0, 10), ("0.35mm", 0.35, 4),
                                      ("hairline", 0.001, 1)):
        path = _path("stroke_%s.png" % label)
        fr.render_marks(_marks(100, 50, strokes=[
            fm.Stroke('outline', ((10, 25), (90, 25)), False, width_mm, BLACK),
        ]), path, dpi=DPI)
        rows = _load(path)
        column = _ink(rows, limit=160)[:, 500]
        ys = np.nonzero(column)[0]
        count = int(column.sum())
        centre = float(ys.mean()) if count else None
        out[label] = {"rows": count, "centre_y": centre}
        checks[label + "_rows"] = expected - 1 <= count <= expected + 1 and count >= 1
        checks[label + "_centred_on_250"] = (
            centre is not None and abs(centre - 249.5) <= 1.0)
        # The line stops at its ends: 10..90 mm, so 50 px and 950 px are bare.
        row = _ink(rows, limit=160)[250]
        checks[label + "_starts_at_100"] = (not row[90]) and bool(row[110])
        checks[label + "_ends_at_900"] = bool(row[890]) and (not row[910])
    out["checks"] = checks
    return out


@_needs_gpu
def test_a_dashed_stroke_has_gaps_and_closed_closes():
    """Dash (5 mm on, 5 mm off) over 80 mm is 8 dashes, half the length inked.

    Counted along the line's own row: runs of ink == 8 exactly, and inked
    fraction of the 800 px span within 5% of one half. A solid line gives one
    run and fraction 1; a dash pattern read in pixels instead of millimetres
    gives 80 runs.

    `closed`: an L of three points. Closed, the hypotenuse back to the first
    point is drawn; open, it is not. The midpoint of that edge is tested in
    both, so the flag has both bounds.
    """
    path = _path("dash.png")
    fr.render_marks(_marks(100, 50, strokes=[
        fm.Stroke('align', ((10, 25), (90, 25)), False, 0.5, BLACK, (5.0, 5.0)),
    ]), path, dpi=DPI)
    row = _ink(_load(path), limit=160)[250, 100:900]
    runs = int(np.count_nonzero(np.diff(np.concatenate(([0], row.astype(np.int8)))) == 1))
    fraction = float(row.mean())

    tri = ((10, 10), (40, 10), (10, 40))
    hits = {}
    for label, closed in (("closed", True), ("open", False)):
        p = _path("tri_%s.png" % label)
        fr.render_marks(_marks(50, 50, strokes=[
            fm.Stroke('outline', tri, closed, 0.5, BLACK)]), p, dpi=DPI)
        ink = _ink(_load(p), limit=160)
        # Midpoint of (40,10)-(10,40) is (25,25) -> px (250,250).
        hits[label] = {
            "hypotenuse": bool(ink[246:255, 246:255].any()),
            "top_edge": bool(ink[96:105, 250].any()),
            "closing_edge": bool(ink[250, 96:105].any()),
        }
    return {
        "dash_runs": runs, "dash_fraction": round(fraction, 3), "tri": hits,
        "checks": {
            "eight_dashes": runs == 8,
            "half_inked": 0.45 <= fraction <= 0.55,
            "open_draws_two_edges": hits["open"]["top_edge"]
                                    and hits["open"]["hypotenuse"],
            "open_leaves_the_last_edge": not hits["open"]["closing_edge"],
            "closed_draws_the_last_edge": hits["closed"]["closing_edge"],
        },
    }


@_needs_gpu
def test_text_is_upright_and_not_mirrored():
    """The letter F settles both flips at once.

    F has no symmetry at all. Its two bars are in its top half and its stem
    is on its left, so within its own ink bounding box:

      * more ink in the top half than the bottom  -- fails if upside down;
      * more ink in the left half than the right  -- fails if mirrored.

    A 180 degree turn fails both. A letter like A or T would catch only the
    vertical flip, and E or C only the horizontal one; a digit string could
    be read either way by a count like this. The previous attempt's text came
    out mirrored and nothing would have said so.

    The box must also sit around the label's point, or an F somewhere else on
    the sheet would pass.
    """
    path = shot.next_path("raster_F_orientation")
    fr.render_marks(_marks(100, 100, labels=[
        fm.Label('number', 50, 50, 60.0, "F", BLACK)]), path, dpi=DPI)
    ink = _ink(_load(path))
    x0, y0, x1, y1 = _ink_box(ink)
    box = ink[y0:y1 + 1, x0:x1 + 1]
    h, w = box.shape
    top, bottom = int(box[:h // 2].sum()), int(box[h - h // 2:].sum())
    left, right = int(box[:, :w // 2].sum()), int(box[:, w - w // 2:].sum())
    return {
        "example": path,
        "ink_box": [x0, y0, x1, y1],
        "top": top, "bottom": bottom, "left": left, "right": right,
        "checks": {
            "there_is_ink": int(ink.sum()) > 5000,
            "upright_more_ink_on_top": top > bottom * 1.2,
            "not_mirrored_more_ink_on_left": left > right * 1.2,
            "box_contains_the_label_point": x0 < 500 < x1 and y0 < 500 < y1,
        },
    }


@_needs_gpu
def test_label_alignment_and_size():
    """Where a label sits relative to its point, and how big it is.

    size_mm is the SVG font-size: 20 mm at 254 dpi is a 200 px em. The
    tolerances are fractions of that em, because that is what "centred"
    means to someone reading the sheet.

      middle    the ink of a capital is centred on y: within 5% of the em.
      baseline  the bottom of a capital sits on y: within 5% of the em, and
                the ink is above it, not below.
      both      horizontally centred on x: within 5% of the em.
      size      foam_marks says cap height is roughly 0.7 of size_mm; the
                ink of an H must be between 0.6 and 0.8 em tall. A blf size
                read as points, or scaled by a UI factor, would not be.

    H is used because it is symmetric both ways and has no overshoot, so its
    ink box is its metrics.
    """
    em = 200.0
    out, checks = {}, {}
    for valign in ("middle", "baseline"):
        path = _path("label_%s.png" % valign)
        fr.render_marks(_marks(100, 100, labels=[
            fm.Label('angle', 50, 50, 20.0, "H", BLACK, valign)]), path, dpi=DPI)
        x0, y0, x1, y1 = _ink_box(_ink(_load(path)))
        cx, cy = (x0 + x1 + 1) / 2.0, (y0 + y1 + 1) / 2.0
        height = y1 - y0 + 1
        out[valign] = {"box": [x0, y0, x1, y1], "cx": cx, "cy": cy,
                       "cap_over_em": round(height / em, 3)}
        checks[valign + "_centred_on_x"] = abs(cx - 500.0) <= 0.05 * em
        checks[valign + "_cap_height_is_0.6_to_0.8_em"] = 0.6 * em <= height <= 0.8 * em
        if valign == "middle":
            checks["middle_ink_centred_on_y"] = abs(cy - 500.0) <= 0.05 * em
        else:
            checks["baseline_bottom_on_y"] = abs((y1 + 1) - 500.0) <= 0.05 * em
            checks["baseline_ink_is_above_y"] = cy < 500.0 - 0.25 * em
    out["checks"] = checks
    return out


@_needs_gpu
def test_draw_order_is_fills_strokes_labels():
    """Later covers earlier: fills in sequence, then strokes, then labels.

    A red fill, then a blue fill overlapping half of it: the overlap is blue.
    A black stroke across both: it is black over either. A white label "H"
    large enough to cross the stroke: where the label's ink is, the stroke is
    hidden. Each claim is checked with its opposite -- the uncovered part of
    each layer still shows -- so "everything is black" cannot pass.
    """
    path = _path("order.png")
    fr.render_marks(_marks(100, 50, fills=[
        fm.Fill('piece', (_rect(10, 10, 60, 40),), RED),
        fm.Fill('piece', (_rect(40, 10, 90, 40),), (0.0, 0.0, 1.0)),
    ], strokes=[
        fm.Stroke('outline', ((5, 25), (95, 25)), False, 2.0, BLACK),
    ], labels=[
        # H's crossbar sits on the middle line: 20 px thick stroke, covered.
        fm.Label('number', 50, 25, 30.0, "H", (1.0, 1.0, 1.0)),
    ]), path, dpi=DPI)
    rows = _load(path)
    white_on_stroke = int(_is_mask(rows[242:258, 400:600], (255, 255, 255)).sum())
    return {
        "red_only": rows[150, 200].tolist(), "overlap": rows[150, 500].tolist(),
        "stroke_on_red": rows[250, 200].tolist(),
        "white_px_on_stroke_band": white_on_stroke,
        "checks": {
            "first_fill_shows_alone": _is(rows[150, 200], (255, 0, 0)),
            "second_fill_covers_first": _is(rows[150, 500], (0, 0, 255)),
            "stroke_covers_fills": _is(rows[250, 200], (0, 0, 0))
                                   and _is(rows[250, 800], (0, 0, 0)),
            "label_covers_stroke": white_on_stroke > 500,
        },
    }


@_needs_gpu
def test_png_and_jpeg_both_load_back():
    """Both formats, the same picture, the right size.

    JPEG is lossy, so its red is checked to a looser tolerance -- but it must
    still be red on white the right way up, which is what a wrong channel
    order or an alpha-premultiplied write would break. The written file's
    first bytes are checked too: a PNG saved under a .jpg name loads fine and
    is not a JPEG.
    """
    marks = _marks(100, 50, fills=[fm.Fill('piece', (_rect(5, 5, 25, 15),), RED)])
    out, checks = {}, {}
    for fmt, name, magic in (('PNG', "both.png", b"\x89PNG"),
                             ('JPEG', "both.jpg", b"\xff\xd8\xff")):
        path = _path(name)
        info = fr.render_marks(marks, path, dpi=DPI, file_format=fmt)
        rows = _load(path)
        with open(path, "rb") as handle:
            head = handle.read(4)
        out[fmt] = {"shape": list(rows.shape), "centre": rows[100, 150].tolist(),
                    "outside": rows[400, 850].tolist(), "writer": info["writer"],
                    "bytes": os.path.getsize(path)}
        checks[fmt + "_size"] = rows.shape == (500, 1000, 3)
        checks[fmt + "_red_top_left"] = _is(rows[100, 150], (255, 0, 0), tol=12)
        checks[fmt + "_white_elsewhere"] = (
            _is(rows[400, 150], (255, 255, 255), tol=12)
            and _is(rows[100, 850], (255, 255, 255), tol=12))
        checks[fmt + "_is_really_that_format"] = head.startswith(magic)
    try:
        fr.render_marks(marks, _path("bad.tif"), file_format='TIFF')
        refused = False
    except ValueError:
        refused = True
    checks["unknown_format_is_refused"] = refused
    out["checks"] = checks
    return out


@_needs_gpu
def test_jpeg_at_4096_is_quick():
    """A 4096 px JPEG in seconds, not minutes.

    The first attempt converted every channel of every pixel to a Python
    float in a list -- 67 million of them at this size. The budget here is
    10 s, which a numpy/OIIO path beats by an order of magnitude and a
    per-pixel Python loop cannot meet. The time is reported so a regression
    is visible before it reaches the limit.
    """
    marks = _marks(200, 200, fills=[
        fm.Fill('piece', (_rect(10, 10, 190, 190), _rect(80, 80, 120, 120)), RED),
    ], strokes=[
        fm.Stroke('outline', _rect(10, 10, 190, 190), True, 0.35, BLACK),
    ], labels=[fm.Label('number', 50, 50, 20.0, "7", BLACK)])
    out, checks = {}, {}
    for fmt, name in (('JPEG', "big.jpg"), ('PNG', "big.png")):
        path = _path(name)
        started = time.perf_counter()
        info = fr.render_marks(marks, path, size=4096, file_format=fmt)
        seconds = time.perf_counter() - started
        image = bpy.data.images.load(path, check_existing=False)
        size = list(image.size)
        bpy.data.images.remove(image)
        out[fmt] = {"seconds": round(seconds, 2), "size": size,
                    "dpi": info["dpi"], "bytes": os.path.getsize(path)}
        checks[fmt + "_is_4096_square"] = size == [4096, 4096]
        checks[fmt + "_under_10_s"] = seconds < 10.0
    # 4096 px over 200 mm: 4096 / (200 / 25.4) = 520.19 dpi.
    checks["size_mode_reports_its_dpi"] = abs(out['JPEG']["dpi"] - 520.19) < 0.01
    out["checks"] = checks
    return out


@_needs_gpu
def test_dpi_is_written_into_the_file():
    """A print dialog reads the resolution from the file, so it must be there.

    Read back with OpenImageIO, which is what exposes the metadata. PNG keeps
    it as pixels per metre in an integer: 254 dpi is exactly 10000, and 300
    dpi is 11811.02 -> 11811, i.e. 299.9994 -- so the tolerance is 0.01 dpi,
    tight enough that the default of 72 or a value in the wrong unit
    (118.11 px/cm) cannot pass.
    """
    if fr.oiio is None:
        return {"skipped": "OpenImageIO not bundled; DPI metadata is only "
                           "written through it", "checks": {}}
    marks = _marks(100, 50)
    out, checks = {}, {}
    for fmt, ext in (('PNG', "png"), ('JPEG', "jpg")):
        for dpi in (254, 300):
            path = _path("dpi_%d.%s" % (dpi, ext))
            fr.render_marks(marks, path, dpi=dpi, file_format=fmt)
            source = fr.oiio.ImageInput.open(path)
            spec = source.spec()
            xres = spec.getattribute("XResolution")
            yres = spec.getattribute("YResolution")
            unit = spec.getattribute("ResolutionUnit")
            source.close()
            # Normalise whatever unit the reader reports to inches.
            per_inch = {"in": 1.0, "inch": 1.0, "cm": 2.54, "m": 0.0254}.get(unit)
            key = "%s_%d" % (fmt, dpi)
            out[key] = {"XResolution": xres, "YResolution": yres, "unit": unit}
            checks[key + "_x"] = (xres is not None and per_inch is not None
                                  and abs(xres * per_inch - dpi) < 0.01)
            checks[key + "_y"] = (yres is not None and per_inch is not None
                                  and abs(yres * per_inch - dpi) < 0.01)
    out["checks"] = checks
    return out


@_needs_gpu
def test_the_fallback_writer_agrees():
    """Without OpenImageIO the picture is the same, the same way up.

    Blender bundles OIIO, so this path never runs unless a build lacks it --
    which is exactly the kind of code that is wrong when it finally does.
    The two writers want opposite row orders, so a fallback that forgot would
    put the red square bottom-left.
    """
    marks = _marks(100, 50, fills=[fm.Fill('piece', (_rect(5, 5, 25, 15),), RED)])
    saved = fr.oiio
    out, checks = {}, {}
    fr.oiio = None
    try:
        for fmt, name in (('PNG', "fallback.png"), ('JPEG', "fallback.jpg")):
            path = _path(name)
            info = fr.render_marks(marks, path, dpi=DPI, file_format=fmt)
            rows = _load(path)
            out[fmt] = {"writer": info["writer"], "shape": list(rows.shape),
                        "centre": rows[100, 150].tolist()}
            checks[fmt + "_used_the_fallback"] = info["writer"] == "bpy"
            checks[fmt + "_size"] = rows.shape == (500, 1000, 3)
            checks[fmt + "_red_top_left"] = _is(rows[100, 150], (255, 0, 0), tol=12)
            checks[fmt + "_not_flipped"] = _is(rows[400, 150], (255, 255, 255), tol=12)
            checks[fmt + "_not_mirrored"] = _is(rows[100, 850], (255, 255, 255), tol=12)
    finally:
        fr.oiio = saved
    out["leftover_images"] = [i.name for i in bpy.data.images
                              if i.name.startswith("foamcraft_raster")]
    checks["no_image_datablock_left_behind"] = not out["leftover_images"]
    out["checks"] = checks
    return out


def test_the_operator_is_registered_as_the_panel_expects():
    """The idname the panel's button calls, with the properties it sets.

    Asked the way the user reaches it (CLAUDE.md, "Verify the verification"):
    `bpy.ops...get_rna_type().properties`, not `cls.bl_rna.properties`, which
    lists the Operator base and reports every operator as having none.

    Needs the add-on enabled in this Blender, which the launcher's instance
    has. Registration comes from the module loaded at startup, so this is a
    statement about the code on disk when Blender started.
    """
    try:
        rna = bpy.ops.foamcraft_drachis.export_pattern_image.get_rna_type()
    except Exception as exc:                           # noqa: BLE001
        return {"error": str(exc), "checks": {"operator_exists": False}}
    props = {p.identifier: p for p in rna.properties}
    menu = bpy.types.TOPBAR_MT_file_export
    in_menu = [f.__module__ for f in menu._dyn_ui_initialize()
               if f.__module__.endswith("foam_raster")]
    formats = ([i.identifier for i in props["file_format"].enum_items]
               if "file_format" in props else [])
    modes = ([i.identifier for i in props["resolution_mode"].enum_items]
             if "resolution_mode" in props else [])
    return {
        "label": rna.name, "properties": sorted(props), "menu_entries": in_menu,
        "checks": {
            "label": rna.name == "Foam Pattern Image (.png/.jpg)",
            "formats_are_png_and_jpeg": formats == ['PNG', 'JPEG'],
            "modes_are_dpi_and_pixels": modes == ['DPI', 'PIXELS'],
            "dpi_defaults_to_300": "dpi" in props and props["dpi"].default == 300,
            "size_defaults_to_4096": "size" in props and props["size"].default == 4096,
            "has_a_filepath": "filepath" in props,
            # Exactly one: two would mean a reload registered it twice.
            "in_file_export_once": len(in_menu) == 1,
        },
    }


def _seamed_cube():
    """A 0.2 m cube, every edge a seam: 6 islands (see tests/test_export_svg.py).

    Prop-sized on purpose. The 2 m cube the SVG tests use unwraps to a sheet
    4 m by 6 m, which at any printable dpi hits the 16384 px clamp -- so the
    example came out as a 10837 x 16384 image at 68 dpi. That clamp is
    checked by arithmetic above; here the point is an ordinary export.
    """
    sandbox_guard.clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=0.2)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def _big_colours(rows, share=0.005):
    """Distinct colours each covering at least `share` of the image.

    Flat fills are large areas of one exact colour; antialiased text and line
    edges are many colours of a few pixels each, and fall under the share.
    """
    flat = rows.reshape(-1, 3).astype(np.uint32)
    keys = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    floor = share * keys.size
    return [((int(v) >> 16) & 255, (int(v) >> 8) & 255, int(v) & 255)
            for v, c in zip(values, counts) if c >= floor]


@_needs_gpu
def test_a_seamed_cube_through_render():
    """The whole path: mesh -> build_marks -> image, on a known shape.

    A cube with every edge seamed is 6 islands. With island_fill on, each is
    one flat colour, so the image has exactly 6 large-area colours that are
    neither paper white nor ink black -- 5 means two pieces share a colour or
    one is missing, 7 means something else is being filled.

    With it off there are none, which is the other bound: the six are the
    fills and not something that is always there.

    `build_marks` is being extracted from the SVG exporter by other work. If
    it still raises NotImplementedError this case reports itself pending
    rather than failing, since the drawer is not what is missing.
    """
    obj = _seamed_cube()
    settings = bpy.context.scene.foamcraft_drachis
    saved = settings.island_fill
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    out, checks = {"faces": len(bm.faces)}, {}
    try:
        uv_layer = bm.loops.layers.uv.active
        for label, fill in (("filled", True), ("unfilled", False)):
            settings.island_fill = fill
            path = (shot.next_path("raster_cube_filled") if fill
                    else _path("cube_unfilled.png"))
            try:
                info = fr.render(bm, uv_layer, settings, path, dpi=150,
                                 scale_length=1.0,
                                 matrix=obj.matrix_world.copy())
            except NotImplementedError as exc:
                return {"skipped": "PENDING: foam_marks.build_marks is still "
                                   "a stub (%s). render_marks is covered by "
                                   "the cases above; rerun once it lands."
                                   % exc, "checks": {}}
            rows = _load(path)
            colours = [c for c in _big_colours(rows)
                       if c != (255, 255, 255) and sum(c) > 60]
            out[label] = {"info": {k: v for k, v in info.items() if k != "path"},
                          "example": path, "fill_colours": colours,
                          "ink_pixels": int(_ink(rows).sum())}
            checks[label + "_is_not_blank"] = int(_ink(rows).sum()) > 100
            checks[label + "_size_matches_report"] = (
                [rows.shape[1], rows.shape[0]] == info["pixels"])
        checks["six_faces"] = out["faces"] == 6
        # 150 was asked for and this sheet is small enough to get it.
        checks["dpi_is_the_one_asked_for"] = (
            out["filled"]["info"]["dpi"] == 150.0
            and out["filled"]["info"]["clamped"] is False)
        checks["six_fill_colours_when_on"] = len(out["filled"]["fill_colours"]) == 6
        checks["no_fill_colours_when_off"] = len(out["unfilled"]["fill_colours"]) == 0
    finally:
        settings.island_fill = saved
        bm.free()
    out["checks"] = checks
    return out


@_needs_gpu
def test_the_operator_writes_a_file():
    """The operator, called the way the panel button calls it.

    From object mode and from edit mode, since it handles the BMesh
    differently in each, and with a path whose extension disagrees with the
    format -- the file that appears must have the format's own extension.
    Pending, like the case above, while build_marks is a stub.
    """
    obj = _seamed_cube()
    try:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        try:
            fm.build_marks(bm, bm.loops.layers.uv.active,
                           bpy.context.scene.foamcraft_drachis, 1.0,
                           obj.matrix_world.copy())
        finally:
            bm.free()
    except NotImplementedError as exc:
        return {"skipped": "PENDING: foam_marks.build_marks is still a stub "
                           "(%s)" % exc, "checks": {}}
    out, checks = {}, {}
    for mode in ('OBJECT', 'EDIT'):
        bpy.ops.object.mode_set(mode=mode)
        asked = _path("op_%s.png" % mode.lower())
        wrote = _path("op_%s.jpg" % mode.lower())
        status = bpy.ops.foamcraft_drachis.export_pattern_image(
            filepath=asked, file_format='JPEG', resolution_mode='PIXELS',
            size=1024)
        exists = os.path.exists(wrote)
        size = None
        if exists:
            image = bpy.data.images.load(wrote, check_existing=False)
            size = list(image.size)
            bpy.data.images.remove(image)
        out[mode] = {"status": sorted(status), "size": size}
        checks[mode + "_finished"] = status == {'FINISHED'}
        checks[mode + "_wrote_jpg_not_png"] = exists and not os.path.exists(asked)
        checks[mode + "_longest_edge_1024"] = size is not None and max(size) == 1024
    bpy.ops.object.mode_set(mode='OBJECT')
    out["checks"] = checks
    return out


@_needs_gpu
def test_example_sheet_for_looking_at():
    """One of everything on a sheet, saved to screenshots/ to be looked at.

    Numbers above say each mark is where it should be; this is the picture a
    person can check them against. The only assertion is that it was written
    at the expected size -- the judgement is the reviewer's.
    """
    path = shot.next_path("raster_example_sheet")
    ring_outer = tuple((60 + 25 * c, 40 + 25 * s) for c, s in _circle(48))
    ring_inner = tuple((60 + 12 * c, 40 + 12 * s) for c, s in _circle(48))
    marks = _marks(160, 80, fills=[
        fm.Fill('piece', (_rect(5, 5, 30, 30),), (0.95, 0.6, 0.6)),
        fm.Fill('piece', (ring_outer, ring_inner), (0.6, 0.75, 0.95)),
    ], strokes=[
        fm.Stroke('outline', _rect(5, 5, 30, 30), True, 0.35, BLACK),
        fm.Stroke('outline', ring_outer, True, 0.35, BLACK),
        fm.Stroke('outline', ring_inner, True, 0.35, BLACK),
        fm.Stroke('align', ((5, 50), (150, 50)), False, 0.5, (0.8, 0.0, 0.0),
                  (4.0, 2.0)),
        fm.Stroke('ruler', ((100, 70), (150, 70)), False, 1.0, BLACK),
    ], labels=[
        fm.Label('number', 17.5, 17.5, 10.0, "1", BLACK),
        fm.Label('number', 60, 21.5, 8.0, "2", BLACK),
        fm.Label('angle', 120, 20, 8.0, "45.0\u00b0 F", (0.0, 0.3, 0.8)),
        fm.Label('ruler', 125, 68, 5.0, "50 mm", BLACK, 'baseline'),
    ])
    info = fr.render_marks(marks, path, dpi=150)
    return {
        "example": path, "pixels": info["pixels"],
        "checks": {
            # 160 mm at 150 dpi = 944.88 -> 945; 80 mm -> 472.44 -> 472.
            "size_is_945_by_472": info["pixels"] == [945, 472],
            "file_exists": os.path.getsize(path) > 1000,
        },
    }


def _circle(n):
    import math
    return [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
            for i in range(n)]


TESTS = (
    test_gpu_context_probe,
    test_pixel_size_is_arithmetic,
    test_sheet_at_254_dpi_is_1000_by_500,
    test_a_fill_lands_where_its_millimetres_say,
    test_an_annulus_keeps_its_hole,
    test_stroke_width_is_millimetres,
    test_a_dashed_stroke_has_gaps_and_closed_closes,
    test_text_is_upright_and_not_mirrored,
    test_label_alignment_and_size,
    test_draw_order_is_fills_strokes_labels,
    test_png_and_jpeg_both_load_back,
    test_jpeg_at_4096_is_quick,
    test_dpi_is_written_into_the_file,
    test_the_fallback_writer_agrees,
    test_the_operator_is_registered_as_the_panel_expects,
    test_a_seamed_cube_through_render,
    test_the_operator_writes_a_file,
    test_example_sheet_for_looking_at,
)

result = {}
_all_pass = True
for _test in TESTS:
    try:
        _report = _test()
        _report["PASS"] = all(_report.get("checks", {}).values())
    except Exception:
        _report = {"ERROR": traceback.format_exc().splitlines()[-8:],
                   "PASS": False}
    result[_test.__name__] = _report
    _all_pass = _all_pass and _report["PASS"]
result["ALL_PASS"] = _all_pass
