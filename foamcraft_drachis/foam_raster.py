"""Render the foam pattern to an image, the way Export UV Layout does.

WHY THIS EXISTS

The SVG export is a drawing described in XML, and what reaches the maker
depends on what opens it. Illustrator applied the stylesheet differently than
a browser does, the document came in at the wrong size, and marks that are
plainly visible in Blender were missing from the page. Every test written for
that exporter checks the SVG's *structure* -- how many elements, where they
sit -- and none of them can see what another program decides the file means.

A raster has no such argument. What is drawn is what arrives.

WHAT IT DRAWS

Only a `foam_marks.Marks`. The first attempt at this file worked the marks
out for itself -- islands, rings, piece numbers -- and so had no angles, no
brackets and no ruler, and disagreed with the SVG about the rest. This one is
a drawer: `render_marks` never sees the mesh or the settings, and cannot
decide whether a mark exists. Fills, then strokes, then labels, in order.

HOW IT WORKS, AND WHOSE SEMANTICS IT COPIES

Blender already rasterises a UV layout, in
`scripts/addons_core/io_mesh_uv_layout/export_uv_png.py`, and this follows it
rather than inventing a second way:

  * `GPUOffScreen(width, height)`, bound, cleared;
  * `FLAT_COLOR` + `TRIS` for filled areas, with `tessellate_polygon` doing
    the concave and holed cases -- which matters here, because a panel
    unwrapped from a band is an annulus and a naive fan fills its hole in;
  * `POLYLINE_UNIFORM_COLOR` for lines, which needs `viewportSize` and
    `lineWidth` uniforms set before it will draw;
  * `fb.read_color(..., 'UBYTE')` then OpenImageIO if it is there, falling
    back to `bpy.data.images`.

WHICH WAY UP, DECIDED ONCE

A `Marks` is in sheet millimetres with y growing *down*; a framebuffer has y
growing *up*. `_Canvas.px` is the one place that converts, and everything --
fills, lines, text anchors -- goes through it. So the framebuffer holds the
sheet the right way up as GL sees it, and `blf`, which only ever draws glyphs
upright in the space it is given, is upright too without being told anything.

The built-in exporter instead flips its projection when OIIO is present, to
suit OIIO's top-down rows. That works for it because a UV layout has no text.
The first attempt here copied the flip and the digits came out mirrored:
geometry obeyed the projection and the glyphs did not agree with it. Keeping
the framebuffer upright removes the thing they could disagree about.

What is left is the file writer's row order, which is a property of the
writer and not of the drawing: `fb.read_color` returns the bottom row first,
OIIO wants the top row first, `bpy.data.images` wants the bottom first.
`_read_rows_top_down` reverses once into the conventional top-down array, and
`_save` hands each writer what it expects.

Text is the part Export UV Layout has no answer for, because a UV layout has
none. It is drawn with `blf` after the geometry, in the same pixel space.

WITHOUT A WINDOW

A background Blender has no GPU context and refuses the offscreen buffer --
but the refusal names `gpu.init()`, and after calling it everything here
works: tests/test_export_raster.py passes in full headless. So `_ensure_gpu`
calls it when `bpy.app.background`, and the export can be scripted with
`blender -b`. A windowed Blender already has a context and is left alone.

TRUE TO SCALE

Given `dpi`, the image is the sheet's millimetres at that many pixels per
inch, and the figure is written into the file (`XResolution`), so a program
that honours it prints at 1:1 without being asked. Given `size` instead, the
longest edge is that many pixels and the dpi that results is reported.

PAGES

No home printer takes a metre-wide sheet, so `pages='LETTER'` (or 'A4') also
cuts the rendered sheet into paper-sized files, 1:1, at the same dpi. This is
deliberately light: the pieces are not re-packed, the sheet is simply cropped.
The arithmetic -- page sizes, a 6.35 mm margin, a 10 mm overlap to tape on --
is tools/tile_svg.py's, copied here because `tools/` is stripped from releases.
Each page file is white paper with its tile inside the margins, thin corner
ticks at the printable area's corners, and its place in the grid in the
FILENAME (`name_p03_r1c3.png`); no text is drawn into the page itself.
"""

import os
from math import cos, radians, sin

import bpy
import bmesh
import gpu
import blf
import numpy as np
from mathutils import Matrix
from mathutils.geometry import tessellate_polygon
from gpu_extras.batch import batch_for_shader
from bpy_extras.io_utils import ExportHelper

from . import foam_log
from . import foam_marks

# Use OIIO if available, else Blender for writing the image -- as the built-in
# UV layout exporter does.
try:
    import OpenImageIO as oiio
except ImportError:
    oiio = None

_log = foam_log.get_logger()

# Any wider than this and the offscreen allocation starts failing on modest
# hardware (it is also the usual maximum texture size).
MAX_PIXELS = 16384
MM_PER_INCH = 25.4
JPEG_QUALITY = 95

# Page arithmetic copied from tools/tile_svg.py (PAGE_SIZES, DEFAULT_MARGIN_MM,
# DEFAULT_OVERLAP_MM, CROP_MM); keep the two in step.
PAGE_SIZES_MM = {
    'LETTER': (8.5 * MM_PER_INCH, 11.0 * MM_PER_INCH),
    'A4': (210.0, 297.0),
}
PAGE_MARGIN_MM = 6.35
PAGE_OVERLAP_MM = 10.0
PAGE_CROP_MM = 5.0
PAGE_MARK_MM = 0.25

_FORMATS = {
    # file_format: (OIIO format name, extensions accepted, extension written)
    'PNG': ("png", (".png",), ".png"),
    'JPEG': ("jpeg", (".jpg", ".jpeg"), ".jpg"),
}


def pixel_size(width_mm, height_mm, size=2048, dpi=None):
    """(width_px, height_px, effective_dpi, clamped) for a sheet.

    With `dpi`, pixels are the sheet's millimetres at that resolution. Without,
    the longest edge is `size`. Either way the longest edge is held to
    MAX_PIXELS, and the dpi returned is the one the image really has -- which
    is the requested one unless the clamp bit.
    """
    if width_mm <= 0.0 or height_mm <= 0.0:
        raise ValueError("the sheet has no area (%r x %r mm)"
                         % (width_mm, height_mm))
    longest_mm = max(width_mm, height_mm)
    if dpi:
        px_per_mm = float(dpi) / MM_PER_INCH
    else:
        px_per_mm = max(1, int(size)) / longest_mm
    clamped = longest_mm * px_per_mm > MAX_PIXELS + 0.5
    if clamped:
        px_per_mm = MAX_PIXELS / longest_mm
    width_px = max(1, int(round(width_mm * px_per_mm)))
    height_px = max(1, int(round(height_mm * px_per_mm)))
    return width_px, height_px, px_per_mm * MM_PER_INCH, clamped


class _Canvas:
    """Sheet millimetres to framebuffer pixels. The only place y is turned."""

    def __init__(self, width_px, height_px, width_mm, height_mm):
        self.width_px = width_px
        self.height_px = height_px
        self.sx = width_px / width_mm
        self.sy = height_px / height_mm
        # One number for sizes that have no direction (line width, font).
        self.px_per_mm = 0.5 * (self.sx + self.sy)

    def px(self, x_mm, y_mm):
        # Sheet y runs down from the top edge; framebuffer y runs up from the
        # bottom. This subtraction is the whole of the orientation logic.
        return (x_mm * self.sx, self.height_px - y_mm * self.sy)

    def projection(self):
        """Pixels to normalised device coordinates, y up, nothing flipped."""
        m = Matrix.Identity(4)
        m[0][0] = 2.0 / self.width_px
        m[1][1] = 2.0 / self.height_px
        m.col[3][0] = -1.0
        m.col[3][1] = -1.0
        return m


def _rgba(color):
    return (float(color[0]), float(color[1]), float(color[2]), 1.0)


def _draw_fills(fills, canvas):
    """Every fill in one batch, in sequence order.

    `tessellate_polygon` takes the outer boundary and its holes together and
    returns triangles that leave the holes empty, as indices into the rings
    laid end to end.
    """
    coords, colors, indices = [], [], []
    for fill in fills:
        rings = [ring for ring in fill.rings if len(ring) >= 3]
        if not rings:
            continue
        offset = len(coords)
        polys = [[canvas.px(x, y) for x, y in ring] for ring in rings]
        count = 0
        for poly in polys:
            coords.extend(poly)
            count += len(poly)
        colors.extend([_rgba(fill.color)] * count)
        for tri in tessellate_polygon(polys):
            indices.append(tuple(i + offset for i in tri))
    if not indices:
        return
    shader = gpu.shader.from_builtin('FLAT_COLOR')
    batch = batch_for_shader(shader, 'TRIS',
                             {"pos": coords, "color": colors},
                             indices=indices)
    batch.draw(shader)


def _dash(points, on, off):
    """Cut a pixel polyline into its drawn pieces, phase carried round bends.

    Returns (start, end) pairs. A pattern that is not positive in both parts
    is drawn solid, since it has no meaning as a dash.
    """
    pairs = []
    if on <= 0.0 or off <= 0.0:
        return [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    drawing, left = True, on
    for i in range(len(points) - 1):
        (ax, ay), (bx, by) = points[i], points[i + 1]
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if length <= 0.0:
            continue
        at = 0.0
        while at < length:
            step = min(left, length - at)
            if drawing:
                t0, t1 = at / length, (at + step) / length
                pairs.append(((ax + (bx - ax) * t0, ay + (by - ay) * t0),
                              (ax + (bx - ax) * t1, ay + (by - ay) * t1)))
            at += step
            left -= step
            if left <= 1e-9:
                drawing = not drawing
                left = on if drawing else off
    return pairs


def _stroke_pairs(stroke, canvas):
    points = [canvas.px(x, y) for x, y in stroke.points]
    if len(points) < 2:
        return []
    if stroke.closed:
        points.append(points[0])
    if stroke.dash:
        return _dash(points, stroke.dash[0] * canvas.px_per_mm,
                     stroke.dash[1] * canvas.px_per_mm)
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def _draw_strokes(strokes, canvas):
    """Lines, in sequence order, one batch per run of like colour and width."""
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    viewport = (float(canvas.width_px), float(canvas.height_px))

    def flush(key, coords):
        if not coords:
            return
        color, width_px = key
        shader.bind()
        # Both uniforms are required; the shader draws nothing without them.
        shader.uniform_float("viewportSize", viewport)
        shader.uniform_float("lineWidth", width_px)
        shader.uniform_float("color", color)
        batch_for_shader(shader, 'LINES', {"pos": coords}).draw(shader)

    key, coords = None, []
    for stroke in strokes:
        # A hairline finer than a pixel would otherwise vanish at low
        # resolution; one pixel is the least that can be drawn.
        this = (_rgba(stroke.color),
                max(1.0, stroke.width_mm * canvas.px_per_mm))
        if this != key:
            flush(key, coords)
            key, coords = this, []
        for a, b in _stroke_pairs(stroke, canvas):
            coords.append(a)
            coords.append(b)
    flush(key, coords)


def _draw_labels(labels, canvas):
    """Text, upright because the framebuffer is.

    blf works in pixels against the bound framebuffer, which is the space
    everything else is already drawn in.
    """
    font = 0
    for label in labels:
        if not label.text:
            continue
        size_px = max(1.0, label.size_mm * canvas.px_per_mm)
        blf.size(font, size_px)
        blf.color(font, *_rgba(label.color))
        x, y = canvas.px(label.x, label.y)
        width, _ = blf.dimensions(font, label.text)
        drop = 0.0
        if label.valign == 'middle':
            # Centre on the capital height, measured from the font rather
            # than taken as a fraction of the size, so digits and capitals --
            # which is what a pattern is labelled with -- sit with as much ink
            # above y as below it. Measuring the text itself would let a
            # comma or a descender shift a whole label.
            drop = blf.dimensions(font, "X")[1] * 0.5
        turn = getattr(label, 'rotation_deg', 0.0) or 0.0
        if not turn:
            blf.position(font, x - width * 0.5, y - drop, 0.0)
            blf.draw(font, label.text)
            continue
        # A turned label (foam_marks: counter-clockwise on the page). The
        # framebuffer is y-up, so blf's counter-clockwise rotation is the
        # page's, and the pen starts half the width back along the reading
        # direction and `drop` down along the text's own "down".
        a = radians(turn)
        ca, sa = cos(a), sin(a)
        px = x - ca * width * 0.5 + sa * drop
        py = y - sa * width * 0.5 - ca * drop
        blf.enable(font, blf.ROTATION)
        try:
            blf.rotation(font, a)
            blf.position(font, px, py, 0.0)
            blf.draw(font, label.text)
        finally:
            blf.disable(font, blf.ROTATION)


def _read_rows_top_down(fb, width, height):
    """The framebuffer as a (height, width, 3) uint8 array, top row first.

    `read_color` returns the bottom row first. Alpha is dropped: the sheet is
    opaque white paper, and blending text over an opaque clear can leave
    alpha below one at glyph edges that a viewer would then show as grey.
    """
    pixels = fb.read_color(0, 0, width, height, 4, 0, 'UBYTE')
    pixels.dimensions = width * height * 4
    rows = np.array(pixels, dtype=np.uint8).reshape(height, width, 4)
    return np.ascontiguousarray(rows[::-1, :, :3])


def _save(filepath, rows, file_format, dpi):
    """Write top-down RGB rows. Returns the name of the writer used."""
    height, width = rows.shape[0], rows.shape[1]
    if oiio:
        spec = oiio.ImageSpec(width, height, 3, "uint8")
        spec.attribute("XResolution", float(dpi))
        spec.attribute("YResolution", float(dpi))
        spec.attribute("ResolutionUnit", "in")
        if file_format == 'JPEG':
            spec.attribute("Compression", "jpeg:%d" % JPEG_QUALITY)
        image = oiio.ImageOutput.create(_FORMATS[file_format][0])
        if image is None:
            raise OSError("no image writer for %s" % file_format)
        # Calling write_image() on an output that failed to open crashes
        # OIIO, so the open() result must be checked (Blender #159686).
        if not image.open(filepath, spec):
            raise OSError(image.geterror() or "could not open %r" % filepath)
        if not image.write_image(rows):
            error = image.geterror()
            image.close()
            raise OSError(error or "could not write %r" % filepath)
        image.close()
        return "oiio"

    # Blender's own images fill from the bottom row, and take floats.
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[:, :, :3] = rows[::-1].astype(np.float32) / 255.0
    image = bpy.data.images.new("foamcraft_raster", width, height,
                                alpha=False)
    try:
        # The bytes are display values already; do not let saving convert.
        image.colorspace_settings.name = 'sRGB'
        image.pixels.foreach_set(rgba.ravel())
        image.filepath_raw = filepath
        image.file_format = file_format
        image.save()
    finally:
        bpy.data.images.remove(image)
    return "bpy"


def page_geometry(dpi, page='LETTER', landscape=False,
                  margin_mm=PAGE_MARGIN_MM):
    """(page_w_px, page_h_px, margin_px) of one sheet of paper at `dpi`."""
    if page not in PAGE_SIZES_MM:
        raise ValueError("unknown page size %r; known: %s"
                         % (page, ", ".join(sorted(PAGE_SIZES_MM))))
    w_mm, h_mm = PAGE_SIZES_MM[page]
    if landscape:
        w_mm, h_mm = h_mm, w_mm
    px = dpi / MM_PER_INCH
    return round(w_mm * px), round(h_mm * px), round(margin_mm * px)


def _starts(length, usable, step):
    """Offsets of the crops along one axis: as few as cover `length`."""
    if length <= usable:
        return [0]
    count = -(-(length - usable) // step) + 1
    return [i * step for i in range(count)]


def page_grid(width_px, height_px, dpi, page='LETTER', landscape=False,
              margin_mm=PAGE_MARGIN_MM, overlap_mm=PAGE_OVERLAP_MM):
    """Crop rectangles that cover a sheet, as [(row, col, x0, y0, x1, y1)].

    Pixels, top-left origin, x1/y1 exclusive, reading order, rows and columns
    counted from 0. A crop is at most one page's printable area (the page
    less its margins); neighbours share `overlap_mm` of pixels to tape along.
    """
    page_w, page_h, margin = page_geometry(dpi, page, landscape, margin_mm)
    usable_w, usable_h = page_w - 2 * margin, page_h - 2 * margin
    overlap = round(overlap_mm / MM_PER_INCH * dpi)
    if usable_w <= overlap or usable_h <= overlap:
        raise ValueError("margins and overlap leave no usable page area")
    xs = _starts(width_px, usable_w, usable_w - overlap)
    ys = _starts(height_px, usable_h, usable_h - overlap)
    return [(r, c, x, y, min(x + usable_w, width_px),
             min(y + usable_h, height_px))
            for r, y in enumerate(ys) for c, x in enumerate(xs)]


def compose_page(rows, crop, dpi, page='LETTER', landscape=False,
                 margin_mm=PAGE_MARGIN_MM):
    """One page's pixels: white paper, the crop inside the margins, and
    black ticks at the printable area's corners pointing inward (as
    tools/tile_svg.py draws them)."""
    _r, _c, x0, y0, x1, y1 = crop
    page_w, page_h, margin = page_geometry(dpi, page, landscape, margin_mm)
    out = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    out[margin:margin + (y1 - y0), margin:margin + (x1 - x0)] = \
        rows[y0:y1, x0:x1]
    arm = max(1, round(PAGE_CROP_MM / MM_PER_INCH * dpi))
    thick = max(1, round(PAGE_MARK_MM / MM_PER_INCH * dpi))
    right, bottom = page_w - margin, page_h - margin
    for cx, sx in ((margin, 1), (right, -1)):
        for cy, sy in ((margin, 1), (bottom, -1)):
            xa, xb = sorted((cx, cx + sx * arm))
            ya, yb = sorted((cy, cy + sy * arm))
            hy = slice(cy, cy + thick) if sy > 0 else slice(cy - thick, cy)
            vx = slice(cx, cx + thick) if sx > 0 else slice(cx - thick, cx)
            out[hy, xa:xb] = 0
            out[ya:yb, vx] = 0
    return out


def write_pages(rows, filepath, file_format, dpi, page='LETTER',
                landscape=False):
    """Write every page of the sheet next to `filepath`. Returns the paths,
    named `<stem>_p01_r1c1.<ext>` in reading order."""
    height_px, width_px = rows.shape[0], rows.shape[1]
    grid = page_grid(width_px, height_px, dpi, page, landscape)
    stem, ext = os.path.splitext(filepath)
    paths = []
    for n, crop in enumerate(grid, 1):
        path = "%s_p%02d_r%dc%d%s" % (stem, n, crop[0] + 1, crop[1] + 1, ext)
        _save(path, compose_page(rows, crop, dpi, page, landscape),
              file_format, dpi)
        paths.append(path)
    return paths


def _ensure_gpu():
    """Give a background Blender the GPU context a windowed one already has."""
    if bpy.app.background and hasattr(gpu, "init"):
        gpu.init()


def render_marks(marks, filepath, size=2048, file_format='PNG', dpi=None,
                 pages='NONE'):
    """Draw `marks` and write the image. Returns a dict of what was done.

    `size` is the longest edge in pixels, as Export UV Layout's own `size`
    is. `dpi`, when given, replaces it: the image is then the sheet's
    millimetres at that resolution, true to scale when printed at it.
    `pages` ('LETTER'/'A4') also writes the sheet as paper-sized files, which
    needs `dpi`: pages only mean anything at a known scale.
    """
    if pages != 'NONE':
        if not dpi:
            raise ValueError("Pages need DPI mode: they are only true to "
                             "scale at a known resolution. Use DPI, not "
                             "Pixels.")
        if pages not in PAGE_SIZES_MM:
            raise ValueError("pages must be 'NONE', 'LETTER' or 'A4', not %r"
                             % (pages,))
    if file_format not in _FORMATS:
        raise ValueError("file_format must be 'PNG' or 'JPEG', not %r"
                         % (file_format,))
    width_px, height_px, effective_dpi, clamped = pixel_size(
        marks.width_mm, marks.height_mm, size, dpi)
    if pages != 'NONE' and clamped:
        raise ValueError("The sheet is too large for %d DPI, so pages would "
                         "not be true to scale. Lower the DPI." % dpi)
    canvas = _Canvas(width_px, height_px, marks.width_mm, marks.height_mm)

    _ensure_gpu()
    offscreen = gpu.types.GPUOffScreen(width_px, height_px)
    offscreen.bind()
    try:
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(1.0, 1.0, 1.0, 1.0))
        gpu.state.blend_set('ALPHA')
        with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
            gpu.matrix.load_matrix(Matrix.Identity(4))
            gpu.matrix.load_projection_matrix(canvas.projection())
            _draw_fills(marks.fills, canvas)
            _draw_strokes(marks.strokes, canvas)
            _draw_labels(marks.labels, canvas)
        gpu.state.blend_set('NONE')
        rows = _read_rows_top_down(fb, width_px, height_px)
    finally:
        offscreen.unbind()
        offscreen.free()

    writer = _save(filepath, rows, file_format, effective_dpi)
    page_files = (write_pages(rows, filepath, file_format, effective_dpi,
                              pages) if pages != 'NONE' else [])

    info = {
        "path": filepath,
        "format": file_format,
        "writer": writer,
        "pixels": [width_px, height_px],
        "sheet_mm": [round(marks.width_mm, 2), round(marks.height_mm, 2)],
        "dpi": round(effective_dpi, 2),
        "clamped": clamped,
        "page_files": page_files,
        "fills": len(marks.fills),
        "strokes": len(marks.strokes),
        "labels": len(marks.labels),
        # Carried out to the operator, which is the only place that can
        # tell the maker; a warning left in the Marks is a warning unread.
        "warnings": list(marks.warnings),
    }
    _log.info(foam_log.summary("raster_export", **{
        k: v for k, v in info.items() if k not in ("path", "page_files")},
        pages=len(page_files)))
    return info


def render(bm, uv_layer, settings, filepath, size=2048, scale_length=1.0,
           file_format='PNG', matrix=None, dpi=None, pages='NONE'):
    """Build the marks for this mesh and draw them to `filepath`."""
    marks = foam_marks.build_marks(bm, uv_layer, settings, scale_length,
                                   matrix)
    return render_marks(marks, filepath, size=size, file_format=file_format,
                        dpi=dpi, pages=pages)


class FOAMCRAFT_DRACHIS_OT_export_pattern_image(bpy.types.Operator,
                                                ExportHelper):
    """Export the flat foam cut pattern as an image, true to scale at its DPI"""

    bl_idname = "foamcraft_drachis.export_pattern_image"
    bl_label = "Foam Pattern Image (.png/.jpg)"
    bl_options = {'PRESET'}

    filter_glob: bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg",
                                          options={'HIDDEN'})
    file_format: bpy.props.EnumProperty(
        name="Format",
        items=(('PNG', "PNG", "Lossless; sharpest lines and text"),
               ('JPEG', "JPEG", "Smaller file, for print services that "
                                "ask for one")),
        default='PNG')
    resolution_mode: bpy.props.EnumProperty(
        name="Resolution",
        items=(('DPI', "DPI", "Size the image from the pattern's real size, "
                              "so printing at this DPI is true to scale"),
               ('PIXELS', "Pixels", "Set the image's longest edge in pixels")),
        default='DPI')
    dpi: bpy.props.IntProperty(
        name="DPI", description="Pixels per inch of pattern",
        default=300, min=10, max=2400)
    pages: bpy.props.EnumProperty(
        name="Pages",
        items=(('NONE', "Whole sheet only", "One image"),
               ('LETTER', "US Letter pages",
                "Also write the sheet cut into Letter pages (needs DPI "
                "mode), 1:1, with a 10 mm overlap to tape and corner marks"),
               ('A4', "A4 pages", "As Letter, on A4")),
        default='NONE')
    size: bpy.props.IntProperty(
        name="Size", description="Longest edge of the image",
        default=4096, min=64, max=MAX_PIXELS, subtype='PIXEL')

    # ExportHelper reads this to name the file and to correct its extension
    # as the format changes in the file browser.
    @property
    def filename_ext(self):
        return _FORMATS[self.file_format][2]

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "file_format")
        layout.prop(self, "resolution_mode")
        if self.resolution_mode == 'DPI':
            layout.prop(self, "dpi")
        else:
            layout.prop(self, "size")
        layout.prop(self, "pages")

    def execute(self, context):
        if self.pages != 'NONE' and self.resolution_mode != 'DPI':
            self.report({'ERROR'}, "Pages need DPI mode: set Resolution to "
                                   "DPI so the pages are true to scale.")
            return {'CANCELLED'}
        path = self.filepath
        if not path.lower().endswith(_FORMATS[self.file_format][1]):
            path = os.path.splitext(path)[0] + self.filename_ext

        # Every object Export UV Layout would take, overlaid on one sheet
        # (foam_export.export_objects has the rule and where it is from).
        from . import foam_export
        objects = foam_export.export_objects(context)
        if not objects:
            self.report({'ERROR'}, "No selected mesh has an active UV map. "
                                   "Unwrap it first.")
            return {'CANCELLED'}
        with foam_export.SheetSource(objects) as items:
            marks = foam_marks.build_marks_multi(
                items, context.scene.foamcraft_drachis,
                context.scene.unit_settings.scale_length)
        try:
            info = render_marks(
                marks, path, size=self.size, file_format=self.file_format,
                dpi=self.dpi if self.resolution_mode == 'DPI' else None,
                pages=self.pages)
        except (ValueError, OSError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        note = ""
        if info["clamped"]:
            note = (" (too large for the requested resolution; "
                    "print at the DPI shown)")
        # Its own name: this loop used to reuse `note`, so a warning
        # overwrote the clamp note above it in the final report.
        if info["page_files"]:
            note += " + %d %s page files" % (len(info["page_files"]),
                                             self.pages)
        for warning in info.get("warnings", ()):
            self.report({'WARNING'}, warning)
        self.report({'INFO'}, "Wrote foam pattern image to %s: %d x %d px, "
                    "%.0f DPI%s" % (path, info["pixels"][0],
                                    info["pixels"][1], info["dpi"], note))
        return {'FINISHED'}


def menu_func_export(self, context):
    self.layout.operator(FOAMCRAFT_DRACHIS_OT_export_pattern_image.bl_idname,
                         text="Foam Pattern Image (.png/.jpg)")


def register():
    bpy.utils.register_class(FOAMCRAFT_DRACHIS_OT_export_pattern_image)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.utils.unregister_class(FOAMCRAFT_DRACHIS_OT_export_pattern_image)
