"""Render the foam pattern to an image, the way Export UV Layout does.

WHY THIS EXISTS

The SVG export is a drawing described in XML, and what reaches the maker
depends on what opens it. Illustrator applied the stylesheet differently than
a browser does, the document came in at the wrong size, and marks that are
plainly visible in Blender were missing from the page. Every test written for
that exporter checks the SVG's *structure* -- how many elements, where they
sit -- and none of them can see what another program decides the file means.

A raster has no such argument. What is drawn is what arrives.

HOW IT WORKS, AND WHOSE SEMANTICS IT COPIES

Blender already rasterises a UV layout, in
`scripts/addons_core/io_mesh_uv_layout/export_uv_png.py`, and this follows it
rather than inventing a second way:

  * `GPUOffScreen(width, height)`, bound, cleared to transparent;
  * a projection that maps the sheet to normalised device coordinates;
  * `FLAT_COLOR` + `TRIS` for filled areas, with `tessellate_polygon` doing
    the concave and holed cases -- which matters here, because a panel
    unwrapped from a band is an annulus and a naive fan fills its hole in;
  * `POLYLINE_UNIFORM_COLOR` for lines, which needs `viewportSize` and
    `lineWidth` uniforms set before it will draw;
  * `fb.read_color(..., 'UBYTE')` then OpenImageIO if it is there, falling
    back to `bpy.data.images`.

The one detail that is easy to get wrong and expensive to debug: **OpenImageIO
writes rows from the top, and Blender's own image API writes them from the
bottom**, so the projection is flipped vertically when OIIO is present. The
built-in exporter does this and says why; copying it is cheaper than
rediscovering it from an upside-down pattern.

Text is the part Export UV Layout has no answer for, because a UV layout has
none. It is drawn with `blf` under a pixel-space projection after the geometry,
so the piece numbers and bevel angles land on top.
"""

import os

import bpy
import gpu
from mathutils import Matrix, Vector
from mathutils.geometry import tessellate_polygon
from gpu_extras.batch import batch_for_shader
import blf

from . import foam_angle_core as core
from . import foam_export as fx
from . import foam_log

try:
    import OpenImageIO as oiio
except ImportError:
    oiio = None

_log = foam_log.get_logger()

# Any wider than this and the offscreen allocation starts failing on modest
# hardware. The ruler is what guarantees the print is true to size, so more
# pixels buy detail rather than correctness.
MAX_PIXELS = 16384


def _sheet_projection(width_px, height_px, sheet_w_mm, sheet_h_mm):
    """Map sheet millimetres to normalised device coordinates.

    The same shape as `get_normalize_uvs_matrix` in Blender's UV layout
    exporter, generalised from the unit square to the sheet's real size.

    Deliberately flips nothing. `_Sheet.to_sheet` produces a top-left origin
    with y growing downward, as SVG does, and this maps it straight onto GL's
    bottom-left origin -- so the framebuffer holds the sheet upside down, and
    `_rows_top_down` turns it the right way up once, for every pass at once.

    The first attempt compensated for the row order here instead, in the
    projection. That corrected the geometry and left the text behind: `blf`
    draws glyphs upright in whatever space it is given, so the digits came out
    mirrored while the shapes looked fine. Two compensations that have to
    agree is one more than necessary.
    """
    matrix = Matrix.Identity(4)
    matrix[0][0] = 2.0 / sheet_w_mm
    matrix[1][1] = 2.0 / sheet_h_mm
    matrix.col[3][0] = -1.0
    matrix.col[3][1] = -1.0
    return matrix


def _rows_top_down(pixels, width, height):
    """Reverse the row order of an RGBA byte buffer.

    `fb.read_color` hands back row 0 as the framebuffer's *bottom* row, and
    both writers below want row 0 to be the image's *top*. Doing it here, on
    the finished buffer, means geometry and text cannot disagree about which
    way up they are.
    """
    stride = width * 4
    flat = bytearray(pixels)
    out = bytearray(len(flat))
    for row in range(height):
        src = (height - 1 - row) * stride
        out[row * stride:(row + 1) * stride] = flat[src:src + stride]
    # bytes, not a list: OpenImageIO's write_image takes a buffer and rejects
    # a Python sequence outright.
    return bytes(out)


def _fill_triangles(rings):
    """Triangles covering `rings` = [outer, hole, ...], in sheet millimetres.

    `tessellate_polygon` takes the outer boundary and its holes together and
    returns triangles that leave the holes empty. Fanning from a centre point
    instead would fill an annulus solid, which is most of what this add-on
    produces.
    """
    if not rings or len(rings[0]) < 3:
        return []
    polys = [[Vector((x, y, 0.0)) for x, y in ring] for ring in rings]
    flat = [p for ring in polys for p in ring]
    try:
        tris = tessellate_polygon(polys)
    except Exception:
        return []
    out = []
    for tri in tris:
        try:
            out.append(tuple((flat[i].x, flat[i].y) for i in tri))
        except IndexError:
            continue
    return out


def _draw_tris(tris, color):
    if not tris:
        return
    coords, colors = [], []
    for tri in tris:
        for pt in tri:
            coords.append(pt)
            colors.append(color)
    shader = gpu.shader.from_builtin('FLAT_COLOR')
    batch = batch_for_shader(shader, 'TRIS', {"pos": coords, "color": colors})
    batch.draw(shader)


def _draw_lines(segments, color, width_px):
    """`segments` is a flat list of (start, end) pairs in sheet millimetres."""
    if not segments:
        return
    coords = []
    for a, b in segments:
        coords.append((a[0], a[1]))
        coords.append((b[0], b[1]))
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    # Both uniforms are required; the shader draws nothing without them.
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    shader.uniform_float("lineWidth", max(1.0, width_px))
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    batch.draw(shader)


def _ring_segments(ring):
    return [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]


def _draw_text(items, width_px, height_px, sheet_w_mm, sheet_h_mm):
    """Labels, in pixel space, after the geometry.

    blf works in pixels against the bound framebuffer, so the projection is
    swapped for a pixel-space one rather than trying to express a font size
    in millimetres.
    """
    if not items:
        return
    font = 0
    sx = width_px / sheet_w_mm
    sy = height_px / sheet_h_mm
    with gpu.matrix.push_pop():
        gpu.matrix.load_matrix(Matrix.Identity(4))
        ortho = Matrix.Identity(4)
        ortho[0][0] = 2.0 / width_px
        # Negative, so a glyph drawn upright in blf's own space comes out
        # upside down in the framebuffer -- matching the geometry, which is
        # already upside down there. `_rows_top_down` rights the pair.
        ortho[1][1] = -2.0 / height_px
        ortho.col[3][0] = -1.0
        ortho.col[3][1] = 1.0
        gpu.matrix.load_projection_matrix(ortho)
        for x_mm, y_mm, size_mm, color, text in items:
            px = max(6.0, size_mm * sy)
            blf.size(font, px)
            blf.color(font, *color)
            tw, th = blf.dimensions(font, text)
            # Centred on its point, which is what the SVG's
            # text-anchor:middle and dominant-baseline:middle do.
            #
            # Drawn into the same upside-down sheet the geometry uses, so the
            # single flip on the way out rights both together. The glyphs are
            # therefore upside down in the framebuffer, which is correct and
            # looks alarming if you read the buffer directly.
            x = x_mm * sx - tw * 0.5
            y = y_mm * sy - th * 0.5
            blf.position(font, x, y, 0.0)
            blf.draw(font, text)


def render(bm, uv_layer, settings, filepath, size=2048, scale_length=1.0,
           file_format='PNG'):
    """Draw the pattern to `filepath`. Returns a dict of what was drawn.

    `size` is the longest edge in pixels, as Export UV Layout's own `size` is,
    with the other edge following the sheet's aspect so nothing is stretched.
    """
    islands_faces = core.uv_islands(bm, uv_layer, only_selected=False)
    if not islands_faces:
        raise ValueError("no UV islands to draw")

    sheet = fx._Sheet(islands_faces, uv_layer, scale_length)
    islands = [fx._Island(i, faces, uv_layer, sheet)
               for i, faces in enumerate(islands_faces)]

    id_layer = bm.faces.layers.int.get('foam_island_id')
    if id_layer is not None:
        for isl in islands:
            rep = next(iter(isl.faces), None)
            if rep is not None:
                isl.piece_id = rep[id_layer]

    sheet_w, sheet_h = sheet.width, sheet.height
    if settings.show_scale_ruler:
        sheet_h += fx.RULER_STRIP_MM
        sheet_w = max(sheet_w, fx.MARGIN_MM * 2 + 210.0)

    longest = max(sheet_w, sheet_h)
    size = max(64, min(int(size), MAX_PIXELS))
    width_px = max(1, int(round(size * sheet_w / longest)))
    height_px = max(1, int(round(size * sheet_h / longest)))

    # One millimetre in pixels, so line widths and fonts follow the sheet.
    px_per_mm = height_px / sheet_h

    fills, outline_segs, text_items = [], [], []

    for isl in islands:
        rings = core.order_rings_outer_first(
            [[(p.x, p.y) for p in
              (isl.to_sheet(l[uv_layer].uv) for l in cyc)]
             for cyc in isl.cycles])
        if not rings:
            continue
        if settings.island_fill:
            r, g, b = core.island_palette_color(isl.index, len(islands))
            fills.append((_fill_triangles(rings), (r, g, b, 1.0)))
        for ring in rings:
            outline_segs.extend(_ring_segments(ring))

        if settings.show_island_ids:
            (px, py), room = core.pole_of_inaccessibility(rings)
            if px is not None:
                number_mm = sheet.permille(settings.number_size,
                                           settings.overlay_scale)
                size_mm = min(number_mm, max(max(room, 0.0) * 1.2,
                                             number_mm * 0.25))
                text_items.append((px, py, size_mm, (0, 0, 0, 1),
                                   str(isl.piece_id)))

    offscreen = gpu.types.GPUOffScreen(width_px, height_px)
    offscreen.bind()
    try:
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(1.0, 1.0, 1.0, 1.0))
        gpu.state.blend_set('ALPHA')
        with gpu.matrix.push_pop():
            gpu.matrix.load_matrix(
                _sheet_projection(width_px, height_px, sheet_w, sheet_h))
            gpu.matrix.load_projection_matrix(Matrix.Identity(4))
            for tris, color in fills:
                _draw_tris(tris, color)
            _draw_lines(outline_segs, (0.0, 0.0, 0.0, 1.0),
                        max(1.0, 0.35 * px_per_mm))
        _draw_text(text_items, width_px, height_px, sheet_w, sheet_h)
        gpu.state.blend_set('NONE')

        pixels = fb.read_color(0, 0, width_px, height_px, 4, 0, 'UBYTE')
        pixels.dimensions = width_px * height_px * 4
        _save(filepath, _rows_top_down(pixels, width_px, height_px),
              width_px, height_px, file_format)
    finally:
        offscreen.unbind()
        offscreen.free()

    info = {
        "path": filepath,
        "pixels": [width_px, height_px],
        "sheet_mm": [round(sheet_w, 1), round(sheet_h, 1)],
        "dpi": round(width_px / (sheet_w / 25.4), 1),
        "islands": len(islands),
        "labels": len(text_items),
    }
    _log.info(foam_log.summary("raster_export", **{
        k: v for k, v in info.items() if k != "path"}))
    return info


def _save(filepath, pixels, width, height, file_format):
    """Write the pixels out, preferring OIIO exactly as the built-in does."""
    if oiio and file_format == 'PNG':
        spec = oiio.ImageSpec(width, height, 4, "uint8")
        image = oiio.ImageOutput.create(filepath)
        if image is None:
            raise OSError("no image writer for %r" % filepath)
        if not image.open(filepath, spec):
            raise OSError(image.geterror() or "could not open %r" % filepath)
        if not image.write_image(pixels):
            error = image.geterror()
            image.close()
            raise OSError(error or "could not write %r" % filepath)
        image.close()
        return

    # bpy images fill from the bottom, so the top-down rows go back the other
    # way for this writer only.
    stride = width * 4
    bottom_up = []
    for row in range(height - 1, -1, -1):
        bottom_up.extend(pixels[row * stride:(row + 1) * stride])

    image = bpy.data.images.new("foamcraft_raster", width, height, alpha=True)
    try:
        image.filepath_raw = filepath
        image.file_format = file_format
        image.pixels = [v / 255.0 for v in bottom_up]
        image.save()
    finally:
        bpy.data.images.remove(image)
