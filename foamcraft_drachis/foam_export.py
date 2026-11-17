"""SVG cut-pattern export for the foam workflow.

Each UV island becomes one flat piece of foam, so this exporter walks every
island's UV boundary and lays out a print-at-1:1 pattern: an outline, a piece
number, and -- along the boundary -- the hand-drafted convention of tick marks
dividing the edge into runs, each labelled with its bevel angle, plus a relief
dart at every run-to-run corner.

Deliberately does NOT reuse Export Paper Model's Unfolder/page-packing (see
``paper_model_export.py`` for that side of the project) -- Paper Model solves
page packing for a printer, this solves real-world-scale pattern making for a
tracing workflow, and reads UV positions directly off the active UV map.

The heavy lifting (``generate_svg``) is a plain function of ``(bm, uv_layer,
settings, scale_length)`` rather than a method on the operator, so it can be
unit-tested against a stand-in settings object without registering anything or
touching ``context.scene.foamcraft_drachis``.
"""

import bpy
import bmesh

from . import foam_angle_core as core  # noqa: F401  (kept for importers)
from .foam_marks import (build_marks, build_marks_multi, scale_warnings,
                         SheetItem, object_prefix)

# What to draw is decided in foam_marks.build_marks; this module only writes
# it out. The helpers that used to live here moved with that decision, and
# are re-exported under their old names for the tests, tools and modules
# that import them from foam_export.
from .foam_marks import (  # noqa: F401  (re-exports)
    DART_SLIT_WIDTH_MM, MARGIN_MM, RULER_STRIP_MM, RULER_LENGTH_MM,
    RULER_TICK_MM, RULER_FONT_MM, INCH_MM, TICK_MERGE_M,
    _face_uv_area, _polygon_centroid, _Island, _Sheet, _run_polyline,
    _boundary_tangent, _inward_normal, _boundary_angle_deg,
    _edge_canonical_start, _run_tick_params, _remember_tick, _seam_chains,
    _seam_tick_params, chain_direction_start, _shared_vert, piece_name,
)


def _fmt(v):
    return f"{v:.3f}"


def _rgb_to_hex(r, g, b):
    return "#%02x%02x%02x" % (int(max(0.0, min(1.0, r)) * 255),
                              int(max(0.0, min(1.0, g)) * 255),
                              int(max(0.0, min(1.0, b)) * 255))


def _points_attr(points):
    return " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in points)


# Every element below carries its own presentation attributes and nothing
# depends on a <style> block. Illustrator's SVG import applies CSS id
# selectors and `dominant-baseline` poorly (strokes lost their width and
# colour, marks went missing), whereas presentation attributes are read by
# every importer, browser and Inkscape alike.

FONT_FAMILY = "Arial, Helvetica, sans-serif"

# `dominant-baseline="middle"` is not used. To centre a label vertically
# without it, the baseline is moved down from the centre by this fraction of
# the font size. A digit or capital is about 0.7 of the font size tall
# (Label's own comment says the same), so its middle sits 0.35 above the
# baseline; the baseline is therefore 0.35 * size below the centre.
MIDDLE_BASELINE_SHIFT = 0.35


def _stroke_attrs(stroke):
    """stroke colour, width and dash as attributes; fill is the caller's."""
    attrs = (f'stroke="{_rgb_to_hex(*stroke.color)}" '
             f'stroke-width="{_fmt(stroke.width_mm)}"')
    if stroke.dash:
        attrs += (f' stroke-dasharray="{_fmt(stroke.dash[0])},'
                  f'{_fmt(stroke.dash[1])}"')
    return attrs


def _line(stroke):
    (x1, y1), (x2, y2) = stroke.points[0], stroke.points[-1]
    return (f'<line x1="{_fmt(x1)}" y1="{_fmt(y1)}" '
            f'x2="{_fmt(x2)}" y2="{_fmt(y2)}" '
            f'fill="none" {_stroke_attrs(stroke)}/>')


def _polygon(stroke):
    return (f'<polygon points="{_points_attr(stroke.points)}" '
            f'fill="none" {_stroke_attrs(stroke)}/>')


def _text(label):
    y = label.y
    if label.valign == 'middle':
        y += MIDDLE_BASELINE_SHIFT * label.size_mm
    # A turned label (a bevel label, square to its cut line) is rotated
    # about its own (x, y). `rotation_deg` is counter-clockwise on the page;
    # SVG's rotate() is clockwise on screen because y grows downward, so it
    # is written negated. The middle-baseline shift above is applied before
    # the turn and so turns with the text. "→"/"←" are U+2192/U+2190, which
    # Arial (the first family named) has.
    turn = getattr(label, 'rotation_deg', 0.0) or 0.0
    transform = (f' transform="rotate({_fmt(-turn)} {_fmt(label.x)} '
                 f'{_fmt(label.y)})"' if turn else '')
    return (f'<text x="{_fmt(label.x)}" y="{_fmt(y)}" '
            f'font-size="{_fmt(label.size_mm)}" '
            f'font-family="{FONT_FAMILY}" '
            f'fill="{_rgb_to_hex(*label.color)}" '
            f'text-anchor="middle"{transform}>{label.text}</text>')


def _rect(stroke):
    xs = [p[0] for p in stroke.points]
    ys = [p[1] for p in stroke.points]
    x, y = min(xs), min(ys)
    return (f'<rect x="{_fmt(x)}" y="{_fmt(y)}" '
            f'width="{_fmt(max(xs) - x)}" height="{_fmt(max(ys) - y)}" '
            f'fill="none" {_stroke_attrs(stroke)}/>')


def svg_from_marks(marks):
    """Serialise a `foam_marks.Marks` as the pattern SVG.

    Writes only. Every decision about whether a mark exists was made in
    `build_marks`; here a mark's `kind` picks its group and element, and
    nothing else is consulted. Styling is per element, as attributes, so no
    importer has to understand CSS.
    """
    fills = [m for m in marks.fills if m.kind == 'piece']
    relief_fills = [m for m in marks.fills if m.kind == 'relief']
    strokes_of = lambda k: [s for s in marks.strokes if s.kind == k]  # noqa: E731
    labels_of = lambda k: [l for l in marks.labels if l.kind == k]    # noqa: E731

    fill_els = []
    for f in fills:
        subpaths = []
        for ring in f.rings:
            head = "M %s" % " L ".join("%s,%s" % (_fmt(x), _fmt(y))
                                       for x, y in ring)
            subpaths.append(head + " Z")
        fill_els.append(
            f'<path d="{" ".join(subpaths)}" fill-rule="evenodd" '
            f'fill="{_rgb_to_hex(*f.color)}" stroke="none"/>')

    outline_els = [_polygon(s) for s in strokes_of('outline')]

    # A filled relief (a slit) is one polygon, filled and stroked; its
    # outline stroke is the same polygon and is not written twice, but
    # supplies the stroke. An unfilled relief (a wedge) is written fill="none".
    relief_strokes = strokes_of('relief')
    dart_els = []
    for f in relief_fills:
        match = next((s for s in relief_strokes if s.points == f.rings[0]),
                     None)
        stroke_attrs = (_stroke_attrs(match) if match else
                        f'stroke="#000000" '
                        f'stroke-width="{_fmt(marks.hairline_mm * 0.7)}"')
        dart_els.append(f'<polygon points="{_points_attr(f.rings[0])}" '
                        f'fill="{_rgb_to_hex(*f.color)}" {stroke_attrs}/>')
    filled = {f.rings[0] for f in relief_fills}
    dart_els += [_polygon(s) for s in relief_strokes
                 if s.points not in filled]

    tick_els = [_line(s) for s in strokes_of('bracket')]
    align_els = [_line(s) for s in strokes_of('align')]
    align_label_els = [_text(l) for l in labels_of('align')]
    angle_els = [_text(l) for l in labels_of('angle')]
    # One polyline per arrow: tail, tip, barb, tip, barb. Round joins, so
    # the doubled-back point at the tip does not throw a mitre spike.
    arrow_els = [
        f'<polyline points="{_points_attr(s.points)}" fill="none" '
        f'{_stroke_attrs(s)} stroke-linejoin="round" '
        f'stroke-linecap="round"/>'
        for s in strokes_of('bevel_arrow')]
    # A label placed outside its piece (foam_layout) is joined to its cut
    # line by a leader: it belongs with the arrow it leads to, so it goes in
    # the same group, after the arrows.
    arrow_els += [_line(s) for s in strokes_of('leader')]
    number_els = [_text(l) for l in labels_of('number')]

    # Each heat-form run is a polyline followed by its note.
    heat_els = []
    heat_strokes, heat_labels = strokes_of('heat'), labels_of('heat')
    for i in range(max(len(heat_strokes), len(heat_labels))):
        if i < len(heat_strokes):
            s = heat_strokes[i]
            heat_els.append(f'<polyline points="{_points_attr(s.points)}" '
                            f'fill="none" {_stroke_attrs(s)}/>')
        if i < len(heat_labels):
            heat_els.append(_text(heat_labels[i]))

    # The ruler's bar and ticks first, then its square and captions left to
    # right -- the order the sheet has always listed them in.
    ruler_strokes = strokes_of('ruler')
    ruler_els = [_line(s) for s in ruler_strokes if not s.closed]
    rest = [(min(p[0] for p in s.points), _rect(s))
            for s in ruler_strokes if s.closed]
    rest += [(l.x, _text(l)) for l in labels_of('ruler')]
    rest.sort(key=lambda item: item[0])
    ruler_els += [el for _, el in rest]

    def group(gid, elements):
        body = "\n".join(elements)
        return f'<g id="{gid}">\n{body}\n</g>'

    w, h = marks.width_mm, marks.height_mm
    lines = [
        "<?xml version='1.0' encoding='UTF-8' standalone='no'?>",
        # width/height are in millimetres and the viewBox is 1 user unit per
        # millimetre, so a point in the file is a point on the foam. This is
        # the scheme Inkscape and browsers honour (Illustrator reads the mm
        # width/height and rescales to its own points).
        f"<svg xmlns='http://www.w3.org/2000/svg' "
        f"width='{w:.3f}mm' height='{h:.3f}mm' "
        f"viewBox='0 0 {w:.3f} {h:.3f}'>",
        group("fills", fill_els),
        group("outlines", outline_els),
        group("darts", dart_els),
        group("ticks", tick_els),
        group("align", align_els),
        group("alignlabels", align_label_els),
        group("angles", angle_els),
        # Drawn arrows are retired (the glyph is in the label, 2026-11-17),
        # so this group exists only when there is something in it -- a
        # leader from foam_layout, or an arrow a caller built by hand. It
        # stays in its place after "angles"; the others keep their order.
        *([group("arrows", arrow_els)] if arrow_els else []),
        group("numbers", number_els),
        group("heatform", heat_els),
        group("ruler", ruler_els),
        "</svg>",
    ]
    return "\n".join(lines)


def generate_svg(bm, uv_layer, settings, scale_length=1.0, matrix=None):
    """Build the full SVG pattern as a string. Pure function -- no file I/O,
    no bpy.context -- so it can be exercised directly in tests.

    Decides nothing itself: `build_marks` says what is on the sheet, and
    `svg_from_marks` writes it down."""
    return svg_from_marks(build_marks(bm, uv_layer, settings,
                                      scale_length, matrix))


def export_objects(context):
    """The objects a pattern export takes, by Export UV Layout's own rule.

    `{*context.selected_objects, context.active_object}`, keeping meshes
    with an active UV map -- copied from
    scripts/addons_core/io_mesh_uv_layout/__init__.py:198-206
    (`iter_objects_to_export`). In edit mode that is every object being
    edited together, since they are selected. Returned in name order, which
    the built-in does not need (it only draws) and this does (it numbers).
    """
    candidates = {*context.selected_objects, context.active_object}
    out = [obj for obj in candidates
           if obj is not None and obj.type == 'MESH'
           and obj.data.uv_layers.active is not None]
    return sorted(out, key=lambda obj: obj.name)


class SheetSource:
    """`SheetItem`s for `objects`, and the BMeshes behind them, freed on exit.

    Export UV Layout leaves edit mode so the mesh data is current
    (`__init__.py:141-143`, and back at 184-185). The same data is read here
    without the round trip: an object whose mesh is in edit mode is read
    through `bmesh.from_edit_mesh` -- the live edit mesh, which is what the
    mode switch would have flushed -- and any other through a fresh BMesh.
    """

    def __init__(self, objects):
        self.objects = objects
        self.items = []
        self._owned = []

    def __enter__(self):
        for obj in self.objects:
            mesh = obj.data
            if mesh.is_editmode:
                bm = bmesh.from_edit_mesh(mesh)
            else:
                bm = bmesh.new()
                bm.from_mesh(mesh)
                self._owned.append(bm)
            uv_layer = bm.loops.layers.uv.active
            if uv_layer is None:
                continue
            self.items.append(SheetItem(
                bm, uv_layer, obj.matrix_world.copy(),
                object_prefix(obj.name,
                              getattr(obj, "foamcraft_drachis_prefix", "")),
                obj.name))
        return self.items

    def __exit__(self, *exc):
        for bm in self._owned:
            bm.free()
        self._owned = []
        self.items = []
        return False


def _rgba_to_hex(rgba):
    r, g, b = (int(max(0, min(1, c)) * 255) for c in rgba[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


class FOAMCRAFT_DRACHIS_OT_export_panel_pattern(bpy.types.Operator):
    """Export flat foam cut patterns (outlines, darts and bevel labels) as an SVG"""

    bl_idname = "foamcraft_drachis.export_panel_pattern"
    bl_label = "Export Foam Panel Pattern (SVG)"
    bl_options = {'PRESET'}

    filepath: bpy.props.StringProperty(name="File Path", subtype='FILE_PATH',
                                       options={'SKIP_SAVE'})
    filter_glob: bpy.props.StringProperty(default="*.svg", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        # Every object Export UV Layout would take, overlaid on one sheet --
        # not just the active one, which is what a maker editing two objects
        # together got before.
        objects = export_objects(context)
        if not objects:
            self.report({'ERROR'}, "No selected mesh has an active UV map. "
                                   "Unwrap it first.")
            return {'CANCELLED'}
        settings = context.scene.foamcraft_drachis
        with SheetSource(objects) as items:
            # Through each world matrix, as the overlay measures, so an
            # object with unapplied scale gets the same angles on paper as
            # on screen.
            marks = build_marks_multi(
                items, settings, context.scene.unit_settings.scale_length)
        for note in marks.warnings:
            self.report({'WARNING'}, note)
        svg_text = svg_from_marks(marks)

        path = self.filepath
        if not path.lower().endswith(".svg"):
            path += ".svg"
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_text)
        self.report({'INFO'}, f"Wrote foam panel pattern to {path}")
        return {'FINISHED'}


def menu_func_export(self, context):
    self.layout.operator(FOAMCRAFT_DRACHIS_OT_export_panel_pattern.bl_idname,
                         text="Foam Panel Pattern (.svg)")


def register():
    bpy.utils.register_class(FOAMCRAFT_DRACHIS_OT_export_panel_pattern)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.utils.unregister_class(FOAMCRAFT_DRACHIS_OT_export_panel_pattern)
