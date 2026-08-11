"""Shared scene settings for the foam workflow.

Every foam module reads from this one PropertyGroup, reached as
``context.scene.foam_craft``. Keeping it in its own module means the overlay,
the exporter, the numbering operator and the previews can be developed against
a stable contract without importing each other.

Nothing here computes anything -- the geometry lives in ``foam_angle_core``.
"""

import bpy

from . import foam_angle_core as core
from . import foam_log


def _apply_log_level(settings):
    """Push the chosen level into the logger.

    Lives here rather than inline in the property so the lambda stays short and
    the import of foam_log is at module level, where a failure shows up at
    registration instead of the first time the dropdown is touched.
    """
    foam_log.set_level(settings.log_level)



# Physical stock this project actually buys, in metres. Foam grade is a
# *density* code and says nothing about thickness -- the same L400 is sold in
# many thicknesses -- so the two are independent settings and grade never sets
# thickness.
THICKNESS_MIN = 0.003175   # 1/8"
THICKNESS_MAX = 0.0508     # 2"
THICKNESS_DEFAULT = 0.0127  # 1/2"


class FoamPanelSettings(bpy.types.PropertyGroup):
    # --- Material
    foam_grade: bpy.props.EnumProperty(
        name="Foam Grade",
        description="Closed-cell foam density grade. Density affects how far "
                    "the foam bends before it needs a relief cut; it does not "
                    "set the thickness",
        items=[
            ('L200', "L200", "Softest closed-cell PE; bends furthest unaided"),
            ('L400', "L400", "Mid-density closed-cell PE"),
            ('L600', "L600", "Firm closed-cell PE; needs relief cuts soonest"),
            ('EVA', "EVA", "EVA craft foam; different material family"),
            ('CUSTOM', "Custom", "Set the tolerances by hand"),
        ],
        default='L400')
    foam_thickness: bpy.props.FloatProperty(
        name="Thickness",
        description="Thickness of the foam sheet being cut. Independent of "
                    "grade -- read it off the stock you actually have",
        default=THICKNESS_DEFAULT, min=THICKNESS_MIN, max=THICKNESS_MAX,
        subtype='DISTANCE', unit='LENGTH', precision=4)

    # --- Saw
    max_tilt_deg: bpy.props.FloatProperty(
        name="Max Blade Tilt",
        description="Furthest the saw blade tilts off vertical. Seams needing "
                    "more than this are flagged for heat forming rather than "
                    "silently clamped",
        default=45.0, min=0.0, max=89.0)
    kerf: bpy.props.FloatProperty(
        name="Kerf",
        description="Blade width, allowed for when placing cut lines",
        default=0.0032, min=0.0, max=0.02,
        subtype='DISTANCE', unit='LENGTH', precision=4)

    # --- Dart run segmentation
    max_angle_variation_deg: bpy.props.FloatProperty(
        name="Max Angle Variation",
        description="How much the bevel angle may drift along one run. One run "
                    "is one pass at one blade tilt, so this is what decides "
                    "where a run ends and the next begins",
        default=10.0, min=0.5, max=90.0)
    max_chord_deviation: bpy.props.FloatProperty(
        name="Straight Cut Limit",
        description="Only for straight-cut tools. How far a run may bow away "
                    "from a straight line before it is split. Zero means no "
                    "limit, correct for a band saw, which follows a curve at a "
                    "fixed blade tilt. Raise above zero only for a table saw, "
                    "where every pass has to be straight",
        default=0.0, min=0.0, max=0.1,
        subtype='DISTANCE', unit='LENGTH', precision=4)
    dissolve_angle_limit_deg: bpy.props.FloatProperty(
        name="Pre-simplify Angle",
        description="Optional Decimate (Planar) pre-pass, delimited by UV and "
                    "seam so island boundaries survive. Helps on dense sculpts; "
                    "leave at zero to segment the boundary directly",
        default=0.0, min=0.0, max=45.0)

    # --- Darts
    notch_shape: bpy.props.EnumProperty(
        name="Dart Shape",
        items=[
            ('V_WEDGE', "V Wedge", "Wedge cut into the edge that closes when glued"),
            ('SLIT', "Slit", "Straight relief slit"),
        ],
        default='V_WEDGE')
    notch_depth_ratio: bpy.props.FloatProperty(
        name="Dart Depth",
        description="Fraction of the foam thickness the dart cuts through",
        default=0.5, min=0.1, max=0.95)

    # --- Stock, for craftability checks
    sheet_width: bpy.props.FloatProperty(
        name="Sheet Width", default=0.6, min=0.0,
        subtype='DISTANCE', unit='LENGTH')
    sheet_height: bpy.props.FloatProperty(
        name="Sheet Height", default=1.2, min=0.0,
        subtype='DISTANCE', unit='LENGTH')
    stretch_tolerance: bpy.props.FloatProperty(
        name="Stretch Tolerance",
        description="How much a piece may distort when flattened before it is "
                    "called non-developable",
        default=0.25, min=0.01, max=2.0)

    # --- Numbering
    origin_object: bpy.props.PointerProperty(
        name="Numbering Origin", type=bpy.types.Object,
        description="Piece 1 is the island nearest this object, and numbering "
                    "walks outwards across the surface. Falls back to the 3D "
                    "cursor when empty")

    # --- Display
    show_overlay: bpy.props.BoolProperty(name="Show Overlay", default=True)
    show_brackets: bpy.props.BoolProperty(name="Cut Runs", default=True)
    show_island_ids: bpy.props.BoolProperty(name="Piece Numbers", default=True)
    show_heat_form: bpy.props.BoolProperty(name="Heat-form Warnings", default=True)
    show_dart_colors: bpy.props.BoolProperty(name="Convex/Concave Colour", default=True)
    only_selected: bpy.props.BoolProperty(
        name="Only Selected", default=False,
        description="Restrict the overlay to selected faces to declutter a busy mesh")

    overlay_scale: bpy.props.FloatProperty(
        name="Overlay Scale",
        description="Scales ticks and text together, so the same data can be a "
                    "readable full-size transfer pattern or a compact map",
        default=1.0, min=0.05, max=10.0)
    # Sizes relative to the pattern are given in parts per thousand of UV
    # space, as whole numbers. Expressed as a decimal fraction they landed
    # around 0.016, where a click of the arrow keys is far too coarse a step
    # to be usable.
    angle_text_size: bpy.props.IntProperty(
        name="Angle Text",
        description="Height of the angle numbers, in thousandths of the UV "
                    "sheet. They then hold their size against the pattern as "
                    "you zoom, the way they would if printed on it",
        default=16, min=2, max=200)
    mark_size: bpy.props.IntProperty(
        name="Mark Size",
        description="Length of the guide ticks and dart marks, in thousandths "
                    "of the UV sheet",
        default=10, min=1, max=200)
    number_size: bpy.props.IntProperty(
        name="Number Size",
        description="Height of the piece numbers, in thousandths of the UV "
                    "sheet",
        default=35, min=2, max=400)

    # Fixed screen sizes, used only when Scale With Pattern is off.
    tick_length_px: bpy.props.FloatProperty(name="Tick Length (px)",
                                            default=10.0, min=1.0, max=100.0)
    font_size: bpy.props.IntProperty(name="Angle Text (px)", default=13,
                                     min=6, max=60)
    min_label_run_px: bpy.props.IntProperty(
        name="Hide Labels Under",
        description="Runs drawn shorter than this many pixels get no angle "
                    "number. Where several short runs meet, as at a notch, the "
                    "numbers pile into an unreadable clump; the guide and its "
                    "ticks still show, so nothing is lost but the crowding. "
                    "Zoom in and they come back",
        default=34, min=0, max=400)
    log_level: bpy.props.EnumProperty(
        name="Log Detail",
        description="How much FoamCraft writes to its log file. Debug adds a "
                    "line for every overlay rebuild, which is one per timer "
                    "tick while the mesh is changing -- useful when checking a "
                    "number, far too noisy to leave on",
        items=[
            ('INFO', "Normal", "Operators and their results"),
            ('DEBUG', "Debug", "Every overlay rebuild, with counts"),
            ('WARNING', "Problems Only", "Warnings and errors"),
        ],
        default='INFO',
        update=lambda self, ctx: _apply_log_level(self))
    min_run_length: bpy.props.FloatProperty(
        name="Shortest Cut",
        description="Cut runs shorter than this along the real surface are "
                    "flagged with a warning mark. This is a property of the "
                    "piece, not of the view -- it is measured in millimetres "
                    "on the sculpt, so zooming the UV editor or scaling the "
                    "UVs never changes which edges are flagged. Set it to the "
                    "shortest pass you can actually make on the saw",
        default=0.006, min=0.0, max=0.2,
        subtype='DISTANCE', unit='LENGTH')
    corner_angle_deg: bpy.props.FloatProperty(
        name="Corner Angle",
        description="Turn in the piece outline beyond which a vertex counts as "
                    "a corner rather than part of a curve. Corners break a cut "
                    "run, so each straight-ish edge gets its own angle guide "
                    "instead of one guide running through the corner and "
                    "describing neither edge. Lower values break more often",
        default=core.CORNER_ANGLE_DEG, min=5.0, max=170.0)
    scale_text_with_view: bpy.props.BoolProperty(
        name="Scale With Pattern", default=True,
        description="Size every mark and number relative to the pattern, the "
                    "way they would be if printed on it. Off falls back to "
                    "fixed screen sizes")
    island_id_font_size: bpy.props.IntProperty(name="Number Size (px)",
                                               default=20, min=6, max=120)
    decimals: bpy.props.IntProperty(name="Decimals", default=0, min=0, max=2)
    min_angle: bpy.props.FloatProperty(
        name="Hide Below", default=1.0, min=0.0, max=180.0,
        description="Runs bending less than this are not labelled")

    convex_color: bpy.props.FloatVectorProperty(
        name="Convex", default=(0.15, 0.45, 1.0, 1.0),
        min=0, max=1, subtype='COLOR', size=4)
    concave_color: bpy.props.FloatVectorProperty(
        name="Concave", default=(1.0, 0.2, 0.55, 1.0),
        min=0, max=1, subtype='COLOR', size=4)
    suppressed_color: bpy.props.FloatVectorProperty(
        name="Hidden Label", default=(1.0, 0.85, 0.1, 1.0),
        min=0, max=1, subtype='COLOR', size=4)
    heat_form_color: bpy.props.FloatVectorProperty(
        name="Heat Form", default=(1.0, 0.35, 0.0, 1.0),
        min=0, max=1, subtype='COLOR', size=4)
    island_id_color: bpy.props.FloatVectorProperty(
        name="Number", default=(1.0, 1.0, 1.0, 1.0),
        min=0, max=1, subtype='COLOR', size=4)


class FOAM_OT_estimate_tolerances(bpy.types.Operator):
    """Suggest deviation and angle tolerances from the foam grade and thickness"""

    bl_idname = "foam.estimate_tolerances"
    bl_label = "Estimate From Foam"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.foam_craft
        deviation, angle = core.estimate_flex_tolerance(s.foam_grade,
                                                        s.foam_thickness)
        s.max_chord_deviation = deviation
        s.max_angle_variation_deg = angle
        self.report({'INFO'},
                    f"Suggested {deviation * 1000:.1f} mm deviation, "
                    f"{angle:.1f}° angle variation. These are a starting point "
                    f"-- bend a sample strip and adjust.")
        return {'FINISHED'}


_classes = (FoamPanelSettings, FOAM_OT_estimate_tolerances)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.foam_craft = bpy.props.PointerProperty(type=FoamPanelSettings)


def unregister():
    del bpy.types.Scene.foam_craft
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
