"""
Export Paper Model - Fold Angles
----------------------------------
A companion add-on to Blender's bundled "Export Paper Model" add-on
(io_export_paper_model.py, by Addam Dominec). It does NOT reimplement
unfolding, cutting, or page packing -- it calls that add-on's own
Unfolder/Mesh classes to do all of that, then adds the one thing the
stock exporter doesn't do: print the fold/bevel angle, in degrees, at
every edge -- including the cut edges between separate pieces, which
the stock SVG/PDF export deliberately leaves unlabeled (it only colors
angles on folds *within* a single piece, since plain paper doesn't need
a bevel number at a cut).

This writes a second SVG using the exact same page-size/margin math the
real exporter uses, so you can overlay it on the real "Export Paper
Model" output and read both together.

REQUIREMENTS:
  - The "Export Paper Model" add-on must be installed and ENABLED
    (Preferences > Add-ons > search "Paper Model"). This script finds
    it automatically at runtime regardless of what internal module name
    Blender gave it (this varies between older Blender versions and the
    4.2+ Extensions Platform).

INSTALL:
  Edit > Preferences > Add-ons > Install... > select this file > enable it.
  A new item appears at File > Export > "Paper Model Fold Angles (.svg)".

USE:
  1. Mark seams around each scale/panel boundary as you normally would
     for Paper Model (or run its own "Unfold" operator first).
  2. File > Export > Paper Model Fold Angles (.svg). Match the Page
     Width/Height/Margin/Scale to whatever you use in the real
     "Export Paper Model" export so the two overlay correctly.
  3. Open both SVGs in Illustrator/Inkscape, stack them at (0,0), and
     you have your normal net plus a number at every edge.

CAVEATS:
  - Written directly against the add-on's real class/method signatures
    (confirmed by reading its current source), but not run against a
    live Blender session -- treat this as a first draft to debug
    against your specific Blender version, not a guaranteed drop-in.
  - Blue = convex/mountain, pink = concave/valley, matching the
    convention from your paper templates.
"""

bl_info = {
    "name": "Export Paper Model - Fold Angles",
    "author": "(companion add-on)",
    "version": (0, 1),
    "blender": (3, 0, 0),
    "location": "File > Export > Paper Model Fold Angles (.svg)",
    "description": "Labels the fold/bevel angle at every edge, including cuts, "
                    "using Export Paper Model's own unfolding and page layout.",
    "category": "Import-Export",
}

import sys
from math import degrees, pi

import bpy
import mathutils as M


def _find_paper_model_module():
    """Locate the already-loaded Export Paper Model add-on module,
    whatever internal name Blender registered it under."""
    for mod in list(sys.modules.values()):
        if mod is not None and hasattr(mod, "Unfolder") and hasattr(mod, "Mesh") and hasattr(mod, "UnfoldError"):
            return mod
    raise ImportError(
        "Could not find the 'Export Paper Model' add-on. "
        "Enable it first: Preferences > Add-ons > search 'Paper Model'."
    )


class ExportPaperModelAngles(bpy.types.Operator):
    """Export a companion SVG labeling the fold/bevel angle at every edge"""

    bl_idname = "export_mesh.paper_model_angles"
    bl_label = "Export Paper Model Fold Angles (SVG)"
    bl_options = {'PRESET'}

    filepath: bpy.props.StringProperty(name="File Path", subtype='FILE_PATH', options={'SKIP_SAVE'})
    output_size_x: bpy.props.FloatProperty(
        name="Page Width", default=0.210, soft_min=0.105, soft_max=0.841, unit="LENGTH")
    output_size_y: bpy.props.FloatProperty(
        name="Page Height", default=0.297, soft_min=0.148, soft_max=1.189, unit="LENGTH")
    output_margin: bpy.props.FloatProperty(
        name="Page Margin", default=0.005, min=0, soft_max=0.1, unit="LENGTH")
    scale: bpy.props.FloatProperty(
        name="Scale", description="Divisor of all dimensions when exporting -- match your real Paper Model export",
        default=1, soft_min=1.0, soft_max=100.0, precision=1)
    limit_by_page: bpy.props.BoolProperty(
        name="Limit Island Size",
        description="Match this to the 'Limit Island Size' setting in the real Paper Model panel",
        default=False)
    angle_epsilon: bpy.props.FloatProperty(
        name="Hidden Edge Angle", description="Folds bending less than this are treated as flat and skipped",
        default=pi / 360, min=0, soft_max=pi / 4, subtype="ANGLE", unit="ROTATION")
    font_size_mm: bpy.props.FloatProperty(name="Label Size (mm)", default=3.0, min=0.5)

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        try:
            ppm = _find_paper_model_module()
        except ImportError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        ob = context.active_object
        recall_mode = ob.mode
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            unfolder = ppm.Unfolder(ob)
            cage_size = M.Vector((self.output_size_x, self.output_size_y))
            margin = M.Vector((self.output_margin, self.output_margin))
            printable = cage_size - 2 * margin
            unfolder_scale = context.scene.unit_settings.scale_length / self.scale

            # Same pipeline Export Paper Model itself runs (Unfolder.prepare
            # + Mesh.scale_islands + Mesh.finalize_islands + Mesh.fit_islands),
            # so island positions/sizes match the real export exactly.
            unfolder.prepare(cage_size, scale=unfolder_scale, limit_by_page=self.limit_by_page)
            mesh = unfolder.mesh
            mesh.scale_islands(unfolder_scale)
            mesh.finalize_islands(printable)
            mesh.fit_islands(printable)

            self._write_svg(mesh, cage_size, margin)
            self.report({'INFO'}, f"Wrote angle labels for {len(mesh.pages)} page(s)")
        except ppm.UnfoldError as error:
            self.report({'ERROR_INVALID_INPUT'}, error.args[0])
            return {'CANCELLED'}
        finally:
            bpy.ops.object.mode_set(mode=recall_mode)
        return {'FINISHED'}

    def _write_svg(self, mesh, page_size, margin):
        def fmt(v):
            # Mirrors Svg.format_vertex()'s exact coordinate transform in the
            # real add-on, so this file overlays correctly on its own export.
            return (
                (v.x + margin.x) * 1000,
                (page_size.y - v.y - margin.y) * 1000,
            )

        base = self.filepath
        if base.lower().endswith(".svg"):
            base = base[:-4]

        angle_epsilon_deg = degrees(self.angle_epsilon)

        for page in mesh.pages:
            path = f"{base}_{page.name}.svg" if len(mesh.pages) > 1 else f"{base}.svg"
            lines = [
                "<?xml version='1.0' encoding='UTF-8' standalone='no'?>",
                f"<svg xmlns='http://www.w3.org/2000/svg' "
                f"width='{page_size.x*1000:.2f}mm' height='{page_size.y*1000:.2f}mm' "
                f"viewBox='0 0 {page_size.x*1000:.2f} {page_size.y*1000:.2f}'>",
                "<style>text{font-family:sans-serif;text-anchor:middle;}"
                ".fold{fill:#0057ff;} .cut{fill:#ff2f92;}</style>",
            ]
            for island in page.islands:
                visited = set()
                for loop, uvedge in island.edges.items():
                    edge = mesh.edges[loop.edge]
                    if edge.angle is None:
                        continue
                    # same dedup pattern the real Svg.write() uses, so each
                    # physical edge gets exactly one label per island
                    vertex_pair = frozenset((uvedge.va, uvedge.vb))
                    if vertex_pair in visited:
                        continue
                    visited.add(vertex_pair)

                    angle_deg = degrees(edge.angle)
                    if abs(angle_deg) < angle_epsilon_deg:
                        continue

                    is_cut = edge.is_cut(uvedge.uvface.face)
                    mid = (uvedge.va.co + uvedge.vb.co) / 2 + island.pos
                    x, y = fmt(mid)
                    css_class = "cut" if is_cut else "fold"
                    lines.append(
                        f"<text class='{css_class}' x='{x:.2f}' y='{y:.2f}' "
                        f"font-size='{self.font_size_mm}'>{abs(angle_deg):.0f}\u00b0</text>"
                    )
            lines.append("</svg>")
            with open(path, "w") as f:
                f.write("\n".join(lines))


def menu_func_export(self, context):
    self.layout.operator(ExportPaperModelAngles.bl_idname, text="Paper Model Fold Angles (.svg)")


def register():
    bpy.utils.register_class(ExportPaperModelAngles)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.utils.unregister_class(ExportPaperModelAngles)


if __name__ == "__main__":
    register()
