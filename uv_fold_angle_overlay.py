"""
UV Fold Angle Overlay
-----------------------
Draws the fold/bevel angle, in degrees, directly over every edge in the
UV Editor -- as a live overlay, not an export. Reads whatever UV layout
is currently active (your own manual unwrap, or the "Unfolded" map that
Export Paper Model's "Create UVMap" option writes), computes the real
dihedral angle from the 3D mesh for each edge, and draws a colored
marker + label at its UV position. Nothing is written to disk; this is
purely a visual aid so you can judge/tune your unwrap and panel layout
by eye before exporting.

Blue = convex/mountain, pink = concave/valley by default (both colors
are user-editable), matching the pink/blue convention on your paper
templates.

INSTALL:
  Edit > Preferences > Add-ons > Install... > select this file > enable it.

USE:
  1. Select your mesh, enter Edit Mode, open a UV Editor.
  2. Unwrap it (any method -- manual seams + Unwrap, or Export Paper
     Model's own Unfold with "Create UVMap" checked).
  3. Open the UV Editor's sidebar (press N) and find the "Fold Angles"
     tab. Adjust colors, font size, label offset, and the "Hide Below"
     angle threshold to taste. Angles update live as you re-unwrap.

NOTES:
  - Angle math (dihedral angle between the two 3D faces sharing an
    edge, signed for convex/concave) is computed directly from the
    mesh -- it does not depend on Export Paper Model being installed,
    though it's designed to complement it.
  - "Only Selected Faces" is useful for decluttering a large, busy mesh
    like yours -- select one horn cluster at a time to check just that
    region's angles.
  - Every draw call here (blf text, gpu 2D_UNIFORM_COLOR points) is
    checked against Blender's current Python API docs. The one thing
    I could not verify hands-on (no Blender available to test against)
    is how it behaves at extreme zoom levels in the UV editor -- if
    labels misplace when very zoomed in/out, that's the first thing
    to check.
"""

bl_info = {
    "name": "UV Fold Angle Overlay",
    "author": "(companion add-on)",
    "version": (0, 1),
    "blender": (4, 0, 0),
    "location": "UV Editor > Sidebar (N) > Fold Angles",
    "description": "Live overlay of fold/bevel angles on every UV edge, with color/font/placement controls.",
    "category": "UV",
}

from math import degrees

import bpy
import blf
import bmesh
import gpu
from gpu_extras.batch import batch_for_shader


def _dihedral_angle_deg(edge):
    """Signed fold angle between the two 3D faces sharing this edge.
    Positive = convex/mountain, negative = concave/valley. None if the
    edge doesn't have exactly two connected faces (mesh boundary)."""
    if len(edge.link_faces) != 2:
        return None
    f1, f2 = edge.link_faces
    ang = f1.normal.angle(f2.normal)
    edge_vec = (edge.verts[1].co - edge.verts[0].co).normalized()
    sign = 1 if f1.normal.cross(f2.normal).dot(edge_vec) >= 0 else -1
    return sign * degrees(ang)


def _iter_uv_edges(bm, uv_layer, only_selected):
    """Yield (edge, uv_a, uv_b) once for each distinct UV-space occurrence
    of every 3D edge (so a seam between two islands yields two entries,
    one per island, at their respective UV positions)."""
    seen = set()
    for face in bm.faces:
        if only_selected and not face.select:
            continue
        loops = face.loops
        n = len(loops)
        for i in range(n):
            loop = loops[i]
            next_loop = loops[(i + 1) % n]
            uv_a = loop[uv_layer].uv.copy()
            uv_b = next_loop[uv_layer].uv.copy()
            key = (loop.edge.index, round(uv_a.x, 6), round(uv_a.y, 6), round(uv_b.x, 6), round(uv_b.y, 6))
            key_rev = (loop.edge.index, round(uv_b.x, 6), round(uv_b.y, 6), round(uv_a.x, 6), round(uv_a.y, 6))
            if key in seen or key_rev in seen:
                continue
            seen.add(key)
            yield loop.edge, uv_a, uv_b


class FoldAngleOverlaySettings(bpy.types.PropertyGroup):
    show: bpy.props.BoolProperty(name="Show Fold Angles", default=True)
    convex_color: bpy.props.FloatVectorProperty(
        name="Convex (Mountain) Color", description="Color for edges bending toward you",
        default=(0.15, 0.45, 1.0, 1.0), min=0, max=1, subtype='COLOR', size=4)
    concave_color: bpy.props.FloatVectorProperty(
        name="Concave (Valley) Color", description="Color for edges bending away from you",
        default=(1.0, 0.2, 0.55, 1.0), min=0, max=1, subtype='COLOR', size=4)
    font_size: bpy.props.IntProperty(name="Font Size", default=14, min=6, max=60)
    decimals: bpy.props.IntProperty(name="Decimal Places", default=0, min=0, max=2)
    min_angle: bpy.props.FloatProperty(
        name="Hide Below (deg)", description="Edges bending less than this are not labeled",
        default=1.0, min=0.0, max=180.0)
    label_offset: bpy.props.FloatProperty(
        name="Label Offset (px)", description="Distance the text is nudged away from the edge, perpendicular to it",
        default=14.0, min=0.0, max=200.0)
    show_markers: bpy.props.BoolProperty(name="Show Edge Markers", default=True)
    marker_size: bpy.props.FloatProperty(name="Marker Size (px)", default=5.0, min=1.0, max=30.0)
    only_selected: bpy.props.BoolProperty(
        name="Only Selected Faces", description="Only label edges belonging to selected faces -- useful to declutter a busy mesh",
        default=False)


def draw_callback():
    context = bpy.context
    scene = context.scene
    if scene is None or not hasattr(scene, "fold_angle_overlay"):
        return
    settings = scene.fold_angle_overlay
    if not settings.show:
        return

    obj = context.edit_object
    if obj is None or obj.type != 'MESH':
        return
    region = context.region
    if region is None:
        return

    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        return

    v2d = region.view2d
    font_id = 0
    blf.size(font_id, settings.font_size)

    marker_points_by_color = {}

    for edge, uv_a, uv_b in _iter_uv_edges(bm, uv_layer, settings.only_selected):
        angle = _dihedral_angle_deg(edge)
        if angle is None or abs(angle) < settings.min_angle:
            continue

        ax, ay = v2d.view_to_region(uv_a.x, uv_a.y, clip=False)
        bx, by = v2d.view_to_region(uv_b.x, uv_b.y, clip=False)
        sx, sy = (ax + bx) / 2, (ay + by) / 2

        dx, dy = bx - ax, by - ay
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        perp_x, perp_y = -dy / length, dx / length

        color = tuple(settings.convex_color if angle >= 0 else settings.concave_color)

        if settings.show_markers:
            marker_points_by_color.setdefault(color, []).append((sx, sy))

        label_x = sx + perp_x * settings.label_offset
        label_y = sy + perp_y * settings.label_offset
        text = f"{abs(angle):.{settings.decimals}f}\u00b0"
        text_w, text_h = blf.dimensions(font_id, text)
        blf.position(font_id, label_x - text_w / 2, label_y - text_h / 2, 0)
        blf.color(font_id, *color)
        blf.draw(font_id, text)

    if marker_points_by_color:
        shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
        gpu.state.point_size_set(settings.marker_size)
        gpu.state.blend_set('ALPHA')
        for color, pts in marker_points_by_color.items():
            batch = batch_for_shader(shader, 'POINTS', {"pos": pts})
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)
        gpu.state.blend_set('NONE')


class IMAGE_PT_fold_angle_overlay(bpy.types.Panel):
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Fold Angles"
    bl_label = "Fold Angle Overlay"

    def draw(self, context):
        if not hasattr(context.scene, "fold_angle_overlay"):
            return
        s = context.scene.fold_angle_overlay
        layout = self.layout
        layout.prop(s, "show")

        col = layout.column()
        col.enabled = s.show
        col.prop(s, "min_angle")
        col.prop(s, "only_selected")

        box = col.box()
        box.label(text="Text")
        box.prop(s, "font_size")
        box.prop(s, "decimals")
        box.prop(s, "label_offset")

        box2 = col.box()
        box2.label(text="Colors")
        box2.prop(s, "convex_color")
        box2.prop(s, "concave_color")

        box3 = col.box()
        box3.prop(s, "show_markers")
        sub = box3.column()
        sub.enabled = s.show_markers
        sub.prop(s, "marker_size")


_draw_handle = None
_classes = (FoldAngleOverlaySettings, IMAGE_PT_fold_angle_overlay)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.fold_angle_overlay = bpy.props.PointerProperty(type=FoldAngleOverlaySettings)

    global _draw_handle
    _draw_handle = bpy.types.SpaceImageEditor.draw_handler_add(draw_callback, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None

    del bpy.types.Scene.fold_angle_overlay
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
