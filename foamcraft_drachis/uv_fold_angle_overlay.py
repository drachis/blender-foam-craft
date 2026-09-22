"""Live overlay of fold/bevel angles on every UV edge (paper workflow).

Draws a colored marker plus the fold angle in degrees at every edge in the UV
Editor, computed from the real 3D dihedral angle between the two faces sharing
that edge. Not an export -- a visual aid for tuning the unwrap and panel layout
by eye. Blue = convex/mountain, pink = concave/valley by default.

Sidebar: UV Editor > N > Fold Angles.
"""

import bpy
import blf
import bmesh
import gpu
from gpu_extras.batch import batch_for_shader

from . import foam_angle_core


class FOAMCRAFT_DRACHIS_PG_fold_angle_settings(bpy.types.PropertyGroup):
    # Off by default. This overlay labels every mesh edge, which is right for
    # paper but unreadable on a foam pattern, and with both overlays enabled
    # the two sets of numbers sit on top of each other. Whichever workflow the
    # user wants, they should turn it on deliberately.
    show: bpy.props.BoolProperty(name="Show Fold Angles", default=False)
    convex_color: bpy.props.FloatVectorProperty(
        name="Convex (Mountain) Color",
        description="Color for edges bending toward you",
        default=(0.15, 0.45, 1.0, 1.0), min=0, max=1, subtype='COLOR', size=4)
    concave_color: bpy.props.FloatVectorProperty(
        name="Concave (Valley) Color",
        description="Color for edges bending away from you",
        default=(1.0, 0.2, 0.55, 1.0), min=0, max=1, subtype='COLOR', size=4)
    font_size: bpy.props.IntProperty(name="Font Size", default=14, min=6, max=60)
    decimals: bpy.props.IntProperty(name="Decimal Places", default=0, min=0, max=2)
    min_angle: bpy.props.FloatProperty(
        name="Hide Below (deg)",
        description="Edges bending less than this are not labeled",
        default=1.0, min=0.0, max=180.0)
    label_offset: bpy.props.FloatProperty(
        name="Label Offset (px)",
        description="Distance the text is nudged away from the edge, perpendicular to it",
        default=14.0, min=0.0, max=200.0)
    show_markers: bpy.props.BoolProperty(name="Show Edge Markers", default=True)
    marker_size: bpy.props.FloatProperty(name="Marker Size (px)", default=5.0,
                                         min=1.0, max=30.0)
    only_selected: bpy.props.BoolProperty(
        name="Only Selected Faces",
        description="Only label edges belonging to selected faces, to declutter a busy mesh",
        default=False)


def draw_callback():
    context = bpy.context
    scene = context.scene
    if scene is None or not hasattr(scene, "foamcraft_drachis_fold_angles"):
        return
    settings = scene.foamcraft_drachis_fold_angles
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

    for edge, uv_a, uv_b in foam_angle_core.iter_uv_edges(
            bm, uv_layer, settings.only_selected):
        angle = foam_angle_core.dihedral_angle_deg(edge)
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
        text = f"{abs(angle):.{settings.decimals}f}°"
        text_w, text_h = blf.dimensions(font_id, text)
        blf.position(font_id, label_x - text_w / 2, label_y - text_h / 2, 0)
        blf.color(font_id, *color)
        blf.draw(font_id, text)

    if marker_points_by_color:
        # 'UNIFORM_COLOR', not the pre-4.0 '2D_UNIFORM_COLOR' -- the 2D_ prefixed
        # names were removed and raise ValueError on 5.2. Verified live on 5.2.0.
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.point_size_set(settings.marker_size)
        gpu.state.blend_set('ALPHA')
        for color, pts in marker_points_by_color.items():
            batch = batch_for_shader(shader, 'POINTS', {"pos": pts})
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)
        gpu.state.blend_set('NONE')


class IMAGE_PT_foamcraft_drachis_fold_angles(bpy.types.Panel):
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Fold Angles"
    bl_label = "Fold Angle Overlay"

    def draw(self, context):
        if not hasattr(context.scene, "foamcraft_drachis_fold_angles"):
            return
        s = context.scene.foamcraft_drachis_fold_angles
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
_classes = (FOAMCRAFT_DRACHIS_PG_fold_angle_settings, IMAGE_PT_foamcraft_drachis_fold_angles)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.foamcraft_drachis_fold_angles = bpy.props.PointerProperty(
        type=FOAMCRAFT_DRACHIS_PG_fold_angle_settings)

    global _draw_handle
    # Never stack handlers: a reload that registers without unregistering
    # leaves the old callback attached forever, still drawing from dead data.
    if _draw_handle is not None:
        try:
            bpy.types.SpaceImageEditor.draw_handler_remove(_draw_handle, 'WINDOW')
        except ValueError:
            pass
        _draw_handle = None
    _draw_handle = bpy.types.SpaceImageEditor.draw_handler_add(
        draw_callback, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        try:
            bpy.types.SpaceImageEditor.draw_handler_remove(_draw_handle, 'WINDOW')
        except ValueError:
            pass
        _draw_handle = None

    del bpy.types.Scene.foamcraft_drachis_fold_angles
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
