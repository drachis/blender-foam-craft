"""Live UV Editor overlay for the foam workflow.

Draws the cut plan the way the hand-drafted patterns do it: each island's
boundary is divided into runs by perpendicular tick marks, and each run carries
the bevel angle for that single saw pass -- the |--- 45° ---| convention. Piece
numbers sit at island centres.

This is deliberately not the paper overlay in ``uv_fold_angle_overlay``. That
one labels every mesh edge, which is correct for paper (every polygon edge is a
fold) and unreadable for foam (foam bends between darts, so most edges need no
mark at all).

Dart runs are expensive to compute -- an organic sculpt can produce a thousand
of them -- and a draw handler fires on every redraw, so results are cached and
recomputed only when something that affects them actually changes.
"""

from math import acos, degrees
from typing import NamedTuple

import blf
import bmesh
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import foam_angle_core as core

# 'UNIFORM_COLOR', not the pre-4.0 '2D_UNIFORM_COLOR': the 2D_-prefixed names
# were removed and raise ValueError on 5.2. Verified live on 5.2.0.
_SHADER_NAME = 'UNIFORM_COLOR'

# How many loop UVs to sample when fingerprinting the unwrap. A re-unwrap
# changes UV coordinates without changing element counts, so counts alone would
# not notice it; sampling coordinates catches it for a fixed small cost.
_UV_SAMPLE_COUNT = 240

# Distance in UV units used to probe which way "outward" points on screen.
# Small enough to stay local to the run, large enough to survive the float
# precision of the view transform at ordinary zoom levels.
_OUTWARD_PROBE = 1e-3


class _RunMark(NamedTuple):
    """One cut run, reduced to plain numbers.

    Deliberately holds no BMesh references. BMesh elements are only valid while
    their mesh is alive and unmodified: leaving edit mode, undo, or another
    object being edited frees them, and touching a freed BMLoop from a draw
    handler crashes Blender outright rather than raising. Since this cache
    outlives any single draw, everything is copied out as floats up front.
    """
    ax: float
    ay: float
    bx: float
    by: float
    angle_deg: float
    per_side_deg: float
    has_dihedral: bool
    needs_heat_form: bool
    # Unit vector in UV space pointing out of the island. The hand-drafted
    # patterns put every bracket and number in the margin outside the piece,
    # where they cannot be confused with cut lines, so the overlay needs to
    # know which side of a boundary run is "outside" rather than picking a
    # perpendicular arbitrarily.
    ox: float = 0.0
    oy: float = 0.0
    # ((u, v, nu, nv), ...) -- the run's outline with its outward normals, so
    # the bracket can be offset along the real edge rather than drawn as a
    # straight chord between the endpoints of a curved run.
    outline: tuple = ()


class _OverlayCache:
    """Computed overlay geometry, in UV space, plus the state it was built from."""

    __slots__ = ("signature", "runs", "island_labels", "stretch_islands")

    def __init__(self):
        self.signature = None
        self.runs = []               # _RunMark, plain data only
        self.island_labels = []      # (u, v, text)
        self.stretch_islands = set()  # island keys needing heat forming


_cache = _OverlayCache()
_draw_handle = None
# Craftability findings, refreshed by the operator rather than every draw --
# the assessment walks the whole mesh and is far too slow for a draw handler.
_findings = []


def _settings(context):
    return getattr(context.scene, "foam_craft", None)


def _uv_fingerprint(bm, uv_layer):
    """Cheap fingerprint of the current unwrap.

    Samples a fixed number of loop UVs rather than hashing them all, so cost
    stays flat on dense meshes while still noticing a re-unwrap.
    """
    loops = [l for f in bm.faces for l in f.loops]
    if not loops:
        return 0
    step = max(1, len(loops) // _UV_SAMPLE_COUNT)
    total = 0.0
    for i in range(0, len(loops), step):
        uv = loops[i][uv_layer].uv
        total += uv.x * 31.7 + uv.y * 17.3
    return round(total, 5)


def _edit_meshes(context):
    """Every mesh currently open in edit mode, not just the active one.

    Blender supports editing several objects at once and the UV editor shows
    all of their UVs together, so the overlay has to cover all of them or it
    silently annotates only one piece of what the user can see.
    """
    objects = getattr(context, "objects_in_mode", None) or []
    if not objects:
        obj = context.edit_object
        objects = [obj] if obj is not None else []
    out = []
    for obj in objects:
        if obj is None or obj.type != 'MESH' or not obj.data.is_editmode:
            continue
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is not None:
            out.append((obj, bm, uv_layer))
    return out


def _signature(pairs, s):
    per_object = tuple(
        (obj.name, len(bm.verts), len(bm.edges), len(bm.faces),
         _uv_fingerprint(bm, uv_layer))
        for obj, bm, uv_layer in pairs
    )
    return (
        per_object,
        round(s.max_chord_deviation, 6),
        round(s.max_angle_variation_deg, 4),
        round(s.foam_thickness, 6),
        round(s.max_tilt_deg, 4),
        round(s.stretch_tolerance, 4),
        s.only_selected,
    )


def _island_uv_centroid(faces, uv_layer):
    total = Vector((0.0, 0.0))
    count = 0
    for face in faces:
        for loop in face.loops:
            total += loop[uv_layer].uv
            count += 1
    return total / count if count else Vector((0.0, 0.0))


def _rebuild(pairs, s):
    """Recompute overlay geometry across every mesh currently in edit mode.

    Everything is copied out as plain numbers before returning; see _RunMark.
    """
    runs = []
    labels = []
    flagged = set()

    for obj, bm, uv_layer in pairs:
        islands = core.uv_islands(bm, uv_layer, s.only_selected)


        for run in core.iter_boundary_dart_segments(
                bm, uv_layer, s.max_chord_deviation, s.max_angle_variation_deg,
                only_selected=s.only_selected, islands=islands):
            if run.has_dihedral:
                bevel = core.compute_bevel(run.mean_angle_deg, s.foam_thickness,
                                           s.max_tilt_deg)
                per_side, heat = bevel.per_side_deg, bevel.needs_heat_form
            else:
                per_side, heat = 0.0, False

            # Outward comes from the boundary's winding, computed in core.
            ox, oy = (run.uv_outward.x, run.uv_outward.y) \
                if run.uv_outward is not None else (0.0, 0.0)

            outline = tuple(
                (p.x, p.y, n.x, n.y)
                for p, n in zip(run.uv_points, run.uv_normals))

            runs.append(_RunMark(
                run.uv_start.x, run.uv_start.y, run.uv_end.x, run.uv_end.y,
                run.mean_angle_deg, per_side, run.has_dihedral, heat, ox, oy,
                outline))

        # Piece numbers come from the numbering operator's attribute when it
        # has been run; otherwise fall back to island order so the overlay
        # still reads on an un-numbered mesh.
        id_layer = bm.faces.layers.int.get("foam_island_id")
        stretch = core.uv_stretch_map(bm, uv_layer)

        for index, faces in enumerate(islands):
            label = index + 1
            if id_layer is not None:
                for face in faces:
                    if face[id_layer] > 0:
                        label = face[id_layer]
                        break
            centroid = _island_uv_centroid(faces, uv_layer)
            key = (obj.name, index)
            labels.append((centroid.x, centroid.y, str(label), key))

            # An island can be fine at its darted edges and still refuse to lie
            # flat through the middle. That is a heat-forming job, not a dart,
            # so it is flagged from areal distortion, not from a boundary angle.
            vals = [stretch.get(f, 0.0) for f in faces]
            if vals and sum(vals) / len(vals) > s.stretch_tolerance:
                flagged.add(key)

    _cache.runs = runs
    _cache.island_labels = labels
    _cache.stretch_islands = flagged


def _ensure_cache(context, s):
    pairs = _edit_meshes(context)
    if not pairs:
        return False
    signature = _signature(pairs, s)
    if signature != _cache.signature:
        _rebuild(pairs, s)
        _cache.signature = signature
    return True


def _draw_lines(segments_by_color, width):
    if not segments_by_color:
        return
    shader = gpu.shader.from_builtin(_SHADER_NAME)
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(width)
    for color, points in segments_by_color.items():
        batch = batch_for_shader(shader, 'LINES', {"pos": points})
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def draw_callback():
    context = bpy.context
    s = _settings(context)
    if s is None or not s.show_overlay:
        return
    region = context.region
    if region is None:
        return
    try:
        ready = _ensure_cache(context, s)
    except (ReferenceError, AttributeError, RuntimeError):
        # The edit mesh can be freed between the redraw request and the draw.
        # Drop the cache rather than drawing from anything stale.
        _cache.signature = None
        return
    if not ready:
        return

    view = region.view2d
    scale = s.overlay_scale
    tick = s.tick_length_px * scale
    font_id = 0
    blf.size(font_id, max(6, int(s.font_size * scale)))

    convex = tuple(s.convex_color)
    concave = tuple(s.concave_color)
    heat = tuple(s.heat_form_color)

    segments = {}
    labels = []

    for run in _cache.runs:
        ax, ay = view.view_to_region(run.ax, run.ay, clip=False)
        bx, by = view.view_to_region(run.bx, run.by, clip=False)
        # view_to_region can return absurd values at extreme zoom; skip rather
        # than hand nonsense to the GPU.
        if not all(abs(v) < 1e6 for v in (ax, ay, bx, by)):
            continue

        dx, dy = bx - ax, by - ay
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            continue

        # Convert the UV-space outward vector into region space. Doing it by
        # transforming a second point keeps it correct under whatever pan and
        # zoom the view is at, rather than assuming the axes stay aligned.
        if run.ox or run.oy:
            tipx, tipy = view.view_to_region(
                run.ax + run.ox * _OUTWARD_PROBE,
                run.ay + run.oy * _OUTWARD_PROBE, clip=False)
            px, py = tipx - ax, tipy - ay
            plen = (px * px + py * py) ** 0.5
            if plen < 1e-9:
                px, py = -dy / length, dx / length
            else:
                px, py = px / plen, py / plen
        else:
            px, py = -dy / length, dx / length

        if run.needs_heat_form and s.show_heat_form:
            color = heat
        elif not s.show_dart_colors:
            color = convex
        else:
            color = convex if run.angle_deg >= 0 else concave

        # The bracket sits clear of the piece, in the margin, so it can never
        # be mistaken for a cut line -- the way the hand-drafted sheets do it.
        # It follows the edge's own shape: a run can span a curve, and a
        # straight chord across it would not read as belonging to that edge.
        off = tick * 1.2
        bracket = []
        for u, v, nu, nv in run.outline:
            sx, sy = view.view_to_region(u, v, clip=False)
            tx, ty = view.view_to_region(u + nu * _OUTWARD_PROBE,
                                         v + nv * _OUTWARD_PROBE, clip=False)
            dxn, dyn = tx - sx, ty - sy
            nlen = (dxn * dxn + dyn * dyn) ** 0.5
            if nlen < 1e-9:
                dxn, dyn = px, py
            else:
                dxn, dyn = dxn / nlen, dyn / nlen
            bracket.append((sx + dxn * off, sy + dyn * off, sx, sy, dxn, dyn))

        if not bracket:
            bracket = [(ax + px * off, ay + py * off, ax, ay, px, py),
                       (bx + px * off, by + py * off, bx, by, px, py)]

        bax, bay = bracket[0][0], bracket[0][1]
        bbx, bby = bracket[-1][0], bracket[-1][1]

        if s.show_brackets:
            bucket = segments.setdefault(color, [])
            # The offset outline, segment by segment, so it hugs the curve.
            for (x0, y0, *_), (x1, y1, *_) in zip(bracket, bracket[1:]):
                bucket.extend([(x0, y0), (x1, y1)])
            # Ends are drawn as a T: a bar crossing the bracket both ways, which
            # is how a dimension line terminates on the reference sheets.
            for bxp, byp, _ex, _ey, nxp, nyp in (bracket[0], bracket[-1]):
                bucket.extend([(bxp - nxp * tick * 0.5, byp - nyp * tick * 0.5),
                               (bxp + nxp * tick * 0.5, byp + nyp * tick * 0.5)])

        # Relief cuts are a different mark entirely: they sit at the vertices
        # along the edge, they go only into the material, and they are the cuts
        # themselves rather than an annotation. Depth follows how hard the
        # boundary turns there -- a tighter corner has to open further to let
        # the panel curve, so it needs more material taken out.
        if s.show_brackets and len(bracket) > 2:
            bucket = segments.setdefault(color, [])
            for k in range(1, len(bracket) - 1):
                _bx0, _by0, ex, ey, nx0, ny0 = bracket[k]
                prev_e = bracket[k - 1]
                next_e = bracket[k + 1]
                v1x, v1y = ex - prev_e[2], ey - prev_e[3]
                v2x, v2y = next_e[2] - ex, next_e[3] - ey
                l1 = (v1x * v1x + v1y * v1y) ** 0.5
                l2 = (v2x * v2x + v2y * v2y) ** 0.5
                if l1 < 1e-9 or l2 < 1e-9:
                    continue
                cosine = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (l1 * l2)))
                turn = degrees(acos(cosine))
                depth = tick * (0.45 + 1.05 * min(turn / 45.0, 1.0))
                bucket.extend([(ex, ey), (ex - nx0 * depth, ey - ny0 * depth)])

        if run.has_dihedral and abs(run.angle_deg) >= s.min_angle:
            text = f"{abs(run.per_side_deg):.{s.decimals}f}°"
            if run.needs_heat_form and s.show_heat_form:
                text += " HEAT"
            # Sit the number against the middle of the bracket as drawn, which
            # on a curved run is nowhere near the midpoint of its chord.
            mid = len(bracket) // 2
            if len(bracket) % 2 == 0:
                a_pt, b_pt = bracket[mid - 1], bracket[mid]
                mx, my = (a_pt[0] + b_pt[0]) * 0.5, (a_pt[1] + b_pt[1]) * 0.5
                nx, ny = (a_pt[4] + b_pt[4]) * 0.5, (a_pt[5] + b_pt[5]) * 0.5
            else:
                mx, my, _, _, nx, ny = bracket[mid]
            nlen = (nx * nx + ny * ny) ** 0.5 or 1.0
            labels.append((mx + nx / nlen * tick * 0.9,
                           my + ny / nlen * tick * 0.9, text, color))

    _draw_lines(segments, max(1.0, scale))

    for x, y, text, color in labels:
        width, height = blf.dimensions(font_id, text)
        blf.position(font_id, x - width / 2, y - height / 2, 0)
        blf.color(font_id, *color)
        blf.draw(font_id, text)

    if s.show_island_ids:
        blf.size(font_id, max(6, int(s.island_id_font_size * scale)))
        id_color = tuple(s.island_id_color)
        for u, v, text, key in _cache.island_labels:
            x, y = view.view_to_region(u, v, clip=False)
            if not all(abs(c) < 1e6 for c in (x, y)):
                continue
            color = heat if (s.show_heat_form
                             and key in _cache.stretch_islands) else id_color
            width, height = blf.dimensions(font_id, text)
            blf.position(font_id, x - width / 2, y - height / 2, 0)
            blf.color(font_id, *color)
            blf.draw(font_id, text)


class FOAM_OT_refresh_overlay(bpy.types.Operator):
    """Recompute the cut runs shown in the UV editor"""

    bl_idname = "foam.refresh_overlay"
    bl_label = "Refresh Overlay"

    def execute(self, context):
        _cache.signature = None
        for area in context.screen.areas:
            area.tag_redraw()
        return {'FINISHED'}


class FOAM_OT_check_craftability(bpy.types.Operator):
    """Check whether this mesh and unwrap can actually be built from foam"""

    bl_idname = "foam.check_craftability"
    bl_label = "Check Craftability"

    @classmethod
    def poll(cls, context):
        return context.edit_object is not None and context.edit_object.type == 'MESH'

    def execute(self, context):
        global _findings
        s = _settings(context)
        obj = context.edit_object
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        _findings = core.assess_craftability(
            bm, uv_layer,
            max_tilt_deg=s.max_tilt_deg,
            max_chord_deviation=s.max_chord_deviation,
            max_angle_variation_deg=s.max_angle_variation_deg,
            sheet_width=s.sheet_width or None,
            sheet_height=s.sheet_height or None,
            stretch_tolerance=s.stretch_tolerance)
        blockers = sum(1 for f in _findings if f.severity == 'blocker')
        warnings = sum(1 for f in _findings if f.severity == 'warn')
        self.report({'INFO'},
                    f"{blockers} blocker(s), {warnings} warning(s)")
        for area in context.screen.areas:
            area.tag_redraw()
        return {'FINISHED'}


_SEVERITY_ICON = {'blocker': 'CANCEL', 'warn': 'ERROR',
                  'info': 'INFO', 'ok': 'CHECKMARK'}


class IMAGE_PT_foam_craft(bpy.types.Panel):
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "FoamCraft"
    bl_label = "Foam Cut Plan"

    def draw(self, context):
        layout = self.layout
        s = _settings(context)
        if s is None:
            layout.label(text="FoamCraft settings unavailable", icon='ERROR')
            return

        layout.prop(s, "show_overlay", toggle=True)
        col = layout.column()
        col.enabled = s.show_overlay

        box = col.box()
        box.label(text="Material")
        box.prop(s, "foam_grade")
        box.prop(s, "foam_thickness")

        box = col.box()
        box.label(text="Cut Runs")
        box.prop(s, "max_chord_deviation")
        box.prop(s, "max_angle_variation_deg")
        box.prop(s, "max_tilt_deg")
        box.operator("foam.estimate_tolerances", icon='SHADERFX')
        box.label(text=f"{len(_cache.runs)} run(s) cached")
        box.operator(FOAM_OT_refresh_overlay.bl_idname, icon='FILE_REFRESH')

        box = col.box()
        box.label(text="Show")
        row = box.row(align=True)
        row.prop(s, "show_brackets", toggle=True)
        row.prop(s, "show_island_ids", toggle=True)
        row = box.row(align=True)
        row.prop(s, "show_heat_form", toggle=True)
        row.prop(s, "show_dart_colors", toggle=True)
        box.prop(s, "only_selected")
        box.prop(s, "overlay_scale")


class IMAGE_PT_foam_craft_display(bpy.types.Panel):
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "FoamCraft"
    bl_label = "Display"
    bl_parent_id = "IMAGE_PT_foam_craft"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = _settings(context)
        if s is None:
            return
        layout.prop(s, "tick_length_px")
        layout.prop(s, "font_size")
        layout.prop(s, "island_id_font_size")
        layout.prop(s, "decimals")
        layout.prop(s, "min_angle")
        layout.prop(s, "convex_color")
        layout.prop(s, "concave_color")
        layout.prop(s, "heat_form_color")
        layout.prop(s, "island_id_color")


class IMAGE_PT_foam_craft_advice(bpy.types.Panel):
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "FoamCraft"
    bl_label = "Craftability"
    bl_parent_id = "IMAGE_PT_foam_craft"

    def draw(self, context):
        layout = self.layout
        s = _settings(context)
        if s is None:
            return
        row = layout.row(align=True)
        row.prop(s, "sheet_width")
        row.prop(s, "sheet_height")
        layout.prop(s, "stretch_tolerance")
        layout.operator(FOAM_OT_check_craftability.bl_idname, icon='VIEWZOOM')

        if not _findings:
            layout.label(text="Not checked yet")
            return

        for finding in _findings:
            box = layout.box()
            box.label(text=finding.title,
                      icon=_SEVERITY_ICON.get(finding.severity, 'DOT'))
            for line in _wrap(finding.detail, 44):
                box.label(text=line)
            if finding.severity != 'ok':
                sub = box.column(align=True)
                sub.scale_y = 0.8
                for para in finding.advice.split("\n"):
                    for line in _wrap(para, 46):
                        sub.label(text=line)


def _wrap(text, width):
    """Blender UI labels do not wrap, so break the text ourselves."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


_classes = (
    FOAM_OT_refresh_overlay,
    FOAM_OT_check_craftability,
    IMAGE_PT_foam_craft,
    IMAGE_PT_foam_craft_display,
    IMAGE_PT_foam_craft_advice,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    global _draw_handle
    # Never stack handlers. A reload that registers without unregistering would
    # otherwise leave the previous callback attached forever, and every stale
    # copy keeps drawing from its own dead cache.
    if _draw_handle is not None:
        try:
            bpy.types.SpaceImageEditor.draw_handler_remove(_draw_handle, 'WINDOW')
        except ValueError:
            pass
        _draw_handle = None
    _cache.signature = None
    _draw_handle = bpy.types.SpaceImageEditor.draw_handler_add(
        draw_callback, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    global _draw_handle, _findings
    if _draw_handle is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    _findings = []
    _cache.signature = None
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
