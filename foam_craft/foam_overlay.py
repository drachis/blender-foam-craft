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

# Where the draw handle is parked so it outlives a module reload. See
# _drop_existing_handler().
_HANDLE_KEY = "foam_craft_overlay_draw_handle"


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
    # ((u, v, tu, tv, nu, nv), ...) -- where a relief dart goes, with the edge
    # tangent and outward normal there, so the mark can be drawn as a T lying
    # along the boundary with its stem cut into the material.
    darts: tuple = ()


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


def _selection_fingerprint(bm):
    """Cheap fingerprint of which faces are selected.

    Without this the overlay only noticed a selection change when some other
    setting happened to change too, so "Only Selected" appeared to need
    toggling before it would follow a new selection. Sampled at a fixed cost
    like the UV fingerprint, since this runs on every draw.
    """
    faces = bm.faces
    if not faces:
        return 0
    step = max(1, len(faces) // _UV_SAMPLE_COUNT)
    total = 0
    count = 0
    for i in range(0, len(faces), step):
        if faces[i].select:
            total += i + 1
            count += 1
    return (count, total)


def _signature(pairs, s):
    per_object = tuple(
        (obj.name, len(bm.verts), len(bm.edges), len(bm.faces),
         _uv_fingerprint(bm, uv_layer),
         _selection_fingerprint(bm) if s.only_selected else 0)
        for obj, bm, uv_layer in pairs
    )
    return (
        per_object,
        round(s.max_chord_deviation, 6),
        round(s.max_angle_variation_deg, 4),
        round(s.foam_thickness, 6),
        round(s.max_tilt_deg, 4),
        round(s.stretch_tolerance, 4),
        round(s.corner_angle_deg, 4),
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
    heat_islands = set()

    for obj, bm, uv_layer in pairs:
        islands = core.uv_islands(bm, uv_layer, s.only_selected)


        for run in core.iter_boundary_dart_segments(
                bm, uv_layer, s.max_chord_deviation, s.max_angle_variation_deg,
                only_selected=s.only_selected, islands=islands,
                corner_angle_deg=s.corner_angle_deg):
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

            dart_marks = []
            for k in run.dart_indices:
                pt = run.uv_points[k]
                nrm = run.uv_normals[k]
                nxt = run.uv_points[min(k + 1, len(run.uv_points) - 1)]
                prv = run.uv_points[max(k - 1, 0)]
                tangent = nxt - prv
                if tangent.length > 1e-12:
                    tangent = tangent.normalized()
                dart_marks.append((pt.x, pt.y, tangent.x, tangent.y,
                                   nrm.x, nrm.y))
            darts = tuple(dart_marks)

            if heat:
                heat_islands.add((obj.name, run.island_index))

            runs.append(_RunMark(
                run.uv_start.x, run.uv_start.y, run.uv_end.x, run.uv_end.y,
                run.mean_angle_deg, per_side, run.has_dihedral, heat, ox, oy,
                outline, darts))

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
    # Heat forming is a property of a piece, not of a single cut: either some
    # seam on it is too steep to saw, or the piece will not lie flat at all.
    # Marking every affected run cluttered the sheet without telling the user
    # anything the piece-level flag does not.
    _cache.stretch_islands = flagged | heat_islands


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
    font_id = 0

    # Screen pixels per unit of UV space at the current zoom. Sizing the text
    # from this keeps it a constant fraction of the pattern, so a number holds
    # its place against the piece it labels no matter how far in you zoom --
    # a fixed pixel size instead shrinks relative to the pattern as it grows.
    origin_x, origin_y = view.view_to_region(0.0, 0.0, clip=False)
    unit_x, unit_y = view.view_to_region(1.0, 0.0, clip=False)
    px_per_uv = abs(unit_x - origin_x) or 1.0

    if s.scale_text_with_view:
        # Sizes are thousandths of the UV sheet.
        angle_px = int(s.angle_text_size / 1000.0 * px_per_uv * scale)
        tick = s.mark_size / 1000.0 * px_per_uv * scale
        number_px = int(s.number_size / 1000.0 * px_per_uv * scale)
    else:
        angle_px = int(s.font_size * scale)
        tick = s.tick_length_px * scale
        number_px = int(s.island_id_font_size * scale)
    angle_px = max(6, min(angle_px, 400))
    tick = max(2.0, min(tick, 600.0))
    blf.size(font_id, angle_px)

    convex = tuple(s.convex_color)
    concave = tuple(s.concave_color)
    heat = tuple(s.heat_form_color)

    segments = {}
    labels = []
    hidden = []
    dart_seen = {}

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

        if not s.show_dart_colors:
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

        # Relief cuts are a different mark from the angle guide: they are the
        # cuts themselves. Drawn as a T lying on the boundary -- the bar runs
        # along the edge, the stem goes into the material and never out of it.
        # Collected per vertex rather than per run so a vertex shared by two
        # runs gets one mark, not two crossing into an X.
        if s.show_brackets:
            for u, v, tu, tv, nu, nv in run.darts:
                key = (round(u, 6), round(v, 6))
                if key in dart_seen:
                    continue
                dart_seen[key] = (tu, tv, nu, nv, color)

        # Skip the number when the run is too short on screen to hold it.
        # This is what stops a notch, where several short runs converge, from
        # becoming a heap of overlapping digits.
        run_px = ((bbx - bax) ** 2 + (bby - bay) ** 2) ** 0.5
        labelled = run.has_dihedral and abs(run.angle_deg) >= s.min_angle
        if labelled and run_px < s.min_label_run_px:
            # Mark the spot rather than dropping it silently: an angle that is
            # simply absent looks like a run that needs no cut.
            mid = len(bracket) // 2
            if len(bracket) % 2 == 0:
                a_pt, b_pt = bracket[mid - 1], bracket[mid]
                mx, my = (a_pt[0] + b_pt[0]) * 0.5, (a_pt[1] + b_pt[1]) * 0.5
                nx, ny = (a_pt[4] + b_pt[4]) * 0.5, (a_pt[5] + b_pt[5]) * 0.5
            else:
                mx, my, _, _, nx, ny = bracket[mid]
            nlen = (nx * nx + ny * ny) ** 0.5 or 1.0
            hidden.append((mx, my, nx / nlen, ny / nlen))
        if labelled and run_px >= s.min_label_run_px:
            text = f"{abs(run.per_side_deg):.{s.decimals}f}°"
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
            # Keep the normal with the anchor rather than resolving the
            # position now: the text has to clear its own height to stay off
            # the piece, and that is only known once the font is measured.
            labels.append((mx, my, nx / nlen, ny / nlen, text, color))

    for (u, v), (tu, tv, nu, nv, color) in dart_seen.items():
        px0, py0 = view.view_to_region(u, v, clip=False)
        if not all(abs(c) < 1e6 for c in (px0, py0)):
            continue
        tx, ty = view.view_to_region(u + tu * _OUTWARD_PROBE,
                                     v + tv * _OUTWARD_PROBE, clip=False)
        nx2, ny2 = view.view_to_region(u + nu * _OUTWARD_PROBE,
                                       v + nv * _OUTWARD_PROBE, clip=False)
        tdx, tdy = tx - px0, ty - py0
        ndx, ndy = nx2 - px0, ny2 - py0
        tl = (tdx * tdx + tdy * tdy) ** 0.5
        nl = (ndx * ndx + ndy * ndy) ** 0.5
        if tl < 1e-9 or nl < 1e-9:
            continue
        tdx, tdy = tdx / tl, tdy / tl
        ndx, ndy = ndx / nl, ndy / nl
        # A single stroke running inward from the boundary. No crossbar: the
        # mark's job is to say "this cut goes into the material from here", and
        # a bar lying on the edge reads as part of the outline instead.
        bucket = segments.setdefault(color, [])
        stem = tick * 1.1
        bucket.extend([(px0, py0), (px0 - ndx * stem, py0 - ndy * stem)])

    _draw_lines(segments, max(1.0, scale))

    # Warn where a number was withheld for want of room.
    if hidden:
        warn_color = tuple(s.suppressed_color)
        blf.size(font_id, max(8, int(angle_px * 0.9)))
        for ax0, ay0, nx, ny in hidden:
            width, height = blf.dimensions(font_id, "!")
            clear = abs(nx) * width * 0.5 + abs(ny) * height * 0.5 + tick * 0.35
            blf.position(font_id, ax0 + nx * clear - width / 2,
                         ay0 + ny * clear - height / 2, 0)
            blf.color(font_id, *warn_color)
            blf.draw(font_id, "!")
        blf.size(font_id, angle_px)

    for ax0, ay0, nx, ny, text, color in labels:
        width, height = blf.dimensions(font_id, text)
        # Push the text out far enough that its own box clears the boundary.
        # Projecting the half-extents onto the normal handles labels sitting
        # against a diagonal edge, where half the height alone is not enough.
        clearance = abs(nx) * width * 0.5 + abs(ny) * height * 0.5 + tick * 0.35
        x = ax0 + nx * clearance
        y = ay0 + ny * clearance
        blf.position(font_id, x - width / 2, y - height / 2, 0)
        blf.color(font_id, *color)
        blf.draw(font_id, text)

    if s.show_island_ids:
        blf.size(font_id, max(6, min(number_px, 500)))
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
        layout.prop(s, "corner_angle_deg")
        layout.separator()
        layout.prop(s, "scale_text_with_view")
        if s.scale_text_with_view:
            layout.prop(s, "angle_text_size")
            layout.prop(s, "mark_size")
            layout.prop(s, "number_size")
        else:
            layout.prop(s, "font_size")
            layout.prop(s, "tick_length_px")
            layout.prop(s, "island_id_font_size")
        layout.prop(s, "decimals")
        layout.prop(s, "min_angle")
        layout.prop(s, "min_label_run_px")
        layout.prop(s, "convex_color")
        layout.prop(s, "concave_color")
        layout.prop(s, "suppressed_color")
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
        # Real-world size of what is being built, so the sheet figures below
        # can be judged against it without leaving the panel.
        objects = getattr(context, "objects_in_mode", None) or []
        if not objects and context.edit_object:
            objects = [context.edit_object]
        meshes = [o for o in objects if o and o.type == 'MESH']
        if meshes:
            unit = context.scene.unit_settings.scale_length
            box = layout.box()
            box.label(text="Model Size", icon='FIXED_SIZE')
            for obj in meshes:
                w, d, h = (v * unit for v in obj.dimensions)
                if len(meshes) > 1:
                    box.label(text=obj.name)
                # One row each: the sidebar is narrow enough that two
                # dimensions on a line get truncated mid-number.
                col = box.column(align=True)
                col.scale_y = 0.85
                for axis, value in (("W", w), ("D", d), ("H", h)):
                    col.label(text=f"{axis}  {_fmt_len(value)}")

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


def _fmt_len(metres):
    """Length in whichever unit reads naturally, plus inches for the workshop.

    Foam stock is bought in imperial here while Blender works in metres, so
    showing both saves converting in your head at the saw.
    """
    inches = metres * 39.3701
    if metres >= 1.0:
        return f"{metres:.2f} m ({inches / 12.0:.1f} ft)"
    if metres >= 0.01:
        return f"{metres * 100.0:.1f} cm ({inches:.1f} in)"
    return f"{metres * 1000.0:.0f} mm ({inches:.2f} in)"


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


def _drop_existing_handler():
    """Remove any handler this add-on previously installed, including one left
    by an earlier copy of this module.

    The handle is parked in driver_namespace rather than only in a module
    global because reloading during development replaces the module object:
    the new copy starts with _draw_handle None while Blender still holds the
    old copy's callback, so both draw and every mark appears twice. A namespace
    that outlives the module is the only place a successor can find it.
    """
    global _draw_handle
    for handle in (_draw_handle, bpy.app.driver_namespace.get(_HANDLE_KEY)):
        if handle is None:
            continue
        try:
            bpy.types.SpaceImageEditor.draw_handler_remove(handle, 'WINDOW')
        except (ValueError, TypeError):
            pass
    _draw_handle = None
    bpy.app.driver_namespace.pop(_HANDLE_KEY, None)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    global _draw_handle
    _drop_existing_handler()
    _cache.signature = None
    _draw_handle = bpy.types.SpaceImageEditor.draw_handler_add(
        draw_callback, (), 'WINDOW', 'POST_PIXEL')
    bpy.app.driver_namespace[_HANDLE_KEY] = _draw_handle


def unregister():
    global _findings
    _drop_existing_handler()
    _findings = []
    _cache.signature = None
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
