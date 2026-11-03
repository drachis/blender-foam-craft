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

from math import radians, tan

import bpy
import bmesh
from mathutils import Vector

from . import foam_angle_core as core

# Real-world sizes for pattern elements that only make sense printed at 1:1 --
# distinct from the overlay's screen-space settings (font_size, tick_length_px,
# overlay_scale), which scale with UV editor zoom rather than paper size.
# Darts are physical cuts, so their size is in millimetres of real foam and
# comes from the stock thickness, not from how the pattern is drawn.
DART_SLIT_WIDTH_MM = 0.6
MARGIN_MM = 10.0

# Annotation sizes are *not* physical. They are parts per thousand of the UV
# sheet, exactly as the overlay draws them (foam_overlay.py:480-482 computes
# `permille / 1000 * px_per_uv * overlay_scale`; here the same expression runs
# against millimetres per UV unit instead of pixels). Printing the same
# numbers the user is looking at is the whole point -- the SVG is a render of
# the overlay, not a second opinion about how big a label should be.


def _face_uv_area(face, uv_layer):
    """Shoelace area of one face in UV space.

    Mirrors ``foam_angle_core._uv_area`` -- kept local rather than imported
    since that name is private to the core module and this is the only other
    place that needs it.
    """
    uvs = [l[uv_layer].uv for l in face.loops]
    area = 0.0
    for i in range(1, len(uvs) - 1):
        u, v = uvs[i] - uvs[0], uvs[i + 1] - uvs[0]
        area += abs(u.x * v.y - u.y * v.x) * 0.5
    return area


def _polygon_centroid(pts):
    """Area-weighted centroid of a closed polygon; falls back to the simple
    mean for degenerate (near-zero-area) polygons such as a two-point sliver."""
    if len(pts) < 3:
        return sum(pts, Vector((0.0, 0.0))) / max(len(pts), 1)
    area = 0.0
    cx = 0.0
    cy = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-12:
        return sum(pts, Vector((0.0, 0.0))) / n
    cx /= (6 * area)
    cy /= (6 * area)
    return Vector((cx, cy))


class _Island:
    """One UV island's geometry, pre-converted to millimetres and placed on
    the output sheet. Built once per island so every SVG group below can share
    the same coordinate transform."""

    def __init__(self, index, faces, uv_layer, sheet):
        self.index = index
        self.faces = faces
        self.piece_id = index + 1  # overwritten below if the attribute exists
        self.cycles = core.island_boundary_cycles(faces, uv_layer)
        self.sheet = sheet

        # The island's own UV bounds. Used for its centroid, which decides
        # which way a dart points -- a local question. Nothing about where the
        # island lands on the sheet is decided here any more; that is the
        # sheet's single transform, shared by every island.
        all_uvs = [l[uv_layer].uv for cyc in self.cycles for l in cyc]
        if not all_uvs:
            all_uvs = [l[uv_layer].uv for f in faces for l in f.loops]
        self.min_u = min(p.x for p in all_uvs)
        self.max_u = max(p.x for p in all_uvs)
        self.min_v = min(p.y for p in all_uvs)
        self.max_v = max(p.y for p in all_uvs)

    def to_sheet(self, uv):
        return self.sheet.to_sheet(uv)


class _Sheet:
    """The one transform from UV space to millimetres on the page.

    There used to be a packer here, which re-arranged the islands into rows
    and scaled each one independently by its own area distortion. Both are
    gone, and deliberately: **the UV layout is the arrangement.** The maker
    positions pieces in the UV editor to suit their stock, and an exporter
    that quietly re-lays them discards that work and produces a sheet that
    cannot be checked against what is on screen.

    CLAUDE.md records this exact mistake once already, against the flat layout
    preview -- a bespoke shelf packer invented for a problem whose answer was
    u->x, v->y scaled to the mesh. This was the same bug in a second place.

    So: any rearranging belongs in Blender's own UV tools, where it is visible
    and undoable. This renders what is there.

    The consequence worth stating is that one uniform scale cannot also make
    each island individually true to size. Where UV distortion differs between
    islands, a piece is as true as its unwrap is -- which is a property of the
    unwrap, shown honestly, rather than a correction applied invisibly at
    export time. Minimum Stretch exists for exactly that, and the craftability
    check already reports pieces that distort too far.
    """

    def __init__(self, islands_faces, uv_layer, scale_length):
        area_3d = sum(f.calc_area() for fs in islands_faces for f in fs)
        area_uv = sum(_face_uv_area(f, uv_layer)
                      for fs in islands_faces for f in fs)
        # One ratio for the whole mesh, so every piece shares a scale and the
        # sheet is a true picture of the UV layout.
        m_per_uv = (area_3d / area_uv) ** 0.5 if area_uv > 1e-12 else 1.0
        self.scale = m_per_uv * scale_length * 1000.0  # mm per UV unit

        uvs = [l[uv_layer].uv for fs in islands_faces for f in fs
               for l in f.loops]
        if uvs:
            self.min_u = min(p.x for p in uvs)
            self.max_u = max(p.x for p in uvs)
            self.min_v = min(p.y for p in uvs)
            self.max_v = max(p.y for p in uvs)
        else:
            self.min_u = self.max_u = self.min_v = self.max_v = 0.0

        self.width = (self.max_u - self.min_u) * self.scale + 2 * MARGIN_MM
        self.height = (self.max_v - self.min_v) * self.scale + 2 * MARGIN_MM

    def to_sheet(self, uv):
        """UV point -> millimetre point on the page.

        SVG y grows downward and UV v grows upward, so v is measured from the
        top of the whole sheet -- not, as it once was, from each island's own
        bounding box, which is what let the islands drift apart.
        """
        x = (uv.x - self.min_u) * self.scale + MARGIN_MM
        y = (self.max_v - uv.y) * self.scale + MARGIN_MM
        return Vector((x, y))

    def permille(self, parts, overlay_scale):
        """A size given in thousandths of the UV sheet, in millimetres.

        Mirrors foam_overlay.py's `permille / 1000 * px_per_uv * scale`, with
        millimetres per UV unit standing in for pixels, so a label prints at
        the size it is drawn at.
        """
        return parts / 1000.0 * self.scale * overlay_scale


def _run_polyline(run, uv_layer, isl):
    pts = [isl.to_sheet(l[uv_layer].uv) for l in run.loops]
    pts.append(isl.to_sheet(run.uv_end))
    return pts


def _boundary_tangent(pt_before, pt_at):
    d = pt_at - pt_before
    return d.normalized() if d.length > 1e-9 else Vector((1.0, 0.0))


def _inward_normal(tangent, boundary_pt, centroid):
    """Perpendicular to the boundary, picked to point into the piece.

    UV winding direction isn't guaranteed consistent across islands (it
    depends on face-normal orientation, which artists flip freely), so rather
    than assume a winding convention, pick whichever perpendicular actually
    points toward the island's own centroid.
    """
    n = Vector((-tangent.y, tangent.x))
    return n if n.dot(centroid - boundary_pt) > 0 else -n


def _fmt(v):
    return f"{v:.3f}"


def _boundary_angle_deg(run_a, run_b):
    """Angle to bevel a dart at the corner between two consecutive runs.

    Where both sides of the corner carry a real seam, split the difference;
    where only one side does (the other is an open rim with no facing panel),
    use that one; where neither does, there's nothing to bevel.
    """
    if run_a.has_dihedral and run_b.has_dihedral:
        return (run_a.mean_angle_deg + run_b.mean_angle_deg) / 2.0
    if run_a.has_dihedral:
        return run_a.mean_angle_deg
    if run_b.has_dihedral:
        return run_b.mean_angle_deg
    return None


def generate_svg(bm, uv_layer, settings, scale_length=1.0):
    """Build the full SVG pattern as a string. Pure function -- no file I/O,
    no bpy.context -- so it can be exercised directly in tests."""
    islands_faces = core.uv_islands(bm, uv_layer, only_selected=False)
    sheet = _Sheet(islands_faces, uv_layer, scale_length)
    islands = [_Island(i, faces, uv_layer, sheet)
              for i, faces in enumerate(islands_faces)]

    id_layer = bm.faces.layers.int.get('foam_island_id')
    if id_layer is not None:
        for isl in islands:
            rep = next(iter(isl.faces), None)
            if rep is not None:
                isl.piece_id = rep[id_layer]

    total_width, total_height = sheet.width, sheet.height

    thickness_mm = settings.foam_thickness * scale_length * 1000.0
    dart_depth_mm = settings.notch_depth_ratio * thickness_mm

    # Annotation sizes, in millimetres, from the same permille settings the
    # overlay draws with -- so the printed sheet matches the screen.
    overlay_scale = settings.overlay_scale
    angle_font_mm = sheet.permille(settings.angle_text_size, overlay_scale)
    number_font_mm = sheet.permille(settings.number_size, overlay_scale)
    tick_length_mm = sheet.permille(settings.mark_size, overlay_scale)
    # Hairlines scale with the sheet too, or a 2 m pattern gets invisible
    # outlines and a 100 mm one gets a smeared blob.
    stroke_mm = sheet.permille(2, overlay_scale)

    all_runs = list(core.iter_boundary_dart_segments(
        bm, uv_layer, settings.max_chord_deviation,
        settings.max_angle_variation_deg, islands=islands_faces))
    runs_by_island = [[] for _ in islands]
    for r in all_runs:
        runs_by_island[r.island_index].append(r)

    outlines, darts, ticks, angles, numbers, heatform = [], [], [], [], [], []

    for isl in islands:
        # --- outlines: one polygon per boundary cycle (an island with a hole
        # yields more than one).
        for cycle in isl.cycles:
            if len(cycle) < 2:
                continue
            pts = [isl.to_sheet(l[uv_layer].uv) for l in cycle]
            pts_str = " ".join(f"{_fmt(p.x)},{_fmt(p.y)}" for p in pts)
            outlines.append(f'<polygon points="{pts_str}"/>')

        # --- numbers: piece ID at the outer cycle's centroid.
        if settings.show_island_ids and isl.cycles:
            outer = max(isl.cycles, key=len)
            pts = [isl.to_sheet(l[uv_layer].uv) for l in outer]
            c = _polygon_centroid(pts)
            numbers.append(
                f'<text x="{_fmt(c.x)}" y="{_fmt(c.y)}" '
                f'font-size="{_fmt(number_font_mm)}">{isl.piece_id}</text>')

        # --- angles + heatform: one entry per run.
        for run in runs_by_island[isl.index]:
            if not run.has_dihedral or abs(run.mean_angle_deg) < settings.min_angle:
                continue
            bevel = core.compute_bevel(run.mean_angle_deg, settings.foam_thickness,
                                       settings.max_tilt_deg)
            mid = (isl.to_sheet(run.uv_start) + isl.to_sheet(run.uv_end)) / 2
            label = f"{abs(run.mean_angle_deg):.{settings.decimals}f}°"
            if bevel.needs_heat_form:
                if settings.show_heat_form:
                    pts = _run_polyline(run, uv_layer, isl)
                    pts_str = " ".join(f"{_fmt(p.x)},{_fmt(p.y)}" for p in pts)
                    heatform.append(f'<polyline points="{pts_str}" fill="none"/>')
                    heatform.append(
                        f'<text x="{_fmt(mid.x)}" y="{_fmt(mid.y)}" '
                        f'font-size="{_fmt(angle_font_mm)}">{label} (heat-form)</text>')
                # Needs a heat-form joint, not a saw cut -- never also present
                # it as an ordinary sawable bevel in the angles group.
                continue
            if settings.show_brackets:
                if settings.show_dart_colors:
                    css_class = "convex" if run.mean_angle_deg >= 0 else "concave"
                else:
                    css_class = "neutral"
                angles.append(
                    f'<text class="{css_class}" x="{_fmt(mid.x)}" y="{_fmt(mid.y)}" '
                    f'font-size="{_fmt(angle_font_mm)}">{label}</text>')

        # --- ticks + darts: one per run-to-run corner, grouped by boundary
        # cycle so the wraparound (last run's end == first run's start)
        # closes correctly.
        cycle_loop_sets = [set(cyc) for cyc in isl.cycles]
        runs_by_cycle = [[] for _ in isl.cycles]
        for run in runs_by_island[isl.index]:
            first = run.loops[0]
            for ci, loopset in enumerate(cycle_loop_sets):
                if first in loopset:
                    runs_by_cycle[ci].append(run)
                    break

        centroid_uv = Vector(((isl.min_u + isl.max_u) / 2, (isl.min_v + isl.max_v) / 2))
        centroid_pt = isl.to_sheet(centroid_uv)

        for cyc_runs in runs_by_cycle:
            n = len(cyc_runs)
            if n == 0:
                continue
            for k in range(n):
                run_a = cyc_runs[k]
                run_b = cyc_runs[(k + 1) % n]
                boundary_pt = isl.to_sheet(run_a.uv_end)

                if settings.show_brackets:
                    prev_loop = run_a.loops[-1]
                    pt_before = isl.to_sheet(prev_loop[uv_layer].uv)
                    tangent = _boundary_tangent(pt_before, boundary_pt)
                    normal = Vector((-tangent.y, tangent.x))
                    tick_a = boundary_pt + normal * (tick_length_mm / 2)
                    tick_b = boundary_pt - normal * (tick_length_mm / 2)
                    ticks.append(
                        f'<line x1="{_fmt(tick_a.x)}" y1="{_fmt(tick_a.y)}" '
                        f'x2="{_fmt(tick_b.x)}" y2="{_fmt(tick_b.y)}"/>')

                boundary_angle = _boundary_angle_deg(run_a, run_b)
                if boundary_angle is None or n < 2:
                    continue
                bevel = core.compute_bevel(boundary_angle, settings.foam_thickness,
                                           settings.max_tilt_deg)
                prev_loop = run_a.loops[-1]
                pt_before = isl.to_sheet(prev_loop[uv_layer].uv)
                tangent = _boundary_tangent(pt_before, boundary_pt)
                inward = _inward_normal(tangent, boundary_pt, centroid_pt)
                half_angle = radians(min(abs(bevel.per_side_deg), 89.0))

                if settings.notch_shape == 'V_WEDGE':
                    half_width = dart_depth_mm * tan(half_angle)
                    p_left = boundary_pt - tangent * half_width
                    p_right = boundary_pt + tangent * half_width
                    apex = boundary_pt + inward * dart_depth_mm
                    pts_str = (f"{_fmt(p_left.x)},{_fmt(p_left.y)} "
                              f"{_fmt(apex.x)},{_fmt(apex.y)} "
                              f"{_fmt(p_right.x)},{_fmt(p_right.y)}")
                    darts.append(f'<polygon points="{pts_str}" fill="none"/>')
                else:  # SLIT
                    half_width = DART_SLIT_WIDTH_MM / 2
                    p1 = boundary_pt - tangent * half_width
                    p2 = boundary_pt + tangent * half_width
                    p3 = p2 + inward * dart_depth_mm
                    p4 = p1 + inward * dart_depth_mm
                    pts_str = " ".join(f"{_fmt(p.x)},{_fmt(p.y)}" for p in (p1, p2, p3, p4))
                    darts.append(f'<polygon points="{pts_str}"/>')

    def group(gid, css_class, elements):
        body = "\n".join(elements)
        return f'<g id="{gid}" class="{css_class}">\n{body}\n</g>'

    convex_hex = _rgba_to_hex(settings.convex_color)
    concave_hex = _rgba_to_hex(settings.concave_color)
    heat_hex = _rgba_to_hex(settings.heat_form_color)

    style = (
        "<style>"
        f"#outlines polygon{{fill:none;stroke:#000;"
        f"stroke-width:{_fmt(stroke_mm)};}}"
        f"#darts polygon{{stroke:#000;stroke-width:{_fmt(stroke_mm * 0.7)};}}"
        f"#ticks line{{stroke:#000;stroke-width:{_fmt(stroke_mm * 0.7)};}}"
        "#numbers text{text-anchor:middle;dominant-baseline:middle;fill:#000;}"
        f"#angles text{{text-anchor:middle;dominant-baseline:middle;}}"
        f"#angles text.convex{{fill:{convex_hex};}}"
        f"#angles text.concave{{fill:{concave_hex};}}"
        "#angles text.neutral{fill:#000;}"
        f"#heatform{{stroke:{heat_hex};stroke-dasharray:2,1;}}"
        f"#heatform text{{fill:{heat_hex};text-anchor:middle;}}"
        "</style>"
    )

    lines = [
        "<?xml version='1.0' encoding='UTF-8' standalone='no'?>",
        f"<svg xmlns='http://www.w3.org/2000/svg' "
        f"width='{total_width:.3f}mm' height='{total_height:.3f}mm' "
        f"viewBox='0 0 {total_width:.3f} {total_height:.3f}'>",
        style,
        group("outlines", "outlines", outlines),
        group("darts", "darts", darts),
        group("ticks", "ticks", ticks),
        group("angles", "angles", angles),
        group("numbers", "numbers", numbers),
        group("heatform", "heatform", heatform),
        "</svg>",
    ]
    return "\n".join(lines)


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
        obj = context.active_object
        mesh = obj.data
        owns_bm = obj.mode != 'EDIT'
        bm = bmesh.new() if owns_bm else bmesh.from_edit_mesh(mesh)
        if owns_bm:
            bm.from_mesh(mesh)
        try:
            uv_layer = bm.loops.layers.uv.active
            if uv_layer is None:
                self.report({'ERROR'}, "This mesh has no active UV map. Unwrap it first.")
                return {'CANCELLED'}

            settings = context.scene.foamcraft_drachis
            svg_text = generate_svg(bm, uv_layer, settings,
                                    context.scene.unit_settings.scale_length)
        finally:
            if owns_bm:
                bm.free()

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
