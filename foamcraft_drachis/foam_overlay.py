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

import logging
from math import acos, cos, degrees, radians, sin
from typing import NamedTuple

import blf
import bmesh
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import foam_angle_core as core
from . import foam_log

_log = foam_log.get_logger()

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
    # Length of the run along the real surface, in metres of world space. This
    # is a property of the sculpt, not of the unwrap or the viewport, so it is
    # what the "too short to cut" warning keys on: scaling the UVs or zooming
    # the editor must not change whether a cut is makeable.
    length_3d: float = 0.0


class _BevelMark(NamedTuple):
    """One run's label as the sheet prints it, mapped back to UV space.

    Plain floats and strings only, like _RunMark.
    """
    u: float
    v: float
    text: str
    # 'convex' / 'concave' / 'neutral' for a bevel label, 'heat' for a
    # heat-form note.
    tone: str
    valign: str
    # The arrow's five points, (u, v) each -- tail, tip, barb, tip, barb --
    # or () when the run has none.
    arrow: tuple
    # The run's two ends, for the on-screen "too short to label" test.
    au: float
    av: float
    bu: float
    bv: float
    # Degrees counter-clockwise as seen (foam_marks.Label.rotation_deg):
    # a bevel label is turned square to its cut line, its arrow glyph
    # pointing the blade's way. The UV editor shows the page the right way
    # up, so the page's angle is the screen's. Added 2026-11-17, defaulted.
    rotation_deg: float = 0.0


class _OverlayCache:
    """Computed overlay geometry, in UV space, plus the state it was built from."""

    __slots__ = ("signature", "runs", "island_labels", "stretch_islands",
                 "align_ticks", "align_labels", "bevel_marks",
                 "run_end_marks")

    def __init__(self):
        self.signature = None
        self.runs = []               # _RunMark, plain data only
        self.island_labels = []      # (u, v, text)
        self.stretch_islands = set()  # island keys needing heat forming
        # Alignment (registration) ticks, exactly the ones the exported
        # sheet prints, mapped back into UV space. Plain floats only.
        self.align_ticks = []        # (u0, v0, u1, v1), tail then tip
        self.align_labels = []       # (u, v, text)
        # Seam number, blade-tilt arrow and degrees per run, and the
        # mid-seam run-end marks: the sheet's own, from foam_marks.
        self.bevel_marks = []        # _BevelMark
        self.run_end_marks = []      # (u0, v0, u1, v1), on the line then in


_cache = _OverlayCache()
_draw_handle = None
# Craftability findings, refreshed by the operator rather than every draw --
# the assessment walks the whole mesh and is far too slow for a draw handler.
_findings = []


def _settings(context):
    return getattr(context.scene, "foamcraft_drachis", None)


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


# Settings the overlay never reads, and so need not rebuild for. Everything
# else on the settings group is in the signature -- walked from RNA rather
# than listed, because a hand-kept list is how Convex/Concave Colour and the
# colours came to be missing from it. tests/test_settings_refresh.py checks
# both directions: each name here exists, and changing any other setting
# changes the signature.
_SIGNATURE_EXEMPT = frozenset({
    # Objects the grease-pencil bake and the numbering operator use.
    "pattern_source", "gp_object", "origin_object",
    # Where the log is written; draws nothing.
    "log_level",
})


def _settings_key(s):
    """Every overlay-relevant setting's value, as plain hashable data."""
    out = []
    for prop in s.bl_rna.properties:
        name = prop.identifier
        if name in ("rna_type", "name") or name in _SIGNATURE_EXEMPT:
            continue
        value = getattr(s, name)
        if prop.type == 'FLOAT':
            value = (tuple(round(v, 6) for v in value)
                     if getattr(prop, "array_length", 0) else round(value, 6))
        elif prop.type in {'INT', 'BOOLEAN'} and \
                getattr(prop, "array_length", 0):
            value = tuple(value)
        elif prop.type == 'POINTER':
            value = getattr(value, "name", None)
        out.append((name, value))
    return tuple(out)


def _signature(pairs, s):
    per_object = tuple(
        (obj.name, len(bm.verts), len(bm.edges), len(bm.faces),
         _uv_fingerprint(bm, uv_layer),
         _selection_fingerprint(bm) if s.only_selected else 0,
         # The piece names it prints, and where it sits: a shared sheet's
         # scale and the mirror warning both follow the world matrix.
         getattr(obj, "foamcraft_drachis_prefix", ""),
         tuple(round(v, 6) for row in obj.matrix_world for v in row))
        for obj, bm, uv_layer in pairs
    )
    return (per_object, round(_scale_length(s), 6), _settings_key(s))


def _scale_length(s):
    """The scene's unit scale, which the exporter multiplies lengths by.

    Read off the settings' own scene rather than bpy.context, so the rebuild
    asks about the scene it is rebuilding for.
    """
    scene = getattr(s, "id_data", None)
    unit = getattr(scene, "unit_settings", None)
    return getattr(unit, "scale_length", 1.0) or 1.0


def _segments_touch(a, b, c, d, eps=1e-7):
    """True when segment a-b meets segment c-d. Plain (x, y) tuples."""
    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1, d2 = cross(c, d, a), cross(c, d, b)
    d3, d4 = cross(a, b, c), cross(a, b, d)
    return ((d1 <= eps and d2 >= -eps) or (d1 >= -eps and d2 <= eps)) and \
           ((d3 <= eps and d4 >= -eps) or (d3 >= -eps and d4 <= eps))


class _Model:
    """The sheet `foam_marks.build_marks` would build for one object, kept
    for the length of one rebuild so the screen can ask the model what it
    prints instead of deciding again.

    Nothing about *where* a mark goes is decided in this module. foam_marks
    decides it for the sheet, on the `_Sheet`/`_Island` objects `build_marks`
    itself constructs, and the overlay asks those same functions and maps
    their millimetres back to UV. A second implementation would agree with
    the first only until one of them changed, and then the screen would
    show marks the paper does not carry.

    The way back is derived from `sheet.to_sheet` itself (the transform is
    affine, so three probe points determine its inverse) rather than written
    out again, so it cannot drift from the forward transform either.

    Only Selected: the *decision* is always made on the whole mesh, because
    the export ignores selection and a walk over a subset of seams starts in
    different places -- the marks would move when the selection changed, and
    stop matching the sheet. Selection only *hides*.

    Holds BMesh elements, so it must not outlive `_rebuild`. Built from the
    timer's rebuild, never from a draw handler.
    """

    def __init__(self, part, sheet, s, shown_islands):
        from . import foam_marks as fm
        self.fm = fm
        self.uv_layer, self.s = part.item.uv_layer, s
        self.matrix = part.item.matrix
        self.shown_islands = shown_islands
        self.scale_length = _scale_length(s)

        # One sheet for every object being edited, exactly as the export
        # builds it (`foam_marks._layout_sheet`): its scale comes from all
        # of them, and so do the seam numbers and piece names, which carry
        # on from one object to the next. A sheet per object, as this used
        # to build, sized each object's labels and ticks by its own scale
        # and numbered every object from 1 -- island 8 on each of two
        # objects both read "8" on a shared UV editor.
        self.sheet = sheet
        self.islands = part.islands
        self.runs_by_island = part.runs_by_island
        self.ok = bool(self.islands)
        if not self.ok:
            return

        self.angle_font_mm = self.sheet.permille(s.angle_text_size,
                                                 s.overlay_scale)
        self.stroke_mm = self.sheet.permille(2, s.overlay_scale)

        # Invert the sheet transform from the transform itself.
        o = self.sheet.to_sheet(Vector((0.0, 0.0)))
        ex = self.sheet.to_sheet(Vector((1.0, 0.0))) - o
        ey = self.sheet.to_sheet(Vector((0.0, 1.0))) - o
        det = ex.x * ey.y - ex.y * ey.x
        if abs(det) < 1e-18:
            self.ok = False
            return

        def to_uv(x, y):
            dx, dy = x - o.x, y - o.y
            return ((dx * ey.y - dy * ey.x) / det,
                    (dy * ex.x - dx * ex.y) / det)
        self.to_uv = to_uv

        # Which of the model's islands (always whole-mesh) are on screen.
        if s.only_selected:
            shown_faces = {f.index for faces in shown_islands for f in faces}
            self.island_shown = [any(f.index in shown_faces for f in isl.faces)
                                 for isl in self.islands]
        else:
            self.island_shown = [True] * len(self.islands)

    def run_end_marks(self):
        """The sheet's run-end marks, as ((u0, v0, u1, v1), ...).

        `foam_marks._run_end_marks` is what `build_marks` draws them with:
        only where a run ends part way along a seam, never at a corner or a
        junction, from the cut line inward. Only Selected keeps the marks
        of pieces that are shown.
        """
        if not self.ok:
            return []
        to_uv, s = self.to_uv, self.s
        out = []
        for island_index, stroke in self.fm._run_end_marks(
                self.islands, self.uv_layer, self.runs_by_island, s,
                s.tick_length * self.scale_length * 1000.0,
                self.stroke_mm * 0.7):
            if not self.island_shown[island_index]:
                continue
            (x0, y0), (x1, y1) = stroke.points[0], stroke.points[-1]
            out.append(to_uv(x0, y0) + to_uv(x1, y1))
        return out

    def align_ticks(self):
        """The alignment ticks the export will print, as UV-space floats.

        Returns (ticks, labels): ((u0, v0, u1, v1), ...) and
        ((u, v, text), ...). Tick length needs no handling of its own: the
        sheet is one uniform scale, so a tick `tick_length` metres long on
        paper is that same physical length against the pieces on screen,
        at any zoom.

        Only Selected: a tick is kept when it touches the boundary of a
        shown piece.
        """
        if not self.ok:
            return [], []
        s, to_uv, uv_layer = self.s, self.to_uv, self.uv_layer
        strokes, labels = self.fm._alignment_ticks(
            self.islands, uv_layer, self.runs_by_island, s,
            s.tick_length * self.scale_length * 1000.0,
            self.angle_font_mm * 0.7, self.stroke_mm * 0.9,
            self.matrix)

        ticks = []
        for stroke in strokes:
            (x0, y0), (x1, y1) = stroke.points[0], stroke.points[-1]
            ticks.append(to_uv(x0, y0) + to_uv(x1, y1))
        marks = [to_uv(l.x, l.y) + (str(l.text),) for l in labels]

        if s.only_selected:
            boundary = []
            for faces in self.shown_islands:
                for face in faces:
                    for loop in face.loops:
                        if core.uv_boundary_edge(loop.edge, uv_layer):
                            a = loop[uv_layer].uv
                            b = loop.link_loop_next[uv_layer].uv
                            boundary.append(((a.x, a.y), (b.x, b.y)))
            keep = [any(_segments_touch(t[0:2], t[2:4], a, b)
                        for a, b in boundary) for t in ticks]
            # A number belongs to the dash it was emitted with; when the two
            # lists do not pair up there is no telling which, so none are
            # shown rather than some on the wrong piece.
            marks = ([m for m, k in zip(marks, keep) if k]
                     if len(marks) == len(ticks) else [])
            ticks = [t for t, k in zip(ticks, keep) if k]

        return ticks, marks

    def bevel_marks(self):
        """Seam number, arrow and degrees per run, as the sheet prints them.

        `foam_marks._bevel_marks` is the function `build_marks` itself
        calls; this maps its labels and arrows back to UV. Only Selected
        keeps the marks of pieces that are shown.
        """
        if not self.ok:
            return []
        to_uv = self.to_uv
        out = []
        for mark in self.fm._bevel_marks(
                self.islands, self.uv_layer, self.runs_by_island, self.s,
                self.angle_font_mm, self.stroke_mm):
            if mark.label is None or not self.island_shown[mark.island_index]:
                continue
            label = mark.label
            tone = 'heat' if label.kind == 'heat' else label.tone
            arrow = (tuple(to_uv(x, y) for x, y in mark.arrow.points)
                     if mark.arrow is not None else ())
            (ax, ay), (bx, by) = mark.run_ends
            out.append(_BevelMark(
                *to_uv(label.x, label.y), str(label.text), tone,
                label.valign, arrow, *to_uv(ax, ay), *to_uv(bx, by),
                float(getattr(label, 'rotation_deg', 0.0) or 0.0)))
        return out


def _island_uv_centroid(faces, uv_layer):
    """Where to put this island's piece number, in UV space.

    Not the centroid, despite the name kept for its callers. The centroid of
    a ring lies in the hole and the centroid of a crescent lies outside the
    shape, so the number was drawn on no piece at all -- plainly visible on a
    panelled sphere, where every island is an annulus and all eight numbers
    floated in their own holes.

    The exporter already places numbers at the pole of inaccessibility, which
    is inside by construction. The overlay has to agree with it: the sheet is
    supposed to be a render of what the UV editor shows, and a number in a
    different place on each is the most confusing kind of disagreement.
    """
    cycles = core.island_boundary_cycles(faces, uv_layer)
    rings = core.order_rings_outer_first(
        [[(l[uv_layer].uv.x, l[uv_layer].uv.y) for l in cyc]
         for cyc in cycles])
    if rings:
        (px, py), _room = core.pole_of_inaccessibility(rings)
        if px is not None:
            return Vector((px, py))

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
    align_ticks = []
    align_labels = []
    bevel_marks = []
    run_end_marks = []

    from . import foam_marks as fm
    # `_seam_chains` and the per-piece de-duplication key on vertex and edge
    # indices, which the export path has fresh; refreshed here at the
    # timer's safe point, never from a draw.
    for _obj, bm, _uv in pairs:
        bm.verts.index_update()
        bm.edges.index_update()
        bm.faces.index_update()
    # The sheet the export would print for these same objects: one
    # transform, pieces and seams numbered on from one object to the next.
    sheet, parts = fm._layout_sheet(
        [fm.SheetItem(bm, uv_layer, obj.matrix_world.copy(),
                      fm.object_prefix(
                          obj.name,
                          getattr(obj, "foamcraft_drachis_prefix", "")),
                      obj.name)
         for obj, bm, uv_layer in pairs], s, _scale_length(s))
    part_of = {part.item.name: part for part in parts}

    for obj, bm, uv_layer in pairs:
        part = part_of[obj.name]
        islands = core.uv_islands(bm, uv_layer, s.only_selected)

        # Kept as a list, not consumed as a generator: the alignment ticks
        # below are decided from these same runs.
        object_runs = list(core.iter_boundary_dart_segments(
                bm, uv_layer, s.max_chord_deviation, s.max_angle_variation_deg,
                only_selected=s.only_selected, islands=islands,
                corner_angle_deg=s.corner_angle_deg,
                matrix=obj.matrix_world,
                dart_wedge_deg=s.dart_wedge_deg,
                max_darts_per_run=s.max_darts_per_run))

        # What the sheet prints, asked of the model once per object.
        if s.show_align_ticks or s.show_brackets or s.show_heat_form:
            model = _Model(part, sheet, s, islands)
            if s.show_align_ticks:
                ticks, tick_labels = model.align_ticks()
                align_ticks.extend(ticks)
                align_labels.extend(tick_labels)
            bevel_marks.extend(model.bevel_marks())
            if s.show_brackets:
                run_end_marks.extend(model.run_end_marks())
            # It holds BMesh elements; nothing of it may outlive this call.
            del model

        for run in object_runs:
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
            for slot, k in enumerate(run.dart_indices):
                pt = run.uv_points[k]
                nrm = run.uv_normals[k]
                nxt = run.uv_points[min(k + 1, len(run.uv_points) - 1)]
                prv = run.uv_points[max(k - 1, 0)]
                tangent = nxt - prv
                if tangent.length > 1e-12:
                    tangent = tangent.normalized()
                # Sign of the curvature this dart absorbs. Positive domes and
                # wants a wedge cut out; negative saddles and wants the sheet
                # to spread instead. Carried through so the two are never drawn
                # as the same instruction.
                load = (run.dart_loads[slot]
                        if slot < len(run.dart_loads) else 0.0)
                dart_marks.append((pt.x, pt.y, tangent.x, tangent.y,
                                   nrm.x, nrm.y, load))
            darts = tuple(dart_marks)

            if heat:
                heat_islands.add((obj.name, run.island_index))

            runs.append(_RunMark(
                run.uv_start.x, run.uv_start.y, run.uv_end.x, run.uv_end.y,
                run.mean_angle_deg, per_side, run.has_dihedral, heat, ox, oy,
                outline, darts, run.length_3d))

        # Piece names are the sheet's own (`_layout_sheet`): the numbering
        # operator's attribute when it has been run, else island order, in
        # either case continued from the objects before this one. Found by
        # face, since Only Selected shows a subset of the whole-mesh islands
        # the sheet numbered.
        name_of_face = {f.index: isl.label
                        for isl in part.islands for f in isl.faces}
        stretch = core.uv_stretch_map(bm, uv_layer)

        for index, faces in enumerate(islands):
            label = next((name_of_face[f.index] for f in faces
                          if f.index in name_of_face), str(index + 1))
            centroid = _island_uv_centroid(faces, uv_layer)
            key = (obj.name, index)
            labels.append((centroid.x, centroid.y, label, key))

            # An island can be fine at its darted edges and still refuse to lie
            # flat through the middle. That is a heat-forming job, not a dart,
            # so it is flagged from areal distortion, not from a boundary angle.
            vals = [stretch.get(f, 0.0) for f in faces]
            if vals and sum(vals) / len(vals) > s.stretch_tolerance:
                flagged.add(key)

    _cache.runs = runs
    _cache.island_labels = labels
    _cache.align_ticks = align_ticks
    _cache.align_labels = align_labels
    _cache.bevel_marks = bevel_marks
    _cache.run_end_marks = run_end_marks
    # Heat forming is a property of a piece, not of a single cut: either some
    # seam on it is too steep to saw, or the piece will not lie flat at all.
    # Marking every affected run cluttered the sheet without telling the user
    # anything the piece-level flag does not.
    _cache.stretch_islands = flagged | heat_islands

    # DEBUG, not INFO: the rebuild runs off a timer, so this is one line per
    # tick whenever the mesh is dirty. Switch it on from the sidebar when a
    # number needs checking, and leave it off otherwise.
    if _log.isEnabledFor(logging.DEBUG):
        lengths = [r.length_3d for r in runs]
        short = sum(1 for r in runs
                    if r.has_dihedral and r.length_3d < s.min_run_length)
        _log.debug(foam_log.summary(
            "rebuild",
            objects=len(pairs),
            runs=len(runs),
            islands=len(labels),
            bevelled=sum(1 for r in runs if r.has_dihedral),
            flagged_short=short,
            heat_islands=len(_cache.stretch_islands),
            min_len=min(lengths) if lengths else 0.0,
            max_len=max(lengths) if lengths else 0.0,
            min_run_length=s.min_run_length))


_dirty = True
# Set when a drag has just ended and the model has not caught up yet. The
# overlay stays hidden until it has, so the frame after release never shows
# marks at the positions the geometry has already left.
_await_refresh = False
_was_modal = False
# Seconds between checks for pending work. Short enough that the marks
# reappear promptly when a drag ends, long enough that a busy mesh is not
# rebuilt on every mouse move.
_TIMER_INTERVAL = 0.05


def mark_dirty():
    global _dirty
    _dirty = True


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph):
    # Only raises a flag. Anything heavier here runs inside the depsgraph
    # evaluation of a modal transform, which is exactly the unsafe moment.
    mark_dirty()


# Only these count as "mid-drag". The UV editor keeps view navigation
# operators such as IMAGE_OT_view_pan registered as modal handlers at all
# times, so testing for any modal operator reports a drag permanently -- which
# stopped the model ever rebuilding and the overlay ever drawing.
_TRANSFORM_PREFIX = "TRANSFORM_OT_"


def _modal_running(context):
    """True while a transform is actually dragging.

    A transform mutates UVs continuously while leaving the mesh unchanged, so
    there is nothing to re-derive until it ends -- and reading the mesh during
    it is what crashed Blender.
    """
    window = getattr(context, "window", None)
    if window is None:
        return False
    return any(op.bl_idname.startswith(_TRANSFORM_PREFIX)
               for op in getattr(window, "modal_operators", ()))


def _timer_tick():
    """Rebuild pending work at a safe point in the main loop.

    Timers fire between operator executions, not inside one, which is the
    closest Python equivalent to the sync point where Blender extracts its own
    draw batches.
    """
    global _dirty, _await_refresh, _was_modal
    context = bpy.context
    s = _settings(context)
    if s is None or not s.show_overlay:
        return _TIMER_INTERVAL

    modal = _modal_running(context)
    if modal:
        _was_modal = True
        return _TIMER_INTERVAL
    if _was_modal:
        # The drag just ended. Whatever is cached describes where the UVs used
        # to be, so hold the overlay back until it has been rebuilt.
        _was_modal = False
        _await_refresh = True
        _dirty = True

    if not _dirty:
        return _TIMER_INTERVAL
    try:
        _ensure_cache(context, s)
        _dirty = False
        _await_refresh = False
    except (ReferenceError, AttributeError, RuntimeError, IndexError):
        _cache.signature = None
        _cache.runs = []
        _cache.island_labels = []
        _cache.align_ticks = []
        _cache.align_labels = []
        _cache.bevel_marks = []
        _cache.run_end_marks = []
        _dirty = False
        _await_refresh = False
    for area in getattr(context.screen, "areas", ()):
        if area.type == 'IMAGE_EDITOR':
            area.tag_redraw()
    return _TIMER_INTERVAL


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
    # Blender's own UV overlay never walks BMesh while drawing -- it submits
    # batches built earlier, at a controlled sync point. Doing mesh work here
    # instead is what corrupted geometry and crashed Blender during a modal
    # transform. This callback now only renders what the timer prepared.
    if not _cache.runs and not _cache.island_labels:
        return

    # Nothing is drawn while a drag is in progress. The cache cannot be
    # refreshed mid-transform without reading the mesh at the moment that
    # crashes, so the alternative is marks frozen where the geometry used to
    # be -- an angle sitting over the wrong piece is worse than no angle at
    # all. Blender still draws the islands themselves, so what is being moved
    # stays visible; the marks return, correctly placed, on release.
    # Never draw known-stale marks. The depsgraph raises the dirty flag the
    # moment UVs change -- including on the transform's final update -- so the
    # first draw after a release finds it set and holds off until the timer
    # has rebuilt. Without this the frame right after release showed the marks
    # where the geometry used to be.
    if _modal_running(context) or _dirty or _await_refresh:
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
            # The ends used to be drawn as a T, a bar crossing the bracket
            # both ways at every run end -- so at every corner of every
            # piece, crossing its neighbour's. The maker: "There is no need
            # for alignment marks at the corners." Where a run ends part way
            # along a seam the sheet's own run-end mark is drawn instead
            # (_cache.run_end_marks, below), and at a corner nothing is.

        # Relief cuts are a different mark from the angle guide: they are the
        # cuts themselves. Drawn as a T lying on the boundary -- the bar runs
        # along the edge, the stem goes into the material and never out of it.
        # Collected per vertex rather than per run so a vertex shared by two
        # runs gets one mark, not two crossing into an X.
        # A run bending less than the minimum cut angle gets no darts, because
        # it gets no cut: the foam simply flexes through it. Relief marks on a
        # seam that is never opened are instructions to damage the piece.
        if s.show_darts and run.has_dihedral \
                and abs(run.angle_deg) >= s.min_angle:
            for u, v, tu, tv, nu, nv, load in run.darts:
                key = (round(u, 6), round(v, 6))
                if key in dart_seen:
                    continue
                if load < 0.0 and not s.allow_gores:
                    # A gore mark here would contradict the advisor, which is
                    # blocking this piece as unmakeable from flat stock. The
                    # blocker crosses mark the spot instead; drawing a "set
                    # material in here" instruction as well would be telling the
                    # maker to build something the tool has just refused.
                    continue
                # A relief cut is coloured by what it does to the material, not
                # by the run's fold direction: a piece can dome along one part
                # of a boundary and saddle along another, and the two need
                # opposite work.
                dart_color = (tuple(s.concave_color) if load < 0.0
                              else tuple(s.convex_color))
                dart_seen[key] = (tu, tv, nu, nv, dart_color, load)

        # Two separate questions, deliberately kept apart.
        #
        # "Is there room on screen for the number?" is a legibility question.
        # It depends on zoom and on how the UVs happen to be scaled, and the
        # answer changes as you navigate. Failing it hides the number, and
        # nothing more -- zoom in and it comes back.
        #
        # "Is this run too short to cut?" is a craftability question, and the
        # warning mark belongs to it. It is measured along the real surface in
        # metres, so it is the same answer at any zoom and at any UV scale.
        # These were previously one test, which made the warning flicker on and
        # off while zooming and moved it between edges when the UVs were scaled
        # 10x or 0.1x -- reporting a property of the view as if it were a
        # property of the work.
        run_px = ((bbx - bax) ** 2 + (bby - bay) ** 2) ** 0.5
        labelled = run.has_dihedral and abs(run.angle_deg) >= s.min_angle
        if labelled and run.length_3d < s.min_run_length:
            mid = len(bracket) // 2
            if len(bracket) % 2 == 0:
                a_pt, b_pt = bracket[mid - 1], bracket[mid]
                mx, my = (a_pt[0] + b_pt[0]) * 0.5, (a_pt[1] + b_pt[1]) * 0.5
                nx, ny = (a_pt[4] + b_pt[4]) * 0.5, (a_pt[5] + b_pt[5]) * 0.5
            else:
                mx, my, _, _, nx, ny = bracket[mid]
            nlen = (nx * nx + ny * ny) ** 0.5 or 1.0
            # The bevel label is no longer written out here in the margin
            # -- it is inside the piece, where the sheet prints it -- so the
            # warning has this anchor to itself.
            hidden.append((mx, my, nx / nlen, ny / nlen, False))

    for (u, v), (tu, tv, nu, nv, color, load) in dart_seen.items():
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
        tipx, tipy = px0 - ndx * stem, py0 - ndy * stem
        bucket.extend([(px0, py0), (tipx, tipy)])
        if load < 0.0:
            # Not a dart at all -- a gore.
            #
            # A dart removes material where the flat sheet has too much. Here
            # the sheet has too *little*: the piece saddles, and its boundary
            # needs to be longer than flat stock provides. Nothing can be cut
            # away to achieve that. The fixes are to add material (glue a wedge
            # into a slit), to stretch it (heat forming, flagged separately), or
            # to move the seam onto flatter ground.
            #
            # A V-groove scored partway through does not help either, however
            # natural the idea: a groove is a hinge, and hinging is free --
            # flat foam already wraps a cylinder with no cuts at all. A groove
            # changes how the sheet *bends*, not how much material it has.
            #
            # Drawn with a bar across the inner tip, so it differs in shape as
            # well as colour. The two call for opposite work, and a colour-blind
            # reader, a greyscale print or a faded marker must not be able to
            # confuse "cut this out" with "fill this in".
            bar = tick * 0.45
            bucket.extend([(tipx - tdx * bar, tipy - tdy * bar),
                           (tipx + tdx * bar, tipy + tdy * bar)])

    # Blockers, marked at the exact spot rather than only counted in the panel.
    # A finding that says "island 42 cannot be made" leaves the user to hunt for
    # which part of it is the problem; a saddle usually occupies one stretch of
    # a boundary and the rest of the piece is fine.
    if s.show_blockers and _findings:
        cross = tick * 0.9
        bucket = segments.setdefault(tuple(s.blocker_color), [])
        for finding in _findings:
            if finding.severity != 'blocker':
                continue
            for u, v in getattr(finding, "sites", ()):
                x, y = view.view_to_region(u, v, clip=False)
                if not all(abs(c) < 1e6 for c in (x, y)):
                    continue
                bucket.extend([(x - cross, y - cross), (x + cross, y + cross),
                               (x - cross, y + cross), (x + cross, y - cross)])

    # Alignment ticks: the registration marks the exported sheet prints,
    # cached in UV space by the rebuild. Both ends go through the view
    # transform, so a tick is its real length against the piece at any zoom
    # -- it is a pen mark of a fixed physical size, not an annotation, and
    # deliberately does not follow mark_size or overlay_scale. The sheet
    # prints them black on white; here they take the piece-number colour,
    # which is the overlay's neutral ink.
    tick_marks = []
    if s.show_align_ticks and _cache.align_ticks:
        bucket = segments.setdefault(tuple(s.island_id_color), [])
        for u0, v0, u1, v1 in _cache.align_ticks:
            x0, y0 = view.view_to_region(u0, v0, clip=False)
            x1, y1 = view.view_to_region(u1, v1, clip=False)
            if not all(abs(c) < 1e6 for c in (x0, y0, x1, y1)):
                continue
            bucket.extend([(x0, y0), (x1, y1)])
        if s.tick_numbers:
            for u, v, text in _cache.align_labels:
                x, y = view.view_to_region(u, v, clip=False)
                if all(abs(c) < 1e6 for c in (x, y)):
                    tick_marks.append((x, y, text))

    # Run-end marks, where one bevel hands over to the next part way along
    # a seam: the sheet's own, from the line inward. None at corners.
    if s.show_brackets and _cache.run_end_marks:
        bucket = segments.setdefault(tuple(s.island_id_color), [])
        for u0, v0, u1, v1 in _cache.run_end_marks:
            x0, y0 = view.view_to_region(u0, v0, clip=False)
            x1, y1 = view.view_to_region(u1, v1, clip=False)
            if all(abs(c) < 1e6 for c in (x0, y0, x1, y1)):
                bucket.extend([(x0, y0), (x1, y1)])

    # Bevel labels: seam number, blade-tilt arrow and degrees, at the place
    # and with the text the exported sheet prints -- cached in UV space by
    # the rebuild from foam_marks._bevel_marks, so screen and paper cannot
    # disagree about what a seam is called or which way to tilt.
    #
    # Hidden where the run is too short on screen to carry a number, the
    # same legibility test the old margin labels used (zoom in and they
    # come back); that is about the view, never about the pattern.
    neutral = convex if not s.show_dart_colors else tuple(s.island_id_color)
    tones = {'convex': convex, 'concave': concave, 'neutral': neutral,
             'heat': heat}
    for mark in _cache.bevel_marks:
        ax, ay = view.view_to_region(mark.au, mark.av, clip=False)
        bx, by = view.view_to_region(mark.bu, mark.bv, clip=False)
        x, y = view.view_to_region(mark.u, mark.v, clip=False)
        if not all(abs(c) < 1e6 for c in (ax, ay, bx, by, x, y)):
            continue
        if ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 < s.min_label_run_px:
            continue
        color = tones.get(mark.tone, neutral)
        labels.append((x, y, mark.text, color, mark.valign,
                       mark.rotation_deg))
        if mark.arrow:
            pts = [view.view_to_region(u, v, clip=False)
                   for u, v in mark.arrow]
            bucket = segments.setdefault(color, [])
            for p, q in zip(pts, pts[1:]):
                bucket.extend([p, q])

    _draw_lines(segments, max(1.0, scale))

    if tick_marks:
        # 0.7 of the angle text, the same ratio the sheet uses.
        blf.size(font_id, max(6, int(angle_px * 0.7)))
        blf.color(font_id, *tuple(s.island_id_color))
        for x, y, text in tick_marks:
            width, height = blf.dimensions(font_id, text)
            blf.position(font_id, x - width / 2, y - height / 2, 0)
            blf.draw(font_id, text)
        blf.size(font_id, angle_px)

    # Warn where a run is too short to cut cleanly. This is about the piece,
    # not about the view: the run is a few millimetres of real edge, and no
    # amount of zooming will make it sawable.
    if hidden:
        warn_color = tuple(s.suppressed_color)
        blf.size(font_id, max(8, int(angle_px * 0.9)))
        for ax0, ay0, nx, ny, _numbered in hidden:
            width, height = blf.dimensions(font_id, "!")
            clear = abs(nx) * width * 0.5 + abs(ny) * height * 0.5 + tick * 0.35
            blf.position(font_id, ax0 + nx * clear - width / 2,
                         ay0 + ny * clear - height / 2, 0)
            blf.color(font_id, *warn_color)
            blf.draw(font_id, "!")
        blf.size(font_id, angle_px)

    for x, y, text, color, valign, turn in labels:
        width, height = blf.dimensions(font_id, text)
        # Where the sheet prints it. 'middle' centres the text on the point;
        # 'baseline' (the heat-form note) sits its baseline there.
        drop = height / 2 if valign == 'middle' else 0.0
        blf.color(font_id, *color)
        if not turn:
            blf.position(font_id, x - width / 2, y - drop, 0)
            blf.draw(font_id, text)
            continue
        # Turned as the sheet turns it, about its own centre: the pen starts
        # half the width back along the reading direction and `drop` down
        # along the text's own "down". Plain cached floats; nothing read.
        a = radians(turn)
        ca, sa = cos(a), sin(a)
        blf.enable(font_id, blf.ROTATION)
        try:
            blf.rotation(font_id, a)
            blf.position(font_id, x - ca * width / 2 + sa * drop,
                         y - sa * width / 2 - ca * drop, 0)
            blf.draw(font_id, text)
        finally:
            blf.disable(font_id, blf.ROTATION)

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


class FOAMCRAFT_DRACHIS_OT_refresh_overlay(bpy.types.Operator):
    """Recompute the cut runs shown in the UV editor"""

    bl_idname = "foamcraft_drachis.refresh_overlay"
    bl_label = "Refresh Overlay"

    def execute(self, context):
        _cache.signature = None
        for area in context.screen.areas:
            area.tag_redraw()
        return {'FINISHED'}


class FOAMCRAFT_DRACHIS_OT_check_craftability(bpy.types.Operator):
    """Check whether this mesh and unwrap can actually be built from foam"""

    bl_idname = "foamcraft_drachis.check_craftability"
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
            stretch_tolerance=s.stretch_tolerance,
            # Same threshold and same world matrix the overlay's warning mark
            # uses, so a run flagged with a "!" in the UV editor is the same
            # run counted in the fragmented-runs finding here.
            min_run_length=s.min_run_length,
            matrix=obj.matrix_world,
            allow_gores=s.allow_gores)
        blockers = sum(1 for f in _findings if f.severity == 'blocker')
        warnings = sum(1 for f in _findings if f.severity == 'warn')
        # The status bar shows two totals and then goes away. The log keeps
        # every finding's code, so two unwraps can be compared after the fact
        # rather than remembered.
        _log.info(foam_log.summary(
            "craftability", object=obj.name,
            blockers=blockers, warnings=warnings, findings=len(_findings),
            min_run_length=s.min_run_length,
            codes=",".join(f.code for f in _findings) or "none"))
        self.report({'INFO'},
                    f"{blockers} blocker(s), {warnings} warning(s)")
        for area in context.screen.areas:
            area.tag_redraw()
        return {'FINISHED'}


_SEVERITY_ICON = {'blocker': 'CANCEL', 'warn': 'ERROR',
                  'info': 'INFO', 'ok': 'CHECKMARK'}


class IMAGE_PT_foamcraft_drachis(bpy.types.Panel):
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
        box.operator("foamcraft_drachis.estimate_tolerances", icon='SHADERFX')
        box.label(text=f"{len(_cache.runs)} run(s) cached")
        box.operator(FOAMCRAFT_DRACHIS_OT_refresh_overlay.bl_idname, icon='FILE_REFRESH')

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


class IMAGE_PT_foamcraft_drachis_display(bpy.types.Panel):
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "FoamCraft"
    bl_label = "Display"
    bl_parent_id = "IMAGE_PT_foamcraft_drachis"
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
        layout.prop(s, "show_darts")
        layout.prop(s, "allow_gores")
        layout.prop(s, "show_blockers")
        col = layout.column(align=True)
        col.enabled = s.show_darts
        col.prop(s, "dart_wedge_deg")
        col.prop(s, "max_darts_per_run")
        layout.prop(s, "min_label_run_px")
        layout.prop(s, "min_run_length")
        layout.separator()
        layout.prop(s, "log_level")
        # Show where the log actually is. Otherwise the only way to find it is
        # to already know, which makes the log useless to anyone debugging.
        col = layout.column(align=True)
        col.enabled = False
        col.label(text=foam_log.log_path(), icon='TEXT')
        layout.prop(s, "convex_color")
        layout.prop(s, "concave_color")
        layout.prop(s, "blocker_color")
        layout.prop(s, "suppressed_color")
        layout.prop(s, "heat_form_color")
        layout.prop(s, "island_id_color")


class IMAGE_PT_foamcraft_drachis_advice(bpy.types.Panel):
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "FoamCraft"
    bl_label = "Craftability"
    bl_parent_id = "IMAGE_PT_foamcraft_drachis"

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
        layout.operator(FOAMCRAFT_DRACHIS_OT_check_craftability.bl_idname, icon='VIEWZOOM')

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


def _operator_registered(idname):
    """True when "category.name" is a registered operator.

    bpy.ops attribute access is lazy, so hasattr() is True for any name
    (measured on 5.2: a bogus idname passes). get_rna_type() is what actually
    resolves it and raises KeyError for an unregistered operator. Reads no
    mesh data.
    """
    category, _, name = idname.partition(".")
    try:
        getattr(getattr(bpy.ops, category), name).get_rna_type()
    except (AttributeError, KeyError, TypeError):
        return False
    return True


class IMAGE_PT_foamcraft_drachis_export(bpy.types.Panel):
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "FoamCraft"
    bl_label = "Print / Export"
    bl_parent_id = "IMAGE_PT_foamcraft_drachis"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = _settings(context)
        if s is None:
            layout.label(text="FoamCraft settings unavailable", icon='ERROR')
            return

        box = layout.box()
        box.label(text="Alignment Marks")
        box.prop(s, "show_align_ticks")
        col = box.column()
        col.enabled = s.show_align_ticks
        col.prop(s, "tick_mode")
        if s.tick_mode == 'BY_DISTANCE':
            col.prop(s, "tick_spacing")
        else:
            col.prop(s, "ticks_per_run")
        col.prop(s, "tick_length")
        col.prop(s, "tick_numbers")

        box = layout.box()
        box.label(text="Bevels")
        box.prop(s, "show_seam_numbers")
        box.prop(s, "pattern_surface")
        box.prop(s, "min_label_mm")

        box = layout.box()
        box.label(text="Pieces")
        box.prop(s, "piece_label_style")
        obj = context.active_object
        if obj is not None and obj.type == 'MESH':
            row = box.row()
            row.enabled = s.piece_label_style == 'NAME'
            row.prop(obj, "foamcraft_drachis_prefix")
            if not obj.foamcraft_drachis_prefix:
                # Say what an empty field prints, rather than leave the
                # maker to discover it on paper.
                from . import foam_marks as fm
                note = box.row()
                note.enabled = False
                note.label(text="Uses %s" % fm.object_prefix(obj.name, ""))

        box = layout.box()
        box.label(text="Sheet")
        box.prop(s, "island_fill")
        box.prop(s, "show_scale_ruler")

        for idname, text, icon in (
                ("foamcraft_drachis.export_pattern_image", "Export Image", 'IMAGE_DATA'),
                ("foamcraft_drachis.export_panel_pattern", "Export SVG", 'EXPORT')):
            if _operator_registered(idname):
                layout.operator(idname, text=text, icon=icon)
            else:
                row = layout.row()
                row.enabled = False
                row.label(text=f"{text} (unavailable)", icon=icon)


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
    FOAMCRAFT_DRACHIS_OT_refresh_overlay,
    FOAMCRAFT_DRACHIS_OT_check_craftability,
    IMAGE_PT_foamcraft_drachis,
    IMAGE_PT_foamcraft_drachis_display,
    IMAGE_PT_foamcraft_drachis_advice,
    IMAGE_PT_foamcraft_drachis_export,
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

    handlers = bpy.app.handlers.depsgraph_update_post
    for stale in [h for h in handlers if h.__name__ == "_on_depsgraph_update"]:
        handlers.remove(stale)
    handlers.append(_on_depsgraph_update)

    if not bpy.app.timers.is_registered(_timer_tick):
        bpy.app.timers.register(_timer_tick, first_interval=_TIMER_INTERVAL,
                                persistent=True)
    mark_dirty()


def unregister():
    global _findings
    if bpy.app.timers.is_registered(_timer_tick):
        bpy.app.timers.unregister(_timer_tick)
    handlers = bpy.app.handlers.depsgraph_update_post
    for stale in [h for h in handlers if h.__name__ == "_on_depsgraph_update"]:
        handlers.remove(stale)
    _drop_existing_handler()
    _findings = []
    _cache.signature = None
    _cache.runs = []
    _cache.island_labels = []
    _cache.align_ticks = []
    _cache.align_labels = []
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
