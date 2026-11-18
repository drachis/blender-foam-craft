"""Live UV Editor overlay for the foam workflow.

Draws the cut plan the way the hand-drafted patterns do it: each island's
boundary is divided into runs by perpendicular tick marks, and each run carries
the bevel angle for that single saw pass -- the |--- 45° ---| convention. Piece
numbers sit at island centres.

This is deliberately not the old paper overlay (``uv_fold_angle_overlay``,
removed 2026-11-17). That one labelled every mesh edge, which is correct for paper (every polygon edge is a
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
    # The label's printed size in sheet mm, after `foam_layout.layout_labels`
    # has shrunk its piece to fit. Drawn at `angle_px * size_mm /
    # _cache.angle_font_mm`: the same mapping the overlay already uses for
    # the sheet's full angle size, so a label shrunk on paper is shrunk on
    # screen by the same factor. 0 means "the full size".
    size_mm: float = 0.0
    # ((u, v), (u, v)): the 'leader' stroke the layout drew from just off
    # this label to its cut line, when the label had to go outside its
    # piece; () otherwise. Hidden with its label.
    leader: tuple = ()
    # `BevelAnchor.piece` -- the island's sheet-wide ordinal -- or -1 for a
    # heat-form note, which has no anchor.
    piece: int = -1


class _OverlayCache:
    """Computed overlay geometry, in UV space, plus the state it was built from."""

    __slots__ = ("signature", "runs", "island_labels", "stretch_islands",
                 "align_ticks", "align_labels", "bevel_marks",
                 "run_end_marks", "angle_font_mm", "number_font_mm")

    def __init__(self):
        self.signature = None
        self.runs = []               # _RunMark, plain data only
        # (u, v, text, key, size_mm): each piece name where and as big as
        # the sheet prints it (`foam_layout.fit_piece_names`); size_mm 0
        # means "the set size" (only when the model printed no name).
        self.island_labels = []
        self.stretch_islands = set()  # island keys needing heat forming
        # Alignment (registration) ticks, exactly the ones the exported
        # sheet prints, mapped back into UV space. Plain floats only.
        self.align_ticks = []        # (u0, v0, u1, v1), tail then tip
        self.align_labels = []       # (u, v, text)
        # Seam number, blade-tilt arrow and degrees per run, and the
        # mid-seam run-end marks: the sheet's own, from foam_marks.
        self.bevel_marks = []        # _BevelMark
        self.run_end_marks = []      # (u0, v0, u1, v1), on the line then in
        # The sheet's full angle-label size in mm, which `_BevelMark.size_mm`
        # is measured against when drawing.
        self.angle_font_mm = 0.0
        # The sheet's set Number Size in mm, which a cached name's size_mm
        # is drawn against, as bevel labels are against angle_font_mm.
        self.number_font_mm = 0.0


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
         tuple(round(v, 6) for row in obj.matrix_world for v in row),
         # A Mirror modifier's axes, mirror object and merge distance
         # decide which open edges are seams (foam_marks.is_join).
         tuple((tuple(round(c, 6) for c in p.point),
                tuple(round(c, 6) for c in p.normal), round(p.tolerance, 9))
               for p in _mirror_planes(obj)))
        for obj, bm, uv_layer in pairs
    )
    return (per_object, round(_scale_length(s), 6), _settings_key(s))


def _mirror_planes(obj):
    from . import foam_marks as fm
    return fm.mirror_planes(obj)


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

        self.to_uv = _sheet_to_uv(self.sheet)
        if self.to_uv is None:
            self.ok = False
            return

        # Which of the model's islands (always whole-mesh) are on screen.
        self.island_shown = _island_shown(self.islands, s, shown_islands)

    def run_end_marks(self):
        """The sheet's run-end marks, each a closed ring of UV points:
        (((u, v), ...), ...).

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
            out.append(tuple(tuple(to_uv(x, y)) for x, y in stroke.points))
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
            self.sheet.permille(s.tick_number_size, s.overlay_scale),
            self.stroke_mm * 0.9,
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


def _sheet_to_uv(sheet):
    """The inverse of `sheet.to_sheet` as a plain function of (x, y) mm, or
    None when the transform is degenerate.

    Derived from the transform itself (it is affine, so three probe points
    determine it) rather than written out again, so it cannot drift from
    the forward transform.
    """
    o = sheet.to_sheet(Vector((0.0, 0.0)))
    ex = sheet.to_sheet(Vector((1.0, 0.0))) - o
    ey = sheet.to_sheet(Vector((0.0, 1.0))) - o
    ox, oy = float(o.x), float(o.y)
    exx, exy, eyx, eyy = float(ex.x), float(ex.y), float(ey.x), float(ey.y)
    det = exx * eyy - exy * eyx
    if abs(det) < 1e-18:
        return None

    def to_uv(x, y):
        dx, dy = x - ox, y - oy
        return ((dx * eyy - dy * eyx) / det,
                (dy * exx - dx * exy) / det)
    return to_uv


def _island_shown(islands, s, shown_islands):
    """Per whole-mesh island, whether Only Selected leaves it on screen."""
    if not s.only_selected:
        return [True] * len(islands)
    shown_faces = {f.index for faces in shown_islands for f in faces}
    return [any(f.index in shown_faces for f in isl.faces) for isl in islands]


def _laid_out_bevel_marks(sheet, parts, s, shown_ordinals, to_uv,
                          names_out=None):
    """The bevel labels and leaders the sheet prints, mapped back to UV.

    Asks `foam_marks.marks_for_sheet` -- the export's own `Marks`, built on
    this same sheet and laid out by `foam_layout.layout_labels` -- so every
    label is where, as big, and as turned as the paper has it, and a label
    the layout moved outside its piece brings its leader. Nothing about a
    label is decided here.

    'angle' labels are kept when their `anchor.piece` (the sheet-wide
    island ordinal) is in `shown_ordinals`. Heat-form notes have no anchor
    and are not moved by the layout, so they are taken from the same build's
    `BevelMark`s, which name their island.

    `names_out`, when a list, receives the printed piece names as
    `foam_marks.marks_for_sheet` reports them: (ordinal, Label, how).

    Holds nothing past the call: plain floats and strings out.
    """
    from . import foam_marks as fm
    bevel_out = []
    marks = fm.marks_for_sheet(sheet, parts, s, _scale_length(s),
                               bevel_out=bevel_out, names_out=names_out)
    leaders = {}
    for st in marks.strokes:
        if st.kind == 'leader' and st.anchor is not None:
            leaders.setdefault(st.anchor, []).append(st)

    out = []
    for label in marks.labels:
        anchor = label.anchor
        if label.kind != 'angle' or anchor is None:
            continue
        if anchor.piece not in shown_ordinals:
            continue
        leader = ()
        own = leaders.get(anchor)
        if own:
            (x0, y0), (x1, y1) = own[0].points[0], own[0].points[-1]
            leader = (to_uv(x0, y0), to_uv(x1, y1))
        (ax, ay), (bx, by) = anchor.run[0], anchor.run[-1]
        out.append(_BevelMark(
            *to_uv(label.x, label.y), str(label.text), label.tone,
            label.valign, (), *to_uv(ax, ay), *to_uv(bx, by),
            float(label.rotation_deg or 0.0), float(label.size_mm),
            leader, int(anchor.piece)))

    for part_index, mark in bevel_out:
        label = mark.label
        if label is None or label.kind != 'heat':
            continue
        isl = parts[part_index].islands[mark.island_index]
        if isl.ordinal not in shown_ordinals:
            continue
        (ax, ay), (bx, by) = mark.run_ends
        out.append(_BevelMark(
            *to_uv(label.x, label.y), str(label.text), 'heat',
            label.valign, (), *to_uv(ax, ay), *to_uv(bx, by),
            float(label.rotation_deg or 0.0), float(label.size_mm)))
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
    # (text, key, ordinal, faces, uv_layer) per shown island. The BMFaces
    # are read only within this call, for the fallback position below.
    name_slots = []

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
                      obj.name, tuple(fm.mirror_planes(obj)))
         for obj, bm, uv_layer in pairs], s, _scale_length(s))
    part_of = {part.item.name: part for part in parts}
    # Sheet-wide island ordinals (`BevelAnchor.piece`) left on screen by
    # Only Selected, gathered per object below.
    shown_ordinals = set()

    for obj, bm, uv_layer in pairs:
        part = part_of[obj.name]
        islands = core.uv_islands(bm, uv_layer, s.only_selected)
        shown_ordinals.update(
            isl.ordinal for isl, shown in zip(
                part.islands, _island_shown(part.islands, s, islands))
            if shown)

        # Kept as a list, not consumed as a generator: the alignment ticks
        # below are decided from these same runs.
        object_runs = list(core.iter_boundary_dart_segments(
                bm, uv_layer, s.max_chord_deviation, s.max_angle_variation_deg,
                only_selected=s.only_selected, islands=islands,
                corner_angle_deg=s.corner_angle_deg,
                matrix=obj.matrix_world,
                dart_wedge_deg=s.dart_wedge_deg,
                max_darts_per_run=s.max_darts_per_run,
                mirror_planes=tuple(part.item.mirror_planes)))

        # What the sheet prints, asked of the model once per object.
        if s.show_align_ticks or s.show_brackets:
            model = _Model(part, sheet, s, islands)
            if s.show_align_ticks:
                ticks, tick_labels = model.align_ticks()
                align_ticks.extend(ticks)
                align_labels.extend(tick_labels)
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
        ordinal_of_face = {f.index: isl.ordinal
                           for isl in part.islands for f in isl.faces}
        stretch = core.uv_stretch_map(bm, uv_layer)

        for index, faces in enumerate(islands):
            label = next((name_of_face[f.index] for f in faces
                          if f.index in name_of_face), str(index + 1))
            ordinal = next((ordinal_of_face[f.index] for f in faces
                            if f.index in ordinal_of_face), None)
            key = (obj.name, index)
            # Placed and sized from the printed sheet once it is built,
            # below; the ordinal finds this island's name there.
            name_slots.append((label, key, ordinal, faces, uv_layer))

            # An island can be fine at its darted edges and still refuse to lie
            # flat through the middle. That is a heat-forming job, not a dart,
            # so it is flagged from areal distortion, not from a boundary angle.
            vals = [stretch.get(f, 0.0) for f in faces]
            if vals and sum(vals) / len(vals) > s.stretch_tolerance:
                flagged.add(key)

    # Bevel labels and leaders: the laid-out ones the export prints, for the
    # whole sheet at once -- the layout fits labels against every object's
    # marks together, so it cannot be asked one object at a time.
    # Piece names too: sized by the model to fit their pieces, clear of
    # the ticks and the outline, so screen and paper print the same name
    # in the same place at the same size.
    to_uv = _sheet_to_uv(sheet)
    printed_names = []
    if (s.show_brackets or s.show_heat_form or s.show_island_ids) \
            and to_uv is not None and any(part.islands for part in parts):
        laid = _laid_out_bevel_marks(sheet, parts, s, shown_ordinals, to_uv,
                                     names_out=printed_names)
        if s.show_brackets or s.show_heat_form:
            bevel_marks = laid
    name_at = {ordinal: tuple(to_uv(label.x, label.y)) + (float(label.size_mm),)
               for ordinal, label, _how in printed_names}
    for text, key, ordinal, faces, layer in name_slots:
        if ordinal in name_at:
            u, v, size_mm = name_at[ordinal]
        else:
            centroid = _island_uv_centroid(faces, layer)
            u, v, size_mm = centroid.x, centroid.y, 0.0
        labels.append((float(u), float(v), text, key, size_mm))
    del name_slots

    _cache.runs = runs
    _cache.island_labels = labels
    _cache.align_ticks = align_ticks
    _cache.align_labels = align_labels
    _cache.bevel_marks = bevel_marks
    _cache.angle_font_mm = (float(sheet.permille(s.angle_text_size,
                                                 s.overlay_scale))
                            if any(part.islands for part in parts) else 0.0)
    _cache.number_font_mm = (float(sheet.permille(s.number_size,
                                                  s.overlay_scale))
                             if any(part.islands for part in parts) else 0.0)
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


def _label_px(angle_px, size_mm, full_mm, ceiling=400):
    """On-screen pixel size of a bevel label printed `size_mm` tall.

    The sheet's full angle size, `full_mm`, is drawn at `angle_px` (the
    overlay's existing mapping of `angle_text_size`); a label the layout
    shrank is drawn smaller by the same factor, so screen and paper keep
    the same proportions. `size_mm` 0 (unknown) draws at `angle_px`.
    Clamped to the overlay's usual 6..`ceiling` px (piece names allow
    500, as they always have). The same mapping draws a piece name at the
    size the model fitted it to, against the set Number Size.
    """
    if size_mm <= 0.0 or full_mm <= 0.0:
        return angle_px
    return max(6, min(int(round(angle_px * size_mm / full_mm)), ceiling))


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

        # The run's outline offset into the margin. It is no longer DRAWN:
        # it used to be, as a convex/concave-coloured "bracket" beside every
        # run, and the sheet prints nothing there -- so the screen showed
        # lines the paper did not, and they ran straight through outside
        # labels and their leaders ("32, 17°" on HE P, 2026-11-17). What it
        # said is all on the sheet's own marks now: where a run starts and
        # ends (the run-end marks, below), and which way it folds (the
        # bevel label's colour and blade arrow). The offset points are kept
        # only to anchor the too-short-to-cut "!" warning, which is a
        # screen-only craftability flag that sits in that margin.
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

        # (Its ends were once drawn as a T at every corner as well; the
        # maker: "There is no need for alignment marks at the corners.")

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
        for ring in _cache.run_end_marks:
            pts = [view.view_to_region(u, v, clip=False) for u, v in ring]
            if all(abs(c) < 1e6 for p in pts for c in p):
                for k, p in enumerate(pts):
                    bucket.extend([p, pts[(k + 1) % len(pts)]])

    # Bevel labels: seam number, blade-tilt arrow and degrees, at the place,
    # size and turn the exported sheet prints them -- the laid-out labels
    # (foam_layout.layout_labels), cached in UV space by the rebuild from
    # the export's own Marks, so screen and paper cannot disagree about what
    # a seam is called, which way to tilt, or where the label went. A label
    # the layout put outside its piece is drawn there with its leader.
    #
    # Hidden where the run is too short on screen to carry a number, the
    # same legibility test the old margin labels used (zoom in and they
    # come back); that is about the view, never about the pattern.
    neutral = convex if not s.show_dart_colors else tuple(s.island_id_color)
    tones = {'convex': convex, 'concave': concave, 'neutral': neutral,
             'heat': heat}
    full_mm = _cache.angle_font_mm
    for mark in _cache.bevel_marks:
        ax, ay = view.view_to_region(mark.au, mark.av, clip=False)
        bx, by = view.view_to_region(mark.bu, mark.bv, clip=False)
        x, y = view.view_to_region(mark.u, mark.v, clip=False)
        if not all(abs(c) < 1e6 for c in (ax, ay, bx, by, x, y)):
            continue
        if ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 < s.min_label_run_px:
            continue
        color = tones.get(mark.tone, neutral)
        px = _label_px(angle_px, mark.size_mm, full_mm)
        labels.append((x, y, mark.text, color, mark.valign,
                       mark.rotation_deg, px))
        if mark.leader:
            (u0, v0), (u1, v1) = mark.leader
            p = view.view_to_region(u0, v0, clip=False)
            q = view.view_to_region(u1, v1, clip=False)
            if all(abs(c) < 1e6 for c in (*p, *q)):
                segments.setdefault(color, []).extend([p, q])
        if mark.arrow:
            pts = [view.view_to_region(u, v, clip=False)
                   for u, v in mark.arrow]
            bucket = segments.setdefault(color, [])
            for p, q in zip(pts, pts[1:]):
                bucket.extend([p, q])

    _draw_lines(segments, max(1.0, scale))

    if tick_marks:
        # Tick Number Size against Angle Text: the ratio the sheet prints
        # them at, both being thousandths of the sheet.
        blf.size(font_id, max(6, int(angle_px * s.tick_number_size
                                     / max(1, s.angle_text_size))))
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

    current_px = angle_px
    for x, y, text, color, valign, turn, px in labels:
        if px != current_px:
            blf.size(font_id, px)
            current_px = px
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
        # Each name at the size the sheet prints it: the model shrank it to
        # fit its piece clear of the outline and ticks. Drawing every name
        # at the full Number Size instead ran neighbouring names into each
        # other and over the bevel labels (2026-11-17).
        number_px = max(6, min(number_px, 500))
        id_color = tuple(s.island_id_color)
        for u, v, text, key, size_mm in _cache.island_labels:
            x, y = view.view_to_region(u, v, clip=False)
            if not all(abs(c) < 1e6 for c in (x, y)):
                continue
            blf.size(font_id, _label_px(number_px, size_mm,
                                        _cache.number_font_mm, 500))
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


def _section(layout, idname, title, toggle=None, settings=None,
             default_closed=True):
    """A collapsible section inside a panel, Blender's own layout.panel().

    `toggle` names a boolean drawn as a checkbox in the section header, the
    way Blender's own sub-panels carry an enable switch; the body is then
    drawn disabled while it is off. Returns the body layout, or None when the
    section is collapsed.
    """
    header, body = layout.panel(idname, default_closed=default_closed)
    if toggle is not None:
        header.use_property_split = False
        header.prop(settings, toggle, text="")
    header.label(text=title)
    if body is not None:
        body.use_property_split = True
        body.use_property_decorate = False
        if toggle is not None:
            body.enabled = bool(getattr(settings, toggle))
    return body


def _wysiwyg_note(layout):
    note = layout.row()
    note.enabled = False
    note.label(text="The UV overlay previews the printout", icon='INFO')


class IMAGE_PT_foamcraft_drachis_export(bpy.types.Panel):
    """Everything that decides what is printed, with the export at the top.

    The overlay is a preview of the sheet (both draw the same foam_marks
    model), so every setting here changes the overlay and the file alike.
    Settings that only change the screen live in "On Screen" instead.
    """
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "FoamCraft"
    bl_label = "Foam Pattern"
    bl_order = 0

    def draw_header(self, context):
        s = _settings(context)
        if s is not None:
            self.layout.prop(s, "show_overlay", text="")

    def draw(self, context):
        layout = self.layout
        s = _settings(context)
        if s is None:
            layout.label(text="FoamCraft settings unavailable", icon='ERROR')
            return

        col = layout.column(align=True)
        col.scale_y = 1.4
        for idname, text, icon in (
                ("foamcraft_drachis.export_pattern_image", "Export Image", 'IMAGE_DATA'),
                ("foamcraft_drachis.export_panel_pattern", "Export SVG", 'EXPORT')):
            if _operator_registered(idname):
                col.operator(idname, text=text, icon=icon)
            else:
                row = col.row()
                row.enabled = False
                row.label(text=f"{text} (unavailable)", icon=icon)
        _wysiwyg_note(layout)

        body = _section(layout, "fc_bevels", "Bevel Labels",
                        toggle="show_brackets", settings=s, default_closed=False)
        if body:
            body.prop(s, "pattern_surface")
            body.prop(s, "show_seam_numbers")
            body.prop(s, "min_angle", text="Min Angle")
            body.prop(s, "decimals")
            body.prop(s, "angle_text_size", text="Text Size")
            body.prop(s, "min_label_mm", text="Min Text Size")
            body.prop(s, "max_tilt_deg")
            body.prop(s, "show_heat_form")

        body = _section(layout, "fc_ticks", "Alignment Marks",
                        toggle="show_align_ticks", settings=s)
        if body:
            body.prop(s, "tick_mode")
            if s.tick_mode == 'BY_DISTANCE':
                body.prop(s, "tick_spacing")
                row = body.row(align=True)
                row.prop(s, "tick_count_min", text="Min")
                row.prop(s, "tick_count_max", text="Max")
            else:
                body.prop(s, "ticks_per_run")
            body.prop(s, "tick_skip_below")
            body.prop(s, "tick_length")
            body.prop(s, "tick_numbers")
            sub = body.column()
            sub.enabled = s.tick_numbers
            sub.prop(s, "tick_number_size", text="Number Size")
            body.prop(s, "corner_angle_deg")
            # Where the mirror line comes from: the object's own Mirror
            # modifier, so say whether one is being read.
            obj = getattr(context, "active_object", None)
            mirrors = [m for m in getattr(obj, "modifiers", ())
                       if getattr(m, "type", None) == 'MIRROR'
                       and m.show_viewport]
            if mirrors:
                axes = "".join(a for m in mirrors
                               for a, on in zip("XYZ", m.use_axis) if on)
                centre = next((m.mirror_object.name for m in mirrors
                               if m.mirror_object is not None), "origin")
                body.label(text=f"Mirror {axes or '-'} about {centre}: "
                                "open edges on it are seams",
                           icon='MOD_MIRROR')
            else:
                body.label(text="Add a Mirror modifier to mark a centre line",
                           icon='MOD_MIRROR')

        body = _section(layout, "fc_darts", "Relief Darts",
                        toggle="show_darts", settings=s)
        if body:
            body.prop(s, "dart_wedge_deg")
            body.prop(s, "max_darts_per_run")
            body.prop(s, "notch_shape")
            body.prop(s, "notch_depth_ratio")
            body.prop(s, "allow_gores")

        body = _section(layout, "fc_pieces", "Piece Names",
                        toggle="show_island_ids", settings=s)
        if body:
            body.prop(s, "piece_label_style")
            obj = context.active_object
            if obj is not None and obj.type == 'MESH':
                row = body.row()
                row.enabled = s.piece_label_style == 'NAME'
                row.prop(obj, "foamcraft_drachis_prefix")
                if not obj.foamcraft_drachis_prefix:
                    # Say what an empty field prints, rather than leave the
                    # maker to discover it on paper.
                    from . import foam_marks as fm
                    note = body.row()
                    note.enabled = False
                    note.label(text="Uses %s" % fm.object_prefix(obj.name, ""))
            body.prop(s, "number_size", text="Text Size")
            body.prop(s, "origin_object")
            body.operator("foamcraft_drachis.assign_island_ids",
                          text="Renumber Pieces", icon='LINENUMBERS_ON')

        body = _section(layout, "fc_sheet", "Sheet & Colour", settings=s)
        if body:
            body.prop(s, "overlay_scale", text="Mark Scale")
            body.prop(s, "island_fill")
            body.prop(s, "show_scale_ruler")
            body.prop(s, "show_dart_colors")
            sub = body.column()
            sub.enabled = s.show_dart_colors
            sub.prop(s, "convex_color")
            sub.prop(s, "concave_color")
            sub = body.column()
            sub.enabled = s.show_heat_form
            sub.prop(s, "heat_form_color")


class IMAGE_PT_foamcraft_drachis_advice(bpy.types.Panel):
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "FoamCraft"
    bl_label = "Craftability"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        s = _settings(context)
        if s is None:
            return

        row = layout.row()
        row.scale_y = 1.4
        row.operator(FOAMCRAFT_DRACHIS_OT_check_craftability.bl_idname,
                     icon='VIEWZOOM')

        if not _findings:
            layout.label(text="Not checked yet")
        else:
            counts = {}
            for finding in _findings:
                counts[finding.severity] = counts.get(finding.severity, 0) + 1
            # Counts beside their icons only: words truncate in a sidebar of
            # default width (measured in the 2026-11-17 capture).
            row = layout.row()
            for severity in ('blocker', 'warn', 'info'):
                if counts.get(severity):
                    row.label(text=str(counts[severity]),
                              icon=_SEVERITY_ICON[severity])
            if not (counts.get('blocker') or counts.get('warn')):
                row.label(text="Buildable", icon='CHECKMARK')
            # One line per finding; the explanation opens on demand, so the
            # verdict reads at a glance instead of being buried in text.
            for k, finding in enumerate(_findings):
                header, body = layout.panel(
                    f"fc_finding_{finding.code}_{k}", default_closed=True)
                header.label(text=finding.title,
                             icon=_SEVERITY_ICON.get(finding.severity, 'DOT'))
                if body is None:
                    continue
                col = body.column(align=True)
                for line in _wrap(finding.detail, 44):
                    col.label(text=line)
                if finding.severity != 'ok':
                    col.separator()
                    sub = body.column(align=True)
                    sub.scale_y = 0.8
                    for para in finding.advice.split("\n"):
                        for line in _wrap(para, 46):
                            sub.label(text=line)

        body = _section(layout, "fc_check_limits", "Limits", settings=s)
        if body:
            body.prop(s, "sheet_width")
            body.prop(s, "sheet_height")
            body.prop(s, "stretch_tolerance")
            body.prop(s, "min_run_length")

        # Real-world size of what is being built, so the sheet figures above
        # can be judged against it without leaving the panel.
        objects = getattr(context, "objects_in_mode", None) or []
        if not objects and context.edit_object:
            objects = [context.edit_object]
        meshes = [o for o in objects if o and o.type == 'MESH']
        if meshes:
            body = _section(layout, "fc_model_size", "Model Size", settings=s)
            if body:
                unit = context.scene.unit_settings.scale_length
                for obj in meshes:
                    w, d, h = (v * unit for v in obj.dimensions)
                    if len(meshes) > 1:
                        body.label(text=obj.name)
                    # One row each: the sidebar is narrow enough that two
                    # dimensions on a line get truncated mid-number.
                    col = body.column(align=True)
                    col.scale_y = 0.85
                    for axis, value in (("W", w), ("D", d), ("H", h)):
                        col.label(text=f"{axis}  {_fmt_len(value)}")


class IMAGE_PT_foamcraft_drachis(bpy.types.Panel):
    """How the UV boundaries are split into cut runs, and the foam."""
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "FoamCraft"
    bl_label = "Cut Runs"
    bl_order = 2
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = _settings(context)
        if s is None:
            layout.label(text="FoamCraft settings unavailable", icon='ERROR')
            return
        layout.use_property_split = True
        layout.use_property_decorate = False

        col = layout.column()
        col.prop(s, "foam_grade")
        col.prop(s, "foam_thickness")
        col.separator()
        col.prop(s, "max_chord_deviation")
        col.prop(s, "max_angle_variation_deg")
        col.operator("foamcraft_drachis.estimate_tolerances", icon='SHADERFX')
        row = layout.row()
        row.label(text=f"{len(_cache.runs)} run(s)")
        row.operator(FOAMCRAFT_DRACHIS_OT_refresh_overlay.bl_idname,
                     text="Refresh", icon='FILE_REFRESH')


class IMAGE_PT_foamcraft_drachis_display(bpy.types.Panel):
    """Settings that change the UV editor and never the printout."""
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "FoamCraft"
    bl_label = "On Screen Only"
    bl_order = 3
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = _settings(context)
        if s is None:
            return
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(s, "only_selected")
        layout.prop(s, "min_label_run_px")
        layout.prop(s, "show_blockers")
        layout.prop(s, "scale_text_with_view")
        col = layout.column()
        if s.scale_text_with_view:
            col.prop(s, "mark_size")
        else:
            col.prop(s, "font_size")
            col.prop(s, "tick_length_px")
            col.prop(s, "island_id_font_size")

        body = _section(layout, "fc_screen_colours", "Colours", settings=s)
        if body:
            body.prop(s, "island_id_color")
            body.prop(s, "blocker_color")
            body.prop(s, "suppressed_color")

        body = _section(layout, "fc_debug", "Debug Log", settings=s)
        if body:
            body.prop(s, "log_level")
            # Show where the log actually is. Otherwise the only way to find
            # it is to already know, which makes the log useless to anyone
            # debugging.
            col = body.column(align=True)
            col.enabled = False
            col.label(text=foam_log.log_path(), icon='TEXT')


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
    IMAGE_PT_foamcraft_drachis_export,
    IMAGE_PT_foamcraft_drachis_advice,
    IMAGE_PT_foamcraft_drachis,
    IMAGE_PT_foamcraft_drachis_display,
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
