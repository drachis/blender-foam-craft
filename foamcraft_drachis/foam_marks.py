"""The one description of what gets drawn on a pattern sheet.

WHY THIS EXISTS

Three places used to work out what to draw -- the overlay, the SVG exporter
and the raster exporter -- and each re-derived the marks and re-applied the
settings its own way. They disagreed: the export asked the core for runs with
different arguments than the overlay did, the raster had no angles or ticks,
and a fix to one left the other two wrong.

So the marks are decided once, here, and written down as plain data. A
backend only draws. It must not consult settings to decide *whether* a mark
exists, and it must not touch the mesh at all.

THE CONTRACT

Everything is in **sheet millimetres**, origin top-left, y growing downward
(the SVG convention, and what `foam_export._Sheet.to_sheet` produces). Colours
are (r, g, b) floats 0..1. Nothing here holds a BMesh element: `build_marks`
copies out floats and returns, so a `Marks` outlives the BMesh it came from.

Draw order is fills, then strokes, then labels, each in sequence order.

`kind` says what a mark *is*, so a backend can style or group by it without
re-deriving anything:

    Fill.kind    'piece'    a piece's coloured interior
                 'relief'   a relief wedge to cut out
                 'ruler'    the 1 inch square's interior, if filled
    Stroke.kind  'outline'  a piece boundary -- the cut line
                 'bracket'  a run-end mark: where one bevel angle stops
                            and the next starts *part way along* a seam.
                            Two points, the first on the cut line, the
                            second inside the piece, square to the line,
                            `tick_length` long. None at corners or
                            junctions (`_run_end_marks`).
                 'align'    an alignment (registration) mark. Two points,
                            the first on the cut line, the second inside
                            the piece, perpendicular to the line there.
                 'bevel_arrow'  which way to tilt the blade for a run. One
                            open polyline of five points: tail, tip, barb,
                            tip, barb -- so points[0] -> points[1] is the
                            direction, perpendicular to the cut line. It
                            points to the side of the line the blade's tip
                            (the far face of the sheet) ends up on. Absent
                            for a straight cut. See `_blade_tip_inward`.
                 'relief'   a relief cut drawn open
                 'heat'     a heat-form run marker
                 'ruler'    scale bar and 1 inch square
    Label.kind   'number'   a piece number, inside its piece
                 'angle'    a run's bevel label: "<seam>, <deg>°" -- the
                            seam number the two glued pieces share, and the
                            blade tilt for this piece (per side, absolute).
                            Just "<seam>" for a straight cut (fold below
                            `min_angle`); just "<deg>°" with seam numbers
                            switched off. The comma is the maker's: "4 3°"
                            read as "43°".
                 'align'    an alignment mark's number
                 'heat'     a heat-form note, "<seam>, <deg>° (heat-form)"
                 'ruler'    scale bar captions
"""

from math import acos, ceil, degrees, radians, tan
from typing import NamedTuple, Optional, Tuple

from mathutils import Vector

from . import foam_angle_core as core

Point = Tuple[float, float]
Color = Tuple[float, float, float]


class Fill(NamedTuple):
    kind: str
    # Outer ring first, then holes; filled even-odd. An annulus is two rings.
    rings: Tuple[Tuple[Point, ...], ...]
    color: Color


class Stroke(NamedTuple):
    kind: str
    points: Tuple[Point, ...]
    closed: bool
    width_mm: float
    color: Color
    # (on_mm, off_mm) or None for a solid line.
    dash: Optional[Tuple[float, float]] = None


class Label(NamedTuple):
    kind: str
    x: float
    y: float
    # Cap height is roughly 0.7 of this; it is the SVG font-size in mm.
    size_mm: float
    text: str
    color: Color
    # Horizontally always centred on x. Vertically: 'middle' centres the
    # text on y, 'baseline' sits its baseline on y.
    valign: str = 'middle'
    # Added after the original contract, defaulted so nothing written
    # against it breaks. For an 'angle' label: 'convex', 'concave', or
    # 'neutral' when bevel colouring is off. Already reflected in `color`,
    # so a backend that only paints can ignore it; the SVG writer needs it
    # because the sheet names the sign as a class, and the colour alone
    # cannot say it when the two colours happen to coincide. '' elsewhere.
    tone: str = ''


class Marks(NamedTuple):
    width_mm: float
    height_mm: float
    fills: Tuple[Fill, ...]
    strokes: Tuple[Stroke, ...]
    labels: Tuple[Label, ...]
    # Added after the original contract, defaulted. Not marks: the sheet's
    # base line width (every stroke width is a multiple of it, except the
    # ruler's absolute 0.3 mm and the heat-form line's 1 mm, which is SVG's
    # default stroke width -- the sheet never set one) and its colour key. The SVG states all of
    # them in its <style> block whether or not a mark of that kind exists,
    # so they cannot be recovered from the marks themselves.
    hairline_mm: float = 0.0
    convex_color: Color = (0.0, 0.0, 0.0)
    concave_color: Color = (0.0, 0.0, 0.0)
    heat_color: Color = (0.0, 0.0, 0.0)
    # Added after the original contract, defaulted. Plain-language notes for
    # the maker about a sheet that is drawn but cannot be fully trusted --
    # currently only unapplied non-uniform scale (see `scale_warnings`). A
    # backend may show or log them; it must not alter the marks for them.
    warnings: Tuple[str, ...] = ()


# Real-world sizes for pattern elements that only make sense printed at 1:1 --
# distinct from the overlay's screen-space settings (font_size, tick_length_px,
# overlay_scale), which scale with UV editor zoom rather than paper size.
# Darts are physical cuts, so their size is in millimetres of real foam and
# comes from the stock thickness, not from how the pattern is drawn.
DART_SLIT_WIDTH_MM = 0.6
MARGIN_MM = 10.0

# The scale strip. Every number here is an absolute millimetre and must stay
# that way: the ruler's whole job is to be measurable against a real ruler
# held to the paper, so it is the one thing on the sheet that must NOT scale
# with the pattern.
RULER_STRIP_MM = 44.0
RULER_LENGTH_MM = 100.0
RULER_TICK_MM = 4.0
RULER_FONT_MM = 4.0
INCH_MM = 25.4

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


def _face_world_area(face, matrix):
    """A face's area after `matrix`, by Newell's method -- the same
    `area_poly_v3` construction `BMFace.calc_area` uses, applied to the
    transformed corners, in Python doubles."""
    pts = [tuple(matrix @ v.co) for v in face.verts]
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0, z0 = pts[i]
        x1, y1, z1 = pts[(i + 1) % n]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return 0.5 * (nx * nx + ny * ny + nz * nz) ** 0.5


# Scale axes further apart than this (relative) are reported. 0.1%: well
# above float noise in a world matrix, and below anything a maker could see
# on a printed piece (1 mm in a metre).
NON_UNIFORM_SCALE_TOL = 1e-3


def scale_warnings(matrix):
    """Plain-language notes about a world matrix the sheet cannot honour.

    One UV->mm factor serves the whole sheet, so a scale that differs per
    axis makes every piece wrong in a different direction. The fix is the
    maker's, in Blender, and is said as such rather than corrected here.
    """
    if matrix is None:
        return ()
    axes = [abs(c) for c in matrix.to_scale()]
    if min(axes) <= 0.0:
        return ("object has zero scale on an axis; the pattern cannot be "
                "sized",)
    if max(axes) / min(axes) - 1.0 > NON_UNIFORM_SCALE_TOL:
        return ("object has non-uniform scale; apply scale (Ctrl+A) for a "
                "true-size pattern",)
    return ()


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
        self.uv_layer = uv_layer
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

    def __init__(self, islands_faces, uv_layer, scale_length, matrix=None):
        # Areas through the world matrix when one is given: an object with
        # unapplied scale is that size in the scene, and the pattern is for
        # the thing being built, not for its mesh data. A uniform scale s then
        # gives a sheet exactly s times larger each way. A non-uniform one
        # cannot be honoured by one UV->mm factor -- see `scale_warnings`.
        if matrix is None:
            area_3d = sum(f.calc_area() for fs in islands_faces for f in fs)
        else:
            area_3d = sum(_face_world_area(f, matrix)
                          for fs in islands_faces for f in fs)
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


def _run_midpoint_frame(run, uv_layer, isl):
    """(point, inward unit normal, loop index) half way along a run.

    The point is half way along the run *as drawn*, which on a curved run is
    nowhere near the middle of its chord. The normal is perpendicular to the
    one edge the point sits on -- not to the chord -- because an arrow drawn
    there has to read as square to the line the maker is looking at. Its
    sign comes from the face behind that edge, which is inside the piece by
    definition (an island centroid is not: on a ring it sits in the hole).

    The loop index says which of the run's edges carries the midpoint, and
    so which seam the run is labelled with when it spans more than one.
    """
    pts = _run_polyline(run, uv_layer, isl)
    lengths = [(pts[i + 1] - pts[i]).length for i in range(len(pts) - 1)]
    half = sum(lengths) / 2
    index = min(len(pts) // 2, len(lengths) - 1)
    mid = pts[len(pts) // 2].copy()
    for i, seg in enumerate(lengths):
        if seg >= half and seg > 1e-12:
            index, mid = i, pts[i].lerp(pts[i + 1], half / seg)
            break
        half -= seg
    inward, _room = _inward_at(isl, uv_layer, mid, run.loops[index])
    return mid, inward, index


def _loop_sheet_dir(isl, uv_layer, loop):
    """Unit direction of a boundary loop's edge on the sheet, or None."""
    d = (isl.to_sheet(loop.link_loop_next[uv_layer].uv)
         - isl.to_sheet(loop[uv_layer].uv))
    return d.normalized() if d.length > 1e-9 else None


def _inward_at(isl, uv_layer, point, loop, other=None):
    """(inward unit normal, room) at `point` on a boundary loop's edge.

    Perpendicular to that edge on the sheet. With `other` -- the boundary
    loop on the far side of a vertex the point sits exactly on -- it is
    perpendicular to the average of the two edge directions instead, so a
    mark on a bend leans into neither side.

    "Inward" is the side of the edge the face behind `loop` lies on, read
    from that face's winding on the sheet: a face is to the left of its own
    loops when its signed (shoelace) area is positive and to the right when
    negative, in whatever frame both are computed in. It used to be "towards
    the face's centre", which is right for a convex face and undecided for a
    concave one -- an L-shaped face has its vertex average exactly on its
    reflex corner, so for the two edges meeting there the centre lay on the
    edge's own line and the sign was whatever rounding said
    (tests/test_marks_model.py, the L prism).

    `room` is still the distance to the face's centre: a cheap stand-in for
    how much piece there is to draw into.
    """
    direction = _loop_sheet_dir(isl, uv_layer, loop)
    if other is not None:
        second = _loop_sheet_dir(isl, uv_layer, other)
        if direction is None:
            direction = second
        elif second is not None and (direction + second).length > 1e-9:
            direction = (direction + second).normalized()
    if direction is None:
        direction = Vector((1.0, 0.0))
    face_pts = [isl.to_sheet(l[uv_layer].uv) for l in loop.face.loops]
    area2 = sum(a.x * b.y - b.x * a.y
                for a, b in zip(face_pts, face_pts[1:] + face_pts[:1]))
    left = Vector((-direction.y, direction.x))
    inward = left if area2 > 0.0 else -left
    face_pt = Vector((sum(p.x for p in face_pts) / len(face_pts),
                      sum(p.y for p in face_pts) / len(face_pts)))
    return inward, (face_pt - point).length


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


def _edge_canonical_start(edge):
    """The end of `edge` that both its faces agree to measure from.

    `edge.verts[0]` is arbitrary and does not survive a rebuild (see
    foam_gp_pattern._edge_ends and
    RESEARCH_theta_seam_edge_nondeterminism.md), and here it matters for a
    second reason: a tick has to land at the *same* point when measured from
    either of the two faces sharing the seam, or the two pieces carry marks
    that do not line up -- which is the one thing an alignment mark must not
    do. Ordering by coordinate gives both sides the same answer.
    """
    a, b = edge.verts[0], edge.verts[1]
    ac, bc = a.co, b.co
    return a if (ac.x, ac.y, ac.z) <= (bc.x, bc.y, bc.z) else b


def _run_tick_params(run, settings):
    """The OLD Per Cut Run placement: ticks spread along one saw run.

    No longer used to place marks (see `_segment_tick_params`). Kept because
    foam_export re-exports the name and the before/after measurement in
    tests/test_marks_model.py asks it what the old placement was.

    Why it was replaced: a run ends wherever the bevel angle drifts past
    `max_angle_variation_deg`, not only at corners, so distributing per run
    restarted the marks at every such break. Measured on a 6-gore sphere:
    36 runs over 6 corner-to-corner seams, 72 placements where the maker
    asked for 12, crowded towards run ends, and the two sides of a seam
    breaking their runs in different places. The maker asked for marks
    spread from corner to corner of an edge.
    """
    edges = []
    total = 0.0
    for loop in run.loops:
        edge = loop.edge
        length = edge.calc_length()
        if length <= 1e-9:
            continue
        edges.append((edge, loop, length, total))
        total += length
    if total <= 1e-9 or not edges:
        return []

    count = max(1, int(settings.ticks_per_run))
    distances = [total * (i + 1) / (count + 1) for i in range(count)]

    out = []
    for d in distances:
        for edge, loop, length, start in edges:
            if d <= start + length or edge is edges[-1][0]:
                local = (d - start) / length
                local = max(0.0, min(1.0, local))
                # Express it from the edge's canonical end.
                t = local if loop.vert is _edge_canonical_start(edge) \
                    else 1.0 - local
                out.append((edge, round(t, 6)))
                break
    return out


# Two marks closer together than this along a seam are the same mark. A dash
# is a few millimetres wide, so anything at this range is not a second one.
TICK_MERGE_M = 0.0005


def _remember_tick(wanted, edge, t):
    """Record a mark at `t` on `edge`, unless one is already there.

    The parameter is merged by *distance on the mesh*, not by its own value.
    Rounding `t` to six places sounds precise and is not a physical test: on
    a 30 mm edge it resolves to 30 nanometres, so two runs arriving at the
    same point by slightly different arithmetic -- 0.666666 and 0.666667 --
    were kept as two marks and drawn as two dashes on top of each other.
    Measured on a six-gore sphere: twelve such pairs.
    """
    length = edge.calc_length()
    if length <= 1e-12:
        return
    params = wanted.setdefault(edge, [])
    for existing in params:
        if abs(existing - t) * length <= TICK_MERGE_M:
            return
    params.append(t)


def _seam_chains(islands):
    """The seams, as chains of connected 3D edges.

    A seam is what a maker tapes together, and it is a property of the mesh,
    not of either piece: the same chain of edges bounds both islands that
    meet along it. Marks spaced along *this* are evenly spaced on the foam;
    marks spaced along one island's boundary runs are not, because the runs
    break wherever the bevel angle changes.

    Chains break at a vertex where more than two boundary edges meet. That is
    a corner where three or more pieces come together, and carrying a
    measurement straight through one would be arbitrary -- there is no single
    "onwards" to continue along.
    """
    edges = []
    seen = set()
    for isl in islands:
        for cycle in isl.cycles:
            for loop in cycle:
                e = loop.edge
                if e.index not in seen:
                    seen.add(e.index)
                    edges.append(e)

    by_vert = {}
    for e in edges:
        for v in e.verts:
            by_vert.setdefault(v.index, []).append(e)

    chains, used = [], set()
    def _walk(start_edge, start_vert):
        chain = [start_edge]
        used.add(start_edge.index)
        vert = start_edge.other_vert(start_vert)
        while vert is not None:
            nxt = [e for e in by_vert.get(vert.index, ())
                   if e.index not in used]
            # Only continue through a plain two-edge join.
            if len(by_vert.get(vert.index, ())) != 2 or len(nxt) != 1:
                break
            edge = nxt[0]
            chain.append(edge)
            used.add(edge.index)
            vert = edge.other_vert(vert)
        return chain

    # Open chains first, from their loose ends, so a chain that has ends is
    # not entered in the middle and split into two. The ends are considered
    # in canonical order, not edge order.
    for e in edges:
        if e.index in used:
            continue
        start, far = _edge_canonical_start(e), None
        far = e.other_vert(start)
        for v in (start, far):
            if len(by_vert.get(v.index, ())) != 2:
                chains.append(_walk(e, v))
                break
    # Whatever is left is a closed loop. "Any edge will do as a start" is
    # true; any *end* of that edge is not. A closed chain walked from
    # verts[0] runs one way round or the other, and By Distance measures
    # from the start, so every mark on that seam moves with it.
    #
    # Scope, measured rather than assumed, because a guard that claims more
    # than it fixes hides the next bug. Five fresh BMeshes of one *fixed*
    # mesh give identical edge orientation -- so this is NOT what makes two
    # exports of one .blend disagree. It is what makes an export disagree
    # with itself across a *rebuilt* mesh: regenerating the same primitive
    # flips verts[0] on about 670 of 9120 edges, which is how it surfaced,
    # as one differing sheet in a 1422-record golden run whose fixtures are
    # rebuilt each time. Cheap, and it buys reproducibility across a
    # re-import; it is not load-bearing for the common case.
    for e in edges:
        if e.index not in used:
            chains.append(_walk(e, _edge_canonical_start(e)))
    return chains


def _co_key(vert):
    c = vert.co
    return (c.x, c.y, c.z)


def _outline_corner_verts(islands, corner_angle_deg):
    """Indices of the vertices where some piece's outline has a corner.

    The same test `core.segment_boundary_cycle` breaks runs with: the turn
    between the UV directions arriving at and leaving a boundary vertex,
    `acos` of their dot product, compared `>= corner_angle_deg`. It is
    evaluated on every island's own outline, and a vertex is a corner if
    *either* piece turns there -- the two sides of a seam are unwrapped
    separately and need not turn by the same amount, and a mark must be
    decided once for both, so the stricter reading wins.

    Not taken from the runs: a run also ends where the bevel angle drifts
    (`max_angle_variation_deg`) or the chord bows (`max_chord_deviation`),
    and those are saw-pass limits, not corners of the piece. Restarting the
    tick distribution at them is what crowded marks onto run ends.
    """
    corners = set()
    for isl in islands:
        uv_layer = isl.uv_layer
        for cycle in isl.cycles:
            n = len(cycle)
            if n < 3:
                continue
            for k in range(n):
                prev, loop = cycle[k - 1], cycle[k]
                a = loop[uv_layer].uv - prev[uv_layer].uv
                b = loop.link_loop_next[uv_layer].uv - loop[uv_layer].uv
                if a.length < 1e-12 or b.length < 1e-12:
                    continue
                cosine = max(-1.0, min(1.0, a.normalized().dot(b.normalized())))
                if degrees(acos(cosine)) >= corner_angle_deg:
                    corners.add(loop.vert.index)
    return corners


def _tick_segments(islands, corner_angle_deg, matrix=None):
    """The seams, cut into corner-to-corner segments, oriented geometrically.

    A segment is what alignment marks are distributed along: a maximal
    stretch of seam between the same two pieces, ended by a junction (a
    third piece, or an open rim, meets it -- `_seam_chains` already breaks
    there) or by a corner of either piece's outline.

    Each segment is a list of (edge, entry_vert, length), walked from one
    end. Which end is chosen by coordinate, never by element order: the
    lower-coordinate end of an open segment; for a closed ring, its
    lowest-coordinate corner (or vertex, when it has none), walked towards
    the lower of its two neighbours. The distributions below are symmetric
    under reversal anyway, so this only fixes the last bits of arithmetic,
    which is what a reproducible export across a rebuilt mesh needs.

    Lengths are measured through `matrix` when given, so a spacing in real
    units means real units on a scaled object.
    """
    corners = _outline_corner_verts(islands, corner_angle_deg)

    def length(edge):
        a, b = edge.verts[0].co, edge.verts[1].co
        if matrix is not None:
            a, b = matrix @ a, matrix @ b
        return (b - a).length

    segments = []
    for chain in _seam_chains(islands):
        verts = [chain_direction_start(chain, 0)]
        for e in chain:
            verts.append(e.other_vert(verts[-1]))
        closed = len(chain) > 1 and verts[-1] is verts[0]
        if closed:
            ring = verts[:-1]
            # Edges: chain[k] joins ring[k] and ring[k + 1].
            cut = [k for k, v in enumerate(ring) if v.index in corners]
            pool = cut if cut else range(len(ring))
            s = min(pool, key=lambda k: _co_key(ring[k]))
            forward = _co_key(ring[(s + 1) % len(ring)]) <= \
                _co_key(ring[s - 1])
            if forward:
                order = [(chain[(s + k) % len(chain)], ring[(s + k) % len(ring)])
                         for k in range(len(chain))]
            else:
                order = [(chain[(s - 1 - k) % len(chain)],
                          ring[(s - k) % len(ring)])
                         for k in range(len(chain))]
        else:
            order = list(zip(chain, verts[:-1]))
            if _co_key(verts[-1]) < _co_key(verts[0]):
                order = [(e, e.other_vert(v)) for e, v in reversed(order)]

        current = []
        for k, (edge, entry) in enumerate(order):
            if k > 0 and entry.index in corners:
                segments.append(current)
                current = []
            current.append((edge, entry, length(edge)))
        if current:
            segments.append(current)
    return segments


def _segment_end_verts(islands, corner_angle_deg):
    """Indices of the vertices where a seam segment ends.

    The same two things `_tick_segments` ends a segment at: a junction (a
    boundary vertex where other than two boundary edges meet -- a third
    piece or an open rim joins) and a corner of either piece's outline.
    Nothing is drawn at these. A corner aligns itself: the two pieces can
    only meet one way there, so a mark on it tells the maker nothing and
    crosses the marks of the neighbouring edge.
    """
    degree = {}
    seen = set()
    for isl in islands:
        for cycle in isl.cycles:
            for loop in cycle:
                e = loop.edge
                if e.index in seen:
                    continue
                seen.add(e.index)
                for v in e.verts:
                    degree[v.index] = degree.get(v.index, 0) + 1
    ends = {v for v, n in degree.items() if n != 2}
    ends |= _outline_corner_verts(islands, corner_angle_deg)
    return ends


def _segment_key(segment):
    """A segment's place in the numbering: its vertices' coordinates.

    Coordinates, because they are the one key that survives both a fresh
    BMesh and a renumbering (CLAUDE.md, "What survives rebuilding a
    BMesh"). The whole sorted vertex list rather than the two ends, since
    two seams can run between the same two corners.
    """
    return tuple(sorted({_co_key(v) for edge, _entry, _len in segment
                         for v in edge.verts}))


def _seam_numbers(islands, corner_angle_deg):
    """edge index -> seam number, 1..N, one per corner-to-corner segment.

    A seam number is what the maker writes on both pieces that glue
    together along one stretch, so it belongs to the segment
    (`_tick_segments`), not to a piece or a cut run. An open rim has no
    neighbouring piece and gets no number.
    """
    seams = [seg for seg in _tick_segments(islands, corner_angle_deg)
             if any(len(edge.link_faces) == 2 for edge, _v, _l in seg)]
    seams.sort(key=_segment_key)
    numbers = {}
    for number, seg in enumerate(seams, 1):
        for edge, _entry, _len in seg:
            numbers[edge.index] = number
    return numbers


def _run_seams(run, seam_of_edge):
    """The distinct seam numbers a run's edges belong to, in run order."""
    out = []
    for loop in run.loops:
        number = seam_of_edge.get(loop.edge.index)
        if number is not None and number not in out:
            out.append(number)
    return out


def _blade_tip_inward(angle_deg, pattern_surface):
    """True when the blade's tip ends up on the piece's side of the line.

    The maker's rule: the arrow points "in the direction the tip of the
    blade will need to be from the edge line". The outline is drawn on the
    face of the foam that is up on the table; the blade enters on the
    outline and its tip leaves through the far face, to one side.

    Derivation, pattern on the OUTER skin (thickness t goes inward). Take
    the top of a cube: the piece is the slab z in [1 - t, 1] and its
    outline edge is the cube's edge x = 1, z = 1. The neighbouring piece is
    the slab x in [1 - t, 1]. Left square, the two slabs both claim the
    corner block x, z in [1 - t, 1]; the mitre that shares it is the plane
    bisecting the two faces, which passes through the outer edge and runs
    towards the cube's centre: x = z. On the far face of the top piece,
    z = 1 - t, that plane is at x = 1 - t -- *inside* the outline by
    t * tan(45 deg). So at a convex (mountain, positive) seam the far face
    is smaller than the outline, the tip is inside the piece, and the arrow
    points INTO the piece. The tilt off square is half the fold angle:
    `compute_bevel(...).per_side_deg`, 45 on a cube.

    A concave (valley, negative) seam is the mirror image: the bisecting
    plane leans away from the piece as it goes down through the foam, the
    far face has to reach past the outline to meet its neighbour, and the
    arrow points OUT of the piece.

    Pattern on the INNER skin (thickness goes outward) swaps "far face"
    for the other one, so both cases reverse.

    Sign convention as pinned in CLAUDE.md: positive = convex.
    """
    convex = angle_deg >= 0.0
    return convex if pattern_surface != 'INNER' else not convex


# Arrow proportions, as multiples of the angle font size so the arrow reads
# as part of its label at any sheet scale.
BEVEL_ARROW_GAP_EM = 0.3      # cut line to the arrow's near end
BEVEL_ARROW_LENGTH_EM = 1.2
BEVEL_ARROW_HEAD_EM = 0.4     # barb length; half that to each side
BEVEL_LABEL_PAD_EM = 0.3      # clear space between marks and text


def _bevel_arrow(mid, inward, font_mm, tip_inward, width_mm, colour):
    """The arrow for one run: tail, tip, barb, tip, barb.

    It lies on the piece's side of the cut line whichever way it points,
    square to the line, starting a small gap in: an inward arrow has its
    tail by the line and its tip further in, an outward one has its tip by
    the line -- pointing at it, and so out of the piece -- because an
    arrow drawn outside the outline is cut off with the waste.
    """
    near = mid + inward * (font_mm * BEVEL_ARROW_GAP_EM)
    far = near + inward * (font_mm * BEVEL_ARROW_LENGTH_EM)
    tail, tip = (near, far) if tip_inward else (far, near)
    along = (tip - tail).normalized()
    across = Vector((-along.y, along.x))
    head = font_mm * BEVEL_ARROW_HEAD_EM
    barb_a = tip - along * head + across * (head * 0.5)
    barb_b = tip - along * head - across * (head * 0.5)
    return Stroke('bevel_arrow',
                  (_pt(tail), _pt(tip), _pt(barb_a), _pt(tip), _pt(barb_b)),
                  False, width_mm, colour)


def _text_clearance(inward, text, font_mm):
    """How far along `inward` a centred label must sit for its box to clear
    a line through the origin square to `inward`."""
    width = len(text) * font_mm * GLYPH_ADVANCE_EM
    return abs(inward.x) * width / 2 + abs(inward.y) * font_mm / 2


class BevelMark(NamedTuple):
    """What one cut run is labelled with. Any of the three may be None."""
    island_index: int
    label: Optional[Label]        # kind 'angle', or 'heat'
    arrow: Optional[Stroke]       # kind 'bevel_arrow'
    heat_stroke: Optional[Stroke]  # kind 'heat'
    # The run's two ends in sheet mm, plain floats: the overlay hides a
    # label whose run is too short on screen to carry one.
    run_ends: Tuple[Point, Point]


def _bevel_marks(islands, uv_layer, runs_by_island, settings, font_mm,
                 stroke_mm):
    """Seam number, blade-tilt arrow and degrees for every cut run.

    One `BevelMark` per run that has anything to say, in island then run
    order. This is the single decision for both the sheet and the screen:
    `build_marks` sorts the parts into its groups, and the overlay asks
    this same function and maps the millimetres back to UV.

    Per run with a neighbouring piece (`has_dihedral`):

      * fold below `min_angle` -- a straight cut. The seam number alone, no
        arrow, no degrees. The maker's convention: no arrow means cut at
        90 degrees.
      * too steep to saw (`needs_heat_form`) -- the heat-form note, seam
        number first, when Heat-form Warnings is on. No arrow: it is not a
        saw cut.
      * otherwise "<seam>, <deg>" and an arrow. Degrees are the blade tilt
        for this one piece -- half the fold -- which is what the saw is set
        to. The sheet used to print the whole fold angle while the screen
        printed the half; both now print the half.

    A run labelled with a seam is labelled with the one its midpoint lies
    on. A run normally lies within one seam segment, but need not: runs
    break at this piece's own corners, segments also break where a third
    piece joins on the far side, and a straight edge shared with two
    neighbours is one run over two seams. Measured in
    tests/test_marks_model.py (`test_a_cut_run_can_span_two_seams`): never
    on the gore sphere, twice on a cube with its top cut in two -- and
    there the second seam's number is then written on its other piece
    only, never on this one. A known gap, not yet decided.

    `show_brackets` ("Cut Runs") gates the labels and arrows, as it gated
    the angle labels before; colour follows `show_dart_colors`.
    """
    seam_of_edge = _seam_numbers(islands, settings.corner_angle_deg)
    show_numbers = getattr(settings, 'show_seam_numbers', True)
    surface = getattr(settings, 'pattern_surface', 'OUTER')
    convex_rgb = tuple(settings.convex_color[:3])
    concave_rgb = tuple(settings.concave_color[:3])
    heat_rgb = tuple(settings.heat_form_color[:3])

    out = []
    for isl in islands:
        for run in runs_by_island[isl.index]:
            if not run.has_dihedral:
                continue
            mid, inward, index = _run_midpoint_frame(run, uv_layer, isl)
            seam = seam_of_edge.get(run.loops[index].edge.index)
            number = str(seam) if (show_numbers and seam is not None) else ""
            ends = (_pt(isl.to_sheet(run.uv_start)),
                    _pt(isl.to_sheet(run.uv_end)))

            if abs(run.mean_angle_deg) < settings.min_angle:
                # A straight cut: the number says which seam, and the
                # absence of an arrow says "square".
                if settings.show_brackets and number:
                    at = mid + inward * (
                        _text_clearance(inward, number, font_mm)
                        + font_mm * BEVEL_LABEL_PAD_EM)
                    out.append(BevelMark(isl.index, Label(
                        'angle', at.x, at.y, font_mm, number, BLACK,
                        'middle', 'neutral'), None, None, ends))
                continue

            bevel = core.compute_bevel(run.mean_angle_deg,
                                       settings.foam_thickness,
                                       settings.max_tilt_deg)
            degrees_text = f"{abs(bevel.per_side_deg):.{settings.decimals}f}°"
            # A comma between them, the maker's request: "4 3°" was read
            # as "43°". "4, 3°" cannot be.
            text = f"{number}, {degrees_text}" if number else degrees_text

            if bevel.needs_heat_form:
                # Needs a heat-form joint, not a saw cut -- never also
                # present it as an ordinary sawable bevel among the angles.
                if settings.show_heat_form:
                    chord = (isl.to_sheet(run.uv_start)
                             + isl.to_sheet(run.uv_end)) / 2
                    pts = tuple(_pt(p)
                                for p in _run_polyline(run, uv_layer, isl))
                    out.append(BevelMark(
                        isl.index,
                        Label('heat', chord.x, chord.y, font_mm,
                              f"{text} (heat-form)", heat_rgb, 'baseline'),
                        None,
                        Stroke('heat', pts, False, 1.0, heat_rgb, (2.0, 1.0)),
                        ends))
                continue

            if not settings.show_brackets:
                continue
            if settings.show_dart_colors:
                if run.mean_angle_deg >= 0:
                    tone, colour = "convex", convex_rgb
                else:
                    tone, colour = "concave", concave_rgb
            else:
                tone, colour = "neutral", BLACK
            arrow = _bevel_arrow(
                mid, inward, font_mm,
                _blade_tip_inward(run.mean_angle_deg, surface),
                stroke_mm, colour)
            # Line, gap, arrow, gap, text: the text starts where the arrow
            # ends, so the two never touch and both stay on the piece.
            at = mid + inward * (
                font_mm * (BEVEL_ARROW_GAP_EM + BEVEL_ARROW_LENGTH_EM
                           + BEVEL_LABEL_PAD_EM)
                + _text_clearance(inward, text, font_mm))
            out.append(BevelMark(isl.index, Label(
                'angle', at.x, at.y, font_mm, text, colour, 'middle', tone),
                arrow, None, ends))
    return out


def _runs_by_cycle(isl, runs):
    """An island's runs grouped by the boundary cycle they lie on, in run
    order, so a cycle's last run ends where its first starts."""
    cycle_loop_sets = [set(cyc) for cyc in isl.cycles]
    out = [[] for _ in isl.cycles]
    for run in runs:
        first = run.loops[0]
        for ci, loopset in enumerate(cycle_loop_sets):
            if first in loopset:
                out[ci].append(run)
                break
    return out


def _run_end_marks(islands, uv_layer, runs_by_island, settings,
                   tick_length_mm, width_mm):
    """(island index, 'bracket' Stroke) for each run end part way along a
    seam segment.

    A run ends where this piece's outline turns a corner, or where the bevel
    angle drifts or the chord bows too far for one saw pass. Only the second
    kind gets a mark: the bevel label changes there, and nothing else on the
    sheet says where. A corner or a junction (`_segment_end_verts`) gets
    none -- the maker: "There is no need for alignment marks at the corners.
    The corners are implicitly aligning." There used to be one on every
    corner of every piece, crossing the neighbouring piece's.

    Drawn as an alignment mark is: starting on the cut line and going into
    the piece only, square to the line -- to the average of the two edges
    when it sits on a bend -- `tick_length` long, shortened where the piece
    is too narrow to take it.

    The caller decides whether to draw them (`show_brackets`).
    """
    segment_ends = _segment_end_verts(islands, settings.corner_angle_deg)
    out = []
    for isl in islands:
        for cyc_runs in _runs_by_cycle(isl, runs_by_island[isl.index]):
            n = len(cyc_runs)
            if n < 2:
                continue
            for k in range(n):
                run_a, run_b = cyc_runs[k], cyc_runs[(k + 1) % n]
                end_vert = run_a.loops[-1].link_loop_next.vert
                if end_vert.index in segment_ends:
                    continue
                at = isl.to_sheet(run_a.uv_end)
                inward, room = _inward_at(isl, uv_layer, at,
                                          run_a.loops[-1], run_b.loops[0])
                reach = min(tick_length_mm, max(room * 0.6, 0.2))
                out.append((isl.index, Stroke(
                    'bracket', (_pt(at), _pt(at + inward * reach)),
                    False, width_mm, BLACK)))
    return out


def _segment_distances(total, settings, mode=None):
    """Where along a segment of length `total` its marks go.

    PER_RUN: N marks at i/(N+1) of the length -- never at either end.
    BY_DISTANCE: every `tick_spacing`, the leftover split equally between
    the two ends; a segment no longer than one spacing gets one mark in its
    middle. Both are symmetric under reversing the segment.
    """
    mode = mode or settings.tick_mode
    if mode == 'BY_DISTANCE':
        step = max(float(settings.tick_spacing), 1e-4)
        # Gaps strictly inside the segment, so no mark lands on an end. The
        # epsilon keeps a segment of exactly k steps (0.2 m at 0.05) from
        # flipping between k-1 and k gaps on its last bit of arithmetic.
        gaps = max(0, ceil(total / step - 1e-6) - 1)
        first = (total - gaps * step) / 2.0
        return [first + i * step for i in range(gaps + 1)]
    count = max(1, int(settings.ticks_per_run))
    return [total * i / (count + 1) for i in range(1, count + 1)]


def _segment_tick_params(islands, settings, matrix=None, mode=None):
    """Every alignment mark, as (edge, t) from the edge's canonical start.

    Decided once per geometric segment, so both pieces sharing the seam
    derive the same point and carry the same dash.
    """
    out = []
    for segment in _tick_segments(islands, settings.corner_angle_deg,
                                  matrix):
        total = sum(seg_len for _e, _v, seg_len in segment)
        if total <= 1e-12:
            continue
        index, walked = 0, 0.0
        for d in _segment_distances(total, settings, mode):
            while index < len(segment) - 1 and \
                    d > walked + segment[index][2]:
                walked += segment[index][2]
                index += 1
            edge, entry, seg_len = segment[index]
            if seg_len <= 1e-12:
                continue
            local = max(0.0, min(1.0, (d - walked) / seg_len))
            if entry is not _edge_canonical_start(edge):
                local = 1.0 - local
            out.append((edge, round(local, 6)))
    return out


def _seam_tick_params(islands, settings, matrix=None):
    """By Distance placement, under its old name (re-exported by
    foam_export). See `_segment_tick_params`."""
    return _segment_tick_params(islands, settings, matrix, 'BY_DISTANCE')


def chain_direction_start(chain, index):
    """Which vertex of `chain[index]` the walk entered it from."""
    edge = chain[index]
    if index == 0:
        if len(chain) == 1:
            return edge.verts[0]
        shared = _shared_vert(edge, chain[1])
        if shared is not None:
            return edge.other_vert(shared)
        return edge.verts[0]
    shared = _shared_vert(edge, chain[index - 1])
    if shared is not None:
        return shared
    return edge.verts[0]


def _shared_vert(edge, other):
    """The first of `edge`'s vertices that `other` also uses, or None.

    Walked in `edge.verts` order rather than read out of a set intersection:
    `next(iter(set))` over BMesh elements picks by Blender-side address.
    Two distinct edges share at most one vertex, so this only fixes the
    pathological case -- but it means the question never needs asking."""
    return next((v for v in edge.verts if v in other.verts), None)


BLACK = (0.0, 0.0, 0.0)


def _pt(v):
    """A mathutils Vector as a plain (x, y) of Python floats.

    The arithmetic before this point stays in Vectors, exactly as the SVG
    exporter always did it, so the numbers that reach the sheet are the same
    ones it always printed -- only the moment of copying out moves.
    """
    return (v.x, v.y)


def _alignment_ticks(islands, uv_layer, runs_by_island, settings,
                     tick_length_mm, font_mm, width_mm, matrix=None):
    """Tick marks on every seam, drawn on both pieces that share it.

    Returns (strokes, labels): one 'align' Stroke per dash, and one 'align'
    Label per dash when `settings.tick_numbers` is on.

    Built in two passes, and the split is the whole point.

    First the positions are decided on the *3D mesh* -- a set of (edge, t)
    pairs -- so a tick belongs to the seam rather than to a piece. Then every
    island that borders that edge draws it, mapping t into its own UVs. Two
    pieces glued along a seam therefore carry marks at the same place on the
    foam, which is what makes them registration marks rather than decoration.

    Deciding per island instead would let the two sides disagree: their runs
    are segmented from their own boundaries and need not break at the same
    places.

    Both modes distribute along corner-to-corner segments of the seam
    (`_tick_segments`), not along cut runs. `runs_by_island` is no longer
    read; it stays in the signature for the overlay, which calls this.
    `matrix` (the object's world matrix) makes By Distance spacing a real
    length on a scaled object; None measures in object space.
    """
    wanted = {}
    for edge, t in _segment_tick_params(islands, settings, matrix):
        _remember_tick(wanted, edge, t)
    if not wanted:
        return [], []

    # A stable index per seam, so "tick 3" means the same mark on both of
    # its pieces when the numbers are switched on.
    numbering = {}
    for edge, params in wanted.items():
        params.sort()
        for order, t in enumerate(params):
            numbering[(edge, t)] = order + 1

    strokes, labels = [], []
    for isl in islands:
        # One dash per piece per mark. An island's boundary can in principle
        # traverse the same 3D edge twice -- a seam running up to a point and
        # back down the far side is met from both directions -- and the mark
        # would then be drawn twice at identical coordinates. Physically it
        # is one dash on one piece of foam.
        #
        # Note this was *not* the cause of the duplicates seen on a six-gore
        # sphere, though it was the first explanation offered for them. Those
        # came from two runs recording the same point as 0.666666 and
        # 0.666667, which `_remember_tick` now merges. The guard is kept for
        # the case it actually describes, which this fixture does not reach.
        drawn = set()
        for cycle in isl.cycles:
            for k, loop in enumerate(cycle):
                edge = loop.edge
                params = wanted.get(edge)
                if not params:
                    continue
                uv_a = loop[uv_layer].uv
                uv_b = loop.link_loop_next[uv_layer].uv
                start_vert = _edge_canonical_start(edge)

                for t in sorted(params):
                    if (edge.index, t) in drawn:
                        continue
                    drawn.add((edge.index, t))
                    local = t if loop.vert is start_vert else 1.0 - t
                    uv = uv_a + (uv_b - uv_a) * local
                    point = isl.to_sheet(uv)
                    # Square to the edge the mark sits on. A mark exactly on
                    # a vertex sits on two edges at once, and is squared to
                    # the average of them. (It is never a segment end --
                    # marks are placed strictly inside a segment -- so the
                    # neighbour is the same seam.)
                    other = None
                    if len(cycle) > 1:
                        if local <= 1e-9:
                            other = cycle[k - 1]
                        elif local >= 1.0 - 1e-9:
                            other = cycle[(k + 1) % len(cycle)]
                    # The face this loop belongs to is inside the island by
                    # definition, so its own centre is a locally correct
                    # "in". An island centroid is not: on a ring it sits in
                    # the hole.
                    inward, room = _inward_at(isl, uv_layer, point, loop,
                                              other)
                    # Shorten where the piece cannot take the full mark. The
                    # face this loop belongs to spans the piece locally, so
                    # the distance to its centre is a cheap stand-in for how
                    # much room there is; a gore's tip has very little, and
                    # two full-length marks facing each other across it meet
                    # in the middle.
                    reach = min(tick_length_mm, max(room * 0.6, 0.2))
                    # From the cut line inward, and nothing outside it. The
                    # mark used to carry a short tail across the line "so it
                    # reads as a crossing"; the tail is cut away with the
                    # waste, and the maker asked for marks that start on the
                    # edge and extend inwards, perpendicular to it.
                    tip = point + inward * reach
                    strokes.append(Stroke('align', (_pt(point), _pt(tip)),
                                          False, width_mm, BLACK))
                    if settings.tick_numbers:
                        at = point + inward * (reach * 1.45)
                        labels.append(Label('align', at.x, at.y, font_mm,
                                            str(numbering[(edge, t)]),
                                            BLACK, 'middle'))
    return strokes, labels


RULER_GAP_MM = 5.0
RULER_CAPTION = ("measure before cutting:",
                 "if not exact, the print was scaled")
# A label is centred on its x, and no backend can be asked how wide a
# string is from here, so captions are placed from an upper bound on the
# advance of a sans-serif glyph. Too generous only moves a caption right.
GLYPH_ADVANCE_EM = 0.6


def _ruler_strip_width():
    """The narrowest sheet the whole ruler strip fits on."""
    longest = max(len(c) for c in RULER_CAPTION)
    caption = longest * RULER_FONT_MM * GLYPH_ADVANCE_EM
    return (MARGIN_MM + RULER_LENGTH_MM + 18.0 + INCH_MM + RULER_GAP_MM
            + caption + MARGIN_MM)


def _scale_strip(sheet_width, top_y):
    """A ruler and a one-inch square, in absolute millimetres.

    Returns (strokes, labels), all of kind 'ruler'. The square is a closed
    four-point stroke.

    Printers scale. "Fit to page" silently shrinks a sheet by a few percent,
    and a pattern that is 3% small produces panels that do not meet -- which
    is only discovered after the foam is cut. Measuring a printed ruler is the
    cheapest check there is, so one goes on every sheet.

    Both units, because the stock and the tools rarely agree: foam sheet is
    sold in metric here and blade settings are often imperial.
    """
    strokes, labels = [], []
    width = 0.3

    def line(x1, y1, x2, y2):
        strokes.append(Stroke('ruler', ((x1, y1), (x2, y2)), False,
                              width, BLACK))

    def text(x, y, s):
        labels.append(Label('ruler', x, y, RULER_FONT_MM, s, BLACK,
                            'baseline'))

    x0 = MARGIN_MM
    # The baseline sits a full inch below the strip's top, plus a gap, so
    # the one-inch square stands wholly inside the strip. It used to start
    # from 45% of the strip, which put the square's top 10 mm above the
    # strip -- touching the lowest piece's cut line.
    y = top_y + RULER_GAP_MM + INCH_MM

    # 100 mm bar with a tick every 10 mm, taller every 50.
    line(x0, y, x0 + RULER_LENGTH_MM, y)
    step = 10.0
    n = int(RULER_LENGTH_MM / step)
    for i in range(n + 1):
        tx = x0 + i * step
        tall = RULER_TICK_MM * (1.6 if i % 5 == 0 else 1.0)
        line(tx, y, tx, y - tall)
    text(x0, y + RULER_FONT_MM * 1.6, "0")
    text(x0 + RULER_LENGTH_MM, y + RULER_FONT_MM * 1.6, "100 mm")

    # A one-inch square, which is far easier to check by eye than a bar.
    sq_x = x0 + RULER_LENGTH_MM + 18.0
    sq_y = y - INCH_MM
    strokes.append(Stroke('ruler', ((sq_x, sq_y), (sq_x + INCH_MM, sq_y),
                                    (sq_x + INCH_MM, sq_y + INCH_MM),
                                    (sq_x, sq_y + INCH_MM)),
                          True, width, BLACK))
    text(sq_x + INCH_MM / 2, y + RULER_FONT_MM * 1.6, "1 inch")

    # Two short lines to the right of the square, clear of it. One long
    # line centred 14 mm past the square ran back through the square.
    left = sq_x + INCH_MM + RULER_GAP_MM
    for row, caption in enumerate(RULER_CAPTION):
        half = len(caption) * RULER_FONT_MM * GLYPH_ADVANCE_EM / 2
        text(left + half, y - INCH_MM / 2 + row * RULER_FONT_MM * 1.5,
             caption)
    return strokes, labels


def build_marks(bm, uv_layer, settings, scale_length=1.0, matrix=None):
    """Everything to draw for this mesh, as a `Marks`.

    `matrix` is the object's world matrix; lengths and angles are measured
    through it, as the overlay does. Pure: no bpy.context, no file I/O.

    Every `settings.show_*` and `min_angle` decision is made here and nowhere
    else on the export path; a backend draws what it is given.
    """
    islands_faces = core.uv_islands(bm, uv_layer, only_selected=False)
    sheet = _Sheet(islands_faces, uv_layer, scale_length, matrix)
    islands = [_Island(i, faces, uv_layer, sheet)
               for i, faces in enumerate(islands_faces)]

    id_layer = bm.faces.layers.int.get('foam_island_id')
    if id_layer is not None:
        for isl in islands:
            rep = next(iter(isl.faces), None)
            if rep is not None:
                isl.piece_id = rep[id_layer]

    total_width, total_height = sheet.width, sheet.height
    if settings.show_scale_ruler:
        # Added below the pattern, so the pattern's own coordinates --
        # and therefore its fidelity to the UV layout -- are unchanged.
        total_height += RULER_STRIP_MM
        total_width = max(total_width, _ruler_strip_width())

    thickness_mm = settings.foam_thickness * scale_length * 1000.0
    dart_depth_mm = settings.notch_depth_ratio * thickness_mm

    # Annotation sizes, in millimetres, from the same permille settings the
    # overlay draws with -- so the printed sheet matches the screen.
    overlay_scale = settings.overlay_scale
    angle_font_mm = sheet.permille(settings.angle_text_size, overlay_scale)
    number_font_mm = sheet.permille(settings.number_size, overlay_scale)
    # (mark_size no longer sizes anything on the sheet: run-end marks are
    # now drawn as alignment marks are, at the physical tick_length.)
    # Hairlines scale with the sheet too, or a 2 m pattern gets invisible
    # outlines and a 100 mm one gets a smeared blob.
    stroke_mm = sheet.permille(2, overlay_scale)

    convex_rgb = tuple(settings.convex_color[:3])
    concave_rgb = tuple(settings.concave_color[:3])
    heat_rgb = tuple(settings.heat_form_color[:3])

    # The one call for runs on the export path, asked exactly as the overlay
    # asks it (foam_overlay._rebuild). It used to pass none of the corner,
    # matrix or dart arguments, so the sheet segmented the boundary with the
    # core's defaults and measured angles in object space, while the screen
    # used the user's Corner Angle and the world matrix -- paper and screen
    # disagreed on how many runs a piece had and what angle each one was.
    all_runs = list(core.iter_boundary_dart_segments(
        bm, uv_layer, settings.max_chord_deviation,
        settings.max_angle_variation_deg, islands=islands_faces,
        corner_angle_deg=settings.corner_angle_deg,
        matrix=matrix,
        dart_wedge_deg=settings.dart_wedge_deg,
        max_darts_per_run=settings.max_darts_per_run))
    runs_by_island = [[] for _ in islands]
    for r in all_runs:
        runs_by_island[r.island_index].append(r)

    piece_fills, relief_fills = [], []
    outlines, reliefs, brackets, heat_strokes = [], [], [], []
    angle_labels, number_labels, heat_labels = [], [], []
    arrows = []

    # --- seam number, blade-tilt arrow and degrees: one entry per run.
    for mark in _bevel_marks(islands, uv_layer, runs_by_island, settings,
                             angle_font_mm, stroke_mm):
        if mark.heat_stroke is not None:
            heat_strokes.append(mark.heat_stroke)
        if mark.arrow is not None:
            arrows.append(mark.arrow)
        if mark.label is not None:
            (heat_labels if mark.label.kind == 'heat'
             else angle_labels).append(mark.label)

    # --- run-end marks: only part way along a seam, never at a corner.
    if settings.show_brackets:
        brackets = [stroke for _i, stroke in _run_end_marks(
            islands, uv_layer, runs_by_island, settings,
            settings.tick_length * scale_length * 1000.0, stroke_mm * 0.7)]

    for isl in islands:
        # --- outlines: one polygon per boundary cycle (an island with a hole
        # yields more than one).
        for cycle in isl.cycles:
            if len(cycle) < 2:
                continue
            pts = tuple(_pt(isl.to_sheet(l[uv_layer].uv)) for l in cycle)
            outlines.append(Stroke('outline', pts, True, stroke_mm, BLACK))

        # --- fill: one colour per piece, behind everything else.
        rings = core.order_rings_outer_first(
            [[(p.x, p.y) for p in
              (isl.to_sheet(l[uv_layer].uv) for l in cyc)]
             for cyc in isl.cycles])
        if settings.island_fill and rings:
            # Every cycle as a ring, filled even-odd, so a piece with a hole
            # in it reads as a hole rather than a disc.
            colour = tuple(core.island_palette_color(isl.index, len(islands)))
            piece_fills.append(Fill(
                'piece', tuple(tuple(tuple(p) for p in ring)
                               for ring in rings), colour))

        # --- numbers: piece ID where it actually lands on the piece.
        if settings.show_island_ids and rings:
            # Not the centroid. The centroid of a crescent or a C lies outside
            # the shape, so the number printed there sits on the sheet beside
            # the piece -- and vanishes entirely once the pattern is used as a
            # texture on the model, which is the main reason it is printed.
            # The pole of inaccessibility is inside by construction, and sits
            # in the piece's widest part so the digits have room.
            (px, py), room = core.pole_of_inaccessibility(rings)
            if px is not None:
                # Shrink a number that will not fit rather than letting
                # it spill over the cut lines of a small piece -- but never
                # below something a person can read. The floor used to be
                # 0.1 mm, which is not a smaller label, it is an invisible
                # one; three pieces on a test sphere printed their number at
                # that size and looked simply absent.
                room_mm = max(room, 0.0)
                size = min(number_font_mm, max(room_mm * 1.2,
                                               number_font_mm * 0.25))
                number_labels.append(Label('number', px, py, size,
                                           str(isl.piece_id), BLACK,
                                           'middle'))

        # --- ticks + darts: one per run-to-run corner, grouped by boundary
        # cycle so the wraparound (last run's end == first run's start)
        # closes correctly.
        runs_by_cycle = _runs_by_cycle(isl, runs_by_island[isl.index])

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

                # Relief cuts remove material, so they are opt-in: the same
                # "Relief Cuts" toggle (show_darts) the overlay obeys. They
                # used to be emitted regardless, so every exported sheet
                # carried wedges the screen did not show.
                if not settings.show_darts:
                    continue
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
                    reliefs.append(Stroke(
                        'relief', (_pt(p_left), _pt(apex), _pt(p_right)),
                        True, stroke_mm * 0.7, BLACK))
                else:  # SLIT
                    # A slit is cut out whole: a filled quad, with its
                    # outline stroked too, as the sheet has always drawn it.
                    half_width = DART_SLIT_WIDTH_MM / 2
                    p1 = boundary_pt - tangent * half_width
                    p2 = boundary_pt + tangent * half_width
                    p3 = p2 + inward * dart_depth_mm
                    p4 = p1 + inward * dart_depth_mm
                    quad = tuple(_pt(p) for p in (p1, p2, p3, p4))
                    relief_fills.append(Fill('relief', (quad,), BLACK))
                    reliefs.append(Stroke('relief', quad, True,
                                          stroke_mm * 0.7, BLACK))

    align, align_labels = [], []
    if settings.show_align_ticks:
        # A real length, not a fraction of the sheet. The mark is made with a
        # pen on foam and is the same size whatever the pattern; sizing it
        # from the sheet gave a 13 mm tick on a 838 mm sheet, which on a gore
        # only 30 mm wide at the ends meant the marks from its two seams
        # reached across and crossed each other -- 12 colliding pairs,
        # measured, and it read as doubled ticks.
        align, align_labels = _alignment_ticks(
            islands, uv_layer, runs_by_island, settings,
            settings.tick_length * scale_length * 1000.0,
            angle_font_mm * 0.7, stroke_mm * 0.9, matrix)

    ruler, ruler_labels = [], []
    if settings.show_scale_ruler:
        ruler, ruler_labels = _scale_strip(total_width, sheet.height)

    # Sequence order is the sheet's paint order, group by group.
    return Marks(
        width_mm=total_width,
        height_mm=total_height,
        fills=tuple(piece_fills + relief_fills),
        strokes=tuple(outlines + reliefs + brackets + align + arrows
                      + heat_strokes + ruler),
        labels=tuple(align_labels + angle_labels + number_labels
                     + heat_labels + ruler_labels),
        hairline_mm=stroke_mm,
        convex_color=convex_rgb,
        concave_color=concave_rgb,
        heat_color=heat_rgb,
        warnings=scale_warnings(matrix),
    )
