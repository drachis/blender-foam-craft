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
                 'bracket'  a run-end bracket / tick
                 'align'    an alignment (registration) mark
                 'relief'   a relief cut drawn open
                 'heat'     a heat-form run marker
                 'ruler'    scale bar and 1 inch square
    Label.kind   'number'   a piece number, inside its piece
                 'angle'    a bevel angle
                 'align'    an alignment mark's number
                 'heat'     a heat-form note
                 'ruler'    scale bar captions
"""

from math import radians, tan
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
RULER_STRIP_MM = 34.0
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
    """Where this run wants its ticks, as (edge, t) on the 3D mesh.

    Per Cut Run only. `t` runs from the edge's canonical start, so the two
    faces sharing the edge derive the same point from it.

    By Distance is not done here, and cannot be: a run is one blade-tilt
    pass, and restarting the measurement at each run makes the gaps depend on
    where the bevel happens to change rather than on the foam. Asking for a
    mark every 30 mm and getting them every 12, 30, 9, 30, 30 is not what the
    setting says. It walks the seam instead -- see `_seam_tick_params`.
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


def _seam_tick_params(islands, settings):
    """Marks every `tick_spacing` of real seam, walked continuously.

    Returned as (edge, t) from each edge's canonical start, so both pieces
    sharing the seam derive the same point and carry the same dash.
    """
    step = max(float(settings.tick_spacing), 1e-4)
    out = []
    for chain in _seam_chains(islands):
        lengths = [e.calc_length() for e in chain]
        total = sum(lengths)
        if total <= 1e-9:
            continue
        # Half a step in from the start, so a chain does not open with a mark
        # sitting on its own corner.
        d = step * 0.5
        if d >= total:
            # Shorter than one step. One mark in the middle still beats none:
            # a short seam is exactly where a piece can slip unnoticed.
            d = total * 0.5
        walked = 0.0
        index = 0
        while d < total and index < len(chain):
            while index < len(chain) and d > walked + lengths[index]:
                walked += lengths[index]
                index += 1
            if index >= len(chain):
                break
            edge = chain[index]
            local = (d - walked) / lengths[index]
            start = _edge_canonical_start(edge)
            # The chain is walked from one end; express the parameter from
            # the edge's own canonical end so both faces agree on it.
            if chain_direction_start(chain, index) is not start:
                local = 1.0 - local
            out.append((edge, round(max(0.0, min(1.0, local)), 6)))
            d += step
    return out


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
                     tick_length_mm, font_mm, width_mm):
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
    """
    wanted = {}
    if settings.tick_mode == 'BY_DISTANCE':
        for edge, t in _seam_tick_params(islands, settings):
            _remember_tick(wanted, edge, t)
    else:
        for runs in runs_by_island:
            for run in runs:
                for edge, t in _run_tick_params(run, settings):
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
            for loop in cycle:
                edge = loop.edge
                params = wanted.get(edge)
                if not params:
                    continue
                uv_a = loop[uv_layer].uv
                uv_b = loop.link_loop_next[uv_layer].uv
                start_vert = _edge_canonical_start(edge)
                # The face this loop belongs to is inside the island by
                # definition, so its own centre is a locally correct "in".
                # An island centroid is not: on a ring it sits in the hole.
                face_uvs = [l[uv_layer].uv for l in loop.face.loops]
                face_c = Vector((
                    sum(p.x for p in face_uvs) / len(face_uvs),
                    sum(p.y for p in face_uvs) / len(face_uvs)))
                face_pt = isl.to_sheet(face_c)

                for t in sorted(params):
                    if (edge.index, t) in drawn:
                        continue
                    drawn.add((edge.index, t))
                    local = t if loop.vert is start_vert else 1.0 - t
                    uv = uv_a + (uv_b - uv_a) * local
                    point = isl.to_sheet(uv)
                    tangent = _boundary_tangent(isl.to_sheet(uv_a), point)
                    if tangent.length < 1e-9:
                        continue
                    inward = _inward_normal(tangent, point, face_pt)
                    # Shorten where the piece cannot take the full mark. The
                    # face this loop belongs to spans the piece locally, so
                    # the distance to its centre is a cheap stand-in for how
                    # much room there is; a gore's tip has very little, and
                    # two full-length marks facing each other across it meet
                    # in the middle.
                    room = (face_pt - point).length
                    reach = min(tick_length_mm, max(room * 0.6, 0.2))
                    # Mostly inward. The outward half of a mark drawn across
                    # the line is cut away with the waste, so it would be ink
                    # spent on nothing; a short tail is kept so the mark is
                    # visibly a crossing and not a scratch.
                    tip = point + inward * reach
                    tail = point - inward * (reach * 0.25)
                    strokes.append(Stroke('align', (_pt(tail), _pt(tip)),
                                          False, width_mm, BLACK))
                    if settings.tick_numbers:
                        at = point + inward * (reach * 1.45)
                        labels.append(Label('align', at.x, at.y, font_mm,
                                            str(numbering[(edge, t)]),
                                            BLACK, 'middle'))
    return strokes, labels


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
    y = top_y + RULER_STRIP_MM * 0.45

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

    text(sq_x + INCH_MM + 14.0, y,
         "measure this: if it is not exact, the print was scaled")
    return strokes, labels


def build_marks(bm, uv_layer, settings, scale_length=1.0, matrix=None):
    """Everything to draw for this mesh, as a `Marks`.

    `matrix` is the object's world matrix; lengths and angles are measured
    through it, as the overlay does. Pure: no bpy.context, no file I/O.

    Every `settings.show_*` and `min_angle` decision is made here and nowhere
    else on the export path; a backend draws what it is given.
    """
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
    if settings.show_scale_ruler:
        # Added below the pattern, so the pattern's own coordinates --
        # and therefore its fidelity to the UV layout -- are unchanged.
        total_height += RULER_STRIP_MM
        total_width = max(total_width, MARGIN_MM * 2 + 210.0)

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
                    pts = tuple(_pt(p) for p in _run_polyline(run, uv_layer, isl))
                    heat_strokes.append(Stroke('heat', pts, False, 1.0,
                                               heat_rgb, (2.0, 1.0)))
                    heat_labels.append(Label('heat', mid.x, mid.y,
                                             angle_font_mm,
                                             f"{label} (heat-form)",
                                             heat_rgb, 'baseline'))
                # Needs a heat-form joint, not a saw cut -- never also present
                # it as an ordinary sawable bevel among the angles.
                continue
            if settings.show_brackets:
                if settings.show_dart_colors:
                    if run.mean_angle_deg >= 0:
                        tone, colour = "convex", convex_rgb
                    else:
                        tone, colour = "concave", concave_rgb
                else:
                    tone, colour = "neutral", BLACK
                angle_labels.append(Label('angle', mid.x, mid.y,
                                          angle_font_mm, label, colour,
                                          'middle', tone))

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
                    brackets.append(Stroke('bracket', (_pt(tick_a), _pt(tick_b)),
                                           False, stroke_mm * 0.7, BLACK))

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
            angle_font_mm * 0.7, stroke_mm * 0.9)

    ruler, ruler_labels = [], []
    if settings.show_scale_ruler:
        ruler, ruler_labels = _scale_strip(total_width, sheet.height)

    # Sequence order is the sheet's paint order, group by group.
    return Marks(
        width_mm=total_width,
        height_mm=total_height,
        fills=tuple(piece_fills + relief_fills),
        strokes=tuple(outlines + reliefs + brackets + align
                      + heat_strokes + ruler),
        labels=tuple(align_labels + angle_labels + number_labels
                     + heat_labels + ruler_labels),
        hairline_mm=stroke_mm,
        convex_color=convex_rgb,
        concave_color=concave_rgb,
        heat_color=heat_rgb,
    )
