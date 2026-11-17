"""Bevel labels that fit: slide, shrink, and failing that, a leader line.

WHY THIS EXISTS

`foam_marks.build_marks` puts every run's bevel label -- one turned string,
"→ <seam>, <deg>°", square to the cut line with its arrow glyph at the
line -- at the middle of the run. On a real pattern that is unreadable on
small and dense pieces: labels land on each other, on the piece number and
across cut lines. The maker chose the fix (2026-11-17), in this order:

  1. "Slide and shrink to fit": move the label along its own run to a clear
     spot, shrinking it if need be, never below `min_label_mm`, the
     smallest text the maker's printer and eyes can read. That number is a
     physical limit, not a tuning knob (CLAUDE.md).
  2. "Leader lines outside": if nothing fits inside, put the label outside
     the piece in free sheet space, with a thin 'leader' line to its run.

WHAT IT IS

A pure pass over plain data: `layout_labels(marks) -> Marks`. It reads no
mesh and no settings beyond its arguments. Everything it needs to slide a
label rides on the label itself, in its `foam_marks.BevelAnchor`: the run
portion it labels, the inward normal per segment of it, the piece outline,
and the blade direction. Labels without an anchor, and every other mark,
are obstacles that are never moved -- the piece number included (the maker
did not ask for that to move).

HOW IT DECIDES, so the result can be explained

A label's box is the contract's: `len(text) * size * GLYPH_ADVANCE_EM` by
`size`, centred on (x, y) and TURNED by `rotation_deg` -- every overlap
test here is on that oriented box, never on an axis-aligned one. Lines are
segments. A label is placed by `foam_marks.bevel_label_frame`, the same
function `build_marks` uses: reading square to the cut line at the point
it is moved to (so sliding round a bend turns it), never upside down, the
glyph at the end nearest the line pointing the blade's way, that end the
arrow gap (0.3 em) from the line. A candidate is clear when its box,
padded by `CLEAR_EM`, overlaps no other box and crosses no line, and --
inside -- lies wholly in its own piece.

ONE SIZE PER PIECE (2026-11-17, the coordinator's review: sizes jumping
inside one piece read as accidental). A piece whose labels are all clear
where `build_marks` put them is kept exactly as it is. Every other piece is
laid out as a whole, in a geometric order (by its labels' original
positions; never by element identity), against what is fixed or already
placed:

  for each size z from full down to `min_label_mm` (`SIZE_STEPS`):
      every label of the piece is tried inside at size z, at the run
      fractions in `FRACTIONS`, nearest the middle first;
      at most `leader_allowance(n) = max(1, n // 4)` labels may fail
      inside -- they go outside on leaders at the same size z;
      if more fail, or a leader cannot be placed, undo and try the next z.
  at the floor, whatever fails inside goes on a leader regardless.

So a piece is drawn at the largest size where all its labels fit, letting
one crowded corner (or a quarter of a big piece) go out on leaders rather
than shrinking every label for it, and a leader label is the size of the
labels inside its piece, so it never dominates them.

Outside: the label keeps the turn it would have at that point of the line
and goes out along the outward normal, or turned off it by
`OUTSIDE_TURNS_DEG`, at `OUTSIDE_GAPS_EM` from the line, clear of every
piece and inside the sheet; its glyph moves to the end nearest the line.
The leader starts `LEADER_GAP_EM` off the box and ends on the cut line, at
the point on the run the label was placed for, and is never shorter than
`MIN_LEADER_EM` text heights, so it reads as a line, not a dot.

Piece names are sized here too (`fit_piece_names`), before any of this
runs: each as large as fits inside its own piece, clear of the outline and
the ticks, and from then on a fixed obstacle like any other mark.

A label that fits nowhere keeps its original place at `min_label_mm` at
the least, and is reported as 'unplaced'. Nothing is dropped.
"""

from collections import defaultdict
from math import cos, floor, hypot, radians, sin

from .foam_marks import (ARROW_LEFT, ARROW_RIGHT, BEVEL_ARROW_GAP_EM,
                         GLYPH_ADVANCE_EM, Stroke, bevel_label_frame)

# Positions along a run portion, nearest its middle first. Fifteen interior
# sixteenths: the ends are corners, where a label would read as belonging
# to either edge.
FRACTIONS = tuple(sorted((i / 16.0 for i in range(1, 16)),
                         key=lambda f: (abs(f - 0.5), f)))
# Piece sizes tried, as fractions of the size build_marks chose. Clamped to
# `min_label_mm` from below; the floor itself is always tried last.
SIZE_STEPS = (1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.12)
# Clear space kept around a label's box, in its own em.
CLEAR_EM = 0.15
# A leader stops this far short of its text, so it does not run into it.
LEADER_GAP_EM = 0.25
# The shortest leader drawn, in text heights of its own label. Below about
# a text height a leader reads as a dot or a stray nick, not a line: the
# first layout put labels 0.6 em off the line, and its leaders came out at
# 0.35 em -- under 1 mm at the 2.5 mm floor, which the maker took for a
# speck beside "32, 17°" (2026-11-17). One and a half heights reads as a
# pointer at every label size, because it scales with the label.
MIN_LEADER_EM = 1.5
# Outside: gap from the cut line to the label's box, in label em, and turns
# off the outward normal, in degrees, tried in this order. Every gap leaves
# room for at least a MIN_LEADER_EM leader plus its LEADER_GAP_EM stand-off
# (the leader runs from the box edge to the line, and the box's nearest
# point along the leader is the full gap away); further gaps step out from
# there when nearer places are taken.
OUTSIDE_GAPS_EM = tuple(MIN_LEADER_EM + LEADER_GAP_EM + extra
                        for extra in (0.0, 0.5, 1.25, 2.75, 5.0))
OUTSIDE_TURNS_DEG = (0.0, 35.0, -35.0, 70.0, -70.0)
# A leader is a hairline: it points, it is not a cut.
LEADER_WIDTH_OF_HAIRLINE = 0.6

DEFAULT_MIN_LABEL_MM = 2.5


def leader_allowance(n):
    """How many of a piece's `n` labels may go on leaders before the piece
    is tried at a smaller size."""
    return max(1, n // 4)


# --- plain geometry ----------------------------------------------------------

def _frame(rotation_deg):
    """(reading, up) unit vectors on the sheet (y down) for a turn that is
    counter-clockwise on the page."""
    a = radians(rotation_deg)
    return (cos(a), -sin(a)), (-sin(a), -cos(a))


def _quad(x, y, size, text, rotation_deg=0.0, pad=0.0):
    """The oriented box of a 'middle' label, four corners in order."""
    (rx, ry), (ux, uy) = _frame(rotation_deg)
    hw = len(text) * size * GLYPH_ADVANCE_EM / 2 + pad
    hh = size / 2 + pad
    return tuple((x + sx * rx * hw + sy * ux * hh,
                  y + sx * ry * hw + sy * uy * hh)
                 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)))


def _label_quad(label, pad=0.0):
    rot = getattr(label, 'rotation_deg', 0.0) or 0.0
    if label.valign == 'baseline':
        # Baseline on y: the box sits half a size above the 'middle' one.
        _r, (ux, uy) = _frame(rot)
        return _quad(label.x + ux * label.size_mm / 2,
                     label.y + uy * label.size_mm / 2,
                     label.size_mm, label.text, rot, pad)
    return _quad(label.x, label.y, label.size_mm, label.text, rot, pad)


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segs_cross(p, q, r, s, eps=1e-9):
    d1, d2 = _orient(r, s, p), _orient(r, s, q)
    d3, d4 = _orient(p, q, r), _orient(p, q, s)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True

    def on(a, b, c):
        return (min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
                and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps)
    return ((abs(d1) <= eps and on(r, s, p)) or (abs(d2) <= eps and on(r, s, q))
            or (abs(d3) <= eps and on(p, q, r))
            or (abs(d4) <= eps and on(p, q, s)))


def _in_quad(p, quad):
    """Strictly inside a convex quad, either winding."""
    signs = [_orient(quad[i], quad[(i + 1) % 4], p) for i in range(4)]
    return all(v > 1e-12 for v in signs) or all(v < -1e-12 for v in signs)


def _seg_hits_quad(p, q, quad):
    if _in_quad(p, quad) or _in_quad(q, quad):
        return True
    return any(_segs_cross(p, q, quad[i], quad[(i + 1) % 4])
               for i in range(4))


def _quads_overlap(a, b):
    """Separating-axis test on two convex quads; touching is not overlap."""
    for quad in (a, b):
        for i in range(4):
            p, q = quad[i], quad[(i + 1) % 4]
            ax, ay = q[1] - p[1], p[0] - q[0]
            pa = [ax * c[0] + ay * c[1] for c in a]
            pb = [ax * c[0] + ay * c[1] for c in b]
            tol = 1e-9 * max(1e-12, hypot(ax, ay))
            if max(pa) <= min(pb) + tol or max(pb) <= min(pa) + tol:
                return False
    return True


def _inside(pt, ring):
    x, y = pt
    inside = False
    n = len(ring)
    for i in range(n):
        (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            if x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
    return inside


def _in_rings(pt, rings):
    """Even-odd over every ring: inside an outer, not inside a hole."""
    return sum(1 for r in rings if _inside(pt, r)) % 2 == 1


def _stroke_segments(points, closed):
    out = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    if closed and len(points) > 2:
        out.append((points[-1], points[0]))
    return out


def _point_at(run, fraction):
    """(point, segment index) at `fraction` of the polyline's length."""
    lengths = [hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(run, run[1:])]
    want = sum(lengths) * fraction
    for i, seg in enumerate(lengths):
        if want <= seg and seg > 1e-12:
            t = want / seg
            a, b = run[i], run[i + 1]
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), i
        want -= seg
    return run[-1], max(len(lengths) - 1, 0)


def _body(text):
    """A bevel label's text without its arrow glyph."""
    return text.replace(ARROW_RIGHT, "").replace(ARROW_LEFT, "").strip()


# --- the obstacle index -------------------------------------------------------

class _Grid:
    """Oriented boxes and segments bucketed by a uniform grid, so a
    candidate is only tested against what is near it. Answers are the same
    as testing everything: the grid only prunes. `mark()`/`rollback()` undo
    additions, for trying a piece at one size and abandoning it."""

    def __init__(self, cell):
        self.cell = max(cell, 1e-6)
        self.cells = defaultdict(list)
        self.items = []
        self.log = []

    def _keys(self, bbox):
        c = self.cell
        for i in range(int(floor(bbox[0] / c)), int(floor(bbox[2] / c)) + 1):
            for j in range(int(floor(bbox[1] / c)),
                           int(floor(bbox[3] / c)) + 1):
                yield (i, j)

    def _add(self, owner, kind, geom, bbox):
        index = len(self.items)
        self.items.append((owner, kind, geom))
        for key in self._keys(bbox):
            self.cells[key].append(index)
            self.log.append(key)

    def add_quad(self, owner, quad):
        self._add(owner, 'quad', quad, _bbox(quad))

    def add_seg(self, owner, p, q):
        self._add(owner, 'seg', (p, q), _bbox((p, q)))

    def mark(self):
        return (len(self.items), len(self.log))

    def rollback(self, mark):
        n_items, n_log = mark
        while len(self.log) > n_log:
            self.cells[self.log.pop()].pop()
        del self.items[n_items:]

    def _near(self, bbox):
        seen = set()
        for key in self._keys(bbox):
            for index in self.cells.get(key, ()):
                if index not in seen:
                    seen.add(index)
                    yield self.items[index]

    def quad_clear(self, quad, skip):
        for owner, kind, geom in self._near(_bbox(quad)):
            if owner is not None and owner == skip:
                continue
            if kind == 'quad':
                if _quads_overlap(quad, geom):
                    return False
            elif _seg_hits_quad(geom[0], geom[1], quad):
                return False
        return True

    def seg_clear(self, p, q, skip):
        for owner, kind, geom in self._near(_bbox((p, q))):
            if owner is not None and owner == skip:
                continue
            if kind == 'quad':
                if _seg_hits_quad(p, q, geom):
                    return False
            elif _segs_cross(p, q, geom[0], geom[1]):
                return False
        return True


# --- one label ----------------------------------------------------------------

class _Placed:
    __slots__ = ("x", "y", "size", "rotation", "text", "leader", "how")

    def __init__(self, x, y, size, rotation, text, leader, how):
        self.x, self.y, self.size = x, y, size
        self.rotation, self.text = rotation, text
        self.leader, self.how = leader, how


def _sizes(full, floor_mm):
    out = []
    for step in SIZE_STEPS:
        size = max(full * step, floor_mm)
        if not out or size < out[-1] - 1e-9:
            out.append(size)
    if out[-1] > floor_mm + 1e-9:
        out.append(floor_mm)
    return out


def _place_inside(label, anchor, owner, grid, size):
    body = _body(label.text)
    for fraction in FRACTIONS:
        p, k = _point_at(anchor.run, fraction)
        n = anchor.normals[min(k, len(anchor.normals) - 1)]
        rot, _r, text = bevel_label_frame(n, anchor.tip_inward, body)
        reach = (size * BEVEL_ARROW_GAP_EM
                 + len(text) * size * GLYPH_ADVANCE_EM / 2)
        x, y = p[0] + n[0] * reach, p[1] + n[1] * reach
        padded = _quad(x, y, size, text, rot, size * CLEAR_EM)
        if not all(_in_rings(c, anchor.rings)
                   for c in _quad(x, y, size, text, rot)):
            continue
        if not grid.quad_clear(padded, owner):
            continue
        return _Placed(x, y, size, rot, text, None, 'slid')
    return None


def _leader_start(centre, rotation, hw, hh, p, gap):
    """Where the line from the box's centre to `p` leaves the box, then
    `gap` further on; None when the box is too near `p` for a leader."""
    cx, cy = centre
    (rx, ry), (ux, uy) = _frame(rotation)
    dx, dy = p[0] - cx, p[1] - cy
    lr, lu = dx * rx + dy * ry, dx * ux + dy * uy
    t = min(hw / abs(lr) if abs(lr) > 1e-12 else float('inf'),
            hh / abs(lu) if abs(lu) > 1e-12 else float('inf'))
    if t >= 1.0:
        return None
    edge = (cx + dx * t, cy + dy * t)
    rest = hypot(p[0] - edge[0], p[1] - edge[1])
    if rest <= gap * 2:
        return None
    return (edge[0] + (p[0] - edge[0]) / rest * gap,
            edge[1] + (p[1] - edge[1]) / rest * gap)


def _place_outside(label, anchor, owner, grid, sheet, outline_rings, size):
    width, height = sheet
    body = _body(label.text)
    for gap_em in OUTSIDE_GAPS_EM:
        for fraction in FRACTIONS:
            p, k = _point_at(anchor.run, fraction)
            n = anchor.normals[min(k, len(anchor.normals) - 1)]
            rot, _r, text = bevel_label_frame(n, anchor.tip_inward, body,
                                              outside=True)
            (rx, ry), (ux, uy) = _frame(rot)
            hw = len(text) * size * GLYPH_ADVANCE_EM / 2
            hh = size / 2
            for turn in OUTSIDE_TURNS_DEG:
                c, s = cos(radians(turn)), sin(radians(turn))
                d = (-n[0] * c + n[1] * s, -n[0] * s - n[1] * c)
                support = (abs(d[0] * rx + d[1] * ry) * hw
                           + abs(d[0] * ux + d[1] * uy) * hh)
                reach = gap_em * size + support
                centre = (p[0] + d[0] * reach, p[1] + d[1] * reach)
                quad = _quad(centre[0], centre[1], size, text, rot)
                bb = _bbox(quad)
                if bb[0] < 0 or bb[1] < 0 or bb[2] > width or bb[3] > height:
                    continue
                if any(_in_rings(corner, outline_rings) for corner in quad):
                    continue
                if not grid.quad_clear(
                        _quad(centre[0], centre[1], size, text, rot,
                              size * CLEAR_EM), owner):
                    continue
                start = _leader_start(centre, rot, hw, hh, p,
                                      size * LEADER_GAP_EM)
                if start is None:
                    continue
                # Stop a hair short of the run: the leader ends *on* its
                # own cut line, which is not a crossing.
                length = hypot(p[0] - start[0], p[1] - start[1])
                if length < MIN_LEADER_EM * size:
                    # Guaranteed by the gaps above; checked, not assumed.
                    continue
                trim = min(length * 0.5, 1e-3)
                end_test = (p[0] - (p[0] - start[0]) / length * trim,
                            p[1] - (p[1] - start[1]) / length * trim)
                if not grid.seg_clear(start, end_test, owner):
                    continue
                return _Placed(centre[0], centre[1], size, rot, text,
                               (start, p), 'leader')
    return None


# --- piece names ----------------------------------------------------------------

# Steps of the size search. The fit is monotone in size -- the box grows
# about a fixed centre, so whatever a size hits, every larger size hits --
# so bisection finds the largest size that fits; 24 halvings of the range
# is far finer than a printer can show.
_NAME_BISECTIONS = 24


def fit_piece_names(names, strokes, labels, set_mm, floor_mm):
    """The size each piece name prints at, and how it got there.

    `names` are (text, x, y, rings): the name, its anchor -- the pole of
    inaccessibility, inside the piece by construction and in its widest
    part -- and its piece's rings (outer first, even-odd), all sheet mm.
    `strokes` and `labels` are every mark already on the sheet that the
    name must keep clear of: outlines, alignment and run-end ticks, relief
    cuts, heat-form lines, tick and heat labels. (Not the bevel labels:
    they are laid out afterwards, around the names.)

    Returns [(size_mm, how)] in `names` order, `how` being 'full' (the set
    size fits), 'shrunk' (the largest size whose box, padded by
    `CLEAR_EM`, lies inside its own piece and clear of every obstacle), or
    'floor': not even `floor_mm` fits, so it is printed at that floor
    anyway -- a tiny piece gets a small name, not none -- and the caller
    reports it.

    `floor_mm` is the smallest legible text: the caller passes the
    maker's `min_label_mm`, the same floor the bevel labels keep (a
    physical limit of their printer and eyes, not a tuning knob). It was a
    quarter of the set size; on the maker's 1.35 m sheet that is 5.9 mm,
    more than twice what they can read, and it left 12 of 52 names -- "HE
    P" among them -- too big for their pieces and still over the ticks
    (measured 2026-11-17, tests/test_overlay_ticks.py). On a small sheet a
    quarter is also *below* legible. A floor above the set size is the set
    size.

    The box is the layout's own (`_quad`): `len(text) * GLYPH_ADVANCE_EM`
    wide by 1 tall, times the size, unturned -- the same contract every
    other overlap test here uses, so a name "clear" here is clear to the
    bevel-label layout that follows. Names are not obstacles to each other:
    each lies inside its own piece, and two pieces only share ground where
    objects are drawn overlaid on purpose (Export UV Layout style).
    Pure: reads nothing but its arguments.
    """
    floor_mm = min(floor_mm, set_mm)
    grid = _Grid(max(set_mm * 2.0, 1.0))
    for s in strokes:
        for p, q in _stroke_segments(s.points, s.closed):
            grid.add_seg(None, p, q)
    for l in labels:
        if l.text:
            grid.add_quad(None, _label_quad(l, l.size_mm * CLEAR_EM))

    out = []
    for text, x, y, rings in names:
        def fits(z):
            if not all(_in_rings(c, rings) for c in _quad(x, y, z, text)):
                return False
            return grid.quad_clear(_quad(x, y, z, text, 0.0, z * CLEAR_EM),
                                   None)
        if fits(set_mm):
            out.append((set_mm, 'full'))
        elif not fits(floor_mm):
            out.append((floor_mm, 'floor'))
        else:
            lo, hi = floor_mm, set_mm
            for _ in range(_NAME_BISECTIONS):
                mid = (lo + hi) / 2
                if fits(mid):
                    lo = mid
                else:
                    hi = mid
            out.append((lo, 'shrunk'))
    return out


# --- the pass -----------------------------------------------------------------

def layout_labels(marks, min_label_mm=DEFAULT_MIN_LABEL_MM, report=None):
    """`marks` with every anchored bevel label moved to a clear spot at one
    size per piece, no smaller than `min_label_mm`, or put outside its
    piece on a 'leader'. Pure: reads nothing but its arguments.

    `report`, when a dict, is filled in label order: 'index' (position in
    `marks.labels`), 'piece', and 'how' -- 'kept' (untouched), 'slid'
    (moved and/or resized inside), 'leader', or 'unplaced' (nothing fitted;
    left where it was, at least `min_label_mm`). Also 'piece_size':
    {piece: size chosen, mm}.
    """
    labels = list(marks.labels)
    strokes = list(marks.strokes)
    bevel = [i for i, l in enumerate(labels)
             if l.kind == 'angle' and getattr(l, 'anchor', None) is not None]
    bevel_set = set(bevel)
    owner_of = {li: k for k, li in enumerate(bevel)}
    hows = {}

    sizes = sorted(labels[i].size_mm for i in bevel)
    cell = max(sizes[len(sizes) // 2] * 3.0 if sizes else 10.0,
               min_label_mm * 3.0)

    def fixed_grid():
        g = _Grid(cell)
        for li, l in enumerate(labels):
            if li in bevel_set or not l.text:
                continue
            g.add_quad(None, _label_quad(l, l.size_mm * CLEAR_EM))
        for s in strokes:
            for p, q in _stroke_segments(s.points, s.closed):
                g.add_seg(None, p, q)
        return g

    def add_unit(g, li, got):
        g.add_quad(owner_of[li], _quad(got.x, got.y, got.size, got.text,
                                       got.rotation, got.size * CLEAR_EM))
        if got.leader:
            g.add_seg(owner_of[li], got.leader[0], got.leader[1])

    def as_placed(li, how):
        l = labels[li]
        return _Placed(l.x, l.y, l.size_mm, l.rotation_deg, l.text, None, how)

    outline_rings = [s.points for s in strokes
                     if s.kind == 'outline' and s.closed]
    pieces = defaultdict(list)
    for li in bevel:
        pieces[labels[li].anchor.piece].append(li)

    # Pass 1: a piece whose labels are all clear where they are, judged
    # against every other original, is kept as it is.
    g_all = fixed_grid()
    for li in bevel:
        add_unit(g_all, li, as_placed(li, 'kept'))
    kept_pieces = set()
    for piece, members in pieces.items():
        ok = True
        for li in members:
            l = labels[li]
            if (l.size_mm < min_label_mm - 1e-9
                    or not all(_in_rings(c, l.anchor.rings)
                               for c in _label_quad(l))
                    or not g_all.quad_clear(
                        _label_quad(l, l.size_mm * CLEAR_EM), owner_of[li])):
                ok = False
                break
        if ok:
            kept_pieces.add(piece)

    grid = fixed_grid()
    piece_size = {}
    for piece in sorted(kept_pieces):
        for li in pieces[piece]:
            add_unit(grid, li, as_placed(li, 'kept'))
            hows[li] = 'kept'
        piece_size[piece] = max(labels[li].size_mm for li in pieces[piece])

    def geometric(li):
        return (round(labels[li].y, 6), round(labels[li].x, 6),
                labels[li].text)

    # Pass 2: the other pieces, whole, in a geometric order.
    todo = sorted((p for p in pieces if p not in kept_pieces),
                  key=lambda p: min(geometric(li) for li in pieces[p]))
    sheet = (marks.width_mm, marks.height_mm)
    placed = {}
    for piece in todo:
        members = sorted(pieces[piece], key=geometric)
        full = max(labels[li].size_mm for li in members)
        trial_sizes = _sizes(full, min_label_mm)
        allowance = leader_allowance(len(members))
        for z in trial_sizes:
            last = z == trial_sizes[-1]
            mark = grid.mark()
            result, failed = {}, []
            for li in members:
                got = _place_inside(labels[li], labels[li].anchor,
                                    owner_of[li], grid, z)
                if got is None:
                    failed.append(li)
                    continue
                result[li] = got
                add_unit(grid, li, got)
            ok = len(failed) <= allowance or last
            if ok:
                for li in failed:
                    got = _place_outside(labels[li], labels[li].anchor,
                                         owner_of[li], grid, sheet,
                                         outline_rings, z)
                    if got is None:
                        if not last:
                            ok = False
                            break
                        got = as_placed(li, 'unplaced')
                        got.size = max(got.size, min_label_mm)
                    result[li] = got
                    add_unit(grid, li, got)
            if ok:
                placed.update(result)
                piece_size[piece] = z
                break
            grid.rollback(mark)

    leaders = []
    width = (marks.hairline_mm or 0.2) * LEADER_WIDTH_OF_HAIRLINE
    for li, got in sorted(placed.items()):
        l = labels[li]
        labels[li] = l._replace(x=got.x, y=got.y, size_mm=got.size,
                                text=got.text, rotation_deg=got.rotation)
        if got.leader:
            leaders.append(Stroke('leader', tuple(got.leader), False,
                                  width, l.color, None, l.anchor))
        hows[li] = got.how

    if report is not None:
        report['index'] = list(bevel)
        report['how'] = [hows[li] for li in bevel]
        report['piece'] = [labels[li].anchor.piece for li in bevel]
        report['piece_size'] = dict(sorted(piece_size.items()))
    return marks._replace(labels=tuple(labels),
                          strokes=tuple(strokes) + tuple(leaders))
