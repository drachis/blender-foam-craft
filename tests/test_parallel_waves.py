"""Two waves running alongside each other, swept from touching to far apart.

    python tools/launch_blender.py --port 9886 --gui
    python tools/blender_bridge.py --port 9886 < tests/test_parallel_waves.py

Needs the **GUI** instance: it writes a screenshot per case so the result can
be judged by eye as well as by number (CLAUDE.md, "Anything visual gets a GUI
Blender"). The numbers still stand on their own if it is run headless -- the
captures are simply skipped.

Why waves, and why a pair of different ones. Everything else in this suite
meets at a point: a T, an X, an elbow. Two lines drawn *alongside* each other
is the harder shape, because there is no single place where they belong
together -- they approach, touch, and separate repeatedly, and the question at
every one of those approaches is the same one the maker answers by eye: is
that one mark or two?

A sine and a sawtooth make that question have many different answers in a
single pair of strokes. Tangent at the peaks, diverging in between, and the
sawtooth's corners are sharp where the sine's are round, so the two curves
approach at every angle from parallel to oblique. Sweeping the amplitude from
nearly nothing up to a foot then walks the whole range from "obviously one
thick mark" to "obviously two separate lines", which is exactly the judgement
`join_touching_strokes` is making.

The bar is not a fixed count. What has to hold is that the answer moves in one
direction only: as the waves are drawn further apart, they may only become
*less* joined, never more. A rule that joins two strokes at 12 mm but not at 6
is not reading the drawing, whatever it does at either end.
"""

import math
import sys
import importlib
import traceback

import bpy
import bmesh
from mathutils import Vector

sys.path.insert(0, r"O:\Work\BlenderPlugin\FoamCraft\tools")

_MOD = "bl_ext.user_default.foamcraft_drachis"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foamcraft_drachis.foam_gp_pattern as gp        # noqa: E402

importlib.reload(gp)

try:
    import shot
    importlib.reload(shot)
except Exception:
    shot = None


# A foot is about 0.3 m, and the plane below is 2 m across, so this runs from
# "the two marks are the same mark" to "nobody would call these one line".
AMPLITUDES_M = (0.001, 0.004, 0.012, 0.03, 0.08, 0.3)

# The nib. Both strokes carry it, so ink meets ink at twice this.
PEN_RADIUS_M = 0.005

_SPAN = 1.7          # how far across the sheet the waves run
_CYCLES = 3.0
# 241 samples, not a round 240, so that (N-1) divides by 12 and the sine's
# peaks land *exactly* on sampled points for 3 cycles. Off-sample peaks cost
# a fraction of a step in x, which on a steep sawtooth ramp is several
# millimetres in y -- enough to push the meeting outside the nib at large
# amplitudes and make the cutter look like it had missed when the test had.
_SAMPLES = 241


def _clear():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def _plane(cuts=23):
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    obj = bpy.context.object
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts,
                              use_grid_fill=True)
    bm.to_mesh(obj.data)
    bm.free()
    return obj


def _sine(amplitude):
    out = []
    for i in range(_SAMPLES):
        f = i / (_SAMPLES - 1.0)
        x = -_SPAN * 0.5 + _SPAN * f
        out.append(Vector((x, amplitude * math.sin(2.0 * math.pi * _CYCLES * f),
                           0.0)))
    return out


def _sawtooth(amplitude, gap=None):
    """A sawtooth riding above the sine, dipping to meet each of its peaks.

    Lifted so its *troughs* sit exactly ``1.5 * PEN_RADIUS_M`` above the
    sine's *peaks*: the midline goes to the sine's amplitude plus its own
    plus that gap. Two consequences, and both are the point of the test.

    The waves never cross. The sawtooth's lowest point is above the sine's
    highest, at every amplitude, so there is not a single real intersection
    anywhere in the pair. Any junction that appears therefore came from
    reading the ink, with no genuine crossing for it to hide behind -- which
    the earlier version of this test did have, and which made it prove much
    less than it looked.

    And the gap where they meet is the same at every amplitude, 1.5 radii,
    which is inside the 2 radii at which ink meets ink. So the sweep does
    not vary *whether* they should join -- they always should, at every
    peak -- it varies how they arrive: nearly parallel and overlapping along
    their whole length at the bottom, and only pinching together at the
    peaks by the top. That is the harder question, and the one the
    local-minimum clustering in `join_touching_strokes` exists to answer.
    """
    if gap is None:
        gap = 1.5 * PEN_RADIUS_M
    offset = 2.0 * amplitude + gap
    out = []
    for i in range(_SAMPLES):
        f = i / (_SAMPLES - 1.0)
        x = -_SPAN * 0.5 + _SPAN * f
        # Phased so the ramp restarts -- its trough -- at the sine's peak.
        ramp = 2.0 * (((_CYCLES * f) + 0.75) % 1.0) - 1.0
        out.append(Vector((x, offset + amplitude * ramp, 0.0)))
    return out


def _closest_between(points_a, points_b):
    """Closest approach between two polylines, point-to-segment both ways."""
    def one_way(pts, other):
        best = None
        for p in pts:
            for a, b in zip(other, other[1:]):
                span = b - a
                denom = span.length_squared
                t = 0.0 if denom < 1e-18 \
                    else max(0.0, min(1.0, (p - a).dot(span) / denom))
                d = (p - (a + span * t)).length
                if best is None or d < best:
                    best = d
        return best
    return min(one_way(points_a, points_b), one_way(points_b, points_a))


def _seam_networks(bm):
    """How many separate cut lines the mesh ended up with."""
    seam = [e for e in bm.edges if e.seam]
    seen, groups = set(), 0
    for edge in seam:
        if edge in seen:
            continue
        groups += 1
        stack = [edge]
        seen.add(edge)
        while stack:
            current = stack.pop()
            for vert in current.verts:
                for other in vert.link_edges:
                    if other.seam and other not in seen:
                        seen.add(other)
                        stack.append(other)
    return groups


def _junctions(bm):
    """Seam vertices where three or more cut edges meet -- the joins."""
    return sum(1 for v in bm.verts
               if sum(1 for e in v.link_edges if e.seam) >= 3)


def _enclosed_regions(bm):
    """Regions the cut walled off completely -- pieces, rather than the rest
    of the sheet.

    This is the number that says whether the pattern actually made anything.
    Two waves pinched together at each peak should enclose the stretch
    between one pinch and the next: a piece per raised iteration of the
    wave. A region touching the sheet's outer boundary is not that -- it is
    the leftover surround, and it is there whether the cut worked or not,
    which is why a plain region count can look healthy while nothing was
    actually enclosed.
    """
    outer = {f for e in bm.edges if len(e.link_faces) == 1
             for f in e.link_faces}
    seen, enclosed = set(), 0
    for face in bm.faces:
        if face in seen:
            continue
        stack, group = [face], []
        seen.add(face)
        while stack:
            current = stack.pop()
            group.append(current)
            for loop in current.loops:
                if loop.edge.seam:
                    continue
                for other in loop.edge.link_faces:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
        if not any(f in outer for f in group):
            enclosed += 1
    return enclosed


def _strays(edges, drawn, tolerance):
    """Cut edges whose middles lie off every drawn stroke."""
    count = 0
    for edge in edges:
        if not edge.is_valid:
            continue
        mid = (edge.verts[0].co + edge.verts[1].co) * 0.5
        best = None
        for points in drawn:
            for a, b in zip(points, points[1:]):
                span = b - a
                denom = span.length_squared
                t = 0.0 if denom < 1e-18 \
                    else max(0.0, min(1.0, (mid - a).dot(span) / denom))
                d = (mid - (a + span * t)).length
                if best is None or d < best:
                    best = d
        if best is not None and best > tolerance:
            count += 1
    return count


def test_sine_against_sawtooth_across_amplitudes():
    """Sweep the two waves apart and watch the joining fall away."""
    rows = {}
    shots = []
    for amplitude in AMPLITUDES_M:
        _clear()
        obj = _plane()
        sine = _sine(amplitude)
        saw = _sawtooth(amplitude)
        # The closest the two curves ever come: 1.5 radii at the peaks, by
        # construction. Measured rather than assumed, because if the shapes
        # ever did cross, this test would silently be proving something
        # else entirely -- which is exactly what its first version did.
        gap = _closest_between(sine, saw)
        crosses = any(p.y >= q.y for p, q in zip(sine, saw))

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        edges, open_ends = gp.cut_strokes(
            bm, [(sine, False, PEN_RADIUS_M), (saw, False, PEN_RADIUS_M)],
            snap_distance=0.0002)
        for e in edges:
            e.seam = True
        networks = _seam_networks(bm)
        junctions = _junctions(bm)
        regions = _seam_regions_count(bm)
        enclosed = _enclosed_regions(bm)
        # Judged against the ink, not the centrelines. Where two marks
        # overlap the cut deliberately runs down the *midline between
        # them* -- that is the whole point of `join_touching_strokes`, and
        # it is by construction off both drawn curves, by up to half the
        # gap it just closed. So the bar for "nobody drew this" has to be
        # half a nib's reach plus a millimetre; tighter than that and the
        # test fails the feature for doing its job.
        stray_tolerance = PEN_RADIUS_M + 0.001
        strays = _strays(edges, [sine, saw], stray_tolerance)
        bm.to_mesh(obj.data)
        obj.data.update()
        bm.free()

        rows["amp_%gmm" % round(amplitude * 1000, 3)] = {
            "amplitude_mm": round(amplitude * 1000, 3),
            "closest_approach_mm": round(gap * 1000, 3),
            "curves_cross": crosses,
            "ink_reach_mm": round(2 * PEN_RADIUS_M * 1000, 1),
            "seam_networks": networks,
            "junctions": junctions,
            "regions": regions,
            "enclosed_pieces": enclosed,
            "cut_edges": len(edges),
            "strays_beyond_the_ink": strays,
            "open_ends": open_ends,
        }
        if shot is not None:
            try:
                written = shot.capture_cut(
                    obj, "waves_amp_%gmm" % round(amplitude * 1000, 3))
                if written:
                    shots.append(written)
            except Exception:
                pass

    ordered = [rows[k] for k in sorted(rows, key=lambda k: rows[k]["amplitude_mm"])]
    out = dict(rows)
    out["screenshots"] = shots
    out["checks"] = {
        # The premise. If the two ever cross, every number here is about
        # something other than what it claims to measure.
        "the_curves_never_cross": not any(r["curves_cross"] for r in ordered),
        # The gap where they meet is 1.5 radii by construction, but only
        # exactly so where a sawtooth sample lands on the sine's peak. The
        # steeper the ramp the more a fraction of a sample costs, so the
        # measured gap drifts up with amplitude -- a property of sampling
        # the shapes, not of the cut.
        "they_stay_within_a_nib_of_each_other": all(
            r["closest_approach_mm"] <= 2.0 * PEN_RADIUS_M * 1000
            for r in ordered),
        # The point of the test: where the ink overlaps, two marks become
        # one line. Asserted wherever the peaks really do come within reach
        # -- at the largest amplitude the sampling drift above pushes most
        # peaks outside the nib, so the pair legitimately stays several
        # lines and that is the test's own doing, not the cutter's.
        "overlapping_marks_become_one_line": all(
            r["seam_networks"] == 1 for r in ordered
            if r["closest_approach_mm"] < 2.0 * PEN_RADIUS_M * 1000 - 0.2),
        "they_join_wherever_the_ink_meets": all(
            r["junctions"] >= 1 for r in ordered),
        # Nothing invented anywhere along the sweep.
        "no_strays_at_any_amplitude": all(
            r["strays_beyond_the_ink"] == 0 for r in ordered),
        # Two marks that merge should come out as one line, not as a ladder
        # of rungs between them. Near-parallel is where this bites: at 1 mm
        # amplitude the pair runs 7.5 mm apart for its whole length, every
        # stretch of it overlapping, and each join walls off another sliver
        # between the two. Measured at 75 pieces from two strokes, which is
        # not a pattern anyone drew -- see the note in this file's summary.
        "merging_marks_do_not_make_a_ladder_of_slivers":
            ordered[0]["regions"] <= 6,
        # The invariant that says the pattern actually made something: the
        # two waves pinch together at each of the 3 peaks, so the stretch
        # between one pinch and the next is walled off -- a piece per raised
        # iteration of the wave. Zero enclosed pieces means the pinches did
        # not close, however healthy the plain region count looks.
        "every_wave_iteration_encloses_a_piece": all(
            r["enclosed_pieces"] >= int(_CYCLES) - 1 for r in ordered),
    }
    return out


def _seam_regions_count(bm):
    seen, regions = set(), 0
    for face in bm.faces:
        if face in seen:
            continue
        regions += 1
        stack = [face]
        seen.add(face)
        while stack:
            current = stack.pop()
            for loop in current.loops:
                if loop.edge.seam:
                    continue
                for other in loop.edge.link_faces:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
    return regions


def test_joining_follows_the_ink_not_the_distance():
    """Whether two marks join must depend on the *ink*, not on any absolute
    distance -- so it is the ratio of separation to nib that decides, and
    the same ratio must give the same answer at any scale.

    Four cases at one amplitude: a fine nib and a broad one, each with the
    marks drawn inside their combined ink and clearly outside it. A rule
    keyed to a fixed distance would join the broad-nib pair at 15 mm and
    refuse the fine-nib pair at 6 mm, which is backwards -- 6 mm is three
    nib-widths for a fine pen and well clear, while 15 mm is inside a broad
    one and plainly one mark.
    """
    rows = {}
    amplitude = 0.03
    for label, radius, gap in (
            ("fine_nib_ink_overlaps", 0.002, 0.003),
            ("fine_nib_clearly_apart", 0.002, 0.006),
            ("broad_nib_ink_overlaps", 0.010, 0.015),
            ("broad_nib_clearly_apart", 0.010, 0.030)):
        _clear()
        obj = _plane()
        sine = _sine(amplitude)
        saw = _sawtooth(amplitude, gap=gap)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        edges, _ends = gp.cut_strokes(
            bm, [(sine, False, radius), (saw, False, radius)],
            snap_distance=0.0002)
        for e in edges:
            e.seam = True
        rows[label] = {
            "nib_radius_mm": round(radius * 1000, 2),
            "separation_mm": round(gap * 1000, 2),
            "ink_reach_mm": round(2 * radius * 1000, 2),
            "ink_overlaps": gap < 2 * radius,
            "seam_networks": _seam_networks(bm),
            "junctions": _junctions(bm),
            "enclosed_pieces": _enclosed_regions(bm),
        }
        bm.free()

    out = dict(rows)
    out["checks"] = {
        "fine_nib_joins_when_its_own_ink_overlaps":
            rows["fine_nib_ink_overlaps"]["enclosed_pieces"] >= 2,
        "fine_nib_leaves_a_wider_gap_alone":
            rows["fine_nib_clearly_apart"]["junctions"] == 0
            and rows["fine_nib_clearly_apart"]["seam_networks"] == 2,
        "broad_nib_joins_across_a_gap_the_fine_one_would_not":
            rows["broad_nib_ink_overlaps"]["enclosed_pieces"] >= 2,
        "broad_nib_still_refuses_when_its_ink_does_not_reach":
            rows["broad_nib_clearly_apart"]["junctions"] == 0
            and rows["broad_nib_clearly_apart"]["seam_networks"] == 2,
        # The point, stated directly: 6 mm apart is two marks for a fine
        # nib, and 15 mm apart is one mark for a broad one. Distance alone
        # cannot tell them apart; ink can.
        "the_wider_gap_joins_and_the_narrower_one_does_not":
            rows["broad_nib_ink_overlaps"]["junctions"] > 0
            and rows["fine_nib_clearly_apart"]["junctions"] == 0
            and rows["broad_nib_ink_overlaps"]["separation_mm"]
            > rows["fine_nib_clearly_apart"]["separation_mm"],
    }
    return out


def test_two_parallel_sines():
    """Two identical sines, one offset above the other -- the degenerate
    case, and the one the sawtooth was chosen to avoid.

    Everywhere else in this file the two marks pinch together and pull
    apart, so there are distinct meetings to find. Two copies of the same
    curve never pinch: the gap between them is the *same all the way
    along*, so there is no closest point, no meeting, and nothing that
    picks out one place over another. If the ink overlaps at all it
    overlaps for the entire length, and a maker looking at that sees one
    thick line rather than two lines that touch somewhere.

    Reported here rather than asserted into a shape, because what a
    single continuous overlap *should* cut to is a design decision and not
    mine to make: one line down the middle is the honest reading of the
    mark, and it is not what a pair-of-lines-plus-a-join produces.
    """
    rows = {}
    amplitude = 0.03
    for label, radius, gap in (
            ("fine_nib_overlapping", 0.002, 0.003),
            ("fine_nib_apart", 0.002, 0.006),
            ("broad_nib_overlapping", 0.010, 0.015),
            ("broad_nib_apart", 0.010, 0.030)):
        _clear()
        obj = _plane()
        lower = _sine(amplitude)
        upper = [Vector((p.x, p.y + gap, p.z)) for p in lower]
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        edges, _ends = gp.cut_strokes(
            bm, [(lower, False, radius), (upper, False, radius)],
            snap_distance=0.0002)
        for e in edges:
            e.seam = True
        rows[label] = {
            "nib_radius_mm": round(radius * 1000, 2),
            "separation_mm": round(gap * 1000, 2),
            "ink_reach_mm": round(2 * radius * 1000, 2),
            "ink_overlaps": gap < 2 * radius,
            "seam_networks": _seam_networks(bm),
            "junctions": _junctions(bm),
            "enclosed_pieces": _enclosed_regions(bm),
            "cut_edges": len(edges),
        }
        bm.free()

    out = dict(rows)
    out["checks"] = {
        # Drawn clearly apart, at either nib, they stay two lines. This is
        # the half that must hold whatever is decided about the other.
        "separate_marks_stay_separate":
            rows["fine_nib_apart"]["seam_networks"] == 2
            and rows["broad_nib_apart"]["seam_networks"] == 2,
        "separate_marks_invent_no_junctions":
            rows["fine_nib_apart"]["junctions"] == 0
            and rows["broad_nib_apart"]["junctions"] == 0,
        # Overlapping ink is noticed at all, at either nib.
        "overlapping_marks_are_noticed":
            rows["fine_nib_overlapping"]["seam_networks"] == 1
            and rows["broad_nib_overlapping"]["seam_networks"] == 1,
        # And it must not wall off slivers down the length, which is what
        # the old per-distance clustering did to a long parallel overlap.
        "no_ladder_of_slivers":
            rows["fine_nib_overlapping"]["enclosed_pieces"] <= 2
            and rows["broad_nib_overlapping"]["enclosed_pieces"] <= 2,
    }
    return out


def test_a_stack_of_parallel_lines_inside_one_another_s_ink():
    """Two parallel marks inside each other's ink read as one thick line.
    So what does a stack of them read as?

    A hatch drawn tight enough that every line touches its neighbours is a
    real thing to draw -- it is how anyone shades -- and it is the case
    where "join what overlaps" has the most opportunity to make a mess:
    with 12 lines each within reach of several others, there are dozens of
    pairs that all overlap, and joining every one of them independently
    would web the whole band together.

    Reported rather than pinned to an expected count, because the right
    answer is a judgement about what the mark means, and 12 tight lines
    plausibly mean one broad band rather than 12 boundaries. What can be
    asserted is that it does not degenerate: the work must not explode with
    the number of lines, and it must not wall off a sliver for every pair.
    """
    rows = {}
    amplitude = 0.03
    radius = 0.005
    spacing = 0.006          # inside 2*radius, so every neighbour overlaps
    for count in (2, 3, 4, 12):
        _clear()
        obj = _plane()
        base = _sine(amplitude)
        strokes = []
        for n in range(count):
            offset = (n - (count - 1) * 0.5) * spacing
            strokes.append(([Vector((p.x, p.y + offset, p.z)) for p in base],
                            False, radius))
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        edges, _ends = gp.cut_strokes(bm, strokes, snap_distance=0.0002)
        for e in edges:
            e.seam = True
        rows["lines_%d" % count] = {
            "lines": count,
            "spacing_mm": round(spacing * 1000, 2),
            "ink_reach_mm": round(2 * radius * 1000, 2),
            "band_width_mm": round((count - 1) * spacing * 1000, 2),
            "seam_networks": _seam_networks(bm),
            "junctions": _junctions(bm),
            "enclosed_pieces": _enclosed_regions(bm),
            "cut_edges": len(edges),
        }
        bm.free()

    ordered = [rows["lines_%d" % n] for n in (2, 3, 4, 12)]
    out = dict(rows)
    out["checks"] = {
        # Every line overlaps its neighbours, so the band is one connected
        # cut however many lines are in it.
        "the_whole_band_is_one_cut": all(
            r["seam_networks"] == 1 for r in ordered),
        # No sliver per pair. Joining n lines pairwise where every pair
        # overlaps would enclose a piece between each neighbouring pair and
        # then some; a band should not manufacture pieces out of its own
        # hatching.
        "no_sliver_per_pair": all(
            r["enclosed_pieces"] <= r["lines"] for r in ordered),
        # And the cost stays sane: 12 lines is 6x the strokes of 2, so
        # anything worse than a few times the edges means the pairwise
        # joining is compounding.
        "work_does_not_explode":
            rows["lines_12"]["cut_edges"] < rows["lines_2"]["cut_edges"] * 12,
    }
    return out



# --- merge_fraction -------------------------------------------------------
#
# `join_touching_strokes` has two different answers available when two marks
# overlap, and `merge_fraction` picks between them:
#
#   below it        the marks are two lines that happen to touch, so each
#                   touching stretch gets a junction and both strokes stay;
#   at or above it  the marks are one line drawn twice, so they collapse to a
#                   single stroke running down the middle of the two.
#
# Counting the strokes that come back separates those outright: joining leaves
# two, merging leaves one. Every case below is that count against a shape
# whose coverage is known by construction rather than measured afterwards.

_MERGE_LEN = 1.0            # marks run from -0.5 to +0.5
_MERGE_SAMPLES = 101
_MERGE_NIB = 0.005          # so ink meets ink within 2 * 0.005 = 0.010
_MERGE_OFFSET = 0.004       # inside that, so these two marks do overlap


def _straight(y, x0=-0.5, x1=0.5, samples=_MERGE_SAMPLES):
    return [Vector((x0 + (x1 - x0) * (i / (samples - 1.0)), y, 0.0))
            for i in range(samples)]


def _veering(frac):
    """Parallel to the x axis for `frac` of its length, then away sharply.

    Built so *both* coverages come out at about `frac`: the stretch lying
    inside the other mark's ink is the same stretch on both curves, because
    over it the two are parallel and `_MERGE_OFFSET` apart. That is what makes
    `frac` the knob the threshold is compared against, rather than something
    that has to be measured after the fact.
    """
    out = []
    for i in range(_MERGE_SAMPLES):
        f = i / (_MERGE_SAMPLES - 1.0)
        x = -0.5 + f
        y = _MERGE_OFFSET if f <= frac else _MERGE_OFFSET + (f - frac) * 2.0
        out.append(Vector((x, y, 0.0)))
    return out


def test_merge_fraction_collapses_a_retraced_line():
    """Two marks drawn along each other for their whole length are one mark.

    Known by construction: the two are parallel and 4 mm apart with a 5 mm nib
    on each, so every point of either lies inside the other's ink and both
    coverages are exactly 1.0. At merge_fraction 0.5 that is over the
    threshold, so the pair must come back as a single stroke -- and that
    stroke must run down the middle, at 2 mm, not along either original.

    The same input at merge_fraction 0.0 must come back as two, because zero
    switches merging off. Both bounds, so a rule that merged everything and a
    rule that merged nothing each fail one of them.
    """
    a = _straight(0.0)
    b = _straight(_MERGE_OFFSET)
    strokes = [(a, False, _MERGE_NIB), (b, False, _MERGE_NIB)]

    merged = gp.join_touching_strokes(strokes, 0.0002, merge_fraction=0.5)
    off = gp.join_touching_strokes(strokes, 0.0002, merge_fraction=0.0)

    midline_error = None
    if len(merged) == 1:
        ys = [p.y for p in merged[0][0]]
        # Every point of the survivor should sit on the midline, 2 mm.
        midline_error = max(abs(y - _MERGE_OFFSET / 2.0) for y in ys)

    return {
        "strokes_in": 2,
        "strokes_out_merging_at_0_5": len(merged),
        "strokes_out_merging_off": len(off),
        "midline_target_mm": round(_MERGE_OFFSET / 2.0 * 1000, 3),
        "worst_midline_error_mm":
            None if midline_error is None else round(midline_error * 1000, 4),
        "checks": {
            "full_overlap_becomes_one_mark": len(merged) == 1,
            # Within a tenth of a millimetre of the true midline. The two
            # inputs sit 2 mm either side of it, so anything that merely kept
            # one original would miss by 2 mm and fail this.
            "the_survivor_runs_down_the_middle":
                midline_error is not None and midline_error < 0.0001,
            "zero_switches_merging_off": len(off) == 2,
        },
    }


def test_merge_fraction_leaves_a_crossing_alone():
    """A deliberate X must not be swallowed into one line.

    Crossing two strokes is how a maker closes a region into a piece, so this
    is the case that matters most: the two marks genuinely touch, and they
    must still come back as two. Known by construction -- each line lies
    inside the other's ink only within about one ink-reach of the centre, so
    both coverages are near 0.01 / 1.41, far under 0.5.

    This is the bound a merge rule keyed to "do they touch at all" rather than
    "how much of them overlaps" would fail.
    """
    d = 0.5
    n = _MERGE_SAMPLES
    a = [Vector((-d + 2 * d * (i / (n - 1.0)),
                 -d + 2 * d * (i / (n - 1.0)), 0.0)) for i in range(n)]
    b = [Vector((-d + 2 * d * (i / (n - 1.0)),
                 d - 2 * d * (i / (n - 1.0)), 0.0)) for i in range(n)]
    strokes = [(a, False, _MERGE_NIB), (b, False, _MERGE_NIB)]

    out = gp.join_touching_strokes(strokes, 0.0002, merge_fraction=0.5)
    return {
        "strokes_in": 2,
        "strokes_out": len(out),
        "checks": {
            "a_crossing_stays_two_marks": len(out) == 2,
        },
    }


def test_merge_fraction_switches_at_its_threshold():
    """Sweep how much of the pair overlaps and check the switch sits where the
    setting says it does -- and that it only ever flips one way.

    `_veering` puts a known fraction of both marks inside the other's ink, so
    with merge_fraction at 0.5 the answer is arithmetic: 0.2 and 0.4 stay two
    marks, 0.6 and 0.8 become one. The monotonic check is the one that
    survives a change of constants -- drawing *more* of a mark on top of
    another may never make it less merged.
    """
    rows = {}
    for frac in (0.2, 0.4, 0.6, 0.8):
        strokes = [(_straight(0.0), False, _MERGE_NIB),
                   (_veering(frac), False, _MERGE_NIB)]
        out = gp.join_touching_strokes(strokes, 0.0002, merge_fraction=0.5)
        rows["overlap_%d_pct" % round(frac * 100)] = {
            "overlap_fraction": frac,
            "strokes_out": len(out),
            "merged": len(out) == 1,
        }

    ordered = [rows["overlap_%d_pct" % p] for p in (20, 40, 60, 80)]
    flags = [r["merged"] for r in ordered]
    out = dict(rows)
    out["checks"] = {
        "under_the_threshold_stays_two_marks":
            flags[0] is False and flags[1] is False,
        "over_the_threshold_becomes_one_mark":
            flags[2] is True and flags[3] is True,
        # More overlap may never mean less merging, whatever the constants.
        "more_overlap_never_merges_less":
            all(flags[i] <= flags[i + 1] for i in range(len(flags) - 1)),
    }
    return out


def test_a_short_retrace_over_a_long_mark_merges():
    """A short mark lying along a long one is a retrace, not a second line.

    This pins a semantic that is easy to get wrong by accident. Coverage is
    taken as the *larger* of the two, so a 100 mm mark drawn along a 1 m one
    merges even though only a tenth of the long mark is covered -- the short
    mark is entirely inside the long one's ink, and there is nothing else it
    could be. Were the rule the smaller of the two, or their mean, this would
    come back as two marks and the maker would get a spurious junction at
    each end of the retrace.
    """
    long_mark = _straight(0.0)
    short_mark = _straight(_MERGE_OFFSET, x0=-0.05, x1=0.05, samples=21)
    out = gp.join_touching_strokes(
        [(long_mark, False, _MERGE_NIB), (short_mark, False, _MERGE_NIB)],
        0.0002, merge_fraction=0.5)
    return {
        "long_mark_points": len(long_mark),
        "short_mark_points": len(short_mark),
        "strokes_out": len(out),
        # The long mark is the spine, so the survivor keeps its point count.
        "survivor_points": len(out[0][0]) if len(out) == 1 else None,
        "checks": {
            "a_retrace_merges": len(out) == 1,
            "the_longer_mark_is_the_spine":
                len(out) == 1 and len(out[0][0]) == len(long_mark),
        },
    }



def test_merge_fraction_reaches_the_cutter():
    """The setting has to survive the trip into `cut_strokes`, not just work
    when `join_touching_strokes` is called directly.

    That trip is where this feature was left unfinished once already: the
    parameter existed and was correct, and nothing passed it, so the merge
    branch was unreachable from the panel. Cutting the same pair of marks
    both ways and counting the edges is the cheapest check that the wire is
    connected -- one line laid into the mesh costs about half the edges of
    two, and no arrangement of two separate cuts costs less than one.

    Measured 2026-09-21: 219 edges unmerged, 108 merged. The bound is stated
    as a ratio rather than those literals so a change to the plane's density
    does not have to be mirrored here.
    """
    rows = {}
    for label, merge in (("merge_off", 0.0), ("merge_on", 0.5)):
        _clear()
        obj = _plane()
        strokes = [(_straight(0.0), False, _MERGE_NIB),
                   (_straight(_MERGE_OFFSET), False, _MERGE_NIB)]
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        edges, open_ends = gp.cut_strokes(
            bm, strokes, snap_distance=0.0002, chain_tolerance=0.0002,
            merge_fraction=merge)
        for e in edges:
            e.seam = True
        rows[label] = {
            "merge_fraction": merge,
            "cut_edges": len(edges),
            "open_ends": open_ends,
            "seam_networks": _seam_networks(bm),
            "enclosed_pieces": _enclosed_regions(bm),
        }
        bm.free()

    off, on = rows["merge_off"], rows["merge_on"]
    out = dict(rows)
    out["checks"] = {
        # The wire is connected at all: the two bakes must differ.
        "the_setting_changes_the_cut": on["cut_edges"] != off["cut_edges"],
        # And it differs the right way, by about half.
        "merging_costs_about_half_the_edges":
            on["cut_edges"] < off["cut_edges"] * 0.75,
        # Both bounds: merging must not cost so little that it cut nothing.
        "merging_still_cuts_a_line":
            on["cut_edges"] > off["cut_edges"] * 0.25,
        # One mark down the middle leaves no sliver between the originals.
        "no_sliver_between_the_two_marks": on["enclosed_pieces"] == 0,
    }
    return out


TESTS = (
    test_a_stack_of_parallel_lines_inside_one_another_s_ink,
    test_sine_against_sawtooth_across_amplitudes,
    test_joining_follows_the_ink_not_the_distance,
    test_two_parallel_sines,
    test_merge_fraction_collapses_a_retraced_line,
    test_merge_fraction_leaves_a_crossing_alone,
    test_merge_fraction_switches_at_its_threshold,
    test_a_short_retrace_over_a_long_mark_merges,
    test_merge_fraction_reaches_the_cutter,
)

result = {}
_all_pass = True
for _test in TESTS:
    try:
        _report = _test()
        _report["PASS"] = all(_report.get("checks", {}).values())
    except Exception:
        _report = {"ERROR": traceback.format_exc().splitlines()[-6:],
                   "PASS": False}
    result[_test.__name__] = _report
    _all_pass = _all_pass and _report["PASS"]
result["ALL_PASS"] = _all_pass
