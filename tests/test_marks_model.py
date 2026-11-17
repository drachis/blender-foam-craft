"""The marks model: what foam_marks.build_marks says is on the sheet.

    python tools/launch_blender.py --port 9885
    python tools/blender_bridge.py --port 9885 < tests/test_marks_model.py

`build_marks` is the one place that decides what a pattern sheet carries;
the SVG writer and the raster drawer only draw what it returns. So the
model is checked on its own here, against the cube with every edge seamed --
a known-answer shape (CLAUDE.md): 6 islands, each a square of 4 boundary
edges, 24 boundary runs, every dihedral exactly 90 degrees.

And the contract's one hard rule is checked directly: a `Marks` holds plain
numbers and strings, never a BMesh element. BMesh elements are freed with
their mesh and crash Blender when touched afterwards, rather than raising,
so a Marks that kept one would be a crash waiting for the first backend
that outlived the bmesh.
"""

import sys
import traceback

import bpy
import bmesh

sys.path.insert(0, r"O:\Work\BlenderPlugin\FoamCraft\tools")

import sandbox_guard  # noqa: E402  (guards every _clear below)

_MOD = "bl_ext.user_default.foamcraft_drachis"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foamcraft_drachis.foam_marks as fm         # noqa: E402
import bl_ext.user_default.foamcraft_drachis.foam_angle_core as core  # noqa: E402


def _clear():
    """Empty the scene, but only in a Blender this suite started."""
    sandbox_guard.clear_scene()


def _seamed_cube():
    """A 2 m cube, every edge a seam, unwrapped into six one-quad islands."""
    _clear()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def _marks_of(obj, **overrides):
    """build_marks on a fresh BMesh, with settings overridden and restored.

    The BMesh is freed before returning, deliberately: whatever comes back
    has to survive that, which is the contract.
    """
    s = bpy.context.scene.foamcraft_drachis
    saved = {k: getattr(s, k) for k in overrides}
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    try:
        for k, v in overrides.items():
            setattr(s, k, v)
        uv_layer = bm.loops.layers.uv.active
        islands = len(core.uv_islands(bm, uv_layer, only_selected=False))
        marks = fm.build_marks(bm, uv_layer, s, 1.0, obj.matrix_world)
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
        bm.free()
    return marks, islands


def _inside(pt, ring):
    """Even-odd point-in-polygon."""
    x, y = pt
    inside = False
    n = len(ring)
    for i in range(n):
        (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            if x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
    return inside


_PLAIN = (float, int, str, bool, type(None))


def _foreign_leaves(value, path="marks", out=None):
    """Every leaf in `value` that is not a plain number, string or None.

    Stricter than "no bmesh type": a mathutils Vector is not allowed either,
    because the contract is plain floats and a Vector is how a BMesh-derived
    value usually sneaks through.
    """
    if out is None:
        out = []
    if isinstance(value, tuple):          # NamedTuples are tuples
        for i, item in enumerate(value):
            name = value._fields[i] if hasattr(value, "_fields") else str(i)
            _foreign_leaves(item, "%s.%s" % (path, name), out)
    elif not isinstance(value, _PLAIN):
        out.append("%s: %s.%s" % (path, type(value).__module__,
                                  type(value).__name__))
    return out


def test_cube_marks_have_the_known_counts():
    """Six islands, so six of each per-piece mark -- exactly, both ways.

    Fills are toggled by `island_fill` and nothing else, so on and off are
    both asserted: 6 and 0. An outline is one closed four-point ring per
    square piece. A piece number lands inside its own piece and in no other.
    """
    obj = _seamed_cube()
    on, islands = _marks_of(obj, island_fill=True, show_island_ids=True)
    off, _ = _marks_of(obj, island_fill=False, show_island_ids=True)

    pieces_on = [f for f in on.fills if f.kind == 'piece']
    pieces_off = [f for f in off.fills if f.kind == 'piece']
    outlines = [s for s in on.strokes if s.kind == 'outline']
    numbers = [l for l in on.labels if l.kind == 'number']
    brackets = [s for s in on.strokes if s.kind == 'bracket']
    angles = [l for l in on.labels if l.kind == 'angle']

    # Number i belongs to outline i (both are emitted in island order); it
    # must be inside that one and outside the other five.
    own = [_inside((l.x, l.y), outlines[i].points)
           for i, l in enumerate(numbers)] if len(numbers) == len(outlines) else []
    foreign = sum(_inside((l.x, l.y), o.points)
                  for i, l in enumerate(numbers)
                  for j, o in enumerate(outlines) if i != j)
    return {
        "islands": islands,
        "piece_fills_on": len(pieces_on),
        "piece_fills_off": len(pieces_off),
        "outlines": len(outlines),
        "outline_point_counts": sorted({len(s.points) for s in outlines}),
        "numbers": len(numbers),
        "numbers_inside_own": sum(own),
        "numbers_inside_another": foreign,
        "brackets": len(brackets),
        "angle_texts": sorted({l.text for l in angles}),
        "checks": {
            "six_islands": islands == 6,
            "six_piece_fills_with_fill_on": len(pieces_on) == 6,
            "no_piece_fills_with_fill_off": len(pieces_off) == 0,
            "six_outlines": len(outlines) == 6,
            "every_outline_closed": all(s.closed for s in outlines),
            "every_outline_has_four_points":
                bool(outlines) and all(len(s.points) == 4 for s in outlines),
            "six_numbers": len(numbers) == 6,
            "every_number_inside_its_own_piece":
                len(own) == 6 and all(own),
            "no_number_inside_another_piece": foreign == 0,
            # Was "twenty_four_brackets": one run-end bracket at each of
            # the 24 run ends. Every one of those ends is a corner of the
            # piece, and the maker asked for nothing at corners
            # (2026-11-17): run-end marks are now drawn only part way along
            # a seam, which a cube does not have. So exactly 0.
            "no_brackets_at_the_corners": len(brackets) == 0,
            # Was "all 90": the sheet printed the whole fold. It now prints
            # the blade tilt for this one piece, half the fold -- 45 on a
            # cube -- after the seam number ("<n>, 45°"), as the screen does.
            "twenty_four_angles_all_45_per_side": len(angles) == 24 and all(
                _body(l.text).endswith(", 45\u00b0") for l in angles),
        },
    }


def test_marks_hold_no_bmesh_and_no_vectors():
    """Every leaf of the returned Marks is a plain value.

    The BMesh is freed inside `_marks_of` before this looks at anything, so
    a held element would show here as a bmesh type (or crash, which the
    runner reports as an error -- also a failure). Checked across every mark
    kind the sweep can produce, with the optional ones switched on.
    """
    # The gore sphere as well as the cube: since 2026-11-17 a 'bracket'
    # (run-end mark) is drawn only part way along a seam, and every run end
    # on a cube is a corner -- the gores are where brackets now appear.
    foreign, kinds = [], set()
    for build in (_seamed_cube, _gore_sphere):
        marks, _ = _marks_of(build(), island_fill=True, show_island_ids=True,
                             show_brackets=True, show_align_ticks=True,
                             tick_numbers=True, show_scale_ruler=True)
        foreign += _foreign_leaves(marks)
        kinds |= {m.kind for m in marks.fills + marks.strokes + marks.labels}
    kinds = sorted(kinds)
    return {
        "foreign_leaves": foreign[:10],
        "kinds_present": kinds,
        "checks": {
            "no_foreign_values": foreign == [],
            # Both bounds: an empty Marks would pass the check above.
            # 'bevel_arrow' left this list 2026-11-17: the drawn arrow is
            # retired, the glyph is in the 'angle' label -- so it is now
            # asserted absent.
            "every_optional_kind_was_produced": set(kinds) >= {
                'piece', 'outline', 'bracket', 'align', 'ruler',
                'number', 'angle'},
            "no_drawn_bevel_arrows": 'bevel_arrow' not in kinds,
        },
    }


# The blade-direction glyphs a bevel label carries since 2026-11-17, when
# the drawn 'bevel_arrow' strokes were retired at the maker's request.
_RIGHT, _LEFT = "\u2192", "\u2190"


def _body(text):
    """A bevel label's text without its arrow glyph."""
    return text.replace(_RIGHT, "").replace(_LEFT, "").strip()


def _label_corners(label):
    """A label's box as four corners, turned by its `rotation_deg`
    (counter-clockwise on the page; sheet y is down, so it reads along
    (cos r, -sin r))."""
    a = _math.radians(getattr(label, "rotation_deg", 0.0) or 0.0)
    rx, ry = _math.cos(a), -_math.sin(a)
    ux, uy = -_math.sin(a), -_math.cos(a)
    hw = len(label.text) * label.size_mm * fm.GLYPH_ADVANCE_EM / 2
    hh = label.size_mm / 2
    return [(label.x + sx * rx * hw + sy * ux * hh,
             label.y + sx * ry * hw + sy * uy * hh)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _label_box(label):
    """The axis-aligned bound of a label's (possibly turned) box."""
    pts = _label_corners(label)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _glyph_arrow(label):
    """(glyph end point, unit pointing direction) of a bevel label's arrow
    glyph, or None. The glyph is the first or last character; it points
    along the reading direction for the right arrow, against it for the
    left."""
    a = _math.radians(getattr(label, "rotation_deg", 0.0) or 0.0)
    rx, ry = _math.cos(a), -_math.sin(a)
    hw = len(label.text) * label.size_mm * fm.GLYPH_ADVANCE_EM / 2
    t = label.text
    if t[:1] in (_RIGHT, _LEFT):
        at, g = (label.x - rx * hw, label.y - ry * hw), t[0]
    elif t[-1:] in (_RIGHT, _LEFT):
        at, g = (label.x + rx * hw, label.y + ry * hw), t[-1]
    else:
        return None
    sign = 1.0 if g == _RIGHT else -1.0
    return at, (rx * sign, ry * sign)


def _glyph_count(marks):
    return sum(1 for l in marks.labels
               if l.kind == 'angle' and _glyph_arrow(l) is not None)


def _boxes_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_every_angle_is_written_inside_its_own_piece():
    """A bevel angle belongs to one piece and must stay on it when cut out.

    The labels used to be centred on the cut line. On a cube unwrapped with
    its faces side by side, two pieces wrote "90" on the same spot across the
    seam and the sheet read "9090" -- seen in the first raster export, which
    is why this is asserted on boxes rather than on anchor points.

    Known answer: a cube with every edge seamed has 6 square pieces and 24
    runs, so 24 labels, 4 per piece, each inside exactly one outline.

    Updated 2026-11-17: a label now reads "<seam>, 45°" and has a blade-tilt
    arrow between it and the cut line, so the arrows are held to the same
    rule -- all 24 wholly inside one piece -- and must not run into any
    label's box. Updated again the same day: the arrow is now a glyph in
    the label and the label is turned square to its line, so the box is
    the turned one (`_label_corners`) and there are 0 drawn arrows.
    """
    marks, _ = _marks_of(_seamed_cube(), show_brackets=True)
    outlines = [s for s in marks.strokes if s.kind == 'outline']
    angles = [l for l in marks.labels if l.kind == 'angle']
    arrows = [s for s in marks.strokes if s.kind == 'bevel_arrow']
    arrows_inside = sum(
        1 for a in arrows
        if sum(1 for o in outlines
               if all(_inside(p, o.points) for p in a.points)) == 1)

    def arrow_box(a):
        xs, ys = [p[0] for p in a.points], [p[1] for p in a.points]
        return (min(xs), min(ys), max(xs), max(ys))
    owners = [[i for i, o in enumerate(outlines)
               if _inside((l.x, l.y), o.points)] for l in angles]
    per_piece = [sum(1 for own in owners if own == [i])
                 for i in range(len(outlines))]
    boxes = [_label_box(l) for l in angles]
    overlaps = sum(1 for i in range(len(boxes))
                   for j in range(i + 1, len(boxes))
                   if _boxes_overlap(boxes[i], boxes[j]))
    # Whole box inside the piece: all four corners, not just the anchor.
    corners_inside = sum(
        1 for l, own in zip(angles, owners) if len(own) == 1 and all(
            _inside(c, outlines[own[0]].points) for c in _label_corners(l)))
    return {
        "angle_labels": len(angles),
        "per_piece": per_piece,
        "overlapping_pairs": overlaps,
        "boxes_wholly_inside": corners_inside,
        "arrows": len(arrows),
        "arrows_wholly_inside": arrows_inside,
        "checks": {
            "there_are_24_labels": len(angles) == 24,
            "each_piece_has_exactly_4": per_piece == [4] * 6,
            "no_two_labels_overlap": overlaps == 0,
            "every_box_is_wholly_inside_its_piece": corners_inside == 24,
            "no_drawn_arrows": len(arrows) == 0 and arrows_inside == 0,
            "24_glyph_arrows": _glyph_count(marks) == 24,
        },
    }


def test_the_ruler_strip_stays_clear_of_the_pattern_and_of_itself():
    """The one-inch square is exactly an inch, below every piece, and
    nothing is written through it.

    It used to stand 10 mm above its strip, touching the lowest cut line,
    with its caption centred over it.
    """
    marks, _ = _marks_of(_seamed_cube(), show_scale_ruler=True)
    lowest_piece = max(y for s in marks.strokes if s.kind == 'outline'
                       for _, y in s.points)
    squares = [s for s in marks.strokes if s.kind == 'ruler' and s.closed]
    sq = squares[0].points if squares else ((0, 0),) * 4
    xs, ys = [p[0] for p in sq], [p[1] for p in sq]
    box = (min(xs), min(ys), max(xs), max(ys))
    captions = [l for l in marks.labels if l.kind == 'ruler']
    through = [l.text for l in captions
               if _boxes_overlap(_label_box(l), box)]
    off_sheet = [l.text for l in captions
                 if _label_box(l)[2] > marks.width_mm
                 or _label_box(l)[0] < 0 or l.y > marks.height_mm]
    return {
        "square": [round(v, 3) for v in box],
        "lowest_piece_y": round(lowest_piece, 3),
        "captions": len(captions),
        "captions_through_square": through,
        "captions_off_sheet": off_sheet,
        "checks": {
            "there_is_exactly_one_square": len(squares) == 1,
            "it_is_one_inch_wide": abs(box[2] - box[0] - 25.4) < 1e-6,
            "it_is_one_inch_tall": abs(box[3] - box[1] - 25.4) < 1e-6,
            "it_starts_below_every_piece": box[1] > lowest_piece + 1.0,
            "it_ends_on_the_sheet": box[3] < marks.height_mm,
            # Both bounds: no captions at all would also "not overlap".
            "the_captions_are_there": len(captions) >= 4,
            "no_caption_runs_through_the_square": not through,
            "no_caption_leaves_the_sheet": not off_sheet,
        },
    }


def test_bmesh_vertex_wrappers_compare_by_identity():
    """foam_marks compares vertices with `is` (`loop.vert is start`), so it
    matters whether two reads of one BMVert give the same Python object.

    CLAUDE.md records that `id()` of a re-read wrapper can repeat because the
    temporary was freed and its address reused -- which says nothing either
    way about `is` between two *live* references. Asked directly: hold two
    references at once and compare. Expected: `is` True and `==` True for
    the same vertex, both False for different vertices; if `is` came back
    False, every `is` test in the tick placement would be always-false.
    """
    bm = bmesh.new()
    try:
        bmesh.ops.create_cube(bm, size=1.0)
        bm.edges.ensure_lookup_table()
        e = bm.edges[0]
        a1 = e.verts[0]
        a2 = e.verts[0]
        via_loop = next(l.vert for l in e.link_loops
                        if l.vert == a1)
        b = e.verts[1]
        out = {
            "same_vert_is": a1 is a2,
            "same_vert_eq": a1 == a2,
            "via_loop_is": via_loop is a1,
            "other_vert_is": a1 is b,
            "other_vert_eq": a1 == b,
        }
    finally:
        bm.free()
    out["checks"] = {
        "same_vertex_is_identical": out["same_vert_is"] and out["via_loop_is"],
        "same_vertex_is_equal": out["same_vert_eq"],
        "different_vertices_differ": not out["other_vert_is"]
        and not out["other_vert_eq"],
    }
    return out


def _set(**values):
    """Set scene settings; returns what they were, for `_set(**saved)`."""
    s = bpy.context.scene.foamcraft_drachis
    saved = {k: getattr(s, k) for k in values}
    for k, v in values.items():
        setattr(s, k, v)
    return saved


def _gore_sphere(gores=6, segments=24, rings=16, radius=0.15):
    """A sphere peeled into pole-to-pole gores (as tests/test_export_svg.py
    and tests/test_reproducible_export.py build it). Each gore's two long
    seams are 16 meridian edges of ~29.4 mm: 470.5 mm of seam per chain."""
    import math as _m
    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=segments,
                                         ring_count=rings)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    step = 2.0 * _m.pi / gores

    def _phi(v):
        if abs(v.co.x) < 1e-9 and abs(v.co.y) < 1e-9:
            return None
        return _m.atan2(v.co.y, v.co.x) % (2.0 * _m.pi)

    for e in bm.edges:
        a, b = e.verts
        if abs(a.co.z - b.co.z) < 1e-9:
            continue
        pa, pb = _phi(a), _phi(b)
        phi = pa if pa is not None else pb
        if phi is None:
            continue
        if pa is not None and pb is not None:
            if min(abs(pa - pb), 2 * _m.pi - abs(pa - pb)) > 1e-6:
                continue
        if abs(phi - round(phi / step) * step) < 1e-6:
            e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.04)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def _banded_sphere(segments=24, rings=12):
    """A 1 m sphere with every latitude ring seamed: closed ring seams of 24
    equal edges each, so evenly spaced marks along one fall on whole-edge
    fractions -- the case that puts a mark exactly on a vertex."""
    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=segments,
                                         ring_count=rings)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        a, b = e.verts
        if abs(a.co.z - b.co.z) < 1e-6:
            e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def _cube(size=2.0, scale=None, apply_scale=False):
    """A cube of edge `size`, every edge a seam, unwrapped to six squares.

    `scale` is set on the object afterwards; `apply_scale` then bakes it into
    the mesh with Blender's own Apply Scale -- the pair is how "the same
    object, scale applied or not" is built without inventing either side.
    """
    _clear()
    bpy.ops.mesh.primitive_cube_add(size=size)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    if scale is not None:
        obj.scale = scale
        bpy.context.view_layer.update()
        if apply_scale:
            bpy.ops.object.transform_apply(location=False, rotation=False,
                                           scale=True)
    return obj


def _placements(obj, mode, **overrides):
    """(segments, new placements, old per-run placements) for `obj`.

    New: `_segment_tick_params`, what build_marks draws. Old: the per-run
    placement build_marks used before 2026-11-16 (`_run_tick_params` over
    the cut runs), kept so the change stays measurable. Every placement is
    returned as plain floats: (edge length, t from the canonical start,
    edge index, canonical start vertex index, other vertex index).
    """
    s = bpy.context.scene.foamcraft_drachis
    saved = _set(tick_mode=mode, **overrides)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    try:
        bm.verts.index_update()
        bm.edges.index_update()
        uv_layer = bm.loops.layers.uv.active
        islands_faces = core.uv_islands(bm, uv_layer, only_selected=False)
        sheet = fm._Sheet(islands_faces, uv_layer, 1.0)
        islands = [fm._Island(i, f, uv_layer, sheet)
                   for i, f in enumerate(islands_faces)]

        def flat(params):
            out = []
            for e, t in params:
                a = fm._edge_canonical_start(e)
                out.append((e.calc_length(), t, e.index, a.index,
                            e.other_vert(a).index))
            return out

        segments = []
        for seg in fm._tick_segments(islands, s.corner_angle_deg):
            verts = [seg[0][1]]
            for e, entry, _len in seg:
                verts.append(e.other_vert(entry))
            segments.append({
                "edges": [(e.index, entry.index, ln) for e, entry, ln in seg],
                "length": sum(ln for _e, _v, ln in seg),
                "ends": (verts[0].index, verts[-1].index),
                "end_z": (verts[0].co.z, verts[-1].co.z),
            })
        new = flat(fm._segment_tick_params(islands, s))
        runs = list(core.iter_boundary_dart_segments(
            bm, uv_layer, s.max_chord_deviation, s.max_angle_variation_deg,
            islands=islands_faces, corner_angle_deg=s.corner_angle_deg,
            matrix=obj.matrix_world, dart_wedge_deg=s.dart_wedge_deg,
            max_darts_per_run=s.max_darts_per_run))
        old = flat([p for r in runs for p in fm._run_tick_params(r, s)])
        run_count = len(runs)
    finally:
        _set(**saved)
        bm.free()
    return segments, new, old, run_count


def _along(segment, placement):
    """A placement's distance from its segment's start, or None if the
    placement is not on this segment."""
    _len, t, edge_index, canon, _other = placement
    walked = 0.0
    for e_index, entry, ln in segment["edges"]:
        if e_index == edge_index:
            local = t if entry == canon else 1.0 - t
            return walked + local * ln
        walked += ln
    return None


def _end_report(segments, placements):
    """How close each placement sits to a segment end and to a vertex."""
    at_vertex = sum(1 for ln, t, *_ in placements
                    if min(t, 1.0 - t) * ln <= 1e-6)
    end_verts = {v for seg in segments for v in seg["ends"]}
    at_end = 0
    for ln, t, _e, a, b in placements:
        if (t * ln <= 1e-6 and a in end_verts) or \
                ((1.0 - t) * ln <= 1e-6 and b in end_verts):
            at_end += 1
    return {"placements": len(placements),
            "at_a_mesh_vertex": at_vertex,
            "at_a_segment_end": at_end}


def test_what_the_old_per_run_placement_did():
    """The before/after measurement behind moving marks to segments.

    Reported by the maker: marks sat "at the vertex" instead of being spread
    from corner to corner of an edge. Measured here with the old per-run
    placement (still computable through `_run_tick_params`) against the new
    per-segment one, on two spheres at the default settings, 2 per run.

    Measured 2026-11-16, before -> after:

      gores (6-gore sphere): 36 cut runs on 6 seam segments. Old: 72
        placements, 0 on a mesh vertex, 0 on a segment end. New: 12, 2 per
        segment. The old marks were not on vertices here -- they were
        crowded: each meridian is one segment but 3 runs per side, because
        `max_angle_variation_deg` (10) ends a run wherever the dihedral
        drifts with latitude, and every run restarted its own 1/3, 2/3. The
        two sides' runs also broke in different places, so the sides
        disagreed (72 placements drew 120 strokes, not 144: only some
        merged).
      bands (sphere, every latitude ring seamed): 22 cut runs on 11 closed
        ring segments. Old: 44 placements, all 44 on a mesh vertex (runs of
        a whole number of equal edges, split in thirds). New: 22, still on
        vertices -- 24 equal edges split in thirds is 8 and 16 -- which the
        maker's spec allows; what changed is that they are now a ring's
        thirds, not a run's.

    So the cause of "at the vertex" was the run splitting, not corners:
    distribution restarted at every saw-pass break (angle drift), which on
    regular meshes puts i/(N+1) of a short run on its vertices and in any
    case piles marks up near run ends.

    The checks pin the new placement only: none at a segment end, none at a
    mesh vertex on these fixtures, and marks present (both bounds).
    """
    rows = {}
    for name, build in (("gores", _gore_sphere), ("bands", _banded_sphere)):
        obj = build()
        segments, new, old, runs = _placements(obj, 'PER_RUN',
                                               ticks_per_run=2)
        rows[name] = {
            "cut_runs": runs,
            "segments": len(segments),
            "old": _end_report(segments, old),
            "new": _end_report(segments, new),
        }
    checks = {}
    for name, row in rows.items():
        checks[name + "_new_marks_exist"] = row["new"]["placements"] > 0
        checks[name + "_new_none_at_a_segment_end"] = \
            row["new"]["at_a_segment_end"] == 0
        checks[name + "_two_per_segment"] = \
            row["new"]["placements"] == 2 * row["segments"]
    out = dict(rows)
    out["checks"] = checks
    return out


def test_cube_per_segment_marks_are_at_exact_fractions():
    """Known answer: every one of the cube's 12 edges is one segment.

    Every cube vertex is a junction of three seams, so no segment can run
    past one. With N per segment the marks sit at i/(N+1) of the edge:
    N=2 -> 24 marks, at 1/3 and 2/3; N=1 -> 12 marks, each at the midpoint.

    Strokes: `_alignment_ticks` draws a mark once per island whose boundary
    crosses that edge, and every cube edge bounds exactly two of the six
    islands -- so strokes are exactly twice the marks: 48 and 24.
    """
    obj = _cube(2.0)
    rows = {}
    for n in (1, 2):
        segments, new, _old, _runs = _placements(obj, 'PER_RUN',
                                                 ticks_per_run=n)
        marks, _ = _marks_of(obj, show_align_ticks=True, tick_mode='PER_RUN',
                             ticks_per_run=n)
        strokes = [st for st in marks.strokes if st.kind == 'align']
        want = sorted(i / (n + 1) for i in range(1, n + 1))
        per_edge = {}
        for _ln, t, e, *_ in new:
            per_edge.setdefault(e, []).append(t)
        rows["n%d" % n] = {
            "segments": len(segments),
            "edges_per_segment": sorted({len(sg["edges"]) for sg in segments}),
            "marks": len(new),
            "strokes": len(strokes),
            "worst_fraction_error": max(
                (max(abs(a - b) for a, b in zip(sorted(ts), want))
                 for ts in per_edge.values() if len(ts) == n), default=None),
            "edges_with_n_marks": sum(1 for ts in per_edge.values()
                                      if len(ts) == n),
        }
    a, b = rows["n1"], rows["n2"]
    out = dict(rows)
    out["checks"] = {
        "twelve_one_edge_segments": a["segments"] == 12
        and a["edges_per_segment"] == [1],
        "n1_twelve_marks_twenty_four_strokes":
            a["marks"] == 12 and a["strokes"] == 24,
        "n1_every_edge_one_mark": a["edges_with_n_marks"] == 12,
        "n1_at_the_midpoint": a["worst_fraction_error"] is not None
        and a["worst_fraction_error"] <= 1e-6,
        "n2_twenty_four_marks_forty_eight_strokes":
            b["marks"] == 24 and b["strokes"] == 48,
        "n2_every_edge_two_marks": b["edges_with_n_marks"] == 12,
        "n2_at_thirds": b["worst_fraction_error"] is not None
        and b["worst_fraction_error"] <= 1e-6,
    }
    return out


def test_by_distance_centres_the_marks_on_each_segment():
    """Known answer: a 0.2 m cube at 0.05 m spacing.

    Each edge is one 200 mm segment. Every 50 mm with the leftover split
    equally, and never on an end: 3 gaps of 50 mm span 150 mm, leaving 25 mm
    at each end -- marks at 25, 75, 125, 175 mm. Four per edge, 48 marks,
    96 strokes (two islands per edge, as above).

    Also a segment shorter than the spacing: at 0.3 m spacing each edge gets
    one mark, at its middle (100 mm).
    """
    obj = _cube(0.2)
    rows = {}
    for label, spacing, want_mm in (("at_50mm", 0.05, [25, 75, 125, 175]),
                                    ("at_300mm", 0.3, [100])):
        segments, new, _old, _runs = _placements(
            obj, 'BY_DISTANCE', tick_spacing=spacing)
        marks, _ = _marks_of(obj, show_align_ticks=True,
                             tick_mode='BY_DISTANCE', tick_spacing=spacing)
        strokes = [st for st in marks.strokes if st.kind == 'align']
        worst = 0.0
        bad_edges = 0
        for seg in segments:
            ds = sorted(_along(seg, p) * 1000.0 for p in new
                        if _along(seg, p) is not None)
            if len(ds) != len(want_mm):
                bad_edges += 1
                continue
            worst = max(worst, max(abs(d - w) for d, w in zip(ds, want_mm)))
        rows[label] = {"segments": len(segments), "marks": len(new),
                       "strokes": len(strokes),
                       "edges_with_wrong_count": bad_edges,
                       "worst_position_error_mm": worst}
    a, b = rows["at_50mm"], rows["at_300mm"]
    out = dict(rows)
    out["checks"] = {
        "twelve_segments": a["segments"] == 12 and b["segments"] == 12,
        "four_per_edge": a["marks"] == 48 and a["edges_with_wrong_count"] == 0,
        "ninety_six_strokes": a["strokes"] == 96,
        # Tolerance from the placement itself: t is rounded to 6 places
        # (foam_marks._segment_tick_params), i.e. at most 0.5e-6 of a 200 mm
        # edge = 1e-4 mm. Measured worst: ~6e-6 mm.
        "at_25_75_125_175_mm":
            a["worst_position_error_mm"] <= 0.5e-6 * 200.0,
        "short_segment_one_mark_each": b["marks"] == 12
        and b["edges_with_wrong_count"] == 0 and b["strokes"] == 24,
        "short_segment_mark_in_the_middle":
            b["worst_position_error_mm"] <= 0.5e-6 * 200.0,
    }
    return out


def test_gore_marks_keep_clear_of_segment_ends():
    """On the gore sphere, every mark is at least L/(N+1) from an end.

    Each gore boundary meridian runs pole to pole with no corner sharp enough
    to break it (the poles are junctions of six seams), so it is one segment:
    six segments of 16 edges, both ends at |z| = radius. With N marks at
    i/(N+1), the nearest end of the closest mark is exactly L/(N+1) away and
    no mark is further than L/2 from both ends -- both bounds.
    """
    obj = _gore_sphere()
    n = 3
    segments, new, _old, _runs = _placements(obj, 'PER_RUN', ticks_per_run=n)
    nearest = []
    for seg in segments:
        length = seg["length"]
        for p in new:
            d = _along(seg, p)
            if d is not None:
                nearest.append((min(d, length - d), length))
    ratios = [near / (ln / (n + 1)) for near, ln in nearest]
    return {
        "segments": len(segments),
        "edges_per_segment": sorted({len(sg["edges"]) for sg in segments}),
        "ends_at_poles": all(abs(abs(z) - 0.15) < 1e-4
                             for sg in segments for z in sg["end_z"]),
        "marks": len(new),
        "min_end_distance_over_bound": round(min(ratios), 9) if ratios else None,
        "max_end_distance_over_half_length": round(max(
            near / (ln / 2) for near, ln in nearest), 9) if nearest else None,
        "checks": {
            "six_pole_to_pole_segments": len(segments) == 6 and sorted(
                {len(sg["edges"]) for sg in segments}) == [16],
            "every_mark_placed_once": len(new) == 6 * n
            and len(nearest) == 6 * n,
            "no_mark_nearer_an_end_than_L_over_N_plus_1":
                bool(ratios) and min(ratios) >= 1.0 - 1e-6,
            "no_mark_beyond_the_middle": bool(nearest) and max(
                near / (ln / 2) for near, ln in nearest) <= 1.0 + 1e-6,
        },
    }


def test_relief_cuts_follow_the_relief_cuts_toggle():
    """build_marks emits relief fills/strokes only when show_darts is on.

    Known answer on the seamed cube, V wedge: each square piece is one
    boundary cycle of 4 runs, and a dart goes at each of the 4 run-to-run
    corners, all with a 90 degree dihedral -- 6 x 4 = 24 'relief' strokes
    and no 'relief' fills (a V wedge is an open stroke; only a slit is
    filled). Slit: 24 fills and 24 strokes. Off: zero of either.
    Bevel arrows stay on show_brackets throughout: 24 either way. (This
    used to count brackets, 24 either way; since 2026-11-17 a cube has none,
    every run end on it being a corner, so the arrows carry the check.)
    """
    obj = _cube(2.0)
    rows = {}
    for label, darts, shape in (("off", False, 'V_WEDGE'),
                                ("on_wedge", True, 'V_WEDGE'),
                                ("on_slit", True, 'SLIT')):
        marks, _ = _marks_of(obj, show_darts=darts, notch_shape=shape,
                             show_brackets=True)
        rows[label] = {
            "relief_strokes": sum(1 for st in marks.strokes
                                  if st.kind == 'relief'),
            "relief_fills": sum(1 for f in marks.fills if f.kind == 'relief'),
            # Glyph arrows in the labels since 2026-11-17.
            "arrows": _glyph_count(marks),
        }
    out = dict(rows)
    out["checks"] = {
        "off_draws_no_relief": rows["off"]["relief_strokes"] == 0
        and rows["off"]["relief_fills"] == 0,
        "on_wedge_is_24_strokes": rows["on_wedge"]["relief_strokes"] == 24
        and rows["on_wedge"]["relief_fills"] == 0,
        "on_slit_is_24_fills_and_strokes":
            rows["on_slit"]["relief_strokes"] == 24
            and rows["on_slit"]["relief_fills"] == 24,
        "arrows_unaffected": all(r["arrows"] == 24 for r in rows.values()),
    }
    return out


def _sheet_size(obj, **overrides):
    marks, _ = _marks_of(obj, show_scale_ruler=False, **overrides)
    return marks.width_mm, marks.height_mm, marks.warnings


def test_unapplied_scale_sizes_the_sheet_and_warns_when_non_uniform():
    """The sheet is the size of the object in the scene, not of its mesh.

    Known answers: a 0.2 m cube scaled 2x (unapplied) must give the same
    sheet as the same cube with the scale applied, to 1e-6 relative; and
    the pattern area (sheet less its fixed 10 mm margins) exactly 2x the
    unscaled 0.2 m cube's each way. Non-uniform scale cannot be honoured by
    one UV->mm factor, so it is warned about: present at (1, 1, 2.5),
    absent at uniform 2x and at no scale.
    """
    margin = 2 * fm.MARGIN_MM
    plain_w, plain_h, plain_warn = _sheet_size(_cube(0.2))
    scaled = _cube(0.2, scale=(2.0, 2.0, 2.0))
    s_w, s_h, s_warn = _sheet_size(scaled)
    applied = _cube(0.2, scale=(2.0, 2.0, 2.0), apply_scale=True)
    a_w, a_h, a_warn = _sheet_size(applied)
    applied_scale = tuple(applied.scale)
    stretched = _cube(0.2, scale=(1.0, 1.0, 2.5))
    _w, _h, n_warn = _sheet_size(stretched)

    def rel(x, y):
        return abs(x - y) / max(abs(y), 1e-12)

    return {
        "plain_mm": (round(plain_w, 6), round(plain_h, 6)),
        "scaled_unapplied_mm": (round(s_w, 6), round(s_h, 6)),
        "scaled_applied_mm": (round(a_w, 6), round(a_h, 6)),
        "applied_object_scale": applied_scale,
        "warnings": {"plain": plain_warn, "uniform": s_warn,
                     "applied": a_warn, "non_uniform": n_warn},
        "checks": {
            "apply_really_applied": applied_scale == (1.0, 1.0, 1.0),
            "unapplied_equals_applied":
                rel(s_w, a_w) <= 1e-6 and rel(s_h, a_h) <= 1e-6,
            "exactly_twice_each_way":
                rel(s_w - margin, 2 * (plain_w - margin)) <= 1e-6
                and rel(s_h - margin, 2 * (plain_h - margin)) <= 1e-6,
            "no_warning_without_scale": plain_warn == (),
            "no_warning_for_uniform_scale": s_warn == () and a_warn == (),
            "warning_for_non_uniform_scale": len(n_warn) == 1
            and "non-uniform scale" in n_warn[0] and "Ctrl+A" in n_warn[0],
        },
    }


# ---------------------------------------------------------------------------
# Bevel labels: seam number, blade-tilt arrow, degrees (2026-11-17).
#
# The maker's hand-marked sheets carry "8->" on one piece and "8<-" on the
# piece it glues to: a seam number shared by the two pieces, and an arrow
# pointing to the side of the cut line the blade's tip has to end up on.
# Everything below is asserted geometrically from the Marks -- which side of
# the line a point is on, how far from it -- never by reading back the
# function that decided it.
# ---------------------------------------------------------------------------

import math as _math

# The degree sign is written ° in code: tools/blender_bridge.py does
# not carry a literal one through intact.
# The degree sign is written \u00b0 in code: tools/blender_bridge.py does
# not carry a literal one through intact.
# Every test below sets these, so a changed default cannot move an answer.
BEVEL_BASE = dict(show_brackets=True, show_seam_numbers=True,
                  pattern_surface='OUTER', decimals=0, min_angle=1.0,
                  show_heat_form=True, max_tilt_deg=45.0, show_darts=False,
                  show_align_ticks=True, tick_mode='PER_RUN', ticks_per_run=2,
                  tick_length=0.006, tick_numbers=False)


def _bevel_marks_of(obj, **overrides):
    wanted = dict(BEVEL_BASE)
    wanted.update(overrides)
    marks, _ = _marks_of(obj, **wanted)
    return marks


def _seg_dist(p, a, b):
    """Distance from point p to segment ab, all plain (x, y)."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx
                                               + (p[1] - ay) * dy) / l2))
    return _math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def _ring_edges(ring):
    return [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]


def _dist_to_ring(p, ring):
    return min(_seg_dist(p, a, b) for a, b in _ring_edges(ring))


def _owners(pts, outlines):
    """Indices of the outlines that contain every one of `pts`."""
    return [i for i, o in enumerate(outlines)
            if all(_inside(p, o.points) for p in pts)]


def _ulp32(x):
    """The gap between adjacent float32 values near |x|: mathutils Vectors
    are 32-bit, and every sheet coordinate passes through one."""
    return 2.0 ** (_math.frexp(max(abs(x), 1e-30))[1] - 24)


def _sheet_ulp(marks):
    return _ulp32(max(marks.width_mm, marks.height_mm))


def _arrow_report(marks):
    """Each blade arrow: its owner piece, and whether it points into it.

    Since 2026-11-17 the arrow is a glyph at the line end of a turned bevel
    label (`_glyph_arrow`), no longer a drawn stroke. Into the piece means
    a step from the glyph along its pointing direction moves away from the
    cut line; out of it, towards it. The owner is the one outline holding
    the label's centre.
    """
    outlines = [s for s in marks.strokes if s.kind == 'outline']
    rows = []
    for l in marks.labels:
        if l.kind != 'angle':
            continue
        g = _glyph_arrow(l)
        if g is None:
            continue
        own = _owners([(l.x, l.y)], outlines)
        if len(own) != 1:
            rows.append({"owner": None})
            continue
        ring = outlines[own[0]].points
        at, (dx, dy) = g
        step = l.size_mm * 0.1
        ahead = (at[0] + dx * step, at[1] + dy * step)
        d_at, d_ahead = _dist_to_ring(at, ring), _dist_to_ring(ahead, ring)
        rows.append({"owner": own[0], "inward": d_ahead > d_at,
                     "outward": d_ahead < d_at})
    return rows


def _arrow_counts(rows):
    return {"arrows": len(rows),
            "inward": sum(1 for r in rows if r.get("inward")),
            "outward": sum(1 for r in rows if r.get("outward")),
            "unowned": sum(1 for r in rows if r["owner"] is None)}


def _seam_table(obj):
    """{seam number: (frozenset of island indices of its edges' faces,
    sorted tuple of rounded vertex coordinates)}, read from the mesh, so a
    label can be checked against the geometric edge it names."""
    s = bpy.context.scene.foamcraft_drachis
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    try:
        bm.verts.index_update()
        bm.edges.index_update()
        uv_layer = bm.loops.layers.uv.active
        islands_faces = core.uv_islands(bm, uv_layer, only_selected=False)
        sheet = fm._Sheet(islands_faces, uv_layer, 1.0)
        islands = [fm._Island(i, f, uv_layer, sheet)
                   for i, f in enumerate(islands_faces)]
        island_of = {f.index: i for i, fs in enumerate(islands_faces)
                     for f in fs}
        numbers = fm._seam_numbers(islands, s.corner_angle_deg)
        table = {}
        for e in bm.edges:
            n = numbers.get(e.index)
            if n is None:
                continue
            pieces, coords = table.setdefault(n, (set(), set()))
            pieces |= {island_of[f.index] for f in e.link_faces}
            coords |= {tuple(round(c, 6) for c in v.co) for v in e.verts}
    finally:
        bm.free()
    return {n: (frozenset(p), tuple(sorted(c)))
            for n, (p, c) in table.items()}


def _flipped_cube():
    """The all-seamed cube with every normal reversed: every dihedral -90
    (CLAUDE.md, the sign convention pinned three ways)."""
    obj = _seamed_cube()
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def _from_pydata(name, verts, faces):
    """A mesh from plain data, normals made consistent and outward, every
    edge a seam, unwrapped."""
    _clear()
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    for e in bm.edges:
        e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def _l_prism():
    """An L-shaped prism, 1 m tall, every edge a seam: 8 pieces.

    Known answer: the L outline (0,0) (2,0) (2,1) (1,1) (1,2) (0,2) has five
    convex corners and one reflex one at (1,1). Of the 18 edges, the 12
    round the top and bottom and 5 of the 6 uprights fold +90 (convex); the
    upright at (1,1) folds -90 (concave): 17 convex seams, 1 concave.

    It is also the face shape that defeated the old "towards the face's
    centre" test for inward: the L's vertex average is (1, 1), exactly its
    reflex corner, which lies on the line of both edges meeting there.
    """
    ring = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    verts = [(x, y, 0.0) for x, y in ring] + [(x, y, 1.0) for x, y in ring]
    n = len(ring)
    faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    faces += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return _from_pydata("LPrism", verts, faces)


def _flat_pair():
    """Two coplanar 1 m quads side by side with a seam between: a fold of
    exactly 0, a straight cut. The rim is open (one face per edge), so it
    has no neighbour and no seam number."""
    verts = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)]
    faces = [(0, 1, 4, 3), (1, 2, 5, 4)]
    return _from_pydata("FlatPair", verts, faces)


def _split_top_cube(fraction=0.3):
    """A 2 m cube whose top is cut in two, off-centre, along x: three pieces
    (a side and the two halves of the top) meet at one vertex on the +x
    side's top edge, and three more on the -x side's.

    The side's outline runs straight through that vertex, so its cut run
    does not end there -- but the seam does: one part of the edge glues to
    one top piece, the rest to the other. The known-answer case for a run
    spanning two seam segments. Off-centre so the run's midpoint is not the
    junction itself.
    """
    _clear()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    # The two top edges running along y, at x = -1 and x = +1.
    top = [e for e in bm.edges
           if all(abs(v.co.z - 1.0) < 1e-6 for v in e.verts)
           and abs(e.verts[0].co.x - e.verts[1].co.x) < 1e-6]
    new = []
    for e in sorted(top, key=lambda e: e.verts[0].co.x):
        a = min(e.verts, key=lambda v: v.co.y)
        _e, v = bmesh.utils.edge_split(e, a, fraction)
        new.append(v)
    bmesh.ops.connect_verts(bm, verts=new)
    for e in bm.edges:
        e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def _parse(text):
    """'<n>, <deg>°' -> (n, '<deg>'); '<n>' -> (n, None); '<deg>°' ->
    (None, '<deg>'). The comma is asserted where it matters, not
    stripped silently: see the cube's label check."""
    n = deg = None
    # The arrow glyph (2026-11-17) is not part of the reading.
    for part in _body(text).replace(",", " ").split():
        if part.endswith("\u00b0"):
            deg = part[:-1]
        else:
            n = int(part)
    return n, deg


def test_cube_seams_are_numbered_once_and_shared_by_both_pieces():
    """All-seamed cube, pattern on the outer skin.

    Known answers: 12 edges, each its own seam (every cube vertex is a
    junction of three) -> numbers exactly {1..12}. Each number is written on
    exactly two labels, on two different pieces, and those two are the
    pieces either side of that cube edge. Every fold is +90 (convex), so
    the blade tilt is 45 per side: 24 labels "<n>, 45°" and 24 arrows, every
    one pointing INTO its own piece -- the far face of a 45 degree mitre on
    the outer skin lies inside the outline (foam_marks._blade_tip_inward
    has the derivation).
    """
    obj = _seamed_cube()
    marks = _bevel_marks_of(obj)
    table = _seam_table(obj)
    outlines = [s for s in marks.strokes if s.kind == 'outline']
    angles = [l for l in marks.labels if l.kind == 'angle']
    parsed = [_parse(l.text) for l in angles]
    by_number = {}
    for l, (n, _deg) in zip(angles, parsed):
        by_number.setdefault(n, []).append(
            tuple(_owners([(l.x, l.y)], outlines)))
    pieces_match = bool(by_number) and all(
        len(owners) == 2 and all(len(o) == 1 for o in owners)
        and frozenset(o[0] for o in owners) == table.get(n, (None,))[0]
        and len({o[0] for o in owners}) == 2
        for n, owners in by_number.items())
    counts = _arrow_counts(_arrow_report(marks))
    return {
        "numbers": sorted(n for n in by_number if n is not None),
        "texts": sorted({l.text.split()[-1] for l in angles}),
        "labels": len(angles),
        "arrows": counts,
        "checks": {
            "numbers_are_1_to_12": sorted(by_number) == list(range(1, 13)),
            "mesh_agrees_12_seams": sorted(table) == list(range(1, 13)),
            "each_number_on_the_two_pieces_of_that_edge": pieces_match,
            # "<n>, 45°": the comma is the maker's, so "4 3°" is not
            # read as "43°" (2026-11-17).
            "twenty_four_labels_reading_n_comma_45": len(angles) == 24
            and all(_body(l.text) == "%d, 45\u00b0" % n
                    for l, (n, deg) in zip(angles, parsed) if n is not None)
            and all(n is not None for n, _d in parsed),
            "twenty_four_arrows_all_into_their_piece": counts == {
                "arrows": 24, "inward": 24, "outward": 0, "unowned": 0},
        },
    }


def test_arrows_flip_for_the_inner_skin_and_for_valleys():
    """Which way an arrow points, on three known-answer shapes.

    Cube, pattern on the INNER skin: all 24 point outward, still drawn
    inside the piece. Normals-flipped cube (every fold -90, all valleys),
    OUTER: all 24 outward. L prism, OUTER: 17 convex and 1 concave seam, so
    34 arrows inward and 2 outward. The two outward ones are on the edges of
    the two side pieces that meet at the reflex upright.

    Measured 2026-11-17 with the old "towards the face's centre" inward
    test put back: 32 in, 2 out, and 2 arrows drawn off the piece entirely
    -- the L top's two edges at its reflex corner, where that centre lies
    on the edge's own line. foam_marks._inward_at now reads the face's
    winding instead.
    """
    cube = _seamed_cube()
    a = _arrow_counts(_arrow_report(
        _bevel_marks_of(cube, pattern_surface='INNER')))
    b = _arrow_counts(_arrow_report(_bevel_marks_of(_flipped_cube())))
    l_obj = _l_prism()
    c = _arrow_counts(_arrow_report(_bevel_marks_of(l_obj)))
    l_seams = len(_seam_table(l_obj))
    return {
        "inner_cube": a, "flipped_cube": b, "l_prism": c,
        "l_prism_seams": l_seams,
        "checks": {
            "inner_all_24_outward": a == {"arrows": 24, "inward": 0,
                                          "outward": 24, "unowned": 0},
            "flipped_all_24_outward": b == {"arrows": 24, "inward": 0,
                                            "outward": 24, "unowned": 0},
            "l_prism_has_18_seams": l_seams == 18,
            "l_prism_34_in_2_out": c == {"arrows": 36, "inward": 34,
                                         "outward": 2, "unowned": 0},
        },
    }


def test_a_straight_cut_gets_a_number_and_no_arrow():
    """Below `min_angle` there is no bevel: the number only -- no arrow and
    no degrees, the maker's "no arrow means cut at 90 degrees".

    Flat pair: one seam of fold 0 between two pieces -> exactly 2 labels,
    both "1", and 0 arrows; the open rim gets nothing. Cube with min_angle
    91 (above every fold of 90): 24 number-only labels, 0 arrows. And Seam
    Numbers off on the cube: 24 labels reading "45°", 24 arrows.
    """
    flat = _bevel_marks_of(_flat_pair())
    flat_labels = sorted(l.text for l in flat.labels if l.kind == 'angle')
    flat_arrows = _glyph_count(flat)
    cube = _seamed_cube()
    steep = _bevel_marks_of(cube, min_angle=91.0)
    steep_labels = [l.text for l in steep.labels if l.kind == 'angle']
    steep_arrows = _glyph_count(steep)
    plain = _bevel_marks_of(cube, show_seam_numbers=False)
    # The glyph is part of the text since 2026-11-17; "45°" is its body.
    plain_labels = [_body(l.text) for l in plain.labels if l.kind == 'angle']
    plain_arrows = _glyph_count(plain)
    return {
        "flat_labels": flat_labels, "flat_arrows": flat_arrows,
        "min_angle_91_labels": sorted(set(steep_labels)),
        "min_angle_91_arrows": steep_arrows,
        "numbers_off_labels": sorted(set(plain_labels)),
        "numbers_off_arrows": plain_arrows,
        "checks": {
            "flat_two_labels_both_1": flat_labels == ["1", "1"],
            "flat_no_arrow": flat_arrows == 0,
            # Each of the 12 seam numbers twice, and nothing else.
            "min_angle_24_number_only": sorted(steep_labels, key=int)
            == sorted([str(n) for n in range(1, 13)] * 2, key=int),
            "min_angle_no_arrow": steep_arrows == 0,
            "numbers_off_24_reading_45": plain_labels == ["45\u00b0"] * 24,
            "numbers_off_24_arrows": plain_arrows == 24,
        },
    }


def test_seam_numbers_survive_a_fresh_bmesh():
    """The same seam gets the same number on every export.

    Numbers are ordered by vertex coordinates, never by element order (the
    BMesh tables are stable for one mesh, sets of elements are not --
    CLAUDE.md, "What survives rebuilding a BMesh"). Four fresh BMeshes of
    the gore sphere must give the same {number: (pieces, coordinates)}
    table. The gore sphere has 6 pole-to-pole seams, so numbers 1..6.
    """
    obj = _gore_sphere()
    tables = [_seam_table(obj) for _ in range(4)]
    return {
        "numbers": sorted(tables[0]),
        "checks": {
            "six_seams": sorted(tables[0]) == list(range(1, 7)),
            "identical_four_times": all(t == tables[0] for t in tables[1:]),
        },
    }


def _run_spans(obj):
    """For each cut run with a neighbour, how many seam segments its edges
    lie on. From the mesh; plain numbers out."""
    s = bpy.context.scene.foamcraft_drachis
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    try:
        bm.verts.index_update()
        bm.edges.index_update()
        uv_layer = bm.loops.layers.uv.active
        islands_faces = core.uv_islands(bm, uv_layer, only_selected=False)
        sheet = fm._Sheet(islands_faces, uv_layer, 1.0)
        islands = [fm._Island(i, f, uv_layer, sheet)
                   for i, f in enumerate(islands_faces)]
        numbers = fm._seam_numbers(islands, s.corner_angle_deg)
        runs = list(core.iter_boundary_dart_segments(
            bm, uv_layer, s.max_chord_deviation, s.max_angle_variation_deg,
            islands=islands_faces, corner_angle_deg=s.corner_angle_deg,
            matrix=obj.matrix_world, dart_wedge_deg=s.dart_wedge_deg,
            max_darts_per_run=s.max_darts_per_run))
        spans = [len(fm._run_seams(r, numbers)) for r in runs
                 if r.has_dihedral]
    finally:
        bm.free()
    return spans


def test_a_cut_run_can_span_two_seams():
    """Measured: can one cut run lie on more than one seam segment? Yes.

    Runs break at *this* piece's corners (and where the bevel drifts);
    seam segments also break where a third piece joins on the far side.
    Where the near piece's outline runs straight past such a junction, one
    run covers two seams. The split-top cube is the known answer: the +x
    and -x sides each have one top-edge run over two seams, and no other
    run spans more than one -- exactly 2 runs spanning 2.

    The gore sphere, by contrast: every seam runs pole to pole and every
    run ends at a pole or part way along one seam, so none spans two.

    What the label then says. Until 2026-11-17 it was one label per run,
    with the seam its midpoint lies on, so on the split-top cube 2 seam
    numbers were written on one piece only (seams 4 and 13 in that
    numbering) -- the old assertion here was `len(singles) == 2`. The
    maker: "if a run can cover two seams on a long piece then the run
    should have 2 numbers on the longer seam to remove ambiguity." So now
    (2026-11-17), the property the maker wants:

      * every seam number appears on exactly 2 labels, on 2 different
        pieces, and those are the two pieces either side of that seam
        (read from the mesh, `_seam_table`);
      * each spanning run carries exactly 2 labels -- found as two labels
        on one piece whose run portions meet end to start;
      * each of those labels lies within its own seam's portion: its
        arrow's near end is the arrow gap (0.3 em) from its own portion
        and further than that from the other portion; and its portion's
        share of the run matches its seam's share of the 3D edge (the top
        edge is split 0.3 / 0.7, so the shares are 0.3 and 0.7 -- the
        unwrap of one flat quad is a similarity, so the ratio survives).
    """
    gore = _run_spans(_gore_sphere())
    split = _split_top_cube()
    split_spans = _run_spans(split)
    marks = _bevel_marks_of(split)
    table = _seam_table(split)
    angles = [l for l in marks.labels if l.kind == 'angle']
    numbers = [_parse(l.text)[0] for l in angles]

    by_number = {}
    for l, n in zip(angles, numbers):
        by_number.setdefault(n, []).append(l.anchor.piece
                                           if l.anchor else None)
    on_both_pieces = bool(by_number) and all(
        n is not None and len(pieces) == 2 and len(set(pieces)) == 2
        and frozenset(pieces) == table.get(n, (None,))[0]
        for n, pieces in by_number.items())

    def length(pts):
        return sum(_math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(pts, pts[1:]))

    def dist_to(p, pts):
        return min(_seg_dist(p, a, b) for a, b in zip(pts, pts[1:]))

    def seam_len(n):
        coords = table[n][1]
        return max(_math.dist(a, b) for a in coords for b in coords)

    def straight_on(a, b):
        # a's last segment and b's first point the same way: one straight
        # line through the joint. Consecutive *runs* on a piece meet at a
        # corner (90 degrees on a cube) and are not one run's portions.
        (p, q), (r, t) = a[-2:], b[:2]
        u, v = (q[0] - p[0], q[1] - p[1]), (t[0] - r[0], t[1] - r[1])
        cross = u[0] * v[1] - u[1] * v[0]
        dot = u[0] * v[0] + u[1] * v[1]
        return dot > 0 and abs(cross) <= 1e-3 * _math.hypot(*u) * _math.hypot(*v)

    # Pairs of labels on one piece whose portions continue each other in a
    # straight line, end to start: the two halves of one spanning run.
    pairs = []
    for i, a in enumerate(angles):
        for j, b in enumerate(angles):
            if (i != j and a.anchor and b.anchor
                    and a.anchor.piece == b.anchor.piece
                    and _math.dist(a.anchor.run[-1], b.anchor.run[0]) < 1e-6
                    and straight_on(a.anchor.run, b.anchor.run)):
                pairs.append((i, j))
    spanning_pieces = sorted(angles[i].anchor.piece for i, _j in pairs)

    within = []
    for i, j in pairs:
        for me, other in ((i, j), (j, i)):
            lab, oth = angles[me], angles[other]
            # The glyph end of the label: 0.3 em (the arrow gap) inside
            # its line, at the middle of its own portion (2026-11-17: the
            # drawn arrow's near end used to be measured here).
            glyph = _glyph_arrow(lab)
            gap = lab.size_mm * fm.BEVEL_ARROW_GAP_EM
            if glyph is None:
                within.append(None)
                continue
            near = glyph[0]
            d_own = dist_to(near, lab.anchor.run)
            d_other = dist_to(near, oth.anchor.run)
            share = length(lab.anchor.run) / (length(lab.anchor.run)
                                              + length(oth.anchor.run))
            n_me, n_other = numbers[me], numbers[other]
            want = seam_len(n_me) / (seam_len(n_me) + seam_len(n_other))
            within.append({
                "seam": n_me, "d_own": round(d_own, 4),
                "d_other": round(d_other, 4), "gap": round(gap, 4),
                "share": round(share, 4), "want_share": round(want, 4),
                "ok": abs(d_own - gap) < 1e-3 * max(gap, 1.0)
                and d_other > gap * 2 and abs(share - want) < 0.02})
    return {
        "gore_runs": len(gore),
        "gore_max_span": max(gore) if gore else None,
        "split_runs": len(split_spans),
        "split_spans": sorted(split_spans),
        "split_seams": len(by_number),
        "labels": len(angles),
        "spanning_pairs": len(pairs),
        "spanning_pieces": spanning_pieces,
        "within": within,
        "checks": {
            "gore_has_runs": len(gore) > 0,
            "gore_no_run_spans_two": bool(gore) and max(gore) == 1,
            "split_exactly_two_runs_span_two":
                sorted(split_spans).count(2) == 2 and max(split_spans) == 2,
            "mesh_seams_all_labelled":
                sorted(by_number) == sorted(table),
            "every_number_on_its_two_pieces": on_both_pieces,
            # One (end, start) pair per spanning run -- a pair is only
            # found in run order -- so 2 runs -> 2 pairs on 2 pieces.
            "two_spanning_runs_with_two_labels_each":
                len(pairs) == 2 and len(set(spanning_pieces)) == 2,
            "each_label_within_its_own_portion":
                len(within) == 4 and all(w and w["ok"] for w in within),
        },
    }


def test_nothing_is_drawn_at_a_corner():
    """The maker: "There is no need for alignment marks at the corners. The
    corners are implicitly aligning."

    Cube: every outline vertex is a corner and a junction. Exactly 0
    strokes of any kind other than the outline itself have a point within
    1 mm of one (relief cuts off: they are cuts, not marks, and sit at run
    ends by design) -- while all 48 alignment marks (2 per seam side, 12
    seams, 2 sides) are still drawn, so the zero is not an empty sheet. And
    no 'bracket' at all: every run on a cube ends at a corner.

    Why 1 mm and not 1e-6: sheet coordinates are float32, 4.9e-4 mm apart
    on this ~8 m sheet, so a mark placed *at* a corner could land a few
    steps off it and slip under a micron-sized radius. 1 mm is far above
    that noise and far below the nearest legitimate mark (a third of a
    2000 mm edge). The nearest distance is reported.
    """
    marks = _bevel_marks_of(_seamed_cube())
    corners = [p for s in marks.strokes if s.kind == 'outline'
               for p in s.points]
    dists = [(_math.hypot(p[0] - c[0], p[1] - c[1]), s.kind)
             for s in marks.strokes if s.kind != 'outline'
             for p in s.points for c in corners]
    near = [kind for d, kind in dists if d < 1.0]
    align = sum(1 for s in marks.strokes if s.kind == 'align')
    brackets = sum(1 for s in marks.strokes if s.kind == 'bracket')
    return {
        "outline_corners": len(corners),
        "strokes_at_a_corner": near,
        "nearest_mark_to_a_corner_mm": min(d for d, _k in dists)
        if dists else None,
        "align": align, "brackets": brackets,
        "checks": {
            "24_outline_corners": len(corners) == 24,
            "nothing_at_a_corner": near == [],
            "still_48_alignment_marks": align == 48,
            "no_run_end_marks_on_a_cube": brackets == 0,
        },
    }


def _start_mark_report(marks, kind):
    """Each `kind` stroke: does it start on its piece's outline, end
    strictly inside, run square to the outline edge it starts on, and how
    long is it. The owner is the piece holding its inner end."""
    outlines = [s for s in marks.strokes if s.kind == 'outline']
    rows = []
    for st in (s for s in marks.strokes if s.kind == kind):
        start, end = st.points[0], st.points[-1]
        own = _owners([end], outlines)
        if len(own) != 1:
            rows.append(None)
            continue
        ring = outlines[own[0]].points
        edges = _ring_edges(ring)
        near = sorted(edges, key=lambda e: _seg_dist(start, *e))
        # On a vertex the mark is meant to be square to the average of the
        # two edges meeting there.
        on_vertex = any(_math.hypot(start[0] - v[0], start[1] - v[1]) < 1e-3
                        for v in ring)
        dirs = []
        for a, b in (near[:2] if on_vertex else near[:1]):
            L = _math.hypot(b[0] - a[0], b[1] - a[1])
            dirs.append(((b[0] - a[0]) / L, (b[1] - a[1]) / L))
        if len(dirs) == 2 and (dirs[0][0] * dirs[1][0]
                               + dirs[0][1] * dirs[1][1]) < 0:
            dirs[1] = (-dirs[1][0], -dirs[1][1])
        ex, ey = sum(d[0] for d in dirs), sum(d[1] for d in dirs)
        mx, my = end[0] - start[0], end[1] - start[1]
        length = _math.hypot(mx, my)
        rows.append({"start_off_line": _seg_dist(start, *near[0]),
                     "end_depth": _dist_to_ring(end, ring),
                     "dot": abs(ex * mx + ey * my)
                     / (_math.hypot(ex, ey) * length),
                     "length": length, "on_vertex": on_vertex})
    return rows


def _shortest_outline_edge(marks):
    return min(_math.hypot(b[0] - a[0], b[1] - a[1])
               for s in marks.strokes if s.kind == 'outline'
               for a, b in _ring_edges(s.points))


def test_alignment_marks_go_inward_and_square_to_the_line():
    """The maker: "The alignment marks should extend inwards and be
    perpendicular to the edge they are aligned along."

    Cube (2 m, tick_length 6 mm): each of the 48 'align' strokes starts on
    its piece's outline, ends strictly inside it, is square to the outline
    edge (|cos| < 1e-6) and is 6 mm long (to 4 float32 steps). "On the
    outline" was first asked to 1e-6 mm and measured 1.6e-4 mm: the cube's
    sheet is ~8 m across, where adjacent float32 values are 4.9e-4 mm apart
    (the same finding as TOL_REL in tests/test_overlay_ticks.py), so a
    point interpolated along an edge that is not exactly axis-aligned in UV
    cannot sit closer to it than that. The tolerance is one float32 step.

    Gore sphere: the same, with tolerances from float32 rather than chosen.
    Sheet coordinates are 32-bit (mathutils), so a point is good to about
    one float32 step `u` of the sheet's largest coordinate. A point on a
    slanted edge can then be ~u off it; a 6 mm mark's direction is good to
    ~2u / 6 mm and an edge's to ~2u / its length. So: 4u on position, and
    4u / 6 mm + 4u / (shortest outline edge) on |cos|. Length at most 6 mm
    (+4u) -- shortened at a gore's narrow tip -- and at least one at 6 mm.
    """
    out, checks = {}, {}
    for name, build in (("cube", _seamed_cube), ("gores", _gore_sphere)):
        marks = _bevel_marks_of(build())
        rows = _start_mark_report(marks, 'align')
        good = [r for r in rows if r is not None]
        u = _sheet_ulp(marks)
        if name == "cube":
            pos_tol, dot_tol = u, 1e-6
        else:
            pos_tol = 4 * u
            dot_tol = 4 * u / 6.0 + 4 * u / _shortest_outline_edge(marks)
        out[name] = {
            "align": len(rows), "owned": len(good),
            "on_a_vertex": sum(1 for r in good if r["on_vertex"]),
            "worst_start_off_line": max((r["start_off_line"] for r in good),
                                        default=None),
            "min_end_depth": min((r["end_depth"] for r in good),
                                 default=None),
            "worst_cos": max((r["dot"] for r in good), default=None),
            "length_range": [min(r["length"] for r in good),
                             max(r["length"] for r in good)] if good else None,
            "float32_step_mm": u, "pos_tol": pos_tol, "cos_tol": dot_tol,
        }
        checks[name + "_has_marks_all_owned"] = \
            len(good) > 0 and len(good) == len(rows)
        checks[name + "_start_on_the_line"] = all(
            r["start_off_line"] < pos_tol for r in good)
        checks[name + "_end_strictly_inside"] = all(
            r["end_depth"] > pos_tol for r in good)
        checks[name + "_perpendicular"] = all(r["dot"] < dot_tol for r in good)
        if name == "cube":
            checks["cube_48_marks"] = len(rows) == 48
            checks["cube_length_is_tick_length"] = all(
                abs(r["length"] - 6.0) < 4 * u for r in good)
        else:
            checks["gores_never_longer_than_tick_length"] = all(
                r["length"] <= 6.0 + 4 * u for r in good)
            checks["gores_some_at_full_length"] = any(
                abs(r["length"] - 6.0) < 4 * u for r in good)
    out["checks"] = checks
    return out


def test_run_end_marks_only_part_way_along_a_seam():
    """On the gore sphere the bevel drifts with latitude, so each side of a
    gore breaks into several cut runs along one pole-to-pole seam. Those
    breaks get a run-end mark, inward and square to the line like an
    alignment mark. Both bounds: more than 0 of them, and 0 at a segment
    end -- the poles, where six gores meet, are the only segment ends here,
    and on the sheet each is a gore's topmost or bottommost outline point.

    Tolerances as for the gore's alignment marks, from float32.
    """
    marks = _bevel_marks_of(_gore_sphere())
    rows = _start_mark_report(marks, 'bracket')
    good = [r for r in rows if r is not None]
    u = _sheet_ulp(marks)
    dot_tol = 4 * u / 6.0 + 4 * u / _shortest_outline_edge(marks)
    tips = []
    for s in marks.strokes:
        if s.kind == 'outline':
            ys = sorted(s.points, key=lambda p: p[1])
            tips += [ys[0], ys[-1]]
    at_end = sum(1 for st in marks.strokes if st.kind == 'bracket'
                 and any(_math.hypot(st.points[0][0] - t[0],
                                     st.points[0][1] - t[1]) < 1.0
                         for t in tips))
    return {
        "run_end_marks": len(rows),
        "at_a_pole": at_end,
        "worst_cos": max((r["dot"] for r in good), default=None),
        "cos_tol": dot_tol,
        "checks": {
            "some_mid_seam_marks": len(good) > 0 and len(good) == len(rows),
            "none_at_a_segment_end": at_end == 0,
            "start_on_the_line": all(r["start_off_line"] < 4 * u
                                     for r in good),
            "inward": all(r["end_depth"] > 4 * u for r in good),
            "perpendicular": all(r["dot"] < dot_tol for r in good),
        },
    }

TESTS = (
    test_bmesh_vertex_wrappers_compare_by_identity,
    test_what_the_old_per_run_placement_did,
    test_cube_per_segment_marks_are_at_exact_fractions,
    test_by_distance_centres_the_marks_on_each_segment,
    test_gore_marks_keep_clear_of_segment_ends,
    test_relief_cuts_follow_the_relief_cuts_toggle,
    test_unapplied_scale_sizes_the_sheet_and_warns_when_non_uniform,
    test_every_angle_is_written_inside_its_own_piece,
    test_the_ruler_strip_stays_clear_of_the_pattern_and_of_itself,
    test_cube_marks_have_the_known_counts,
    test_marks_hold_no_bmesh_and_no_vectors,
    test_cube_seams_are_numbered_once_and_shared_by_both_pieces,
    test_arrows_flip_for_the_inner_skin_and_for_valleys,
    test_a_straight_cut_gets_a_number_and_no_arrow,
    test_seam_numbers_survive_a_fresh_bmesh,
    test_a_cut_run_can_span_two_seams,
    test_nothing_is_drawn_at_a_corner,
    test_alignment_marks_go_inward_and_square_to_the_line,
    test_run_end_marks_only_part_way_along_a_seam,
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
