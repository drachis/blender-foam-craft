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
            # 4 runs per square, 24 in all: one bracket at each run's end.
            "twenty_four_brackets": len(brackets) == 24,
            # Every dihedral of a cube is 90, so every angle says so.
            "twenty_four_angles_all_90": len(angles) == 24 and all(
                l.text.startswith("90") for l in angles),
        },
    }


def test_marks_hold_no_bmesh_and_no_vectors():
    """Every leaf of the returned Marks is a plain value.

    The BMesh is freed inside `_marks_of` before this looks at anything, so
    a held element would show here as a bmesh type (or crash, which the
    runner reports as an error -- also a failure). Checked across every mark
    kind the sweep can produce, with the optional ones switched on.
    """
    obj = _seamed_cube()
    marks, _ = _marks_of(obj, island_fill=True, show_island_ids=True,
                         show_brackets=True, show_align_ticks=True,
                         tick_numbers=True, show_scale_ruler=True)
    foreign = _foreign_leaves(marks)
    kinds = sorted({m.kind for m in marks.fills + marks.strokes + marks.labels})
    return {
        "foreign_leaves": foreign[:10],
        "kinds_present": kinds,
        "checks": {
            "no_foreign_values": foreign == [],
            # Both bounds: an empty Marks would pass the check above.
            "every_optional_kind_was_produced": set(kinds) >= {
                'piece', 'outline', 'bracket', 'align', 'ruler',
                'number', 'angle'},
        },
    }


def _label_box(label):
    """A label's box, from the same upper bound the model places it with."""
    half_w = len(label.text) * label.size_mm * fm.GLYPH_ADVANCE_EM / 2
    half_h = label.size_mm / 2
    return (label.x - half_w, label.y - half_h,
            label.x + half_w, label.y + half_h)


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
    """
    marks, _ = _marks_of(_seamed_cube())
    outlines = [s for s in marks.strokes if s.kind == 'outline']
    angles = [l for l in marks.labels if l.kind == 'angle']
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
        1 for box, own in zip(boxes, owners) if len(own) == 1 and all(
            _inside(c, outlines[own[0]].points)
            for c in ((box[0], box[1]), (box[2], box[1]),
                      (box[2], box[3]), (box[0], box[3]))))
    return {
        "angle_labels": len(angles),
        "per_piece": per_piece,
        "overlapping_pairs": overlaps,
        "boxes_wholly_inside": corners_inside,
        "checks": {
            "there_are_24_labels": len(angles) == 24,
            "each_piece_has_exactly_4": per_piece == [4] * 6,
            "no_two_labels_overlap": overlaps == 0,
            "every_box_is_wholly_inside_its_piece": corners_inside == 24,
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
    Brackets stay on show_brackets throughout: 24 either way.
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
            "brackets": sum(1 for st in marks.strokes
                            if st.kind == 'bracket'),
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
        "brackets_unaffected": all(r["brackets"] == 24 for r in rows.values()),
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
