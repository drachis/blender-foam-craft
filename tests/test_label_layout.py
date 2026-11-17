"""Bevel labels: turned text with an arrow glyph, laid out so they fit.

    python tools/launch_blender.py --port 9930
    python tools/blender_bridge.py --port 9930 < tests/test_label_layout.py

Two decisions of the maker's (2026-11-17), checked here together because
the second is laid out on top of the first:

  * The label: one string, "-> 8, 30°", turned square to its cut line at the
    middle of the run, the glyph (U+2192 / U+2190) pointing to the side of
    the line the blade's tip goes, never upside down, the glyph at the end
    nearest the line (`foam_marks.bevel_label_frame`). Drawn arrows are
    retired.
  * The layout (`foam_layout.layout_labels`): where labels collide, slide
    along the run and shrink -- one size per piece, never below
    `min_label_mm` -- and failing that go outside on a leader line.

Every check is geometry against an independent model of a label -- the
contract's box, `len(text) * size * GLYPH_ADVANCE_EM` by `size`, centred on
(x, y), TURNED by `rotation_deg` counter-clockwise on the page (y down, so
reading along (cos r, -sin r)) -- written out again below rather than
imported from foam_layout, so a wrong box in the layout cannot agree with
itself.
"""

import math
import os
import shutil
import sys
import tempfile
import traceback

import bpy
import bmesh

sys.path.insert(0, r"O:\Work\BlenderPlugin\FoamCraft\tools")
import sandbox_guard  # noqa: E402

_MOD = "bl_ext.user_default.foamcraft_drachis"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foamcraft_drachis.foam_marks as fm  # noqa: E402

try:
    import bl_ext.user_default.foamcraft_drachis.foam_layout as fl  # noqa: E402
except ImportError:
    fl = None

ROOT = r"O:\Work\BlenderPlugin\FoamCraft"
SNAPSHOT = os.path.join(ROOT, "tests", "fixtures", "beast_skull_snapshot.blend")
SNAPSHOT_OBJECT = "head_simplified"
RIGHT, LEFT = "\u2192", "\u2190"


# --- the independent label model -----------------------------------------

def _axes(label):
    """(reading, up) on the sheet, y down, for the label's page rotation."""
    a = math.radians(getattr(label, "rotation_deg", 0.0) or 0.0)
    return (math.cos(a), -math.sin(a)), (-math.sin(a), -math.cos(a))


def _quad(label):
    """The label's box as four corners, turned with it."""
    (rx, ry), (ux, uy) = _axes(label)
    hw = len(label.text) * label.size_mm * fm.GLYPH_ADVANCE_EM / 2
    hh = label.size_mm / 2
    cx, cy = label.x, label.y
    if label.valign == 'baseline':
        cx, cy = cx + ux * hh, cy + uy * hh
    return tuple((cx + sx * rx * hw + sy * ux * hh,
                  cy + sx * ry * hw + sy * uy * hh)
                 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)))


def _ends(label):
    """(start, end) of the text along its reading direction."""
    (rx, ry), _u = _axes(label)
    hw = len(label.text) * label.size_mm * fm.GLYPH_ADVANCE_EM / 2
    return ((label.x - rx * hw, label.y - ry * hw),
            (label.x + rx * hw, label.y + ry * hw))


def _glyph(label):
    """(glyph end point, pointing direction) or None for no glyph."""
    (rx, ry), _u = _axes(label)
    start, end = _ends(label)
    t = label.text
    if t.startswith(RIGHT) or t.startswith(LEFT):
        at, g = start, t[0]
    elif t.endswith(RIGHT) or t.endswith(LEFT):
        at, g = end, t[-1]
    else:
        return None
    sign = 1.0 if g == RIGHT else -1.0
    return at, (rx * sign, ry * sign)


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _quads_overlap(a, b, eps=1e-6):
    for quad in (a, b):
        for i in range(4):
            p, q = quad[i], quad[(i + 1) % 4]
            ax, ay = q[1] - p[1], p[0] - q[0]
            n = math.hypot(ax, ay) or 1.0
            pa = [(ax * c[0] + ay * c[1]) / n for c in a]
            pb = [(ax * c[0] + ay * c[1]) / n for c in b]
            if max(pa) <= min(pb) + eps or max(pb) <= min(pa) + eps:
                return False
    return True


def _seg_cross(p, q, r, s, eps=1e-9):
    def on(a, b, c):
        return (min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
                and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps)
    d1, d2 = _orient(r, s, p), _orient(r, s, q)
    d3, d4 = _orient(p, q, r), _orient(p, q, s)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    return ((abs(d1) <= eps and on(r, s, p)) or (abs(d2) <= eps and on(r, s, q))
            or (abs(d3) <= eps and on(p, q, r))
            or (abs(d4) <= eps and on(p, q, s)))


def _in_quad(p, quad):
    s = [_orient(quad[i], quad[(i + 1) % 4], p) for i in range(4)]
    return all(v > 1e-12 for v in s) or all(v < -1e-12 for v in s)


def _seg_hits_quad(p, q, quad):
    if _in_quad(p, quad) or _in_quad(q, quad):
        return True
    return any(_seg_cross(p, q, quad[i], quad[(i + 1) % 4]) for i in range(4))


def _segments(stroke):
    pts = stroke.points
    out = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    if stroke.closed and len(pts) > 2:
        out.append((pts[-1], pts[0]))
    return out


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
    """Even-odd over every ring: inside the outer, outside any hole."""
    return sum(_inside(pt, r) for r in rings) % 2 == 1


def _quad_in_rings(quad, rings):
    if not all(_in_rings(c, rings) for c in quad):
        return False
    return not any(_seg_hits_quad(a, b, quad)
                   for r in rings for a, b in zip(r, r[1:] + r[:1]))


def _seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx
                                               + (p[1] - a[1]) * dy) / l2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _dist_to_polyline(p, pts):
    return min(_seg_dist(p, a, b) for a, b in zip(pts, pts[1:]))


def _nearest_normal(p, anchor):
    """The anchor's inward normal on the run segment nearest `p`."""
    k = min(range(len(anchor.run) - 1),
            key=lambda i: _seg_dist(p, anchor.run[i], anchor.run[i + 1]))
    return anchor.normals[min(k, len(anchor.normals) - 1)]


def _dist_pt_quad(p, quad):
    if _in_quad(p, quad):
        return 0.0
    return min(_seg_dist(p, quad[i], quad[(i + 1) % 4]) for i in range(4))


def _census(marks):
    """The size of the problem, counted the way the maker sees it, with
    turned boxes. Pairs, not labels: two labels on one spot are one
    collision."""
    angles = [l for l in marks.labels if l.kind == 'angle']
    numbers = [l for l in marks.labels if l.kind == 'number']
    outlines = [s for s in marks.strokes if s.kind == 'outline']
    quads = [_quad(l) for l in angles]
    pairs = [(i, j) for i in range(len(quads))
             for j in range(i + 1, len(quads))
             if _quads_overlap(quads[i], quads[j])]
    on_number = [i for i, q in enumerate(quads)
                 if any(_quads_overlap(q, _quad(n)) for n in numbers)]
    cut_segs = [seg for o in outlines for seg in _segments(o)]
    crossing = [i for i, q in enumerate(quads)
                if any(_seg_hits_quad(p, r, q) for p, r in cut_segs)]
    return {"angle_labels": len(angles), "label_label_pairs": pairs,
            "on_number": on_number, "crossing_cut": crossing}


def _owned(marks):
    """[(label, [leaders])] for every anchored bevel label."""
    leaders = {}
    for s in marks.strokes:
        if s.kind == 'leader' and s.anchor is not None:
            leaders.setdefault(s.anchor, []).append(s)
    return [(l, leaders.get(l.anchor, []))
            for l in marks.labels if l.kind == 'angle' and l.anchor]


def _glyph_faults(label):
    """What is wrong with a label's turn and glyph, as a list of words."""
    out = []
    rot = label.rotation_deg
    if not (-90.0 < rot <= 90.0 + 1e-9):
        out.append("upside_down")
    g = _glyph(label)
    if label.anchor.tip_inward is None:
        if g is not None:
            out.append("glyph_on_straight_cut")
        return out
    if g is None:
        return out + ["no_glyph"]
    at, pointing = g
    n = _nearest_normal(at, label.anchor)
    blade = n if label.anchor.tip_inward else (-n[0], -n[1])
    if pointing[0] * blade[0] + pointing[1] * blade[1] < 1 - 1e-6:
        out.append("glyph_not_blade_way")
    # The glyph is the end of the text nearest the line.
    start, end = _ends(label)
    far = end if at == start else start
    if _dist_to_polyline(at, label.anchor.run) \
            > _dist_to_polyline(far, label.anchor.run) + 1e-9:
        out.append("glyph_not_at_line")
    return out


def _audit(marks, min_mm):
    """Everything asked for, counted. A pass is every list empty."""
    owned = _owned(marks)
    census = _census(marks)
    outlines = [s for s in marks.strokes if s.kind == 'outline']
    rings = [o.points for o in outlines if o.closed]
    cut_segs = [seg for o in outlines for seg in _segments(o)]
    small = [l.text for l, _ld in owned if l.size_mm < min_mm - 1e-9]
    homeless, bad_leader, leader_crosses, outside_on_piece = [], [], [], []
    glyph_faults, sizes_by_piece = [], {}
    for l, leaders in owned:
        quad = _quad(l)
        sizes_by_piece.setdefault(l.anchor.piece, set()).add(
            round(l.size_mm, 6))
        glyph_faults += [(l.text, f) for f in _glyph_faults(l)]
        inside = _quad_in_rings(quad, l.anchor.rings)
        if not inside and len(leaders) != 1:
            homeless.append(l.text)
        if inside and leaders:
            homeless.append(l.text)
        for ld in leaders:
            start, end = ld.points[0], ld.points[-1]
            # A small gap off the text -- more than nothing, at most half
            # its height -- and the end on its own cut line.
            gap = _dist_pt_quad(start, quad)
            if not (0.0 < gap <= 0.5 * l.size_mm
                    and _dist_to_polyline(end, l.anchor.run) < 1e-6):
                bad_leader.append(l.text)
            length = math.dist(start, end)
            t = (length - 1e-3) / length if length > 1e-3 else 0.0
            trimmed = (start[0] + (end[0] - start[0]) * t,
                       start[1] + (end[1] - start[1]) * t)
            if any(_seg_cross(start, trimmed, p, q) for p, q in cut_segs):
                leader_crosses.append(l.text)
            if any(_in_rings(c, rings) for c in quad):
                outside_on_piece.append(l.text)
    return {
        "labels": len(owned),
        "label_label_pairs": census["label_label_pairs"],
        "on_number": census["on_number"],
        "crossing_cut": census["crossing_cut"],
        "below_min": small,
        "neither_inside_nor_one_leader": homeless,
        "leader_not_off_box_to_own_run": bad_leader,
        "leader_crosses_cut": leader_crosses,
        "outside_label_on_a_piece": outside_on_piece,
        "glyph_faults": glyph_faults,
        "pieces_with_mixed_sizes": sorted(
            p for p, z in sizes_by_piece.items() if len(z) > 1),
        "leaders": sum(len(ld) for _l, ld in owned),
    }


def _audit_clean(a):
    return all(not v for v in a.values() if isinstance(v, list))


def _counts(a):
    return {k: (len(v) if isinstance(v, list) else v) for k, v in a.items()}


# --- synthetic shapes with known answers ----------------------------------

BASE = dict(show_brackets=True, show_seam_numbers=True,
            pattern_surface='OUTER', decimals=0, min_angle=1.0,
            show_heat_form=True, max_tilt_deg=45.0, show_darts=False,
            show_align_ticks=True, tick_mode='PER_RUN', ticks_per_run=2,
            tick_length=0.006, tick_numbers=False, show_island_ids=True,
            island_fill=False, angle_text_size=16, number_size=24,
            overlay_scale=1.0, show_scale_ruler=False,
            piece_label_style='NUMBER')


def _seamed_box(z_scale=1.0):
    """A 2 m cube (squashed to `z_scale` in z), every edge a seam."""
    sandbox_guard.clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.object
    obj.scale = (1.0, 1.0, z_scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj



def _seamed_box_keep(shift_u):
    """Another all-seamed 2 m cube added WITHOUT clearing the scene, its UVs
    moved `shift_u` along u, so two objects can share one sheet side by
    side."""
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(5.0, 0.0, 0.0))
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    for d in obj.data.uv_layers.active.data:
        d.uv = (d.uv.x + shift_u, d.uv.y)
    return obj

def _transform_island_uvs(obj, polygon, scale=1.0, turn_deg=0.0):
    """Scale and turn one face's UVs about their own centre."""
    me = obj.data
    uv = me.uv_layers.active.data
    idx = list(polygon.loop_indices)
    cu = sum(uv[i].uv.x for i in idx) / len(idx)
    cv = sum(uv[i].uv.y for i in idx) / len(idx)
    c, s = math.cos(math.radians(turn_deg)), math.sin(math.radians(turn_deg))
    for i in idx:
        u, v = uv[i].uv.x - cu, uv[i].uv.y - cv
        uv[i].uv = (cu + (u * c - v * s) * scale, cv + (u * s + v * c) * scale)


def _shrink_top_island(obj, factor):
    top = max(obj.data.polygons, key=lambda p: p.normal.z)
    _transform_island_uvs(obj, top, scale=factor)
    return top.index


def _marks(obj, **overrides):
    s = bpy.context.scene.foamcraft_drachis
    wanted = dict(BASE)
    wanted.update(overrides)
    saved = {k: getattr(s, k) for k in wanted if hasattr(s, k)}
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    try:
        for k, v in wanted.items():
            if hasattr(s, k):
                setattr(s, k, v)
        uv = bm.loops.layers.uv.active
        # layout=False: these tests measure the layout's INPUT and run
        # `fl.layout_labels` on it themselves. build_marks lays the labels
        # out by default since 2026-11-17 (every export prints that).
        return fm.build_marks(bm, uv, s, 1.0, obj.matrix_world,
                              layout=False)
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
        bm.free()


def _min_mm():
    return float(getattr(bpy.context.scene.foamcraft_drachis,
                         "min_label_mm", 2.5))


def _forms(labels):
    """How many labels take each of the four text forms."""
    out = {"glyph_first_right": 0, "glyph_first_left": 0,
           "glyph_last_right": 0, "glyph_last_left": 0, "none": 0}
    for l in labels:
        t = l.text
        if t.startswith(RIGHT):
            out["glyph_first_right"] += 1
        elif t.startswith(LEFT):
            out["glyph_first_left"] += 1
        elif t.endswith(RIGHT):
            out["glyph_last_right"] += 1
        elif t.endswith(LEFT):
            out["glyph_last_left"] += 1
        else:
            out["none"] += 1
    return out


def test_cube_labels_turn_with_their_edges():
    """All-seamed cube: every piece a square with axis-aligned edges.

    Known answers. Every label turns 0 or 90 degrees (left/right edges
    read across at 0; top/bottom read upward at 90 -- never -90 or 180,
    which would be upside down). The glyph points along the blade: into
    the piece on the outer skin (every fold is convex), out of it on the
    inner. And the four text forms split exactly by edge: on a square the
    left and bottom edges read *inward* (glyph first) and the right and top
    edges read *outward* (glyph last), so OUTER gives 12 "-> n, 45°" and 12
    "n, 45° <-", INNER 12 "<- n, 45°" and 12 "n, 45° ->". Every glyph end is
    within one text height of its run's midpoint; every box is wholly inside
    its piece; no drawn arrows remain.
    """
    obj = _seamed_box()
    out = {}
    for surface in ('OUTER', 'INNER'):
        marks = _marks(obj, pattern_surface=surface)
        angles = [l for l in marks.labels if l.kind == 'angle']
        rots = sorted({round(l.rotation_deg, 3) for l in angles})
        faults = [(l.text, f) for l in angles for f in _glyph_faults(l)]
        near_mid = []
        for l in angles:
            run = l.anchor.run
            mid = ((run[0][0] + run[-1][0]) / 2, (run[0][1] + run[-1][1]) / 2)
            at, _d = _glyph(l)
            near_mid.append(math.dist(at, mid) <= l.size_mm)
        out[surface] = {
            "labels": len(angles), "rotations": rots,
            "forms": _forms(angles), "faults": faults[:6],
            "all_near_mid": all(near_mid) and len(near_mid) == 24,
            "all_inside": all(_quad_in_rings(_quad(l), l.anchor.rings)
                              for l in angles),
            "arrows": sum(1 for s in marks.strokes
                          if s.kind == 'bevel_arrow'),
            "axis_aligned": all(min(abs(l.rotation_deg),
                                    abs(l.rotation_deg - 90)) < 1e-3
                                for l in angles),
        }
    o, i = out['OUTER'], out['INNER']
    return {
        **out,
        "checks": {
            "24_labels_each": o["labels"] == 24 and i["labels"] == 24,
            "rotations_0_or_90": o["axis_aligned"] and i["axis_aligned"],
            "no_glyph_faults": not o["faults"] and not i["faults"],
            "outer_forms": o["forms"] == {
                "glyph_first_right": 12, "glyph_first_left": 0,
                "glyph_last_right": 0, "glyph_last_left": 12, "none": 0},
            "inner_forms": i["forms"] == {
                "glyph_first_right": 0, "glyph_first_left": 12,
                "glyph_last_right": 12, "glyph_last_left": 0, "none": 0},
            "glyph_within_one_height_of_mid":
                o["all_near_mid"] and i["all_near_mid"],
            "boxes_inside_their_piece": o["all_inside"] and i["all_inside"],
            "no_drawn_arrows": o["arrows"] == 0 and i["arrows"] == 0,
        },
    }


def test_a_turned_square_turns_its_labels_45():
    """The cube with every square's UVs turned 45 degrees about its own
    centre: every edge is now diagonal on the sheet, so every label turns
    +45 or -45 -- exactly, to float noise -- still never upside down, and
    the glyph still points the blade's way."""
    obj = _seamed_box()
    for poly in obj.data.polygons:
        _transform_island_uvs(obj, poly, turn_deg=45.0)
    marks = _marks(obj)
    angles = [l for l in marks.labels if l.kind == 'angle']
    rots = sorted({round(l.rotation_deg, 3) for l in angles})
    faults = [(l.text, f) for l in angles for f in _glyph_faults(l)]
    return {
        "rotations": rots, "faults": faults[:6],
        "checks": {
            "24_labels": len(angles) == 24,
            "all_plus_or_minus_45": all(
                abs(abs(l.rotation_deg) - 45.0) < 1e-3 for l in angles),
            "both_signs_present": any(l.rotation_deg > 0 for l in angles)
            and any(l.rotation_deg < 0 for l in angles),
            "no_glyph_faults": faults == [],
        },
    }


def test_a_straight_cut_label_turns_with_no_glyph():
    """min_angle above every fold of the cube: 24 number-only labels,
    turned like the others (0 or 90), with no glyph -- no arrow means cut
    square."""
    obj = _seamed_box()
    marks = _marks(obj, min_angle=91.0)
    angles = [l for l in marks.labels if l.kind == 'angle']
    return {
        "texts": sorted({l.text for l in angles})[:5],
        "checks": {
            "24_labels": len(angles) == 24,
            "no_glyphs": _forms(angles)["none"] == 24,
            "all_digits": all(l.text.isdigit() for l in angles),
            "rotations_0_or_90": all(
                min(abs(l.rotation_deg), abs(l.rotation_deg - 90)) < 1e-3
                for l in angles),
        },
    }


def test_svg_turns_the_text_the_page_way():
    """SVG's rotate() is clockwise on screen (y down), so a label turned
    +90 on the page must be written rotate(-90 x y), about its own point;
    an unturned label has no transform at all."""
    import bl_ext.user_default.foamcraft_drachis.foam_export as fx
    up = fm.Label('angle', 50.0, 40.0, 5.0, RIGHT + " 1, 45\u00b0",
                  (0, 0, 0), 'middle', 'neutral', None, 90.0)
    flat = fm.Label('number', 20.0, 20.0, 5.0, "7", (0, 0, 0))
    svg = fx.svg_from_marks(fm.Marks(100.0, 100.0, (), (), (up, flat)))
    return {
        "checks": {
            "turned_label_rotates_minus_90_about_itself":
                'transform="rotate(-90.000 50.000 40.000)"' in svg,
            "glyph_written": RIGHT in svg,
            "only_one_transform": svg.count("transform=") == 1,
            "no_arrows_group_when_empty": 'id="arrows"' not in svg,
        },
    }


def _render_ink(label, path):
    """Render one label alone at 10 px/mm and return its ink as (count,
    centroid x, centroid y) in mm, sheet frame."""
    import numpy as np
    import bl_ext.user_default.foamcraft_drachis.foam_raster as fr
    marks = fm.Marks(60.0, 60.0, (), (), (label,))
    fr.render_marks(marks, path, dpi=254)
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    rows = px.reshape(h, w, 4)[::-1]          # top row first
    ink = rows[:, :, :3].mean(axis=2) < 0.5
    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        return 0, None, None
    return int(len(xs)), float(xs.mean()) / 10.0, float(ys.mean()) / 10.0


def test_the_arrow_glyphs_render_in_blenders_font():
    """Blender's default blf font draws U+2192 and U+2190, and the raster
    turns text the page's way.

    An arrow's head is its heavy end: "->" alone at (30, 30) must put its
    ink centroid right of 30, "<-" left of it -- both bounds, by more than
    half a millimetre on a 20 mm glyph, and with ink at all (a missing
    glyph draws nothing or a box, which is symmetric). Turned +90 on the
    page, "->" points up the page: centroid above 30 in sheet y (smaller y).
    """
    tmp = tempfile.mkdtemp(prefix="foam_glyph_")
    try:
        r = _render_ink(fm.Label('angle', 30.0, 30.0, 20.0, RIGHT,
                                 (0, 0, 0)), os.path.join(tmp, "r.png"))
        l = _render_ink(fm.Label('angle', 30.0, 30.0, 20.0, LEFT,
                                 (0, 0, 0)), os.path.join(tmp, "l.png"))
        u = _render_ink(fm.Label('angle', 30.0, 30.0, 20.0, RIGHT,
                                 (0, 0, 0), 'middle', '', None, 90.0),
                        os.path.join(tmp, "u.png"))
    except Exception as exc:
        return {"skipped": "raster unavailable: %s" % exc, "checks": {}}
    return {
        "right": r, "left": l, "turned_up": u,
        "checks": {
            "right_has_ink": r[0] > 50,
            "left_has_ink": l[0] > 50,
            "right_is_right_heavy": r[1] is not None and r[1] > 30.5,
            "left_is_left_heavy": l[1] is not None and l[1] < 29.5,
            "turned_points_up_the_page": u[2] is not None and u[2] < 29.5
            and abs(u[1] - 30.0) < 3.0,
        },
    }


# --- layout on known shapes ----------------------------------------------

def test_ample_room_changes_nothing():
    """All-seamed cube: four labels per square piece at the edge middles,
    nowhere near each other. The census before is zero (asserted -- if it
    were not, "nothing moved" would be a wrong answer), and after layout
    every label is within 1e-6 mm of where build_marks put it, at the same
    size, turn and text, and no leader is made."""
    obj = _seamed_box()
    before = _marks(obj)
    after = fl.layout_labels(before, _min_mm())
    b_lab = [l for l in before.labels if l.kind == 'angle']
    a_lab = [l for l in after.labels if l.kind == 'angle']
    moved = [b.text for b, a in zip(b_lab, a_lab)
             if math.hypot(a.x - b.x, a.y - b.y) > 1e-6
             or abs(a.size_mm - b.size_mm) > 1e-6
             or a.rotation_deg != b.rotation_deg or a.text != b.text]
    census = _census(before)
    return {
        "labels": len(b_lab), "moved": moved,
        "census_before": _counts(census),
        "checks": {
            "twenty_four_labels": len(b_lab) == 24 and len(a_lab) == 24,
            "census_before_is_zero": not census["label_label_pairs"]
            and not census["on_number"] and not census["crossing_cut"],
            "nothing_moved_or_changed": moved == [],
            "no_leaders": not any(s.kind == 'leader' for s in after.strokes),
            "same_stroke_count": len(after.strokes) == len(before.strokes),
        },
    }


def _strip_long_labels(marks):
    """Labels on the long (2 m) runs of the four 2 m x 0.2 m side strips."""
    out = []
    for l in marks.labels:
        if l.kind != 'angle' or not l.anchor:
            continue
        xs = [p[0] for r in l.anchor.rings for p in r]
        ys = [p[1] for r in l.anchor.rings for p in r]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        run_len = sum(math.dist(a, b) for a, b in
                      zip(l.anchor.run, l.anchor.run[1:]))
        if max(w, h) > 5 * min(w, h) and run_len > 0.5 * max(w, h):
            out.append(l)
    return out


def test_labels_facing_across_a_strip_are_resolved():
    """A 2 x 2 x 0.2 m box, all edges seamed. Each side strip is 2000 by
    200 mm; the labels of its two long edges read across the strip, each
    about 470 mm long at full size (8 glyphs x 0.6 x 98 mm), so from each
    side they run right across it and meet.

    Before: at least one such pair overlaps -- the precondition, asserted.
    After: the whole sheet audits clean (0 overlaps, nothing on a number or
    a cut line, nothing under `min_label_mm`, every label inside or on one
    leader, glyphs right, one size per piece).
    """
    obj = _seamed_box(0.1)
    before = _marks(obj)
    report = {}
    after = fl.layout_labels(before, _min_mm(), report)
    strips = _strip_long_labels(before)
    quads = [_quad(l) for l in strips]
    pairs_before = sum(_quads_overlap(quads[i], quads[j])
                       for i in range(len(quads))
                       for j in range(i + 1, len(quads)))
    audit = _audit(after, _min_mm())
    hows = report.get("how", [])
    return {
        "strip_labels": len(strips), "pairs_before": pairs_before,
        "census_before": _counts(_census(before)),
        "audit_after": _counts(audit),
        "how": {h: hows.count(h) for h in sorted(set(hows))},
        "piece_size": {k: round(v, 2)
                       for k, v in report.get("piece_size", {}).items()},
        "checks": {
            "eight_strip_labels": len(strips) == 8,
            "they_collide_before": pairs_before >= 1,
            "whole_sheet_clean": _audit_clean(audit),
            "same_label_count": audit["labels"] == 24,
        },
    }


def test_a_piece_too_small_puts_its_labels_on_leaders():
    """The all-seamed cube with its top square shrunk to 0.5% in UV: ~10 mm
    square on the sheet, narrower than one label even at the 2.5 mm floor
    ("-> 11, 45°" is 9 x 2.5 x 0.6 = 13.5 mm long). (5% was tried first: a
    100 mm piece, and 2.5 mm labels fit inside it -- the premise was wrong,
    not the layout.) Its 4 labels go outside on exactly one leader each,
    and no other label needs one: exactly 4 leaders. Each leader starts a
    small gap off its label and ends on that label's own run; each outside
    label is on no piece and crosses no cut line; the tiny piece's 4
    labels share one size.

    Piece numbers are OFF for the leader case and on for a second run that
    pins what happens with them. First measured 2026-11-17: a piece number
    then never shrank below a quarter of its full size, so on a 10 mm piece
    it was ~40 mm tall, covered the piece's whole outline, and every leader
    would have crossed it -- all 4 labels came back 'unplaced'. Since the
    same day a name is fitted inside its own piece, down to Smallest Label
    (`foam_layout.fit_piece_names`), so with numbers on the same 4 labels
    go out on leaders exactly as with them off, and nothing is unplaced.
    """
    obj = _seamed_box()
    _shrink_top_island(obj, 0.005)
    with_numbers = {}
    fl.layout_labels(_marks(obj), _min_mm(), with_numbers)
    before = _marks(obj, show_island_ids=False)
    report = {}
    after = fl.layout_labels(before, _min_mm(), report)
    audit = _audit(after, _min_mm())

    def ring_area(rings):
        r = rings[0]
        return abs(sum(a[0] * b[1] - b[0] * a[1]
                       for a, b in zip(r, r[1:] + r[:1]))) / 2
    owned = _owned(after)
    tiny = min({l.anchor.piece: ring_area(l.anchor.rings)
                for l, _ld in owned}.items(), key=lambda kv: kv[1])[0]
    tiny_labels = [(l, ld) for l, ld in owned if l.anchor.piece == tiny]
    return {
        "tiny_piece": tiny,
        "how_with_numbers_on": {h: with_numbers["how"].count(h)
                                for h in sorted(set(with_numbers["how"]))},
        "how": [h for h, p in zip(report["how"], report["piece"])
                if p == tiny],
        "tiny_labels": [(l.text, round(l.size_mm, 2), len(ld))
                        for l, ld in tiny_labels],
        "audit_after": _counts(audit),
        "faults": {k: v for k, v in audit.items()
                   if isinstance(v, list) and v},
        "checks": {
            "four_labels_on_the_tiny_piece": len(tiny_labels) == 4,
            "each_has_exactly_one_leader":
                all(len(ld) == 1 for _l, ld in tiny_labels),
            "exactly_four_leaders_on_the_sheet": audit["leaders"] == 4,
            "each_outside_its_piece": len(tiny_labels) == 4 and not any(
                _quad_in_rings(_quad(l), l.anchor.rings)
                for l, _ld in tiny_labels),
            "whole_sheet_clean": _audit_clean(audit),
            "numbers_on_none_unplaced":
                with_numbers["how"].count("unplaced") == 0,
            "numbers_on_the_same_four_leaders":
                with_numbers["how"].count("leader") == 4,
        },
    }


def test_layout_is_reproducible():
    """Same mesh read into two fresh BMeshes -- fresh element structs, so
    any set-ordered step would move (CLAUDE.md) -- gives identical laid-out
    marks, exactly. On the strip box and the tiny-top cube, where the
    layout actually moves labels."""
    out = {}
    for name, build in (("strip_box", lambda: _seamed_box(0.1)),
                        ("tiny_top", lambda: (_seamed_box(),
                                              _shrink_top_island(
                                                  bpy.context.object, 0.005),
                                              bpy.context.object)[2])):
        obj = build()
        base = _marks(obj, show_island_ids=False)
        a = fl.layout_labels(_marks(obj, show_island_ids=False), _min_mm())
        b = fl.layout_labels(_marks(obj, show_island_ids=False), _min_mm())
        moved = sum(1 for l, m in zip(base.labels, a.labels)
                    if (l.x, l.y, l.size_mm) != (m.x, m.y, m.size_mm))
        out[name] = {"identical": a == b, "labels_moved": moved}
    return {
        **out,
        "checks": {
            "strip_box_identical": out["strip_box"]["identical"],
            "strip_box_layout_did_something":
                out["strip_box"]["labels_moved"] > 0,
            "tiny_top_identical": out["tiny_top"]["identical"],
            "tiny_top_layout_did_something":
                out["tiny_top"]["labels_moved"] > 0,
        },
    }


def test_two_objects_keep_their_pieces_apart():
    """Two all-seamed cubes on one sheet (`build_marks_multi`), the second's
    UVs shifted 1.2 to the right so the two sets of six squares sit side by
    side. Each object has an island 0..5; the layout keys pieces on the
    sheet-wide ordinal (`BevelAnchor.piece`), so there must be exactly 12
    distinct pieces with exactly 4 labels each -- 6 and 8 would mean piece
    k of one cube was taken for piece k of the other -- every label's box
    inside the outline its anchor names, and the laid-out sheet clean.
    """
    a = _seamed_box()
    a.name = "CubeA"
    b = _seamed_box_keep(1.2)
    b.name = "CubeB"
    s = bpy.context.scene.foamcraft_drachis
    wanted = dict(BASE)
    saved = {k: getattr(s, k) for k in wanted if hasattr(s, k)}
    bms = []
    try:
        for k, v in wanted.items():
            if hasattr(s, k):
                setattr(s, k, v)
        items = []
        for obj in (a, b):
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bms.append(bm)
            items.append(fm.SheetItem(bm, bm.loops.layers.uv.active,
                                      obj.matrix_world, "C", obj.name))
        # The raw sheet: laid out below, by hand (see `_marks`).
        before = fm.build_marks_multi(items, s, 1.0, layout=False)
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
        for bm in bms:
            bm.free()
    report = {}
    after = fl.layout_labels(before, _min_mm(), report)
    angles = [l for l in after.labels if l.kind == 'angle' and l.anchor]
    per_piece = {}
    for l in angles:
        per_piece[l.anchor.piece] = per_piece.get(l.anchor.piece, 0) + 1
    audit = _audit(after, _min_mm())
    return {
        "pieces": len(per_piece), "labels_per_piece": sorted(per_piece.values()),
        "audit_after": _counts(audit),
        "checks": {
            "forty_eight_labels": len(angles) == 48,
            "twelve_distinct_pieces": len(per_piece) == 12,
            "four_labels_each": sorted(per_piece.values()) == [4] * 12,
            "every_label_in_its_named_piece": all(
                _quad_in_rings(_quad(l), l.anchor.rings) for l in angles),
            "twelve_piece_sizes_reported":
                len(report.get("piece_size", {})) == 12,
            "whole_sheet_clean": _audit_clean(audit),
        },
    }


# --- the maker's file ------------------------------------------------------

# The snapshot is modelled 100x life size: 108 Blender units tall in a scene
# with unit scale 1.0, for a head about 1.08 m tall (measured 2026-11-17).
# A sheet length -- `min_label_mm`, `tick_length` -- only means anything at
# the size of the thing being built (CLAUDE.md, "Model at real-world
# scale"), so it is exported as the maker would print it: at 0.01.
SNAPSHOT_SCALE_LENGTH = 0.01


def _load_snapshot():
    """Append `head_simplified` and the scene it lives in from a temp copy.

    Copied first, so nothing -- not even a stray save -- can reach the
    git-ignored snapshot; the maker's original under O:\\Projects is never
    referenced at all. Appended rather than opened: opening a file from
    inside a bridge request replaces the session that is running it.
    Returns (object, settings) or (None, reason). The settings are the
    appended scene's own -- the maker's. Piece labels are left at the
    default style, names like "HE M" (2026-11-17, the multi-object work):
    longer than the old numbers, and an obstacle the layout must respect.
    """
    if not os.path.exists(SNAPSHOT):
        return None, "fixture %s is absent (git-ignored, local only)" % SNAPSHOT
    sandbox_guard.clear_scene()
    for sc in [sc for sc in bpy.data.scenes if sc != bpy.context.scene]:
        bpy.data.scenes.remove(sc)
    tmp = os.path.join(tempfile.mkdtemp(prefix="foam_layout_"),
                       "snapshot_copy.blend")
    shutil.copyfile(SNAPSHOT, tmp)
    with bpy.data.libraries.load(tmp, link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n == SNAPSHOT_OBJECT]
        dst.scenes = list(src.scenes[:1])
    obj = next((o for o in dst.objects if o is not None), None)
    if obj is None:
        return None, "object %r not in the snapshot" % SNAPSHOT_OBJECT
    bpy.context.scene.collection.objects.link(obj)
    scene = next((s for s in dst.scenes if s is not None), None)
    settings = getattr(scene, "foamcraft_drachis", None) if scene else None
    if settings is None:
        settings = bpy.context.scene.foamcraft_drachis
    return obj, settings


def _snapshot_marks(obj, settings):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    try:
        uv = bm.loops.layers.uv.active
        # The raw sheet, the "before" of the before/after (see `_marks`).
        return fm.build_marks(bm, uv, settings, SNAPSHOT_SCALE_LENGTH,
                              obj.matrix_world, layout=False)
    finally:
        bm.free()


def _piece_names(marks):
    """island index -> printed piece number, by which piece holds it."""
    out = {}
    numbers = [l for l in marks.labels if l.kind == 'number']
    seen = {}
    for l in marks.labels:
        if l.kind == 'angle' and l.anchor and l.anchor.piece not in seen:
            seen[l.anchor.piece] = l.anchor.rings
    for piece, rings in seen.items():
        holder = [n.text for n in numbers if _in_rings((n.x, n.y), rings)]
        out[piece] = holder[0] if len(holder) == 1 else "?%d" % piece
    return out


def _per_piece(before, after, report, names):
    rows = {}
    b_owned = [l for l in before.labels if l.kind == 'angle' and l.anchor]
    a_owned = [l for l in after.labels if l.kind == 'angle' and l.anchor]
    b_q = [_quad(l) for l in b_owned]
    a_q = [_quad(l) for l in a_owned]
    sizes = report.get("piece_size", {})
    for i, l in enumerate(b_owned):
        name = names.get(l.anchor.piece, "?")
        row = rows.setdefault(name, {
            "labels": 0, "pairs_before": 0, "pairs_after": 0,
            "size_mm": round(sizes.get(l.anchor.piece, 0.0), 2),
            "leaders": 0, "slid": 0, "kept": 0})
        row["labels"] += 1
        row["pairs_before"] += sum(_quads_overlap(b_q[i], b_q[j])
                                   for j in range(i + 1, len(b_q)))
        row["pairs_after"] += sum(_quads_overlap(a_q[i], a_q[j])
                                  for j in range(i + 1, len(a_q)))
        how = report["how"][i]
        if how == "leader":
            row["leaders"] += 1
        elif how in ("slid", "kept"):
            row[how] += 1
    return dict(sorted(rows.items(), key=lambda kv: (len(kv[0]), kv[0])))


def test_snapshot_head():
    """The maker's head, 26 pieces: before and after, measured.

    Before is the size of the problem, reported, not asserted -- except that
    it is not zero, or the after-count proves nothing. After, each of these
    is asserted zero: label/label overlaps, labels on a piece number, labels
    crossing any cut line, labels under `min_label_mm`, labels neither
    inside their own piece nor on exactly one leader, a leader not running
    from just off its label to its own run, a leader or outside label on a
    piece, a glyph not pointing the blade's way or not at the line end or
    text upside down, a piece with labels at two sizes, an unplaced label.

    Then the sheet is rendered at 150 dpi through foam_raster and the
    maker's six example pieces -- 13 and 16 first -- are cropped out.
    """
    obj, settings = _load_snapshot()
    if obj is None:
        return {"skipped": settings, "checks": {}}
    min_mm = float(getattr(settings, "min_label_mm", 2.5))
    before = _snapshot_marks(obj, settings)
    report = {}
    after = fl.layout_labels(before, min_mm, report)
    names = _piece_names(before)
    census = _census(before)
    audit_before = _audit(before, min_mm)
    audit = _audit(after, min_mm)
    hows = report.get("how", [])
    # The maker's six examples, by piece number; printed as names now, so
    # number n is the name ending in its letters (fm.piece_letters).
    wanted = []
    for n in (13, 16, 10, 17, 15, 4):
        letters = fm.piece_letters(n)
        hit = [t for t in names.values()
               if t in (str(n), letters) or t.endswith(" " + letters)]
        wanted.append(hit[0] if hit else str(n))
    shots = _render_and_crop(after, names, wanted)
    return {
        "pieces": sum(1 for l in before.labels if l.kind == 'number'),
        "scale_length": SNAPSHOT_SCALE_LENGTH,
        "sheet_mm": [round(before.width_mm, 1), round(before.height_mm, 1)],
        "angle_font_mm": sorted({round(l.size_mm, 3) for l in before.labels
                                 if l.kind == 'angle'}),
        "min_label_mm": min_mm,
        "before": _counts(audit_before),
        "after": _counts(audit),
        "faults_after": {k: v[:6] for k, v in audit.items()
                         if isinstance(v, list) and v},
        "how": {h: hows.count(h) for h in sorted(set(hows))},
        "per_piece": _per_piece(before, after, report, names),
        "shots": shots,
        "checks": {
            "names_found_for_the_six": all(
                not str(v).startswith("piece not") for v in shots.values()),
            "twenty_six_pieces": sum(1 for l in before.labels
                                     if l.kind == 'number') == 26,
            "there_was_a_problem": len(census["label_label_pairs"]) > 0,
            "same_label_count": audit["labels"] == census["angle_labels"]
            and audit["labels"] > 0,
            "no_label_label_overlap": audit["label_label_pairs"] == [],
            "no_label_on_a_number": audit["on_number"] == [],
            "no_label_crossing_a_cut": audit["crossing_cut"] == [],
            "none_below_min": audit["below_min"] == [],
            "each_inside_or_on_one_leader":
                audit["neither_inside_nor_one_leader"] == [],
            "leaders_off_box_to_own_run":
                audit["leader_not_off_box_to_own_run"] == [],
            "leaders_cross_no_cut": audit["leader_crosses_cut"] == [],
            "outside_labels_on_no_piece":
                audit["outside_label_on_a_piece"] == [],
            "glyphs_right": audit["glyph_faults"] == [],
            "one_size_per_piece": audit["pieces_with_mixed_sizes"] == [],
            "nothing_unplaced": "unplaced" not in hows,
        },
    }


def _render_and_crop(marks, names, wanted, dpi=150):
    """Render the sheet, and crop each wanted piece (by printed number)
    with a margin, into screenshots/ via tools/shot.py. Returns paths, or a
    reason when the raster cannot be drawn."""
    try:
        import bl_ext.user_default.foamcraft_drachis.foam_raster as fr
        from shot import next_path
        import numpy as np
    except Exception as exc:
        return {"skipped": "cannot import raster/shot: %s" % exc}
    path = next_path("label_layout_head_150dpi")
    try:
        info = fr.render_marks(marks, path, dpi=dpi)
    except Exception as exc:
        return {"skipped": "render failed: %s" % exc}
    out = {"sheet": path, "pixels": info.get("pixels"),
           "dpi": info.get("dpi")}
    px_per_mm = info["pixels"][0] / marks.width_mm
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    pix = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(pix)
    pix = pix.reshape(h, w, 4)            # bottom row first
    rings_of = {}
    for l in marks.labels:
        if l.kind == 'angle' and l.anchor:
            rings_of[names.get(l.anchor.piece)] = l.anchor.rings
    for name in wanted:
        rings = rings_of.get(name)
        if not rings:
            out[name] = "piece not found"
            continue
        xs = [p[0] for p in rings[0]]
        ys = [p[1] for p in rings[0]]
        for l in marks.labels:
            if l.kind == 'angle' and l.anchor and l.anchor.rings == rings:
                for c in _quad(l):
                    xs.append(c[0])
                    ys.append(c[1])
        x0 = max(int((min(xs) - 25) * px_per_mm), 0)
        x1 = min(int((max(xs) + 25) * px_per_mm), w)
        y0 = max(int((min(ys) - 25) * px_per_mm), 0)
        y1 = min(int((max(ys) + 25) * px_per_mm), h)
        crop = pix[h - y1:h - y0, x0:x1, :]
        new = bpy.data.images.new("crop_%s" % name, x1 - x0, y1 - y0,
                                  alpha=False)
        new.pixels.foreach_set(crop.reshape(-1))
        cpath = next_path("label_layout_piece_%s" % name)
        new.filepath_raw = cpath
        new.file_format = 'PNG'
        new.save()
        out[name] = cpath
        bpy.data.images.remove(new)
    bpy.data.images.remove(img)
    return out


def test_piece_names_fit_by_arithmetic():
    """`foam_layout.fit_piece_names` on shapes with a known answer.

    A name "AB" is 2 * 0.6 = 1.2 em wide by 1 em tall, padded 0.15 em all
    round: 1.5 z by 1.3 z at size z. Centred in a 100 mm square with nothing
    else on it, it fits while 1.5 z <= 100, so z = 66.67 mm exactly (the
    width binds first). So:
      * set 50 mm  -> 'full', 50;
      * set 80 mm  -> 'shrunk', 66.67 (to the bisection's resolution);
      * a tick from the left edge 30 mm in along the centre line leaves
        50 - 30 = 20 mm to the half-width 0.75 z -> z = 26.67, 'shrunk';
      * a 2 mm square with a 2.5 mm floor -> 'floor', at 2.5;
      * a floor above the set size is the set size (set 2, floor 2.5,
        square 100 -> 'full', 2).
    The floor case is the one the maker's heads never reach (0 of 52,
    tests/test_overlay_ticks.py), so it is pinned here.
    """
    sq = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    tiny = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
    outline = fm.Stroke('outline', sq, True, 0.2, fm.BLACK)
    tick = fm.Stroke('align', ((0.0, 50.0), (30.0, 50.0)), False, 0.2,
                     fm.BLACK)
    tiny_outline = fm.Stroke('outline', tiny, True, 0.2, fm.BLACK)
    f = fl.fit_piece_names
    full = f([("AB", 50.0, 50.0, (sq,))], [outline], [], 50.0, 2.5)[0]
    shrunk = f([("AB", 50.0, 50.0, (sq,))], [outline], [], 80.0, 2.5)[0]
    ticked = f([("AB", 50.0, 50.0, (sq,))], [outline, tick], [], 80.0,
               2.5)[0]
    floor = f([("AB", 1.0, 1.0, (tiny,))], [tiny_outline], [], 50.0,
              2.5)[0]
    capped = f([("AB", 50.0, 50.0, (sq,))], [outline], [], 2.0, 2.5)[0]
    tol = 1e-3
    return {
        "full": full, "shrunk": shrunk, "ticked": ticked, "floor": floor,
        "capped": capped,
        "checks": {
            "fits_whole": full == (50.0, 'full'),
            "shrinks_to_the_width": shrunk[1] == 'shrunk'
            and abs(shrunk[0] - 100.0 / 1.5) < tol,
            "a_tick_is_an_obstacle": ticked[1] == 'shrunk'
            and abs(ticked[0] - 20.0 / 0.75) < tol,
            "too_small_keeps_the_floor": floor == (2.5, 'floor'),
            "floor_never_above_set_size": capped == (2.0, 'full'),
        },
    }


TESTS = (
    test_piece_names_fit_by_arithmetic,
    test_cube_labels_turn_with_their_edges,
    test_a_turned_square_turns_its_labels_45,
    test_a_straight_cut_label_turns_with_no_glyph,
    test_svg_turns_the_text_the_page_way,
    test_the_arrow_glyphs_render_in_blenders_font,
    test_ample_room_changes_nothing,
    test_labels_facing_across_a_strip_are_resolved,
    test_a_piece_too_small_puts_its_labels_on_leaders,
    test_layout_is_reproducible,
    test_two_objects_keep_their_pieces_apart,
    test_snapshot_head,
)

result = {}
_all_pass = True
for _test in TESTS:
    try:
        if fl is None:
            raise ImportError("foamcraft_drachis.foam_layout is missing")
        _report = _test()
        _report["PASS"] = all(_report.get("checks", {}).values())
    except Exception:
        _report = {"PASS": False, "error": traceback.format_exc()}
    result[_test.__name__] = _report
    _all_pass = _all_pass and _report["PASS"]
result["ALL_PASS"] = _all_pass
