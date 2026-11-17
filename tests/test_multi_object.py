"""Several objects on one sheet, and pieces named "HR A".

    python tools/launch_blender.py --port 9920
    python tools/blender_bridge.py --port 9920 < tests/test_multi_object.py

The maker: "opening two objects in a shared UV mode only exports the active
object". The answer they chose is Export UV Layout's: every object it would
take -- `{*context.selected_objects, context.active_object}`, meshes with an
active UV map (io_mesh_uv_layout/__init__.py:198-206) -- drawn overlaid into
one UV square. And: "numbering is not continuing across objects, island 8 on
each object is labeled 8 on a combined uv sheet" -- so seam numbers and piece
letters continue from one object to the next, objects in name order.

Known answers come from the cube. A cube with every edge seamed is 6 pieces,
12 seams and 24 cut runs, all +90 degree folds (CLAUDE.md, "Good test
shapes"). Two of them are 12 pieces, 24 seams -- numbered 1..24, each on
exactly two labels, both on pieces of the same object -- and 48 runs.

Piece letters are bijective base 26, the spreadsheet-column scheme, which is
pure arithmetic and checked as such.
"""

import os
import re
import sys
import shutil
import tempfile
import traceback
from collections import Counter

import bpy
import bmesh
import numpy as np
from mathutils import Matrix

sys.path.insert(0, r"O:\Work\BlenderPlugin\FoamCraft\tools")
import sandbox_guard  # noqa: E402
import shot           # noqa: E402

_MOD = "bl_ext.user_default.foamcraft_drachis"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foamcraft_drachis.foam_marks as fm    # noqa: E402
import bl_ext.user_default.foamcraft_drachis.foam_raster as fr   # noqa: E402

ROOT = r"O:\Work\BlenderPlugin\FoamCraft"
SNAPSHOT = os.path.join(ROOT, "tests", "fixtures", "beast_skull_snapshot.blend")
_TMP = tempfile.mkdtemp(prefix="foamcraft_multi_")


# --- fixtures ---------------------------------------------------------------

def _settings():
    s = bpy.context.scene.foamcraft_drachis
    s.show_brackets = True
    s.show_seam_numbers = True
    s.show_island_ids = True
    s.island_fill = True
    s.piece_label_style = 'NAME'
    return s


def _seamed_cube(name):
    """A 0.2 m cube, every edge a seam, unwrapped: 6 pieces."""
    bpy.ops.mesh.primitive_cube_add(size=0.2)
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def _map_uvs(obj, fn):
    for loop_uv in obj.data.uv_layers.active.data:
        u, v = fn(loop_uv.uv.x, loop_uv.uv.y)
        loop_uv.uv = (u, v)


def _copy_uvs(src, dst):
    for a, b in zip(src.data.uv_layers.active.data,
                    dst.data.uv_layers.active.data):
        b.uv = a.uv


def _two_cubes(layout='SIDE', b_matrix=None):
    """Cubes "CubeA" (prefix CA) and "CubeB" (prefix CB).

    SIDE: A's UVs squeezed into the left half, B's into the right, so no
    piece overlaps another. SAME: B's UVs a copy of A's, full square, so
    every piece lies exactly on its twin. HALF: as SAME, then B's UVs
    scaled by 0.5 about the centre -- a different UV density.
    """
    sandbox_guard.clear_scene()
    a = _seamed_cube("CubeA")
    b = _seamed_cube("CubeB")
    b.location.x = 0.5
    _copy_uvs(a, b)
    if layout == 'SIDE':
        _map_uvs(a, lambda u, v: (u * 0.5, v))
        _map_uvs(b, lambda u, v: (0.5 + u * 0.5, v))
    elif layout == 'HALF':
        _map_uvs(b, lambda u, v: (0.5 + (u - 0.5) * 0.5,
                                  0.5 + (v - 0.5) * 0.5))
    if b_matrix is not None:
        b.matrix_world = b_matrix
    a.foamcraft_drachis_prefix = "CA"
    b.foamcraft_drachis_prefix = "CB"
    bpy.context.view_layer.update()
    return a, b


class _Items:
    """SheetItems for objects, as the operators build them; frees on exit."""

    def __init__(self, objects):
        self.objects = objects
        self.bms = []

    def __enter__(self):
        items = []
        for obj in self.objects:
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            self.bms.append(bm)
            items.append(fm.SheetItem(
                bm, bm.loops.layers.uv.active, obj.matrix_world.copy(),
                fm.object_prefix(obj.name, obj.foamcraft_drachis_prefix),
                obj.name))
        return items

    def __exit__(self, *exc):
        for bm in self.bms:
            bm.free()
        return False


def _seam_of(text):
    """The seam number of a bevel label: "7, 45°" -> 7.

    The first integer anywhere, not at the start: the label now carries an
    arrow glyph that may lead ("→ 7, 45°"), and a fixed prefix pattern
    read every label as None once it did."""
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else None


def _bevel_census(objects, settings):
    """[(object name, island index, seam number)] for every bevel label,
    asked of the model's own `_bevel_marks` per object on the shared sheet,
    so each label's piece and object are known without geometry."""
    out = []
    with _Items(objects) as items:
        sheet, parts = fm._layout_sheet(items, settings, 1.0)
        font = sheet.permille(settings.angle_text_size, settings.overlay_scale)
        stroke = sheet.permille(2, settings.overlay_scale)
        for part in parts:
            for mark in fm._bevel_marks(part.islands, part.item.uv_layer,
                                        part.runs_by_island, settings,
                                        font, stroke):
                if mark.label is not None:
                    out.append((part.item.name, mark.island_index,
                                _seam_of(mark.label.text)))
        names = [(p.item.name, [isl.label for isl in p.islands])
                 for p in parts]
    return out, names


def _outline_key(stroke):
    return frozenset((round(x, 4), round(y, 4)) for x, y in stroke.points)


# --- pure arithmetic --------------------------------------------------------

def test_piece_letters_are_bijective_base_26():
    """1->A ... 26->Z, 27->AA, 52->AZ, 53->BA, 702->ZZ, 703->AAA.

    702 = 26 + 26*26: every one- and two-letter name, the maker's A..ZZ.
    703 is the first three-letter one; the scheme carries on there.
    """
    want = {1: "A", 2: "B", 26: "Z", 27: "AA", 28: "AB", 52: "AZ",
            53: "BA", 702: "ZZ", 703: "AAA", 18278: "ZZZ"}
    got = {n: fm.piece_letters(n) for n in want}
    names = [fm.piece_letters(n) for n in range(1, 703)]
    return {
        "got": got,
        "checks": {
            "known_values": got == want,
            # Both bounds on 1..702: all distinct, and every one is one or
            # two letters (nothing reaches three before 703).
            "first_702_distinct": len(set(names)) == 702,
            "first_702_are_1_or_2_letters":
                all(1 <= len(n) <= 2 for n in names),
            "name_style": fm.piece_name("HR", 2) == "HR B",
            "number_style": fm.piece_name("HR", 2, 'NUMBER') == "2",
            "default_prefix_two_letters":
                fm.default_prefix("head_simplified") == "HE",
            "typed_prefix_wins_and_is_upper":
                fm.object_prefix("head_simplified", "st") == "ST",
            "prefix_capped_at_three":
                fm.object_prefix("x", "abcd") == "ABC",
        },
    }


def test_prefix_property_holds_three_upper_case_characters():
    """The Object property: typed lower case reads back upper case, and a
    fourth character is not kept. The property declared maxlen=4 at first,
    on the reading that RNA counts the terminating null; this case measured
    "ABCD" kept, so maxlen counts characters and is 3."""
    sandbox_guard.clear_scene()
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.object
    obj.foamcraft_drachis_prefix = "hr"
    lower = obj.foamcraft_drachis_prefix
    obj.foamcraft_drachis_prefix = "ABCD"
    long_ = obj.foamcraft_drachis_prefix
    obj.foamcraft_drachis_prefix = ""
    empty = obj.foamcraft_drachis_prefix
    return {
        "lower": lower, "long": long_,
        "checks": {
            "upper_cased": lower == "HR",
            "three_kept_not_four": long_ == "ABC",
            "empty_allowed": empty == "",
        },
    }


# --- one sheet, two objects ---------------------------------------------------

def test_two_cubes_side_by_side_make_one_sheet():
    """12 pieces named CA A..CA F and then CB G..CB L; seams 1..24.

    Letters continue across objects (the maker: "the numbering / labels
    should continue so island edges are unique pairs"), so CubeB's first
    piece is G, not A. Seams: every number 1..24 appears on exactly two
    labels, on two different pieces, both of the same object -- a seam is
    an edge of one mesh.
    """
    a, b = _two_cubes('SIDE')
    s = _settings()
    with _Items([a, b]) as items:
        marks = fm.build_marks_multi(items, s, 1.0)
    outlines = [st for st in marks.strokes if st.kind == 'outline']
    names = sorted(l.text for l in marks.labels if l.kind == 'number')
    census, part_names = _bevel_census([a, b], s)
    by_seam = {}
    for obj_name, island, seam in census:
        by_seam.setdefault(seam, []).append((obj_name, island))
    want_names = sorted(["CA " + c for c in "ABCDEF"]
                        + ["CB " + c for c in "GHIJKL"])
    return {
        "outlines": len(outlines), "names": names,
        "seams": sorted(k for k in by_seam if k is not None),
        "warnings": list(marks.warnings),
        "checks": {
            "twelve_outlines": len(outlines) == 12,
            "twelve_distinct_names": names == want_names,
            "seams_exactly_1_to_24": set(by_seam) == set(range(1, 25)),
            "each_seam_on_two_labels":
                all(len(v) == 2 for v in by_seam.values()),
            "each_seam_on_two_pieces":
                all(len(set(v)) == 2 for v in by_seam.values()),
            "each_seam_within_one_object":
                all(len({o for o, _i in v}) == 1 for v in by_seam.values()),
            # The model's own labels carry the same numbers.
            "sheet_labels_carry_them": sorted(
                _seam_of(l.text) for l in marks.labels
                if l.kind in ('angle', 'heat'))
                == sorted(seam for _o, _i, seam in census),
            "no_warnings": marks.warnings == (),
        },
    }


def test_numbering_does_not_depend_on_order():
    """Same names and seams whichever object is listed (or active) first:
    objects are taken in name order, not in the order handed over."""
    a, b = _two_cubes('SIDE')
    s = _settings()
    runs = []
    for order in ([a, b], [b, a]):
        with _Items(order) as items:
            marks = fm.build_marks_multi(items, s, 1.0)
        runs.append(sorted((l.text, round(l.x, 4), round(l.y, 4))
                           for l in marks.labels
                           if l.kind in ('number', 'angle', 'heat')))
    return {
        "labels": len(runs[0]),
        "checks": {
            "same_either_way": runs[0] == runs[1],
            # 12 names + 48 bevel labels; not an empty list agreeing.
            "sixty_labels": len(runs[0]) == 60,
        },
    }


def test_identical_uvs_are_drawn_overlaid():
    """Both cubes over the whole UV square: 12 outlines in 6 coinciding
    pairs, one of each object -- drawn on top of each other, as Export UV
    Layout draws them, never re-arranged."""
    a, b = _two_cubes('SAME')
    s = _settings()
    with _Items([a, b]) as items:
        marks = fm.build_marks_multi(items, s, 1.0)
    outlines = [st for st in marks.strokes if st.kind == 'outline']
    counts = Counter(_outline_key(st) for st in outlines)
    return {
        "outlines": len(outlines), "distinct": len(counts),
        "checks": {
            "twelve_outlines": len(outlines) == 12,
            "six_distinct_shapes": len(counts) == 6,
            "every_shape_exactly_twice":
                sorted(counts.values()) == [2] * 6,
            "no_warnings": marks.warnings == (),
        },
    }


def test_different_uv_density_is_reported_once():
    """CubeB's UVs at half size: same 3D area over a quarter of the UV
    area, so its metres per UV unit is double CubeA's. Exactly one warning,
    naming both; none when the two are unwrapped alike."""
    a, b = _two_cubes('HALF')
    s = _settings()
    with _Items([a, b]) as items:
        uneven = fm.build_marks_multi(items, s, 1.0).warnings
    a, b = _two_cubes('SAME')
    with _Items([a, b]) as items:
        even = fm.build_marks_multi(items, s, 1.0).warnings
    density = [w for w in uneven if "UV density" in w]
    return {
        "uneven": list(uneven), "even": list(even),
        "checks": {
            "one_density_warning": len(density) == 1,
            "names_both_objects": bool(density) and "CubeA" in density[0]
            and "CubeB" in density[0],
            "only_that_warning": len(uneven) == 1,
            "none_when_equal": even == (),
        },
    }


def test_a_mirrored_copy_is_reported_and_its_bevels_measured():
    """CubeB scaled (-1, 1, 1), with CubeA's UVs: one mirror warning,
    naming CubeB only.

    And the measurement the warning rests on. A mirror image of a solid has
    the same convex edges, so the physically right answer for the mirrored
    cube is still +90 everywhere: convex tone, blade tip inward. If the
    angle were measured through the negative-determinant matrix with a
    sign that flips, every bevel would come out concave and every arrow
    reversed. Recorded both ways: the tones per object, and whether B's
    arrows are point for point A's (they share UVs, so they should be).
    """
    mirror = Matrix.Translation((0.5, 0.0, 0.0)) @ Matrix.Diagonal(
        (-1.0, 1.0, 1.0, 1.0))
    a, b = _two_cubes('SAME', b_matrix=mirror)
    s = _settings()
    s.show_dart_colors = True
    with _Items([a, b]) as items:
        sheet, parts = fm._layout_sheet(items, s, 1.0)
        font = sheet.permille(s.angle_text_size, s.overlay_scale)
        stroke = sheet.permille(2, s.overlay_scale)
        per = {}
        for part in parts:
            marks = fm._bevel_marks(part.islands, part.item.uv_layer,
                                    part.runs_by_island, s, font, stroke)
            per[part.item.name] = {
                "tones": Counter(m.label.tone for m in marks
                                 if m.label is not None),
                "angles": sorted(round(r.mean_angle_deg, 3)
                                 for rs in part.runs_by_island for r in rs),
                # The blade direction, however it is drawn: a stroke arrow,
                # or (since 2026-11-17) a glyph in the label text with the
                # direction in its anchor. Points, text and direction are
                # all compared, so either form has to agree.
                "arrows": sorted(
                    (tuple((round(x, 4), round(y, 4))
                           for x, y in m.arrow.points)
                     if m.arrow is not None else (),
                     # Less the seam number, which differs by design:
                     # the copy's seams are numbered on from the original's.
                     re.sub(r"\d+, ", "", m.label.text),
                     getattr(getattr(m.label, "anchor", None),
                             "tip_inward", None))
                    for m in marks if m.label is not None),
            }
        warnings = fm.sheet_warnings(parts, sheet)
    mirror_notes = [w for w in warnings if "mirrored" in w]
    return {
        "warnings": list(warnings),
        "tones": {k: dict(v["tones"]) for k, v in per.items()},
        "angle_range": {k: [v["angles"][0], v["angles"][-1]]
                        for k, v in per.items() if v["angles"]},
        "checks": {
            "det_is_negative": b.matrix_world.to_3x3().determinant() < 0,
            "one_mirror_warning": len(mirror_notes) == 1,
            "names_the_copy": bool(mirror_notes)
            and mirror_notes[0].startswith("CubeB"),
            "not_the_original": not any(w.startswith("CubeA")
                                        for w in mirror_notes),
            # The physical answer: the mirrored cube is still convex, +90.
            "original_all_convex":
                dict(per["CubeA"]["tones"]) == {"convex": 24},
            "mirror_all_convex":
                dict(per["CubeB"]["tones"]) == {"convex": 24},
            "mirror_angles_equal_original":
                per["CubeA"]["angles"] == per["CubeB"]["angles"],
            "mirror_arrows_equal_original":
                per["CubeA"]["arrows"] == per["CubeB"]["arrows"]
                and len(per["CubeA"]["arrows"]) == 24,
            # Convex, pattern on the outer skin: the blade tip goes into
            # the piece on every run (`_blade_tip_inward`) -- on the copy
            # too. None means this build carries no anchor; then the
            # equality above is the whole check.
            "mirror_blade_tips_inward": all(
                d in (True, None) for _p, _t, d in per["CubeB"]["arrows"]),
        },
    }


# --- the operators ------------------------------------------------------------

def _edit_both(a, b, active):
    # mode_set polls for an active object, which freshly appended objects
    # do not yet have; only leave a mode that is actually entered.
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    a.select_set(True)
    b.select_set(True)
    bpy.context.view_layer.objects.active = active
    bpy.ops.object.mode_set(mode='EDIT')


def _svg_outlines(path):
    text = open(path, encoding="utf-8").read()
    group = re.search(r'<g id="outlines">(.*?)</g>', text, re.S)
    return len(re.findall(r"<polygon", group.group(1))) if group else -1


def _svg_numbers(path):
    text = open(path, encoding="utf-8").read()
    group = re.search(r'<g id="numbers">(.*?)</g>', text, re.S)
    return sorted(re.findall(r">([^<]+)</text>", group.group(1))) \
        if group else []


def test_both_operators_export_both_objects():
    """In multi-object edit mode, each operator writes both objects.

    SVG: 12 outline polygons and the 12 names. PNG: the size the model's
    sheet says, and 12 large flat fill colours -- one per piece; 6 would
    mean only the active object was drawn. Run with each object active in
    turn: the files must not change with it.
    """
    a, b = _two_cubes('SIDE')
    _settings()
    out, checks = {}, {}
    svgs = []
    for active in (a, b):
        _edit_both(a, b, active)
        tag = active.name
        svg = os.path.join(_TMP, "both_%s.svg" % tag)
        png = os.path.join(_TMP, "both_%s.png" % tag)
        st_svg = bpy.ops.foamcraft_drachis.export_panel_pattern(filepath=svg)
        st_png = bpy.ops.foamcraft_drachis.export_pattern_image(
            filepath=png, file_format='PNG', resolution_mode='DPI', dpi=150)
        bpy.ops.object.mode_set(mode='OBJECT')
        svgs.append(open(svg, encoding="utf-8").read())
        image = bpy.data.images.load(png, check_existing=False)
        try:
            w, h = image.size
            flat = np.empty(w * h * 4, dtype=np.float32)
            image.pixels.foreach_get(flat)
        finally:
            bpy.data.images.remove(image)
        rows = (flat.reshape(h, w, 4)[:, :, :3] * 255 + 0.5).astype(np.uint32)
        keys = (rows[:, :, 0] << 16) | (rows[:, :, 1] << 8) | rows[:, :, 2]
        values, counts = np.unique(keys, return_counts=True)
        # Piece fills are pale by design (core.island_palette_color), every
        # channel well above 150; the convex colour of the bevel labels and
        # arrows (#2673ff, measured at 1% of this image) is not, and is
        # the one other large colour here. Paper white is excluded.
        fills = [int(v) for v, c in zip(values, counts)
                 if c >= 0.002 * keys.size and int(v) != 0xFFFFFF
                 and min(int(v) >> 16, (int(v) >> 8) & 255,
                         int(v) & 255) >= 150]
        with _Items([a, b]) as items:
            marks = fm.build_marks_multi(items, bpy.context.scene.foamcraft_drachis,
                                         bpy.context.scene.unit_settings.scale_length)
        want_px = fr.pixel_size(marks.width_mm, marks.height_mm, dpi=150)[:2]
        out[tag] = {"svg_outlines": _svg_outlines(svg),
                    "svg_names": _svg_numbers(svg),
                    "png": [w, h], "png_fills": len(fills)}
        checks[tag + "_svg_finished"] = st_svg == {'FINISHED'}
        checks[tag + "_png_finished"] = st_png == {'FINISHED'}
        checks[tag + "_svg_12_outlines"] = out[tag]["svg_outlines"] == 12
        checks[tag + "_svg_12_names"] = len(set(out[tag]["svg_names"])) == 12
        checks[tag + "_png_sheet_size"] = [w, h] == list(want_px)
        checks[tag + "_png_12_fills"] = len(fills) == 12
    checks["svg_same_whichever_is_active"] = svgs[0] == svgs[1]
    out["checks"] = checks
    return out


# --- the maker's file -----------------------------------------------------------

def _copy_settings(src, dst):
    for prop in src.bl_rna.properties:
        name = prop.identifier
        if name in ("rna_type", "name") or prop.is_readonly:
            continue
        if prop.type == 'POINTER':
            continue
        try:
            setattr(dst, name, getattr(src, name))
        except (AttributeError, TypeError, ValueError):
            pass


def test_the_beast_skull_head_and_its_mirror():
    """head_simplified and head_simplified.001 (scale -1, 1, 1) together,
    through the image operator, from a temp copy of the snapshot.

    Appended, not opened -- opening a file inside a bridge request replaces
    the session running it -- with the snapshot scene's settings and unit
    scale copied onto this scene, since the operator reads context.scene.
    Reports the piece count and the warnings; the image goes to
    screenshots/ to be looked at. Bounds: twice the single head's pieces,
    with no name repeated, and exactly one mirror warning naming the copy.
    """
    names = ("head_simplified", "head_simplified.001")
    if not os.path.exists(SNAPSHOT):
        return {"skipped": "fixture %s is absent (git-ignored, local only)"
                           % SNAPSHOT, "checks": {}}
    sandbox_guard.clear_scene()
    tmp = os.path.join(tempfile.mkdtemp(prefix="foam_multi_"), "copy.blend")
    shutil.copyfile(SNAPSHOT, tmp)
    with bpy.data.libraries.load(tmp, link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n in names]
        dst.scenes = list(src.scenes[:1])
    objs = [o for o in dst.objects if o is not None]
    if len(objs) != 2:
        return {"skipped": "snapshot lacks %r" % (names,), "checks": {}}
    for o in objs:
        bpy.context.scene.collection.objects.link(o)
    scene = next((sc for sc in dst.scenes if sc is not None), None)
    if scene is not None and getattr(scene, "foamcraft_drachis", None):
        _copy_settings(scene.foamcraft_drachis,
                       bpy.context.scene.foamcraft_drachis)
        bpy.context.scene.unit_settings.scale_length = \
            scene.unit_settings.scale_length
    s = bpy.context.scene.foamcraft_drachis
    # Both heads are parented to "retopo", which comes along unlinked. Until
    # the depsgraph evaluates them, matrix_world is whatever the file held
    # and the mirror copy measured a positive determinant -- the mirror
    # warning was missed. Evaluate first, then read.
    bpy.context.view_layer.update()
    by_name = {o.name: o for o in objs}
    a, b = by_name[names[0]], by_name[names[1]]

    with _Items([a]) as items:
        single = fm.build_marks_multi(
            items, s, bpy.context.scene.unit_settings.scale_length)
    with _Items([a, b]) as items:
        both = fm.build_marks_multi(
            items, s, bpy.context.scene.unit_settings.scale_length)
    single_pieces = sum(1 for l in single.labels if l.kind == 'number')
    piece_names = [l.text for l in both.labels if l.kind == 'number']

    # Appended objects are linked but not in this view layer's active
    # slot until made so; mode_set needs an active object.
    bpy.context.view_layer.update()
    _edit_both(a, b, a)
    path = shot.next_path("multi_object_beast_skull")
    status = bpy.ops.foamcraft_drachis.export_pattern_image(
        filepath=path, file_format='PNG', resolution_mode='PIXELS',
        size=3000)
    bpy.ops.object.mode_set(mode='OBJECT')
    mirror_notes = [w for w in both.warnings if "mirrored" in w]
    return {
        "image": path,
        "prefixes": [fm.object_prefix(o.name, o.foamcraft_drachis_prefix)
                     for o in (a, b)],
        "scale_of_copy": list(b.scale),
        "determinants": [a.matrix_world.to_3x3().determinant(),
                         b.matrix_world.to_3x3().determinant()],
        "scale_length": bpy.context.scene.unit_settings.scale_length,
        "single_pieces": single_pieces,
        "pieces_on_sheet": len(piece_names),
        "first_and_last_names": [piece_names[:3], piece_names[-3:]],
        "warnings": list(both.warnings),
        "sheet_mm": [round(both.width_mm, 1), round(both.height_mm, 1)],
        "checks": {
            "operator_finished": status == {'FINISHED'},
            "image_written": os.path.exists(path)
            and os.path.getsize(path) > 1000,
            "twice_the_pieces": single_pieces > 0
            and len(piece_names) == 2 * single_pieces,
            "no_name_repeats": len(set(piece_names)) == len(piece_names),
            "one_mirror_warning": len(mirror_notes) == 1
            and mirror_notes[0].startswith("head_simplified.001"),
        },
    }


TESTS = (
    test_piece_letters_are_bijective_base_26,
    test_prefix_property_holds_three_upper_case_characters,
    test_two_cubes_side_by_side_make_one_sheet,
    test_numbering_does_not_depend_on_order,
    test_identical_uvs_are_drawn_overlaid,
    test_different_uv_density_is_reported_once,
    test_a_mirrored_copy_is_reported_and_its_bevels_measured,
    test_both_operators_export_both_objects,
    test_the_beast_skull_head_and_its_mirror,
)

result = {}
_all_pass = True
for _test in TESTS:
    try:
        _report = _test()
        _report["PASS"] = all(_report.get("checks", {}).values())
    except Exception:
        _report = {"ERROR": traceback.format_exc().splitlines()[-8:],
                   "PASS": False}
    result[_test.__name__] = _report
    _all_pass = _all_pass and _report["PASS"]
result["ALL_PASS"] = _all_pass
