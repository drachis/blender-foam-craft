"""What the SVG exporter actually puts on the sheet.

    python tools/launch_blender.py --port 9885
    python tools/blender_bridge.py --port 9885 < tests/test_export_svg.py

The exporter had no test of any kind until 2026-11-03, which is how two
user-visible defects reached a release: the sheet does not reproduce the UV
layout, and the angle labels can come out empty. Both are checked here.

The fixture is the cube with every edge seamed, which CLAUDE.md lists as a
known-answer shape: 6 islands, 24 boundary runs, every dihedral exactly 90
degrees. Nothing about those numbers is a matter of taste, so a wrong result
cannot survive them.
"""

import math
import sys
import importlib
import re
import traceback

import bpy
import bmesh
from mathutils import Vector

sys.path.insert(0, r"O:\Work\BlenderPlugin\FoamCraft\tools")

import sandbox_guard  # noqa: E402  (guards every _clear below)

_MOD = "bl_ext.user_default.foamcraft_drachis"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foamcraft_drachis.foam_export as fx    # noqa: E402
import bl_ext.user_default.foamcraft_drachis.foam_angle_core as core  # noqa: E402

importlib.reload(core)
importlib.reload(fx)


def _clear():
    """Empty the scene -- but only in a Blender this suite started.

    Delegates to tools/sandbox_guard, which refuses when the session cannot
    prove it is disposable. On 2026-11-03 this function, called directly,
    deleted a user's unsaved sculpt after the launcher adopted their Blender
    off a shared port. The guard is the thing that stops that repeating.
    """
    sandbox_guard.clear_scene()


def _seamed_cube():
    """A 2 m cube with every edge a seam, unwrapped into six separate faces.

    Six islands of one quad each, so every island boundary is four edges and
    every one of those edges still has two faces in 3D -- the seam is a UV
    cut, not a hole. That matters: it is exactly the configuration in which a
    boundary run should still carry a dihedral angle.
    """
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


def _bmesh_of(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    return bm


def _settings():
    return bpy.context.scene.foamcraft_drachis


def test_the_cube_fixture_is_what_we_think():
    """Assert the fixture before asserting anything about the exporter.

    CLAUDE.md: rebuild scene state for every measurement, then assert what you
    built. A cube is 6 faces and 12 edges; all-seamed it unwraps to 6 islands.
    If this is wrong, every number below is about some other shape.
    """
    obj = _seamed_cube()
    bm = _bmesh_of(obj)
    try:
        uv_layer = bm.loops.layers.uv.active
        islands = core.uv_islands(bm, uv_layer, only_selected=False)
        dihedrals = sorted({round(math.degrees(e.calc_face_angle_signed(0.0)), 3)
                            for e in bm.edges if len(e.link_faces) == 2})
        return {
            "faces": len(bm.faces),
            "edges": len(bm.edges),
            "islands": len(islands),
            "distinct_dihedrals_deg": dihedrals,
            "checks": {
                "six_faces": len(bm.faces) == 6,
                "twelve_edges": len(bm.edges) == 12,
                "six_islands": len(islands) == 6,
                # Every edge of a cube is convex 90. Both bounds: a single
                # value, and that value is 90.
                "every_dihedral_is_90": dihedrals == [90.0],
            },
        }
    finally:
        bm.free()


def test_the_sheet_reproduces_the_uv_layout():
    """The UV layout *is* the arrangement, so the sheet must preserve it.

    CLAUDE.md records this exact mistake once already, in the flat layout
    preview: a bespoke shelf packer was invented to arrange pieces, when the
    answer was u->x, v->y scaled to the mesh, because the user packed the UVs
    deliberately. The maker lays pieces out to suit their stock; an exporter
    that re-arranges them silently discards that work.

    The check is a similarity test, not an equality one. The sheet is in
    millimetres and the UVs are in unit space, so the two can differ by one
    overall scale and one translation and still "match" -- what must not
    differ is the *relative* placement of one island against another.

    Measured by taking each island's centroid in UV space and on the sheet,
    then comparing every pairwise distance. Under a single uniform scale all
    those ratios are the same number. If islands have been independently
    rescaled or moved into packer rows, they are not.
    """
    obj = _seamed_cube()
    bm = _bmesh_of(obj)
    try:
        uv_layer = bm.loops.layers.uv.active
        islands_faces = core.uv_islands(bm, uv_layer, only_selected=False)
        sheet = fx._Sheet(islands_faces, uv_layer, 1.0)
        islands = [fx._Island(i, faces, uv_layer, sheet)
                   for i, faces in enumerate(islands_faces)]

        uv_centroids, sheet_centroids = [], []
        for isl in islands:
            uvs = [l[uv_layer].uv for f in isl.faces for l in f.loops]
            uv_c = Vector((sum(p.x for p in uvs) / len(uvs),
                           sum(p.y for p in uvs) / len(uvs)))
            uv_centroids.append(uv_c)
            sheet_centroids.append(isl.to_sheet(uv_c))

        ratios = []
        for i in range(len(islands)):
            for j in range(i + 1, len(islands)):
                d_uv = (uv_centroids[i] - uv_centroids[j]).length
                d_sheet = (sheet_centroids[i] - sheet_centroids[j]).length
                if d_uv > 1e-9:
                    ratios.append(d_sheet / d_uv)

        spread = (max(ratios) / min(ratios)) if ratios else 0.0
        return {
            "islands": len(islands),
            "pair_count": len(ratios),
            "mm_per_uv_min": round(min(ratios), 3) if ratios else None,
            "mm_per_uv_max": round(max(ratios), 3) if ratios else None,
            "spread_max_over_min": round(spread, 3),
            "shared_scale_mm_per_uv": round(sheet.scale, 3),
            "sheet_mm": (round(sheet.width, 1), round(sheet.height, 1)),
            "checks": {
                # One uniform mapping from UV to sheet: every pairwise
                # distance scales by the same factor. 1% allows for float
                # noise and nothing else.
                "layout_is_uniformly_scaled": spread < 1.01,
                # Both bounds: ratios must exist at all, or the above is a
                # statement about an empty list.
                "islands_are_distinct": len(ratios) == 15,
            },
        }
    finally:
        bm.free()


def test_every_run_gets_an_angle_label():
    """A seamed cube's runs are all at 90 degrees, so all of them get labels.

    90 is far above the 5 degree Minimum Cut Angle, and a cube's seams are UV
    cuts rather than holes -- both faces are still there -- so every boundary
    run has a dihedral to report. There is no reading of the settings under
    which a label should be missing here.

    Counting the <text> elements inside the angles group, because that is what
    reaches the printed sheet. Counting runs instead would pass while the SVG
    came out blank, which is the defect being chased.
    """
    obj = _seamed_cube()
    bm = _bmesh_of(obj)
    try:
        uv_layer = bm.loops.layers.uv.active
        settings = _settings()
        svg = fx.generate_svg(bm, uv_layer, settings, 1.0)

        islands_faces = core.uv_islands(bm, uv_layer, only_selected=False)
        runs = list(core.iter_boundary_dart_segments(
            bm, uv_layer, settings.max_chord_deviation,
            settings.max_angle_variation_deg, islands=islands_faces))

        group = re.search(r'<g id="angles".*?</g>', svg, re.S)
        labels = re.findall(r"<text[^>]*>([^<]*)</text>",
                            group.group(0) if group else "")

        return {
            "min_angle_setting": round(settings.min_angle, 3),
            "runs_total": len(runs),
            "runs_with_dihedral": sum(1 for r in runs if r.has_dihedral),
            "run_mean_angles_deg": sorted({round(r.mean_angle_deg, 1)
                                           for r in runs}),
            "angle_labels_in_svg": len(labels),
            "sample_labels": labels[:4],
            "svg_bytes": len(svg),
            "checks": {
                # Every boundary edge of a seamed cube still has two faces.
                "every_run_has_a_dihedral":
                    len(runs) > 0 and all(r.has_dihedral for r in runs),
                # And every one of them is 90, well over the 5 threshold.
                "every_run_is_ninety_degrees":
                    sorted({round(abs(r.mean_angle_deg)) for r in runs}) == [90],
                # So the sheet must carry one label per run.
                "a_label_per_run": len(labels) == len(runs),
                # Both bounds: a file with no angles group would give 0, and a
                # runaway one would give far too many.
                "labels_exist": len(labels) > 0,
            },
        }
    finally:
        bm.free()



def _panelled_sphere(segments=24, rings=12):
    """A sphere cut into panels by seaming whole edge loops.

    Stands in for the real subject: a smooth sculpt covered in panels. Unlike
    the cube, the fold between two neighbouring panels here is the surface's
    own gentle curvature rather than a hard 90 degree corner, which is the
    condition the Minimum Cut Angle threshold interacts with.
    """
    _clear()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=segments,
                                         ring_count=rings)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    # Seam every edge that runs along a ring (constant z between its verts is
    # false for ring edges; use the vertical ones) plus every Nth meridian, so
    # the sphere falls into a grid of panels.
    for e in bm.edges:
        a, b = e.verts
        same_ring = abs(a.co.z - b.co.z) < 1e-6
        if same_ring:
            # A horizontal edge: seam it to separate stacked bands.
            e.seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def test_the_angle_threshold_is_not_what_hides_the_labels():
    """Rules out the obvious explanation for a pattern exporting unlabelled.

    The first guess, on a report of "the export has no annotations", was that
    Minimum Cut Angle was eating them: it defaults to 5 degrees, its own
    description warns that a subdivided sculpt reports a degree or two across
    every edge, and a smooth surface's panel seams might plausibly sit in the
    same range.

    Measured 2026-11-03, and the guess was wrong. On a sphere panelled into
    bands the folds come out at about 15 degrees -- three times the threshold
    -- and every one of the 22 runs is labelled, at the default setting and at
    zero alike. The threshold is not touching them.

    The run count in this docstring said 528 when the case was first written.
    That number is not reproducible: the same fixture and settings now give 22,
    and a separate probe across two sphere radii agrees. Whatever produced 528
    was a code state that no longer exists, so the figure is corrected here
    rather than left to be trusted. The conclusion is unchanged -- the folds
    clear the threshold by three times either way -- but a number nobody can
    reproduce is worse than no number, because the next session will try.

    Kept rather than deleted, because it is the cheapest way to re-rule-out
    the same hypothesis next time, and because the numbers say something
    useful on their own: a sphere this size folds about 15 degrees per band,
    so a maker whose labels vanish on a comparable shape should look
    somewhere other than this setting.

    The real cause turned out to be legibility, not suppression -- see
    test_the_labels_are_large_enough_to_read_on_the_sheet.
    """
    obj = _panelled_sphere()
    bm = _bmesh_of(obj)
    try:
        uv_layer = bm.loops.layers.uv.active
        settings = _settings()
        islands_faces = core.uv_islands(bm, uv_layer, only_selected=False)
        runs = list(core.iter_boundary_dart_segments(
            bm, uv_layer, settings.max_chord_deviation,
            settings.max_angle_variation_deg, islands=islands_faces))

        angles = sorted(abs(r.mean_angle_deg) for r in runs if r.has_dihedral)

        def _label_count(threshold):
            original = settings.min_angle
            settings.min_angle = threshold
            try:
                svg = fx.generate_svg(bm, uv_layer, settings, 1.0)
            finally:
                settings.min_angle = original
            group = re.search(r'<g id="angles".*?</g>', svg, re.S)
            return len(re.findall(r"<text[^>]*>([^<]*)</text>",
                                  group.group(0) if group else ""))

        at_default = _label_count(5.0)
        at_zero = _label_count(0.0)
        below = sum(1 for a in angles if a < 5.0)

        return {
            "islands": len(islands_faces),
            "runs_total": len(runs),
            "runs_with_dihedral": sum(1 for r in runs if r.has_dihedral),
            "fold_angles_deg_min": round(angles[0], 2) if angles else None,
            "fold_angles_deg_max": round(angles[-1], 2) if angles else None,
            "fold_angles_deg_median":
                round(angles[len(angles) // 2], 2) if angles else None,
            "runs_below_5_deg": below,
            "labels_at_default_threshold_5deg": at_default,
            "labels_at_zero_threshold": at_zero,
            "checks": {
                # The folds on this shape sit well clear of the threshold, so
                # lowering it to zero changes nothing. That is the finding.
                "threshold_does_not_suppress_anything_here":
                    at_zero == at_default,
                "every_run_is_labelled_at_the_default": at_default == len(runs),
                # Both bounds: there must be runs in the first place, and
                # their folds must genuinely clear the 5 degree threshold --
                # if they did not, the equality above would be vacuous.
                "the_mesh_has_runs": len(runs) > 0,
                "folds_clear_the_threshold": below == 0,
            },
        }
    finally:
        bm.free()



def test_the_labels_are_large_enough_to_read_on_the_sheet():
    """A label that is in the file but 0.3 px tall is not an annotation.

    The exporter writes every label at a fixed ANGLE_FONT_MM = 4.0, and sets
    the sheet's viewBox from whatever size the packed islands came to. Those
    two numbers are unrelated, so the text's size *relative to the sheet* is
    whatever falls out -- and on a model whose pieces are large, the labels
    shrink to nothing when the file is opened and fitted to the screen.

    Note what is not the cause: the labels are present. test_every_run_gets_
    an_angle_label counts 24 of them in the cube's SVG. The failure here is
    not a missing element, it is an unreadable one, which looks identical to
    the user and has a completely different fix.

    The bar: opened and fitted to a 1000 px tall view, a label must come out
    at least 6 px tall, which is about the smallest text a person can read.
    Stated as a ratio so it does not depend on anyone's screen.
    """
    rows = {}
    for label, builder in (("seamed_cube", _seamed_cube),
                           ("panelled_sphere", _panelled_sphere)):
        obj = builder()
        bm = _bmesh_of(obj)
        try:
            uv_layer = bm.loops.layers.uv.active
            svg = fx.generate_svg(bm, uv_layer, _settings(), 1.0)
            box = re.search(r"viewBox='0 0 ([\d.]+) ([\d.]+)'", svg)
            width = float(box.group(1)) if box else 0.0
            height = float(box.group(2)) if box else 0.0
            # Read the size off an actual label rather than a module constant:
            # the font is computed per sheet now, so the constant is gone and
            # what matters is what reached the file.
            group = re.search(r'<g id="angles".*?</g>', svg, re.S)
            fonts = [float(f) for f in re.findall(
                r'font-size="([\d.]+)"', group.group(0) if group else "")]
            font_mm = fonts[0] if fonts else 0.0
            px_at_1000 = (font_mm / height * 1000.0) if height else 0.0
            rows[label] = {
                "sheet_width_mm": round(width, 1),
                "sheet_height_mm": round(height, 1),
                "angle_font_mm": round(font_mm, 3),
                "font_as_fraction_of_sheet_height":
                    round(font_mm / height, 6) if height else None,
                "label_px_when_fitted_to_1000px": round(px_at_1000, 2),
            }
        finally:
            bm.free()

    out = dict(rows)
    out["checks"] = {
        "cube_labels_readable":
            rows["seamed_cube"]["label_px_when_fitted_to_1000px"] >= 6.0,
        "sphere_labels_readable":
            rows["panelled_sphere"]["label_px_when_fitted_to_1000px"] >= 6.0,
    }
    return out



def test_every_piece_number_lands_inside_its_piece():
    """A number printed outside its piece is worse than no number.

    The sheet doubles as a texture: applied to the model, each panel should
    carry its own number. A label placed at the island's centroid fails that
    whenever the island is not convex -- the centroid of a crescent, a C or a
    ring lies in the gap, so the number sits on the sheet beside the piece and
    disappears entirely once the pattern is on the model.

    Checked by taking every number's coordinates straight out of the SVG and
    asking whether that point is inside the island it belongs to, using the
    same ray-casting test the placement relies on. Both bounds: every number
    inside, and one number per island so none was dropped to make that true.

    And its *size* is checked, which this case originally did not. A label is
    shrunk to fit the room its piece has, and the floor on that was 0.1 mm --
    not a smaller label, an invisible one. Three islands on a panelled sphere
    printed at exactly that, because their boundary cycles had been ordered
    by vertex count: an annulus unwrapped from a sphere band has as many
    vertices inside as out, the comparison tied, and the hole was taken for
    the outline. The region was then empty, every distance negative, and the
    "room" zero.

    The original checks all passed through that, which is the more useful
    lesson. "Inside some island" was asked where "inside its own island" was
    meant, and nothing asked whether the number could actually be seen.
    """
    obj = _seamed_cube()
    bm = _bmesh_of(obj)
    try:
        uv_layer = bm.loops.layers.uv.active
        settings = _settings()
        svg = fx.generate_svg(bm, uv_layer, settings, 1.0)

        islands_faces = core.uv_islands(bm, uv_layer, only_selected=False)
        sheet = fx._Sheet(islands_faces, uv_layer, 1.0)
        islands = [fx._Island(i, faces, uv_layer, sheet)
                   for i, faces in enumerate(islands_faces)]
        rings_per_island = []
        for isl in islands:
            rings_per_island.append(
                [[(p.x, p.y) for p in
                  (isl.to_sheet(l[uv_layer].uv) for l in cyc)]
                 for cyc in sorted(isl.cycles, key=len, reverse=True)])

        group = re.search(r'<g id="numbers".*?</g>', svg, re.S)
        placed = re.findall(
            # Updated 2026-11-16: <text> now also carries font-family, fill
            # and text-anchor attributes (no CSS), so the tag no longer ends
            # at font-size.
            r'<text x="([-\d.]+)" y="([-\d.]+)" font-size="([\d.]+)"'
            r'[^>]*>([^<]*)</text>',
            group.group(0) if group else "")

        # Island index by piece id, so each number is tested against its OWN
        # piece rather than against any of them.
        by_piece = {str(isl.piece_id): rings_per_island[i]
                    for i, isl in enumerate(islands)}

        own_piece, sizes = 0, []
        for x, y, size, label in placed:
            size = float(size)
            # The writer moves a 'middle' label's baseline down by 0.35 size
            # (no dominant-baseline); undo it to get the label's centre.
            px, py = float(x), float(y) - fx.MIDDLE_BASELINE_SHIFT * size
            sizes.append(size)
            rings = by_piece.get(label)
            if rings and core._point_in_rings(px, py, rings):
                own_piece += 1

        readable = [s for s in sizes if s >= 1.0]
        return {
            "islands": len(islands),
            "numbers_placed": len(placed),
            "numbers_inside_their_own_piece": own_piece,
            "font_sizes_mm": [round(s, 2) for s in sorted(sizes)],
            "smallest_font_mm": round(min(sizes), 3) if sizes else None,
            "checks": {
                "one_number_per_island": len(placed) == len(islands),
                # Its own piece, not merely some piece.
                "every_number_is_inside_its_own_piece":
                    own_piece == len(placed),
                # And large enough to see. A millimetre is generous as a
                # floor and still catches a collapse to nothing.
                "every_number_is_legible": len(readable) == len(sizes),
                # Both bounds: zero numbers would satisfy everything above.
                "numbers_were_actually_placed": len(placed) > 0,
            },
        }
    finally:
        bm.free()


def test_a_crescent_piece_gets_its_number_inside():
    """The shape a centroid gets wrong, asserted directly.

    The cube's islands are convex, so the case above would pass even with the
    old centroid placement. This is the one that would not: a C whose centroid
    is provably outside it. Testing the geometry rather than a bake, because
    producing a reliably crescent-shaped UV island is a test about unwrapping
    and this is a test about label placement.
    """
    c_shape = [(0.0, 0.0), (10.0, 0.0), (10.0, 3.0), (3.0, 3.0),
               (3.0, 7.0), (10.0, 7.0), (10.0, 10.0), (0.0, 10.0)]
    cx = sum(p[0] for p in c_shape) / len(c_shape)
    cy = sum(p[1] for p in c_shape) / len(c_shape)
    (px, py), room = core.pole_of_inaccessibility([c_shape])

    # A square, where the answer is arithmetic: dead centre, radius half the
    # side. A wrong implementation cannot survive this.
    square = [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]
    (sx, sy), sr = core.pole_of_inaccessibility([square])

    # A square with a hole: the label must not land in the hole.
    holed = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    hole = [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)]
    (hx, hy), hr = core.pole_of_inaccessibility([holed, hole])

    return {
        "crescent_centroid": [round(cx, 3), round(cy, 3)],
        "crescent_centroid_inside": core._point_in_rings(cx, cy, [c_shape]),
        "crescent_pole": [round(px, 3), round(py, 3)],
        "crescent_pole_inside": core._point_in_rings(px, py, [c_shape]),
        "square_pole": [round(sx, 3), round(sy, 3), round(sr, 3)],
        "holed_pole_inside":
            core._point_in_rings(hx, hy, [holed, hole]),
        "checks": {
            # The failure being fixed, stated as a fact about the fixture.
            "the_centroid_really_is_outside":
                not core._point_in_rings(cx, cy, [c_shape]),
            "the_pole_is_inside":
                core._point_in_rings(px, py, [c_shape]),
            # Arithmetic, to a hundredth: centre of an 8x8 square, radius 4.
            "square_answer_is_exact":
                abs(sx - 4.0) < 0.01 and abs(sy - 4.0) < 0.01
                and abs(sr - 4.0) < 0.01,
            "a_hole_is_avoided": core._point_in_rings(hx, hy, [holed, hole]),
        },
    }


def test_island_fill_colours_are_distinct_and_toggleable():
    """One colour per piece, and a way to turn them off for a home printer."""
    obj = _seamed_cube()
    bm = _bmesh_of(obj)
    try:
        uv_layer = bm.loops.layers.uv.active
        settings = _settings()
        islands = len(core.uv_islands(bm, uv_layer, only_selected=False))

        original = settings.island_fill
        try:
            settings.island_fill = True
            on = fx.generate_svg(bm, uv_layer, settings, 1.0)
            settings.island_fill = False
            off = fx.generate_svg(bm, uv_layer, settings, 1.0)
        finally:
            settings.island_fill = original

        def _fill_colours(svg):
            g = re.search(r'<g id="fills".*?</g>', svg, re.S)
            return re.findall(r'fill="(#[0-9a-f]{6})"', g.group(0) if g else "")

        on_colours = _fill_colours(on)
        off_colours = _fill_colours(off)
        return {
            "islands": islands,
            "fills_when_on": len(on_colours),
            "distinct_colours": len(set(on_colours)),
            "fills_when_off": len(off_colours),
            "sample": on_colours[:4],
            "checks": {
                "one_fill_per_island": len(on_colours) == islands,
                # Distinct, or two neighbouring pieces cannot be told apart,
                # which is the entire purpose.
                "colours_are_all_different":
                    len(set(on_colours)) == len(on_colours),
                "the_toggle_removes_them": len(off_colours) == 0,
            },
        }
    finally:
        bm.free()


def test_the_scale_ruler_is_in_real_millimetres():
    """The ruler must not scale with the pattern -- that is its whole job.

    Everything else on the sheet is sized in thousandths of the UV sheet so it
    reads at any pattern size. The ruler is the exception: it is measured
    against a real ruler held to the paper, so a 100 mm bar has to be 100 mm
    of SVG user units, whatever the pattern around it is doing.

    Checked across two patterns of very different size. The bar and the inch
    square must come out identical in both.
    """
    rows = {}
    for label, builder in (("seamed_cube", _seamed_cube),
                           ("panelled_sphere", _panelled_sphere)):
        obj = builder()
        bm = _bmesh_of(obj)
        try:
            uv_layer = bm.loops.layers.uv.active
            svg = fx.generate_svg(bm, uv_layer, _settings(), 1.0)
            g = re.search(r'<g id="ruler".*?</g>', svg, re.S)
            body = g.group(0) if g else ""
            # The long bar is the widest horizontal line in the group.
            spans = [abs(float(b) - float(a)) for a, b in re.findall(
                r'<line x1="([-\d.]+)" y1="[-\d.]+" x2="([-\d.]+)"', body)]
            # Anchored on x/y so it cannot match stroke-width, which the rect
            # now carries as an attribute (no CSS).
            rect = re.search(r'<rect x="[-\d.]+" y="[-\d.]+" '
                             r'width="([\d.]+)"', body)
            box = re.search(r"viewBox='0 0 ([\d.]+) ([\d.]+)'", svg)
            rows[label] = {
                "bar_mm": round(max(spans), 3) if spans else None,
                "inch_square_mm": round(float(rect.group(1)), 3) if rect else None,
                "sheet_height_mm": round(float(box.group(2)), 1) if box else None,
                "tick_count": len(spans),
            }
        finally:
            bm.free()

    cube, sphere = rows["seamed_cube"], rows["panelled_sphere"]
    out = dict(rows)
    out["checks"] = {
        "bar_is_100mm_on_both":
            cube["bar_mm"] == 100.0 and sphere["bar_mm"] == 100.0,
        # 25.4 mm exactly, or it is not an inch.
        "inch_square_is_25_4_on_both":
            cube["inch_square_mm"] == 25.4 and sphere["inch_square_mm"] == 25.4,
        # Both bounds: the two sheets really are different sizes, so the
        # equality above means the ruler is independent rather than that the
        # two cases happen to match.
        "the_two_sheets_differ_in_size":
            cube["sheet_height_mm"] != sphere["sheet_height_mm"],
    }
    return out



def _tick_lines(svg):
    g = re.search(r'<g id="align".*?</g>', svg, re.S)
    body = g.group(0) if g else ""
    return re.findall(
        r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)"',
        body)


def test_alignment_ticks_are_paired_across_every_seam():
    """Both pieces of a seam must be marked at the same point on the foam.

    This is the property that makes them registration marks rather than
    decoration: the maker brings two ticks together, so a tick on piece 4
    with no partner on piece 7 is worse than no tick at all.

    Guaranteed by construction -- positions are chosen on the shared 3D edge
    and then mapped into each island's own UVs -- and checked here by
    counting. Every seam edge carrying marks has two faces in two different
    islands, so each chosen (edge, t) must produce exactly two drawn lines.

    The seamed cube is the right fixture: all twelve edges are seams, every
    one shared by exactly two of the six islands, so the arithmetic is known
    before the test runs.
    """
    obj = _seamed_cube()
    bm = _bmesh_of(obj)
    try:
        uv_layer = bm.loops.layers.uv.active
        settings = _settings()
        original = (settings.show_align_ticks, settings.tick_mode,
                    settings.ticks_per_run)
        try:
            settings.show_align_ticks = True
            settings.tick_mode = 'PER_RUN'
            settings.ticks_per_run = 2
            svg = fx.generate_svg(bm, uv_layer, settings, 1.0)
        finally:
            (settings.show_align_ticks, settings.tick_mode,
             settings.ticks_per_run) = original

        lines = _tick_lines(svg)

        islands_faces = core.uv_islands(bm, uv_layer, only_selected=False)
        sheet = fx._Sheet(islands_faces, uv_layer, 1.0)
        islands = [fx._Island(i, faces, uv_layer, sheet)
                   for i, faces in enumerate(islands_faces)]
        runs = list(core.iter_boundary_dart_segments(
            bm, uv_layer, settings.max_chord_deviation,
            settings.max_angle_variation_deg, islands=islands_faces))

        # Changed 2026-11-16: marks are placed per corner-to-corner segment
        # of seam (foam_marks._segment_tick_params), no longer per cut run
        # (_run_tick_params), so this asks the function build_marks now uses.
        # On this cube the answer is the same either way -- every run is one
        # edge is one segment -- which is why the counts below did not move.
        import bl_ext.user_default.foamcraft_drachis.foam_marks as fm
        saved_ticks = settings.ticks_per_run
        settings.ticks_per_run = 2
        try:
            placed = fm._segment_tick_params(islands, settings)
        finally:
            settings.ticks_per_run = saved_ticks
        wanted = {}
        for edge, t in placed:
            wanted.setdefault(edge, set()).add(t)
        marks = sum(len(v) for v in wanted.values())

        # How many of those edges really are shared by two islands?
        face_island = {}
        for index, faces in enumerate(islands_faces):
            for f in faces:
                face_island[f] = index
        shared = 0
        for edge in wanted:
            sides = {face_island.get(f) for f in edge.link_faces}
            sides.discard(None)
            if len(sides) == 2:
                shared += 1

        return {
            "islands": len(islands),
            "runs": len(runs),
            "edges_with_marks": len(wanted),
            "edges_shared_by_two_islands": shared,
            "distinct_marks_on_the_mesh": marks,
            "lines_drawn": len(lines),
            "checks": {
                # Every marked edge of a seamed cube joins two pieces.
                "every_marked_edge_is_a_seam": shared == len(wanted),
                # And each mark is therefore drawn twice, once per piece.
                "each_mark_is_drawn_on_both_pieces":
                    len(lines) == marks * 2,
                # Both bounds, and the known answer: 12 edges, 2 marks each.
                "twenty_four_marks": marks == 24,
            },
        }
    finally:
        bm.free()


def test_tick_spacing_responds_to_both_modes():
    """Per Cut Run and By Distance each do what they say.

    Per Cut Run is arithmetic: n ticks on each of r runs is n*r marks, so
    doubling the count doubles them. By Distance is not tied to run count at
    all -- halving the spacing should roughly double the marks, and the two
    modes should disagree, or one of them is not wired up.
    """
    obj = _panelled_sphere()
    bm = _bmesh_of(obj)
    try:
        uv_layer = bm.loops.layers.uv.active
        settings = _settings()
        saved = (settings.show_align_ticks, settings.tick_mode,
                 settings.ticks_per_run, settings.tick_spacing)

        def _count(mode, per_run=2, spacing=0.025):
            settings.show_align_ticks = True
            settings.tick_mode = mode
            settings.ticks_per_run = per_run
            settings.tick_spacing = spacing
            return len(_tick_lines(fx.generate_svg(bm, uv_layer, settings, 1.0)))

        try:
            per_run_2 = _count('PER_RUN', per_run=2)
            per_run_4 = _count('PER_RUN', per_run=4)
            wide = _count('BY_DISTANCE', spacing=0.04)
            narrow = _count('BY_DISTANCE', spacing=0.02)
            settings.show_align_ticks = False
            off = len(_tick_lines(fx.generate_svg(bm, uv_layer, settings, 1.0)))
        finally:
            (settings.show_align_ticks, settings.tick_mode,
             settings.ticks_per_run, settings.tick_spacing) = saved

        return {
            "per_run_2": per_run_2,
            "per_run_4": per_run_4,
            "by_distance_40mm": wide,
            "by_distance_20mm": narrow,
            "when_switched_off": off,
            "checks": {
                "doubling_the_count_doubles_the_marks":
                    per_run_4 == per_run_2 * 2,
                "halving_the_spacing_adds_marks": narrow > wide,
                # The two modes must not be the same code path.
                "the_modes_differ": wide != per_run_2,
                "the_toggle_removes_them": off == 0,
                "marks_were_produced": per_run_2 > 0,
            },
        }
    finally:
        bm.free()


def test_relief_cuts_are_off_by_default():
    """Relief cuts remove material and most panel work does not want them.

    They were on by default and described as the alignment feature, which was
    two mistakes at once: a maker following the sheet would have cut wedges
    out of edges that wanted an ink mark. The code is kept -- a tight compound
    curve can genuinely need one -- but it is now opt-in and named so it
    cannot be mistaken for Alignment Ticks.

    Fixed 2026-11-16. This used to read only the property's metadata, so it
    passed while the export drew relief wedges whatever the toggle said --
    nothing here ever looked at a sheet. It now draws one: the seamed cube,
    whose 6 square pieces have 4 runs each and so 4 run-to-run corners, all
    at 90 degrees -- 24 relief wedges with the toggle on (V wedge, the
    default shape), and 0 with it off, at its default.
    """
    settings = _settings()
    prop = settings.bl_rna.properties.get("show_darts")

    obj = _seamed_cube()
    counts = {}
    saved = _set(notch_shape='V_WEDGE')
    try:
        for label, on in (("default", prop.default if prop else True),
                          ("on", True)):
            _set(show_darts=on)
            bm = _bmesh_of(obj)
            try:
                marks = fx.build_marks(bm, bm.loops.layers.uv.active,
                                       settings, 1.0, obj.matrix_world)
            finally:
                bm.free()
            counts[label] = sum(1 for m in marks.strokes + marks.fills
                                if m.kind == 'relief')
    finally:
        _set(**saved)
        _set(show_darts=prop.default if prop else False)
    return {
        "default": prop.default if prop else None,
        "ui_name": prop.name if prop else None,
        "relief_marks": counts,
        "description_mentions_alignment":
            bool(prop) and "alignment" in prop.description.lower(),
        "checks": {
            "off_by_default": prop is not None and prop.default is False,
            "no_relief_drawn_at_the_default": counts.get("default") == 0,
            "twenty_four_wedges_when_on": counts.get("on") == 24,
            "renamed_away_from_dart": bool(prop) and "Relief" in prop.name,
            # The description has to point at the real feature, or the next
            # person makes the same mistake.
            "points_at_the_alignment_feature":
                bool(prop) and "alignment" in prop.description.lower(),
        },
    }



def test_an_annulus_is_not_mistaken_for_its_own_hole():
    """A ring's outline is the big circle, whatever the vertex counts say.

    pole_of_inaccessibility takes [outer, hole, ...] and treats everything
    after the first entry as a hole, so the order is not a detail -- getting
    it wrong inverts the shape. Ordering by vertex count ties on a ring
    unwrapped from a sphere band, which has as many vertices inside as out.

    Asserted against arithmetic. A ring of outer radius 10 and inner radius 4
    has its widest point in the band, 3 from either boundary, at radius 7
    from the centre. Order the rings the wrong way round and the answer is
    not merely worse, it is negative: there is no region at all.
    """
    import math as _m
    outer = [(10.0 * _m.cos(2 * _m.pi * i / 24.0),
              10.0 * _m.sin(2 * _m.pi * i / 24.0)) for i in range(24)]
    inner = [(4.0 * _m.cos(2 * _m.pi * i / 24.0),
              4.0 * _m.sin(2 * _m.pi * i / 24.0)) for i in range(24)]

    ordered = core.order_rings_outer_first([inner, outer])
    (px, py), room = core.pole_of_inaccessibility(ordered)
    radius = (px * px + py * py) ** 0.5

    # What the old vertex-count ordering produced, kept so the failure is
    # visible rather than merely described.
    (_bx, _by), bad_room = core.pole_of_inaccessibility([inner, outer])

    return {
        "outer_first": ordered[0] is outer,
        "pole_radius_from_centre": round(radius, 2),
        "room": round(room, 2),
        "room_if_ordered_wrongly": round(bad_room, 2),
        "checks": {
            "the_bigger_ring_is_taken_as_the_outline": ordered[0] is outer,
            # Midway through the band: (10 + 4) / 2 = 7, give or take the
            # polygon's own flattening.
            "pole_sits_in_the_band": 6.5 <= radius <= 7.5,
            # Half the band width, 3, less the flattening.
            "room_is_about_half_the_band_width": 2.5 <= room <= 3.1,
            # And the wrong order really is catastrophic, not just worse.
            "wrong_order_finds_no_region": bad_room < 0.0,
        },
    }



def _gore_sphere(gores=6, segments=24, rings=16, radius=0.15):
    """A sphere peeled into pole-to-pole gores.

    The shape the reference prop is built from, and the right fixture for
    alignment marks: a gore's two long seams are what gets registered against
    its neighbours. Latitude bands are not -- their seams are closed rings,
    and a ring never runs out of seam to space marks along.
    """
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


def test_by_distance_puts_a_mark_every_n_units_of_seam():
    """The setting says every N units of seam, so it must mean the seam.

    It did not. Spacing was measured per cut run and restarted at every run
    boundary, so a run shorter than one step still got a mark at its middle
    and the gaps came out wherever the bevel angle happened to change. Asking
    for 30 mm and getting 12, 30, 9, 30 is not what the control says.

    Now the marks are walked along the seam -- the chain of shared 3D edges
    that two pieces are taped along -- which is the thing whose length the
    maker cares about. Checked by reconstructing each mark's distance along
    its chain and taking the gaps: on a 6-gore sphere every chain is a
    470.5 mm meridian, and at 30 mm every single gap must be 30 mm.
    """
    obj = _gore_sphere()
    bm = _bmesh_of(obj)
    try:
        uv_layer = bm.loops.layers.uv.active
        settings = _settings()
        saved = (settings.tick_mode, settings.tick_spacing)
        spacing = 0.030
        try:
            settings.tick_mode = 'BY_DISTANCE'
            settings.tick_spacing = spacing
            islands_faces = core.uv_islands(bm, uv_layer, only_selected=False)
            sheet = fx._Sheet(islands_faces, uv_layer, 1.0)
            islands = [fx._Island(i, f, uv_layer, sheet)
                       for i, f in enumerate(islands_faces)]
            chains = fx._seam_chains(islands)
            params = fx._seam_tick_params(islands, settings)
        finally:
            settings.tick_mode, settings.tick_spacing = saved

        gaps = []
        for chain in chains:
            lengths = [e.calc_length() for e in chain]
            starts, walked = {}, 0.0
            for idx, e in enumerate(chain):
                starts[e.index] = (walked, idx)
                walked += lengths[idx]
            positions = []
            for edge, t in params:
                if edge.index not in starts:
                    continue
                start_at, idx = starts[edge.index]
                entry = fx.chain_direction_start(chain, idx)
                canon = fx._edge_canonical_start(edge)
                local = t if entry is canon else 1.0 - t
                positions.append(start_at + local * lengths[idx])
            positions.sort()
            gaps.extend(positions[i + 1] - positions[i]
                        for i in range(len(positions) - 1))

        chain_mm = sorted({round(sum(e.calc_length() for e in c) * 1000, 1)
                           for c in chains})
        gaps_mm = [round(g * 1000, 2) for g in gaps]
        return {
            "seam_chains": len(chains),
            "chain_lengths_mm": chain_mm,
            "gaps": len(gaps_mm),
            "gap_min_mm": min(gaps_mm) if gaps_mm else None,
            "gap_max_mm": max(gaps_mm) if gaps_mm else None,
            "checks": {
                # One chain per gore boundary.
                "six_seam_chains": len(chains) == 6,
                # Every gap is the requested step, to a hundredth of a mm.
                "every_gap_is_the_requested_spacing":
                    bool(gaps_mm)
                    and abs(min(gaps_mm) - 30.0) < 0.01
                    and abs(max(gaps_mm) - 30.0) < 0.01,
                # Both bounds: a single mark per chain would leave no gaps at
                # all and pass the above vacuously.
                "there_are_gaps_to_measure": len(gaps_mm) >= 60,
            },
        }
    finally:
        bm.free()


def test_a_dash_is_never_drawn_twice_in_one_place():
    """One mark on the foam is one dash on the sheet.

    Two runs can reach the same point on a shared edge by slightly different
    arithmetic and record it as 0.666666 and 0.666667. Keyed on the parameter
    those are two marks; on a 30 mm edge they are 30 nanometres apart, which
    is one mark written down twice, and it printed as two dashes on top of
    each other. Measured on a 6-gore sphere before the fix: 12 such pairs.

    Positions are now merged by distance on the mesh rather than by parameter
    value. Checked in both modes, because the duplicate appeared in only one
    of them and a fix that quietly handled just that one would be half a fix.
    """
    import collections
    obj = _gore_sphere()
    bm = _bmesh_of(obj)
    try:
        uv_layer = bm.loops.layers.uv.active
        settings = _settings()
        saved = (settings.tick_mode, settings.tick_spacing,
                 settings.ticks_per_run)
        rows = {}
        try:
            for label, mode in (("by_distance", 'BY_DISTANCE'),
                                ("per_run", 'PER_RUN')):
                settings.tick_mode = mode
                settings.tick_spacing = 0.030
                settings.ticks_per_run = 2
                svg = fx.generate_svg(bm, uv_layer, settings, 1.0)
                g = re.search(r'<g id="align".*?</g>', svg, re.S)
                lines = re.findall(
                    r'<line x1="([-\d.]+)" y1="([-\d.]+)" '
                    r'x2="([-\d.]+)" y2="([-\d.]+)"',
                    g.group(0) if g else "")
                counts = collections.Counter(lines)
                rows[label] = {
                    "lines": len(lines),
                    "duplicates": sum(v - 1 for v in counts.values() if v > 1),
                }
        finally:
            (settings.tick_mode, settings.tick_spacing,
             settings.ticks_per_run) = saved

        out = dict(rows)
        out["checks"] = {
            "no_duplicates_by_distance": rows["by_distance"]["duplicates"] == 0,
            "no_duplicates_per_run": rows["per_run"]["duplicates"] == 0,
            # Both bounds: zero marks would have zero duplicates.
            "marks_were_drawn": rows["by_distance"]["lines"] > 0
            and rows["per_run"]["lines"] > 0,
        }
        return out
    finally:
        bm.free()



def test_alignment_ticks_are_a_real_length_not_a_fraction_of_the_sheet():
    """A pen mark is the same size whatever the pattern it is drawn on.

    Tick length was taken as a permille of the sheet, like the angle text,
    and that is wrong for this mark in a way that showed up on the foam: it
    gave a 13 mm tick on an 838 mm sheet, while the gore it was drawn on is
    only 30 mm wide near its ends. The marks from the piece's two seams then
    reached across and crossed -- 12 colliding pairs, measured -- and read as
    doubled ticks.

    Angle text scales because it is a label on a drawing and has to stay
    readable at any pattern size. A registration mark does not: it is a thing
    the maker draws on the material with a pen.

    Checked across two sheets of very different size, exactly as the ruler
    is, because that is the property being claimed.
    """
    rows = {}
    settings = _settings()
    saved = settings.tick_length
    try:
        settings.tick_length = 0.006
        for label, builder in (("gore_sphere", _gore_sphere),
                               ("seamed_cube", _seamed_cube)):
            obj = builder()
            bm = _bmesh_of(obj)
            try:
                uv_layer = bm.loops.layers.uv.active
                svg = fx.generate_svg(bm, uv_layer, settings, 1.0)
                g = re.search(r'<g id="align".*?</g>', svg, re.S)
                body = g.group(0) if g else ""
                lines = [tuple(map(float, m)) for m in re.findall(
                    r'<line x1="([-\d.]+)" y1="([-\d.]+)" '
                    r'x2="([-\d.]+)" y2="([-\d.]+)"', body)]
                box = re.search(r"viewBox='0 0 ([\d.]+) ([\d.]+)'", svg)
                lengths = [((c - a) ** 2 + (d - b) ** 2) ** 0.5
                           for a, b, c, d in lines]
                rows[label] = {
                    "sheet_mm": round(float(box.group(1)), 1),
                    "longest_tick_mm": round(max(lengths), 2) if lengths else 0,
                    "ticks": len(lines),
                }
            finally:
                bm.free()
    finally:
        settings.tick_length = saved

    a, b = rows["gore_sphere"], rows["seamed_cube"]
    out = dict(rows)
    out["checks"] = {
        # 6 mm plus a quarter-length tail on the far side = 7.5 mm.
        "gore_tick_is_the_requested_length":
            abs(a["longest_tick_mm"] - 7.5) < 0.1,
        "cube_tick_is_the_same_length":
            abs(b["longest_tick_mm"] - 7.5) < 0.1,
        # Both bounds: the two sheets really are different sizes, so the
        # equality above means the length is independent of the sheet.
        "the_sheets_differ_in_size": a["sheet_mm"] != b["sheet_mm"],
        "ticks_were_drawn": a["ticks"] > 0 and b["ticks"] > 0,
    }
    return out


def _set(**values):
    """Set scene settings; returns what they were, for `_set(**saved)`."""
    s = _settings()
    saved = {k: getattr(s, k) for k in values}
    for k, v in values.items():
        setattr(s, k, v)
    return saved


def _overlay_runs(obj):
    """The overlay's own runs for `obj`, through its own builder.

    The same call tools/golden_marks.py records: `_rebuild` fills the
    module's cache from an edit-mode BMesh. Returned as the cache's plain
    _RunMark tuples, so nothing outlives edit mode.
    """
    import bl_ext.user_default.foamcraft_drachis.foam_overlay as ov
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        ov._rebuild([(obj, bm, bm.loops.layers.uv.active)], _settings())
        return list(ov._cache.runs)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')


def _export_run_count(obj, matrix):
    """Runs on the exported sheet: one bracket closes every run."""
    bm = _bmesh_of(obj)
    try:
        svg = fx.generate_svg(bm, bm.loops.layers.uv.active, _settings(),
                              1.0, matrix)
    finally:
        bm.free()
    m = re.search(r'<g id="ticks".*?</g>', svg, re.S)
    return m.group(0).count("<line") if m else -1


def _seamed_prism():
    """An octagonal prism: both rims seamed, and one vertical edge.

    Three islands with known outlines. Each cap is a flat regular octagon,
    so it unwraps without distortion and its outline turns exactly 45
    degrees at every vertex. The side is a developable strip of 8 quads,
    unwrapping to a rectangle with four 90 degree corners.

    So the run count is arithmetic, not an impression. At a corner angle of
    35 (the default), every octagon vertex is a corner: 8 runs per cap, 4 on
    the strip, 20. At 60, no octagon vertex is: each cap rim has one
    dihedral (90) all the way round and no straightness limit, so it is one
    run, and the strip is still 4 -- 6 in all.

    The vertical seam is the edge whose lower vertex is smallest by
    coordinate, chosen geometrically rather than by index (CLAUDE.md:
    primitives do not number the same way twice).
    """
    _clear()
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=1.0, depth=2.0)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    verticals = []
    for e in bm.edges:
        a, b = e.verts
        if abs(a.co.z - b.co.z) < 1e-9:
            e.seam = True
        else:
            low = min((a.co, b.co), key=lambda c: c.z)
            verticals.append(((round(low.x, 6), round(low.y, 6)), e))
    min(verticals, key=lambda item: item[0])[1].seam = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def test_the_sheet_and_the_screen_break_runs_at_the_same_corners():
    """Corner Angle reaches the export, so paper and screen agree on runs.

    The export used to ask the core for runs with none of the overlay's
    arguments: no corner angle, no world matrix, no dart settings. With
    Corner Angle at its default the two agreed by coincidence (the setting's
    default is the core's); move the slider and the screen re-segmented
    while the sheet kept the default. Here it is moved, on a shape whose run
    counts are known at both values (see `_seamed_prism`).
    """
    obj = _seamed_prism()
    bm = _bmesh_of(obj)
    try:
        islands = len(core.uv_islands(bm, bm.loops.layers.uv.active,
                                      only_selected=False))
    finally:
        bm.free()
    saved = _set(corner_angle_deg=35.0, show_brackets=True)
    try:
        default_sheet = _export_run_count(obj, obj.matrix_world)
        default_screen = len(_overlay_runs(obj))
        _set(corner_angle_deg=60.0)
        moved_sheet = _export_run_count(obj, obj.matrix_world)
        moved_screen = len(_overlay_runs(obj))
    finally:
        _set(**saved)
    return {
        "islands": islands,
        "runs_at_35": {"sheet": default_sheet, "screen": default_screen},
        "runs_at_60": {"sheet": moved_sheet, "screen": moved_screen},
        "checks": {
            "three_islands": islands == 3,
            # The known answers, so agreement cannot be agreement on nonsense.
            "twenty_runs_at_35_on_the_sheet": default_sheet == 20,
            "twenty_runs_at_35_on_screen": default_screen == 20,
            "six_runs_at_60_on_the_sheet": moved_sheet == 6,
            "six_runs_at_60_on_screen": moved_screen == 6,
            "sheet_matches_screen_at_60": moved_sheet == moved_screen,
        },
    }


def test_unapplied_scale_gives_the_sheet_the_screens_run_angles():
    """An object with unapplied non-uniform scale: same runs, same angles.

    The overlay measures runs through the object's world matrix, because
    the straightness limit is a real length in metres and object-local
    coordinates understate a scaled piece. The export did not, so on a
    scaled object the sheet split runs in different places than the screen
    and printed different mean angles. Checked through the export operator
    itself, since passing the matrix is the operator's job.

    A gore sphere stretched 2.5x in z, with a straightness limit set: the
    meridian seams bow, and how far they bow depends on the stretch, so the
    matrix changes where runs break. Their dihedral varies with latitude, so
    a different break gives a different mean angle -- not just a different
    count. The last check is the one that makes the others mean something:
    measured without the matrix, the sheet must come out different.
    """
    import os
    import tempfile

    obj = _gore_sphere()
    obj.scale = (1.0, 1.0, 2.5)
    bpy.context.view_layer.update()
    # Decimals is capped at 2 by the property (measured: 3 reads back as 2),
    # so the screen side is formatted with whatever the setting holds.
    saved = _set(max_chord_deviation=0.004, decimals=2, show_brackets=True,
                 max_tilt_deg=89.0)
    path = os.path.join(tempfile.mkdtemp(prefix="foam_scale_"), "sheet.svg")
    try:
        bpy.context.view_layer.objects.active = obj
        op = bpy.ops.foamcraft_drachis.export_panel_pattern(filepath=path)
        with open(path, encoding="utf-8") as f:
            svg = f.read()
        screen = _overlay_runs(obj)
        object_space_runs = _export_run_count(obj, None)
        s = _settings()
        screen_angles = sorted(
            "%.*f" % (s.decimals, abs(m.angle_deg)) for m in screen
            if m.has_dihedral and not m.needs_heat_form
            and abs(m.angle_deg) >= s.min_angle)
    finally:
        _set(**saved)

    body = re.search(r'<g id="angles".*?</g>', svg, re.S)
    sheet_angles = sorted(re.findall(r'>([\d.]+)' + '\N{DEGREE SIGN}<',
                                     body.group(0) if body else ""))
    ticks = re.search(r'<g id="ticks".*?</g>', svg, re.S)
    sheet_runs = ticks.group(0).count("<line") if ticks else -1
    first_diff = next(((a, b) for a, b in zip(sheet_angles, screen_angles)
                       if a != b), None)
    return {
        "operator": sorted(op),
        "scale": tuple(obj.scale),
        "runs": {"sheet": sheet_runs, "screen": len(screen),
                 "sheet_in_object_space": object_space_runs},
        "angle_labels": {"sheet": len(sheet_angles),
                         "screen": len(screen_angles)},
        "first_differing_angle": first_diff,
        "checks": {
            "operator_finished": op == {'FINISHED'},
            "scale_is_non_uniform_and_unapplied":
                tuple(obj.scale) == (1.0, 1.0, 2.5),
            "sheet_has_the_screens_run_count": sheet_runs == len(screen),
            "sheet_has_the_screens_angles": sheet_angles == screen_angles,
            # Both bounds: equal and empty would pass the line above.
            "there_are_angles_to_compare": len(sheet_angles) >= 12,
            # The matrix has to matter on this fixture, or the agreement
            # above proves nothing about passing it.
            "object_space_would_disagree":
                object_space_runs != sheet_runs,
        },
    }


def _svg_and_marks_for_cube():
    """The seamed cube's SVG text and the Marks it was written from."""
    obj = _seamed_cube()
    bm = _bmesh_of(obj)
    try:
        uv_layer = bm.loops.layers.uv.active
        marks = fx.build_marks(bm, uv_layer, _settings(), 1.0)
        return fx.svg_from_marks(marks), marks
    finally:
        bm.free()


_SVG_NS = "{http://www.w3.org/2000/svg}"


def test_the_svg_has_no_css_to_misread():
    """Illustrator ignored the <style> block (CSS id selectors) and
    dominant-baseline, so strokes had no width or colour. Neither may appear.

    Both bounds: absent, AND the file is not empty (an empty file also has
    no <style>), so the outlines group must hold exactly six elements --
    one per island of the seamed cube.
    """
    import xml.etree.ElementTree as ET
    svg, marks = _svg_and_marks_for_cube()
    root = ET.fromstring(svg)   # also proves the file is well-formed XML
    # Keep a copy to open in other programs (screenshots/ is gitignored).
    import os
    out_dir = r"O:\Work\BlenderPlugin\FoamCraft\screenshots"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "export_svg_attributes.svg"), "w",
              encoding="utf-8") as fh:
        fh.write(svg)
    outlines = [g for g in root.iter(_SVG_NS + "g") if g.get("id") == "outlines"]
    n_outline = len(list(outlines[0])) if outlines else 0
    return {
        "outline_elements": n_outline,
        "checks": {
            "no_style_element": "<style" not in svg,
            "no_dominant_baseline": "dominant-baseline" not in svg,
            "no_css_class_hooks": "class=" not in svg,
            "six_outlines": n_outline == 6,
        },
    }


def test_every_drawn_element_states_its_own_paint():
    """Each shape must carry stroke/fill itself; a stroked one needs a width.

    Parsed with ElementTree, not regex, so attribute order cannot matter.
    The count is asserted > 0 so a parse that finds nothing cannot pass, and
    both stroked and unstroked (piece fills, stroke="none") are asserted
    present so neither branch goes unchecked.
    """
    import xml.etree.ElementTree as ET
    svg, marks = _svg_and_marks_for_cube()
    root = ET.fromstring(svg)
    shapes = [e for t in ("line", "polygon", "polyline", "path", "rect")
              for e in root.iter(_SVG_NS + t)]
    no_paint = [e.tag for e in shapes
                if e.get("stroke") is None and e.get("fill") is None]
    stroked = [e for e in shapes if e.get("stroke") not in (None, "none")]
    bad_width = [e.tag for e in stroked
                 if not (float(e.get("stroke-width", "0")) > 0)]
    unstroked = len(shapes) - len(stroked)
    return {
        "shapes": len(shapes), "stroked": len(stroked),
        "unstroked_fills": unstroked,
        "checks": {
            "shapes_were_found": len(shapes) > 0,
            "every_shape_has_stroke_or_fill": not no_paint,
            "stroked_shapes_exist": len(stroked) > 0,
            "every_stroke_has_a_positive_width": not bad_width,
            "fills_were_checked": unstroked > 0,
        },
    }


def test_every_text_states_its_own_font():
    """font-size > 0, fill, font-family and text-anchor on every <text>."""
    import xml.etree.ElementTree as ET
    svg, marks = _svg_and_marks_for_cube()
    root = ET.fromstring(svg)
    texts = list(root.iter(_SVG_NS + "text"))
    missing = [t.text for t in texts
               if not all(t.get(a) for a in
                          ("fill", "font-family", "text-anchor"))]
    small = [t.text for t in texts if not float(t.get("font-size", "0")) > 0]
    return {
        "texts": len(texts),
        "checks": {
            "texts_were_found": len(texts) > 0,
            # Every label in the Marks reached the file: none dropped.
            "one_text_per_label": len(texts) == len(marks.labels),
            "every_text_has_fill_family_anchor": not missing,
            "every_font_size_is_positive": not small,
            "all_anchored_middle": all(t.get("text-anchor") == "middle"
                                       for t in texts),
        },
    }


def test_a_middle_label_baseline_is_shifted_numerically():
    """Without dominant-baseline, a 'middle' label's y is its centre plus
    0.35 * size (cap height ~0.7 * size, so half of it). The file's y must
    equal the Marks label's y + 0.35 * size, within 1e-3 (the writer rounds
    to 3 places). A 'baseline' label's y is unchanged; the ruler's captions
    are such, so both branches are measured.
    """
    import xml.etree.ElementTree as ET
    svg, marks = _svg_and_marks_for_cube()
    root = ET.fromstring(svg)
    group = next(g for g in root.iter(_SVG_NS + "g") if g.get("id") == "numbers")
    ys = sorted(float(t.get("y")) for t in group)
    numbers = [l for l in marks.labels if l.kind == 'number']
    want = sorted(l.y + 0.35 * l.size_mm for l in numbers)
    err = max((abs(a - b) for a, b in zip(ys, want)), default=1e9)

    ruler = next(g for g in root.iter(_SVG_NS + "g") if g.get("id") == "ruler")
    base = sorted(float(t.get("y")) for t in ruler.iter(_SVG_NS + "text"))
    want_base = sorted(l.y for l in marks.labels
                       if l.kind == 'ruler' and l.valign == 'baseline')
    base_err = max((abs(a - b) for a, b in zip(base, want_base)),
                   default=1e9)
    return {
        "numbers": len(numbers), "max_error": err,
        "ruler_captions": len(want_base), "ruler_error": base_err,
        "checks": {
            "six_numbers": len(numbers) == 6 and len(ys) == 6,
            "middle_shift_applied": err < 1e-3,
            # Both bounds: the shift is really non-zero here, so an
            # unshifted writer cannot satisfy the equality above.
            "shift_is_nonzero": all(0.35 * l.size_mm > 1e-3 for l in numbers),
            "baseline_labels_unshifted":
                len(want_base) > 0 and len(base) == len(want_base)
                and base_err < 1e-3,
        },
    }


TESTS = (
    test_the_cube_fixture_is_what_we_think,
    test_the_sheet_reproduces_the_uv_layout,
    test_every_run_gets_an_angle_label,
    test_the_angle_threshold_is_not_what_hides_the_labels,
    test_the_labels_are_large_enough_to_read_on_the_sheet,
    test_every_piece_number_lands_inside_its_piece,
    test_a_crescent_piece_gets_its_number_inside,
    test_an_annulus_is_not_mistaken_for_its_own_hole,
    test_island_fill_colours_are_distinct_and_toggleable,
    test_the_scale_ruler_is_in_real_millimetres,
    test_alignment_ticks_are_paired_across_every_seam,
    test_tick_spacing_responds_to_both_modes,
    test_by_distance_puts_a_mark_every_n_units_of_seam,
    test_a_dash_is_never_drawn_twice_in_one_place,
    test_alignment_ticks_are_a_real_length_not_a_fraction_of_the_sheet,
    test_relief_cuts_are_off_by_default,
    test_the_sheet_and_the_screen_break_runs_at_the_same_corners,
    test_unapplied_scale_gives_the_sheet_the_screens_run_angles,
    test_the_svg_has_no_css_to_misread,
    test_every_drawn_element_states_its_own_paint,
    test_every_text_states_its_own_font,
    test_a_middle_label_baseline_is_shifted_numerically,
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
