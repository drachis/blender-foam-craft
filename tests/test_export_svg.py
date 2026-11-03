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
    -- and every one of the 528 runs is labelled, at the default setting and
    at zero alike. The threshold is not touching them.

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


TESTS = (
    test_the_cube_fixture_is_what_we_think,
    test_the_sheet_reproduces_the_uv_layout,
    test_every_run_gets_an_angle_label,
    test_the_angle_threshold_is_not_what_hides_the_labels,
    test_the_labels_are_large_enough_to_read_on_the_sheet,
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
