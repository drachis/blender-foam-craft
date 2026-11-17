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


TESTS = (
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
