"""Record every mark FoamCraft draws, deterministically, for refactoring against.

    python tools/golden_marks.py > golden.log

A refactor that must change nothing cannot be checked by looking at a
pattern. Two exports of "the same" sheet are not comparable by eye -- there
are hundreds of marks and a moved label looks like a different label. So the
drawing is written down as text, one line per mark, in a fixed order, and the
lines are compared.

WHAT IS RECORDED, AND WHY THOSE THINGS

The thing being refactored is mark *generation*: three places independently
work out what to draw (the overlay for the screen, generate_svg for the
sheet, and a raster exporter). The refactor collapses them onto one model, so
the baseline has to pin what each currently produces:

  * `overlay`  -- every field of every _RunMark the overlay builds, plus its
                  piece-number labels. This is the drawing the user actually
                  trusts, and the one the refactor must not disturb.
  * `svg`      -- the element count of each group in the exported sheet, and
                  a checksum of the whole file. Counts say *what* changed;
                  the checksum catches a change the counts cannot see.

DETERMINISM

Blender does not hand back the same mesh twice. `primitive_uv_sphere_add`
renumbers faces on every call (a weld ends it, and welds do not order their
output), and edge vertex order moves too -- both measured, both recorded in
CLAUDE.md. So:

  * every fixture is rebuilt from scratch, never inherited;
  * floats are rounded to six places, well past what any of this means
    physically, so last-bit noise is not mistaken for a change;
  * records are sorted by their own text before being numbered, so the order
    of islands or runs cannot leak into the comparison.

The sort is what makes this work at all. Without it a run that produced
identical marks in a different order would read as every line differing.
"""

import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

# NOTE: this template goes through str.format, so every literal brace in it
# must be doubled -- a dict or set literal written normally is read as a
# format field and fails with a KeyError naming its first key.
RECORDER = r'''
import hashlib
import math
import sys

import bpy
import bmesh

sys.path.insert(0, r"{tools}")
import sandbox_guard

_MOD = "bl_ext.user_default.foamcraft_drachis"
for _n in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_n]
import bl_ext.user_default.foamcraft_drachis.foam_angle_core as core
import bl_ext.user_default.foamcraft_drachis.foam_export as fx
import bl_ext.user_default.foamcraft_drachis.foam_overlay as ov


def _r(value):
    """Six places. Finer than that is noise in every number here."""
    return "%.6f" % (value + 0.0)


# ---------------------------------------------------------------- fixtures

def _cube_all_seamed():
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        e.seam = True
    bmesh.update_edit_mesh(obj.data)
    return obj


def _cube_half_seamed():
    """Only the vertical edges, so some boundary and some interior."""
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        a, b = e.verts
        if abs(a.co.z - b.co.z) > 1e-9:
            e.seam = True
    bmesh.update_edit_mesh(obj.data)
    return obj


def _sphere_bands():
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, segments=16, ring_count=8)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for e in bm.edges:
        a, b = e.verts
        if abs(a.co.z - b.co.z) < 1e-9:
            e.seam = True
    bmesh.update_edit_mesh(obj.data)
    return obj


def _sphere_gores():
    gores = 6
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, segments=24,
                                         ring_count=16)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    step = 2.0 * math.pi / gores

    def phi(v):
        if abs(v.co.x) < 1e-9 and abs(v.co.y) < 1e-9:
            return None
        return math.atan2(v.co.y, v.co.x) % (2.0 * math.pi)

    for e in bm.edges:
        a, b = e.verts
        if abs(a.co.z - b.co.z) < 1e-9:
            continue
        pa, pb = phi(a), phi(b)
        p = pa if pa is not None else pb
        if p is None:
            continue
        if pa is not None and pb is not None:
            if min(abs(pa - pb), 2 * math.pi - abs(pa - pb)) > 1e-6:
                continue
        if abs(p - round(p / step) * step) < 1e-6:
            e.seam = True
    bmesh.update_edit_mesh(obj.data)
    return obj


def _torus():
    """CLAUDE.md's craftability reference: seams on both loops."""
    bpy.ops.mesh.primitive_torus_add(major_segments=12, minor_segments=5,
                                     major_radius=0.3, minor_radius=0.1)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    for i, e in enumerate(bm.edges):
        if i % 3 == 0:
            e.seam = True
    bmesh.update_edit_mesh(obj.data)
    return obj


def _normalise_faces(obj):
    """Put the face table in an order that depends only on the geometry.

    Blender's primitives do not hand back the same table twice --
    `primitive_uv_sphere_add` ends in a weld and welds do not order their
    output, measured and recorded in CLAUDE.md. Island order follows face
    order, piece numbers follow island order, and fill colours follow the
    island index, so an unsorted table means the same mesh is numbered and
    coloured differently on every run.

    Sorted by each face's vertex coordinates, not by vertex index: the
    indices move for the same reason the faces do. bmesh's own sort wants a
    number, so the Python-side ranking is handed to it.
    """
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    def geometry_key(face):
        return tuple(sorted(tuple(round(c, 5) for c in v.co)
                            for v in face.verts))

    ranked = sorted(bm.faces, key=geometry_key)
    rank = {{f.index: i for i, f in enumerate(ranked)}}
    bm.faces.sort(key=lambda f: rank[f.index])
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    bmesh.update_edit_mesh(obj.data)


FIXTURES = (
    ("cube_all", _cube_all_seamed),
    ("cube_half", _cube_half_seamed),
    ("sphere_bands", _sphere_bands),
    ("sphere_gores", _sphere_gores),
    ("torus", _torus),
)

# Settings swept per fixture. Each one reaches code the others do not: a
# path the sweep never takes is a path this proves nothing about.
SWEEPS = (
    ("default", {{}}),
    ("perrun", {{"tick_mode": 'PER_RUN', "ticks_per_run": 2}}),
    ("bydist", {{"tick_mode": 'BY_DISTANCE', "tick_spacing": 0.03}}),
    ("nofill", {{"island_fill": False}}),
    ("darts", {{"show_darts": True}}),
    ("noticks", {{"show_align_ticks": False}}),
)

DEFAULTS = {{
    "tick_mode": 'PER_RUN',
    "ticks_per_run": 2,
    "tick_spacing": 0.025,
    "island_fill": True,
    "show_darts": False,
    "show_align_ticks": True,
    "show_island_ids": True,
    "show_brackets": True,
    "show_scale_ruler": True,
    "only_selected": False,
    "tick_numbers": False,
}}

records = []

for name, build in FIXTURES:
    for sweep_name, overrides in SWEEPS:
        sandbox_guard.clear_scene()
        obj = build()
        _normalise_faces(obj)
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.03)
        # Freeze the unwrap. ANGLE_BASED is an iterative solver and its
        # packing is not bit-identical between runs, so the same fixture
        # yields UVs that differ in their last places -- enough to move a
        # pole, a tick or a run boundary and make every comparison below
        # meaningless. Rounding absorbs the solver's noise without changing
        # the layout it found.
        _bm_freeze = bmesh.from_edit_mesh(obj.data)
        _uv_freeze = _bm_freeze.loops.layers.uv.active
        for _f in _bm_freeze.faces:
            for _l in _f.loops:
                _uvv = _l[_uv_freeze].uv
                _uvv.x = round(_uvv.x, 4)
                _uvv.y = round(_uvv.y, 4)
        bmesh.update_edit_mesh(obj.data)

        s = bpy.context.scene.foamcraft_drachis
        for key, value in DEFAULTS.items():
            setattr(s, key, value)
        for key, value in overrides.items():
            setattr(s, key, value)

        tag = "%s/%s" % (name, sweep_name)

        # --- the overlay's own marks, through its own builder -------------
        bm_edit = bmesh.from_edit_mesh(obj.data)
        uv_edit = bm_edit.loops.layers.uv.active
        # _rebuild returns nothing; it fills the module's _cache. Read that.
        try:
            ov._rebuild([(obj, bm_edit, uv_edit)], s)
            runs = list(ov._cache.runs)
            labels = list(ov._cache.island_labels)
            heat = set(ov._cache.stretch_islands)
        except Exception as exc:
            records.append("%s|overlay|ERROR|%s" % (tag, exc))
            runs, labels, heat = [], [], set()

        for m in runs:
            # Every field of the mark, including the shape of its outline and
            # dart lists. A refactor that moved a bracket by a millimetre or
            # dropped a dart has to show up here.
            outline = ";".join(",".join(_r(v) for v in pt) for pt in m.outline)
            darts = ";".join(",".join(_r(v) for v in d) for d in m.darts)
            records.append("|".join((
                tag, "overlay_run",
                _r(m.ax), _r(m.ay), _r(m.bx), _r(m.by),
                _r(m.angle_deg), _r(m.per_side_deg),
                str(int(bool(m.has_dihedral))),
                str(int(bool(m.needs_heat_form))),
                _r(m.ox), _r(m.oy),
                _r(m.length_3d),
                "outline=" + outline, "darts=" + darts)))
        for lab in labels:
            records.append("|".join((
                tag, "overlay_label", _r(lab[0]), _r(lab[1]), str(lab[2]))))
        records.append("%s|overlay_counts|runs=%d|labels=%d|heat=%d"
                       % (tag, len(runs), len(labels), len(heat)))

        bpy.ops.object.mode_set(mode='OBJECT')

        # --- the exported sheet -------------------------------------------
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.edges.index_update()
        uv = bm.loops.layers.uv.active
        try:
            svg = fx.generate_svg(bm, uv, s, 1.0)
        except Exception as exc:
            svg = "ERROR %s" % exc
        bm.free()

        import re as _re
        for gid in ("fills", "outlines", "darts", "ticks", "align",
                    "alignlabels", "angles", "numbers", "heatform", "ruler"):
            m = _re.search(r'<g id="%s".*?</g>' % gid, svg, _re.S)
            body = m.group(0) if m else ""
            records.append("%s|svg_group|%s|n=%d" % (
                tag, gid,
                body.count("<line") + body.count("<text")
                + body.count("<polygon") + body.count("<path")
                + body.count("<rect")))
        box = _re.search(r"viewBox='0 0 ([\d.]+) ([\d.]+)'", svg)
        if box:
            records.append("%s|svg_sheet|%s|%s" % (tag, box.group(1),
                                                   box.group(2)))
        # The whole file, byte for byte. It was briefly hashed over sorted
        # lines to tolerate an element ordering that moved between runs --
        # which was the wrong call: that ordering was a bug in
        # island_boundary_cycles, not a fact to work around, and sorting
        # would have hidden it. Fixed there; strict here.
        records.append("%s|svg_sha|%s" % (
            tag, hashlib.sha256(svg.encode("utf-8")).hexdigest()[:16]))

        bpy.ops.object.mode_set(mode='EDIT')

# Sorted before numbering. Island and run order is not part of the drawing --
# two runs that produce the same marks in a different order are the same
# drawing -- and without this every line would read as changed.
records.sort()
lines = ["%06d|%s" % (i + 1, rec) for i, rec in enumerate(records)]

result = {{"lines": lines, "count": len(lines)}}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=9990,
                        help="port for the Blender this starts and stops")
    parser.add_argument("--keep", action="store_true",
                        help="leave Blender running afterwards")
    args = parser.parse_args()

    launcher = os.path.join(ROOT, "tools", "launch_blender.py")
    started = subprocess.run(
        [sys.executable, launcher, "--port", str(args.port)],
        capture_output=True, text=True)
    if started.returncode != 0:
        sys.stderr.write(started.stdout + started.stderr)
        return 2

    try:
        import blender_bridge
        code = RECORDER.format(tools=os.path.join(ROOT, "tools"))
        reply = blender_bridge.execute(code, port=args.port, timeout=1800)
        if reply.get("status") != "ok":
            sys.stderr.write("recorder failed: %s\n"
                             % reply.get("message", reply))
            if reply.get("stderr"):
                sys.stderr.write(reply["stderr"])
            return 2
        for line in reply["result"]["lines"]:
            print(line)
    finally:
        if not args.keep:
            subprocess.run([sys.executable, launcher, "--port",
                            str(args.port), "--stop"],
                           capture_output=True, text=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
