# Blender Fold-Angle Tooling — Project Context

## Working practices (read first)

**Ask before inventing. Asking costs nothing and is the preferred move.**
The user very often already knows the answer, and a question is cheaper than
any of: reading source, guessing, or building something that has to be undone.
Asking is not an interruption and does not annoy — it is the ideal path. Treat
"I'll design something here" as the signal to stop and ask instead.

Worked example (2026-08-10): the flat layout preview grew a bespoke shelf
packer — gaps, shelf widths, wrap-around — to arrange pieces in the viewport.
It was invented, it was buggy (offsets were added to UV positions rather than
replacing them, so pieces overlapped), and it was unnecessary: the answer was
"use u→x, v→y scaled to the mesh", because the UV layout *is* the arrangement
and the user had already packed it deliberately. One question would have saved
the whole detour and the bug in it.

This generalises. Prefer, in order: ask the user, use the built-in, read the
docs, read the source. Inventing is last, and it needs a stated reason.

**Every check goes in `tests/`, never in a scratch file.**
A measurement written to answer one question is worth keeping, because the
next change to that code needs the same question asked again. Writing it
fresh each time costs more than reading it, loses the reasoning about *why*
that shape or that number was chosen, and leaves no record of what has
actually been verified.

So: no one-off probes. Add the case to the relevant file in `tests/`, with a
comment saying what answer is expected and where that expectation comes from.
When a check turns out to be wrong, fix it in place — the diff then shows how
the understanding changed, which is usually the more valuable half.

    python tools/launch_blender.py --port 9885
    python tools/blender_bridge.py --port 9885 < tests/test_gp_pattern.py
    python tools/blender_bridge.py --port 9885 < tests/test_curvature_darts.py

Two rules the suite already enforces, both learned the hard way:

- **State both bounds.** A check that can only fail in one direction is not a
  check. "Two islands" is asserted `== 2`, never `> 1` — which an
  all-edges-seamed mesh also satisfies — and never `< n_faces`, which anything
  satisfies. This project once reported "2256 of 2256 edges selected" as a
  pass, when the cutter had produced nothing and the measurement had fallen
  through to the whole mesh.
- **Prefer a shape with a known answer.** Assert against arithmetic, not
  against an impression. A cylinder has zero Gaussian curvature, a sphere
  totals 4π, a torus totals zero, a cone's defect is the pie slice you cut
  from a paper disc. Those are checks a wrong result cannot survive; "looks
  about right" is exactly how the dart parameter came to be wrong three times
  running.

**Check the real API and docs before designing around a problem.** Blender
almost always already has the feature. Read the current API reference or probe
the live API before concluding something isn't possible. A "clever" workaround
that reimplements or sidesteps an existing built-in is a defect, not a
solution — it will drift out of maintenance and break between releases.

**When blocked, ask — do not work around it.** If a planned approach appears to
hit a wall, stop and raise it as a question rather than silently substituting a
different design. In auto/unattended mode: park the problem, continue with
other work that isn't blocked, and collect the question for the end of the
run so the user can decide the path forward. Never let an unreviewed
architectural substitution land quietly.

Worked example of the failure mode (2026-08-09): the plan called for the
Decimate modifier in Planar/Dissolve mode to generate candidate dart runs.
This was abandoned mid-build on the grounds that decimated geometry has no
public `origindex` mapping back to the original UVs — and a bespoke boundary
walk was written instead. Both premises were wrong. The Decimate modifier has a
**Delimit** option (`delimit={'UV','SEAM',...}`) that preserves UV island and
seam boundaries, and the *evaluated mesh keeps its UVMap*, so no index mapping
back to the original was ever needed: the simplified boundary can be read
directly from the evaluated mesh. The correct move was to ask, or to read the
modifier's properties, before substituting a hand-rolled algorithm.

**Never touch mesh data from a draw handler.** This crashed Blender and
corrupted geometry, twice, before the cause was understood. The rule is not
"do less work in the draw callback" — it is that a renderer must not read the
model at all. Blender's own UV overlay obeys this: `MeshUVs::edit_object_sync`
in `draw/engines/overlay/overlay_mesh.hh` only submits GPU batches built
earlier by the extractors in `draw/intern/mesh_extractors/`, and invalidation
happens through a dirty tag, never by inspecting the mesh mid-draw. There is
also **no position-only fast path** — a UV transform tags
`BKE_MESH_BATCH_DIRTY_ALL` through the depsgraph and everything is
re-extracted. So the problem was never the *amount* of work; it was doing any
of it at the wrong moment.

The working shape in Python, since add-ons cannot hook the extraction pipeline:

| Piece | Does | Touches mesh |
|---|---|---|
| Draw handler | renders cached plain data | **never** |
| `depsgraph_update_post` | raises a dirty flag, nothing else | **never** |
| `bpy.app.timers` tick | rebuilds when dirty, and never mid-drag | yes, at a safe point |

Two dead ends worth not repeating: splitting the cache into cheap and expensive
halves still read the mesh from the draw handler and turned a freeze into
corruption; moving that read into a depsgraph handler was no better, because
during a modal transform it fires every frame, mid-mutation. A timer fires
*between* operator executions, which is the closest thing Python has to
Blender's sync point.

Related rules learned the same way:
- **Never hold `BMFace`/`BMEdge`/`BMLoop` beyond the call that fetched them.**
  Freed BMesh elements crash outright rather than raising, so `try/except`
  cannot save you. Copy out floats.
- **`ensure_lookup_table()` mutates the bmesh.** Calling it speculatively from
  a draw or depsgraph callback during a transform is itself a corruption
  source. Call it where topology actually changed.
- **Never draw known-stale data.** A flag set by the timer cannot fix a stale
  frame, because the draw beats the timer. Refuse to draw while dirty instead.
- **Detect transforms, not modal operators.** The UV editor keeps
  `IMAGE_OT_view_pan` registered as a modal handler permanently, so testing for
  "any modal operator" reports a drag forever. Match `TRANSFORM_OT_`.

**Interactive features can only be verified interactively.** This is why the
project favours BDD over unit tests. Every scripted check passed while the
overlay was crashing, because none of them entered a modal loop: scripted
`bpy.ops.transform.rotate(value=...)` executes immediately and never drags.
Nudging UV coordinates from a script is not a transform. Until a human has
dragged a UV by hand, an interactive fix is not verified — say so plainly
rather than reporting it fixed.

**Verify the verification.** Three separate checks reported the draw path
wrongly in one session: `inspect.getsource` served stale lines from `linecache`
after a file edit, and two string slices ran past the end of the function. A
claim of "fixed" survived a crash twice because the measurement was as broken
as the code. Extract real function bounds, prefer reading state back out of
Blender over introspecting source text, and distrust a check that has never
failed.

**Rebuild scene state for every measurement.** Inherited state produced wrong
numbers twice. An object named `Torus` once held a 500-face Suzanne left by an
earlier test, and run counts moved between supposedly identical runs. Delete
everything and rebuild, then assert what you built — a 12x5 torus has 60 faces;
if it says 500, stop.

**Prefer documentation to reading the source tree.** A source-reading agent
burned 106k tokens on questions the manual answers. Read the docs first; go to
`blender/` (the submodule) only for questions documentation genuinely does not
cover, and target exact files rather than grepping broadly.

**Treat the pre-existing scripts as unverified.** `uv_fold_angle_overlay.py`
and `export_paper_model_angles.py` were drafted by an LLM without access to
Blender API references, so they contain outdated and hallucinated API usage.
Verify every call against the live 5.2 API before trusting it. Two confirmed
examples are recorded under "Verified against live Blender 5.2" below.

## Background
This project supports a physical costume/prop fabrication workflow: a creature
horn/head sculpt is being covered in individually-cut scale/scute panels (each
labeled with a piece number and a bevel-cut angle in degrees, currently done
by hand on paper). The goal is to move that pattern-and-angle work into
Blender so it's computed rather than hand-drafted.

Blender version target: **5.2 LTS** (released July 14, 2026).

## Files in this folder

### `uv_fold_angle_overlay.py` — primary, most recent, in-progress
A Blender add-on that draws a **live overlay directly in the UV Editor**:
a colored marker + the fold/bevel angle (in degrees) at every edge, computed
from the real 3D dihedral angle between the two faces sharing that edge.
Not an export — a visual aid for tuning the unwrap/seam layout by eye.
Sidebar panel (N key, "Fold Angles" tab) exposes convex/concave color,
font size, decimal places, label offset, marker size, min-angle threshold,
and an "only selected faces" declutter toggle.

**Status: written against verified Blender API docs, but never run inside an
actual Blender session.** This is the first thing to test.

Known-risky spots to check first in a live session:
- `region.view2d.view_to_region(x, y, clip=False)` behavior at extreme
  zoom levels in the UV editor (the bail-out guard for absurd values in
  `draw_callback()` has not been exercised against real data).
- ~~`gpu.shader.from_builtin('2D_UNIFORM_COLOR')`~~ — **resolved: this was
  broken.** See "Verified against live Blender 5.2.0 LTS" above.
- ~~`blf.size(font_id, size)`~~ — resolved, 2-arg confirmed live.
- Whether `bm.loops.layers.uv.active` reliably matches what's actually
  displayed in the UV Editor when multiple UV maps exist. **Still open.**

**Superseded:** this file now lives in the `foam_craft/` package as
`foam_craft/uv_fold_angle_overlay.py`, with its geometry moved to
`foam_craft/foam_angle_core.py`. Both bugs above are fixed there, and the
overlay has been confirmed drawing live against Suzanne UVs.

### `export_paper_model_angles.py` — secondary, companion export
A separate add-on that reuses Blender's bundled "Export Paper Model"
add-on's own `Unfolder`/`Mesh`/`Edge` classes (does not reimplement
unfolding, cutting, or page-packing) and adds an SVG export that labels
the angle at every edge **including cut boundaries between separate
pieces** — which the stock Export Paper Model add-on deliberately does
not do (it only colors angle info on folds *within* a single piece, since
plain paper folding doesn't need a bevel number at a cut).

Requires the "Export Paper Model" add-on to be enabled first
(Preferences > Add-ons > search "Paper Model"). Locates it at runtime via
duck-typing on loaded modules (`hasattr(mod, "Unfolder")` etc.), since its
exact module name varies between legacy installs and Blender 4.2+'s
Extensions Platform.

**Status: written directly against the add-on's real source** (fetched and
read via `curl` from `raw.githubusercontent.com/blender/blender-addons/main/io_export_paper_model.py`,
not from memory) but likewise never run live.

## Verified against live Blender 5.2.0 LTS (2026-08-09)
Measured directly in a running session, not inferred from docs.

- **`gpu.shader.from_builtin('2D_UNIFORM_COLOR')` does not exist on 5.2** — it
  raises `ValueError`. The `2D_`-prefixed names were removed. Valid names are
  `FLAT_COLOR`, `IMAGE*`, `SMOOTH_COLOR`, `UNIFORM_COLOR`, `POLYLINE_*`,
  `POINT_*`. Use `UNIFORM_COLOR`; it batches for both `POINTS` and `LINES`.
  The original overlay could never have drawn a marker.
- **The original hand-rolled convex/concave sign test was inverted.** It derived
  its sign from `edge.verts[1].co - edge.verts[0].co`, but BMEdge vertex order
  has no fixed relationship to the winding of the adjacent faces. Measured: it
  disagreed with Blender on exactly half of a torus's 192 interior edges, and
  was inverted on all 12 edges of a cube (every one of which is convex).
  Use `BMEdge.calc_face_angle_signed(fallback)` instead. Convention pinned three
  ways: cube = +90 (convex), normals-flipped cube = -90, valley crease = -90.
  So **positive = convex/mountain, negative = concave/valley**.
- `blf.size(font_id, size)` — 2-arg form is correct; the 3-arg (`dpi`) form
  raises `TypeError`.
- `Decimate` modifier: `decimate_type` in `{COLLAPSE, UNSUBDIV, DISSOLVE}`;
  `DISSOLVE` exposes `angle_limit`, `use_dissolve_boundaries`, and a flag-set
  `delimit` of `{NORMAL, MATERIAL, SEAM, SHARP, UV}`. **The evaluated mesh keeps
  its UVMap**, so simplified boundaries can be read straight from the evaluated
  mesh — there is no need to map back to original indices (and no public
  `origindex` API to do so anyway).
- Decimate DISSOLVE genuinely simplifies curved surfaces; a subdivided cube
  (1536 faces) dissolves to 204 at 10° and 55 at 20°. **Note:** when driving it
  from Python you must call `depsgraph.update()` after changing `angle_limit`,
  or `evaluated_get()` returns the previous result and it looks like a no-op.
- `mathutils.bvhtree.BVHTree.FromObject(obj, depsgraph).find_nearest(pt)` works
  for closest-point reprojection.
- `mesh.attributes.new(name, type='INT', domain='FACE')` works for per-face
  integer piece IDs.
- Blender Labs MCP add-on socket protocol (port 9876): newline is not the
  delimiter — requests are **null-byte terminated** JSON,
  `{"type": "execute", "code": str, "strict_json": bool}` + `b"\0"`, and the
  executed code must assign a JSON-serializable dict to `result`.
- **`Material.use_nodes = False` silently does not stick.** On a material from
  `bpy.data.materials.new()`, assigning `False` raises no exception but
  `mat.use_nodes` reads back `True` immediately after. Presumably EEVEE Next in
  5.2 requires node-based materials; the legacy non-node diffuse workflow is
  gone. Consequence: to make a material show a **flat solid color in every
  viewport shading mode**, set *both* `mat.diffuse_color` (used by Solid mode's
  "Material Color") *and* the Principled BSDF node's `Base Color` input (used by
  Material Preview and Rendered). `diffuse_color` alone is not enough, and
  turning nodes off to force it is not an option.

## Good test shapes
- **Torus, 12 major x 5 minor** — the craftability reference case: it needs
  seams on *both* major and minor loops before it can be built from flat foam.
  Not a useful test for Decimate.
- **Subdivision surface then Decimate** — the better demonstration of
  curvature-driven simplification.
- Cube with every edge seamed — exact known values: 6 islands, 4 neighbours
  each, 24 boundary runs, all at exactly 90°.
- Suzanne — irregular organic surface for realistic island/UV behaviour.

## What's already been verified (don't re-derive from scratch)
- Export Paper Model's `Edge.calculate_angle()` computes the signed dihedral
  angle (`self.angle`, in radians, positive=convex/mountain, negative=
  concave/valley) for every edge with two connected faces.
- Its SVG exporter (`Svg.write()`) uses that angle only to color/style fold
  lines within a single unbroken piece — never prints the numeric angle,
  and never labels cut/boundary edges at all.
- `SpaceImageEditor.draw_handler_add(callback, args, 'WINDOW', 'POST_PIXEL')`
  is the correct pattern for a UV Editor overlay.
- Blender addon RGBA color-picker convention (confirmed from Export Paper
  Model's own source):
  `bpy.props.FloatVectorProperty(default=(...,...,...,1.0), min=0, max=1, subtype='COLOR', size=4)`

## Testing approach for this session
Use Blender MCP (community `blender-mcp` project, or Anthropic's official
Blender connector which is built on it) to iterate live against a running
Blender 5.2 instance rather than manual copy/paste into Blender's Python
console:

1. Install/enable both add-ons in a real Blender 5.2 session.
2. Via the MCP `execute_blender_code`-equivalent tool: create a simple test
   mesh with visible curvature (e.g. a subdivided+sculpted cube or a
   pre-existing horn asset), mark seams, unwrap it, and confirm the overlay
   draws without errors.
3. Check whether the connected MCP tool's screenshot capability covers the
   UV/Image Editor specifically (confirmed to cover the 3D viewport; UV
   editor coverage was unconfirmed as of the prior session) — if not,
   `execute_blender_code` can likely call Blender's own screenshot
   operator against a specific editor area directly as a workaround.
4. Fix whatever the live session reveals, prioritizing the "known-risky
   spots" list above.

### Setup notes for Blender MCP
- Only one MCP client can hold Blender's local socket connection at a time
  — don't run Claude Desktop and Claude Code against the same Blender
  instance simultaneously.
- The MCP server executes arbitrary Python in Blender with no built-in
  guard against destructive changes. Work against a throwaway/disposable
  `.blend` file during iteration, not the actual prop asset file.
- Community server setup: `claude mcp add blender uvx blender-mcp` (requires
  the `uv` package manager), then enable `addon.py` inside Blender and
  start its server from the BlenderMCP panel in the 3D viewport sidebar (N).
