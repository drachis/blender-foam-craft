# Design: stop keying collections on BMesh elements

## The decision

FoamCraft should carry **ordered sequences and indices**, the way Blender's
own code does, instead of sets and dicts keyed on `BMVert` / `BMFace` /
`BMEdge` / `BMLoop`.

This is a direction, not a diagnosis. The diagnosis is in
[RESEARCH_theta_seam_edge_nondeterminism.md](RESEARCH_theta_seam_edge_nondeterminism.md)
and in CLAUDE.md's "What survives rebuilding a BMesh" entry; read those first.

## Why, in one paragraph

Three separate reproducibility bugs shipped in 0.02 — a boundary walk, a
curvature routing, and a seam walk — and all three were the same mistake:
iterating or seeding from a collection keyed on BMesh elements. Each was
fixed in place with a geometric sort. That works, but it treats the symptom
three times rather than the cause, and there are **38 more element-keyed
collections** in the package waiting to do it a fourth time.

## What was measured, so it is not re-derived

- `BMVert.__hash__` is its own slot wrapper, not `object.__hash__`, and
  `hash(v) != id(v) // 16`. The hash tracks the element's address in
  *Blender's* memory, not the Python wrapper's — four vertices gave four
  hashes while `id()` repeated one address three times.
- Set iteration is therefore plain CPython table order over those hashes:
  sorting 42 vertices by `(hash % 128, hash)` reproduced the observed
  iteration exactly.
- So it is **deterministic and unpredictable** — identical within one BMesh
  whose element addresses are fixed, and different the moment
  `bmesh.new()` + `from_mesh` allocates fresh structs.
- The *tables* are faithful. `bm.verts`, `bm.faces`, `bm.edges`,
  `v.link_edges` and `edge.verts[0]` were all identical across five fresh
  BMeshes of one fixed mesh.

## Blender's own answer

**Not one of Blender's bundled add-ons puts a BMesh element in a `set`.**
Checked across the whole of
`5.2/scripts/addons_core` — `io_mesh_uv_layout`, `io_scene_fbx`,
`io_scene_gltf2`, `node_wrangler` and the rest — zero matches for
`set(bm.verts)`, `set(bm.faces)`, `set(bm.edges)` or any variant.

`io_mesh_uv_layout` does not even use `.index`: it walks the mesh tables in
order and carries plain tuples of coordinates. That is the idiom to copy.

## Scope

Root cause, and the thing to change first:

    foam_angle_core.uv_islands()  ->  "Returns a list of sets of BMFace"

Every consumer inherits that set. Counted element-keyed collections:

| file | sites |
|---|---|
| `foam_gp_pattern.py` | 9 |
| `foam_angle_core.py` | 4 |
| `foam_export.py` | 3 |
| `foam_island_ids.py` | 2 |
| `foam_overlay.py` | 2 |
| `paper_model_export.py` | 1 |

Membership tests are the one legitimate use and are hot — `_next_boundary_loop`
asks `radial.face not in island_faces` per step. Those want a set built
locally from an ordered source, not an ordered source replaced by a set.

## What not to do

- **Do not sort by `hash` or `id`.** Same allocator, same instability.
- **Do not reach for `.index` as the general answer.** It is stable within a
  BMesh and moves when anything renumbers; Blender's primitives renumber
  their face table on every call, measured.
- **Do not add more geometric sorts.** They are correct and they are the
  symptom-level fix already applied three times. The point of this work is
  that the next site should not need one.

Coordinates remain the right key where an *order* genuinely has to be
derived from geometry rather than carried.

## The safety net, which exists and is proven

Do not start without reading `tools/golden_marks.py`.

- It records 1422 marks across 5 fixtures x 6 settings sweeps and reports the
  same numbers on four consecutive runs.
- It is **proven sensitive**: perturbing `compute_bevel` by 0.01 degrees moves
  937 records and names the field.
- Compare with the bundled `golden_compare.py` from the `golden-refactor`
  skill. Save the baseline **before** the first edit, on the current commit.

`tests/test_reproducible_export.py` covers the property directly — two
BMeshes from one mesh must agree — and both load-bearing guards are proven to
fail it when reverted.

## Settled (2026-11-09): lists of faces, in face-table order

`uv_islands` returns a list of *lists* of BMFace, both levels in `bm.faces`
order. Indices were considered and rejected:

- **The hazard is the container, not the element.** A list walked out of
  `bm.faces` is exactly as reproducible as `bm.faces`, which is the table
  measured stable across fresh BMeshes. Indices buy nothing over that.
- **Indices would move `bm` into every helper.** `island_boundary_cycles`,
  `boundary_defect_load`, `find_tight_notches` and `island_angle_defect`
  take bare faces; each would need the BMesh and a lookup table, and would
  still build vertex sets inside.
- **The survey found a class the scope table missed: float sums over a set.**
  The sheet's area totals, island centroids, stretch means, defect totals,
  and `next(iter(isl.faces))` choosing a piece id all iterated the island
  set. An ordered list fixes every one with no edit at the site.

Membership stays a set, built locally and never iterated:
`island_boundary_cycles` builds one per island and hands it to
`_next_boundary_loop`, so returning lists costs no linear scans.

## Audit of the 38 sites

Most were never hazards. A dict keyed on BMesh elements iterates in
*insertion* order, so it is only as unordered as whatever filled it; a set
used only for `in` never exposes its order. Classified:

| kind | outcome |
|---|---|
| island sets iterated, summed, `next(iter())` (every `uv_islands` consumer) | fixed at the root: lists |
| `island_angle_defect`, `boundary_defect_load`: `set(faces)` then iterated | iterate the list, local set for `in` |
| `island_boundary_cycles` index sort over the island set | removed: input is ordered at source |
| `foam_gp_pattern._hit_faces` returning a set, sorted by two callers | returns an index-ordered tuple; one per-site sort removed; the union in `connect` keeps its sort |
| `foam_export.chain_direction_start`: `next(iter(set & set))` | walks `edge.verts` |
| membership-only sets (`pool`, `unvisited`, `seen_*`, `used_verts`, `own`, `boundary`) | correct as they are |
| dicts filled from ordered sources (`face_hits`, `face_edges`, `by_edge`, `wanted`, `ratios`, `owner`) | correct as they are |
| sets of ints or name tuples (`adjacency`, `absorbed`, `flagged`, `heat_islands`) | not BMesh elements |
| `remove_doubles` vertex order (`touched`, sorted by fresh index) | kept: same order as `bm.verts` |
| coordinate sorts in `boundary_defect_load`, `_edge_ends`, `order_rings_outer_first`, `_walk_order` | kept: they guard renumbering, which table order does not |

Verified: golden marks 1422 of 1422 unchanged after every step; making
`uv_islands` return sets again fails both export cases of
`tests/test_reproducible_export.py`, so the root is now what holds the
property.

## Related work in flight

An MVC split was planned on top of this — three independent mark generators
(overlay, SVG, raster) collapsed onto one model. It was deliberately **not**
started, because the baseline had to be trustworthy first. This ordering work
is the foundation that was missing; the MVC refactor follows it, not the
reverse.
