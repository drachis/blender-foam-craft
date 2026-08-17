# Research: gaps in the grease-pencil cutter's seams at crossings and junctions

A research note, not a design doc: this records what was measured about a
defect in `foam_craft/foam_gp_pattern.py`, with reproduction steps, so the
investigation does not have to be redone. See
[DESIGN_grease_pencil_patterning.md](DESIGN_grease_pencil_patterning.md) for
what the cutter is and why it exists; this file is where its bugs get
diagnosed.

## The symptom (2026-08-16)

Reported against the live test file (`GPencil` + `Sphere` + `Sphere_Pattern`,
session on `tools/blender_bridge.py --port 9885`, `bpy.data.filepath == ""` --
unsaved): screenshots of the baked pattern show short **gaps in the red seam
dashes** right where one drawn stroke crosses or meets another, and at a spot
where a stroke doubles back on itself, even though the black Grease Pencil
strokes look continuous and the two lines visibly touch. The user's stated
success criterion: every drawn point should have a seam vertex within a
reasonable distance, and near-misses at junctions should extend the incoming
line to the junction rather than leave a gap.

## Root cause, confirmed by isolated `bmesh` experiment, not by reading

`bmesh.utils.face_split_edgenet(face, edgenet)` requires the edges handed to
it for one face to form a *valid* planar net -- every edge must reach the
face's boundary at both ends (directly, or through a chain of other net edges)
or close into a loop, and no two net edges may cross without sharing a vertex.
When that is violated **anywhere in one face's net, the call silently returns
an empty list for the *whole* face** -- not just the offending edge. Every
edge in that face's net, including ones that were perfectly fine on their own,
comes back `is_wire == True` and is later deleted by `cut_stroke`'s dangling
cleanup (`foam_craft/foam_gp_pattern.py:645-651`). No exception is raised, so
`_split_faces` (`foam_craft/foam_gp_pattern.py:529-557`) never logs a warning
for this path -- only the `except ValueError` branch does, and this failure
mode never reaches it. This is the single biggest reason the gaps have no
trace in `foam_craft.log`.

Two real code paths in `foam_craft/foam_gp_pattern.py` currently produce an
invalid net of exactly this kind:

1. **Order-dependent T-junction snapping.** `snap_stroke_ends`
   (`foam_craft/foam_gp_pattern.py:745-823`) moves an open stroke's end onto a
   point mid-way along *another* stroke, and inserts that same point into the
   target stroke's own point list so both lines pass through it -- but
   `cut_strokes` (`foam_craft/foam_gp_pattern.py:826-907`) still cuts each
   chained stroke into the mesh one at a time, in list order
   (`foam_craft/foam_gp_pattern.py:856-869`). If the *target* stroke is cut
   **after** the stroke whose end was snapped onto it, the snapped point is
   not yet real topology when the first stroke is cut: `_classify`
   (`foam_craft/foam_gp_pattern.py:309-352`) can only match `VERT`/`EDGE`
   against geometry that already exists, so the point comes back `FACE`-kind
   and `_realise` (`foam_craft/foam_gp_pattern.py:462-487`) creates a brand
   new, disconnected `BMVert` there. The final edge of that stroke then reaches
   the face boundary at one end only (a "spur") -- exactly the shape that
   makes `face_split_edgenet` reject the whole face's net.
2. **Self-crossing ("doubling back") strokes.** `_bridge`
   (`foam_craft/foam_gp_pattern.py:362-420`) only computes a crossing between
   **consecutive** drawn points that land in different faces. A stroke that
   loops back over its own earlier path within a *single* face produces two of
   its own net edges that cross each other in the face interior without
   sharing a vertex there -- because nothing in the pipeline ever compares a
   stroke's own segments against each other for self-intersection. That is
   the second shape that invalidates a face's net outright.

Either defect, once it lands in a face that also contains an otherwise-correct
crossing or junction cut (which is the common case right at a junction, since
that is precisely where multiple strokes and a stroke's own doubled-back path
are most likely to share a face), silently erases that good cut too. That is
why the visible gap is often wider than the single bad segment, and why it
clusters exactly at crossings and junctions rather than appearing at random
points along a line.

## Reproduction

Live session used for all of this:

```bash
python tools/launch_blender.py --port 9885   # if not already running
echo "import bpy
result = {'objects': [o.name for o in bpy.data.objects]}" | python tools/blender_bridge.py --port 9885
```

**Isolated proof #1 -- a spur (reaches the boundary at one end only) blocks
its own face, alone:**

```python
import bmesh
bm = bmesh.new()
v0 = bm.verts.new((0,0,0)); v1 = bm.verts.new((1,0,0))
v2 = bm.verts.new((1,1,0)); v3 = bm.verts.new((0,1,0))
face = bm.faces.new((v0,v1,v2,v3))
edge01 = bm.edges.get((v0, v1))
new_edge, mid = bmesh.utils.edge_split(edge01, v0, 0.5)   # a real prior crossing
tip = bm.verts.new((0.5, 0.5, 0.0))                       # a stroke's un-cut tip
spur_edge = bm.edges.new((mid, tip))
made = bmesh.utils.face_split_edgenet(face=face, edgenet=[spur_edge])
# made == [], spur_edge.is_wire == True
```
Result measured this session: `made_faces == []`, `spur_edge.is_wire == True`,
`n_faces_after == 1` (no split happened at all).

**Isolated proof #2 -- a spur poisons an unrelated, otherwise-valid cut in the
same face:**

```python
# same quad face as above, plus:
#  - a clean boundary-to-boundary line p1->p2 (would split the face alone)
#  - an unrelated spur s1->tip elsewhere in the same face
made = bmesh.utils.face_split_edgenet(face=face, edgenet=[valid_edge, spur])
# made == [] -- the clean cut is destroyed too, both edges come back wire
```
Result measured this session: `made_faces_vertcounts == []`,
`n_wire_after == 4` (all four net edges, including the previously-valid one).

**Isolated proof #3 -- two of a stroke's own segments crossing without a
shared vertex (the doubling-back case) does the same:**

```python
# quad face, two segments that cross in the interior with no shared vertex,
# each individually connected to a different boundary vertex
made = bmesh.utils.face_split_edgenet(face=face, edgenet=[edge_a, edge_b, ...])
# made == [], all of edge_a, edge_b, and their boundary connectors come back wire
```
Result measured this session: `made_faces_vertcounts == []`, `n_wire_after == 6`.

**On the real file:** `chain_strokes` merges the file's 11 raw strokes into 5
chained paths (`strokes=5` in `foam_craft.log`); the last bake reported
`open_ends=17`, matching the sum of `dangling=` across the five per-chain log
lines exactly (2 + 9 + 3 + 2 + 1 = 17). Re-running the pipeline stroke-by-stroke
against a fresh copy of the mesh (bypassing the operator so each dangling
edge's endpoints can be captured before `cut_stroke` deletes them, script kept
at the end of this document) shows these 17 are not all lone tip-spurs:

- Chains 0, 2, 3, 4 each end in 1-3 consecutive dangling edges right at one
  tip -- consistent with the order-dependent T-junction spur (finding #1).
- Chain 1 has a run of **8 consecutive** dangling edges forming a path that
  clearly reverses direction partway
  (`(-0.308, 0.734, 0.598)` -> `(-0.329, 0.727, 0.594)` doubles back toward
  `(-0.326, 0.721, 0.602)`, a point already visited two segments earlier) --
  consistent with the self-crossing case (finding #2), and matching the
  "curves doubling back on themselves" scenario the user named directly.

The X-crossing shown in the third screenshot has not yet been pinned to one
specific stroke pair in the file (not enough of the crossing's exact mesh
coordinates were captured before this was written up) -- but proof #2 above
means it needs no new mechanism to explain: any crossing that happens to share
a face with an order-dependent spur or a doubling-back segment from an
unrelated stroke will vanish the same way, which is consistent with a gap
appearing symmetrically on both sides of an otherwise-clean crossing.

## How the real knife avoids this entirely (confirmed from source, 2026-08-16, corrected same day)

The question raised next was whether cutting could be done **iteratively
along a stroke**, one segment at a time, rather than batching a whole
stroke's (or even a whole face's) hits into one `face_split_edgenet` call --
on the reasoning that a self-crossing stroke is only a problem for a
*batched* net; two atomic point-to-point segments committed in the order
they were drawn can't retroactively conflict with each other.

**First pass at this (superseded below): "the knife mutates real BMesh
topology per mouse-move segment."** `knife_add_cut()`
(`editmesh_knife.cc:2299-2375`) does only process the segment from
`kcd->prev` to `kcd->curr`, and during a freehand drag this does fire on
every `MOUSEMOVE` once `kcd->linehits.size() >= 2`
(`editmesh_knife.cc:4481-4490`) -- both true. The mistake was assuming the
per-segment call it makes, `knife_cut_face()` -> `knife_add_single_cut()`
(`editmesh_knife.cc:2056-2068`, `:1975-2050`), touches `BMesh` at all. It
doesn't: `knife_add_single_cut` only allocates `KnifeEdge`/`KnifeVert`
structs (`new_knife_edge`, `new_knife_vert`, `knife_split_edge`) -- the
knife's own lightweight, in-memory preview graph, entirely separate from the
mesh being edited. No `BM_edge_create`, no `BM_vert_create`, nothing that
changes the real mesh, anywhere in that call path. Caught by the user, who
pushed back rather than let the wrong model stand -- exactly the kind of
correction this project's `CLAUDE.md` asks for ("only the logic in Blender
and its source code are authoritative").

**What actually happens, read more carefully:**

1. **During the whole interactive session** (every click, every dragged
   `MOUSEMOVE`), the knife only ever extends its own `KnifeVert`/`KnifeEdge`
   graph -- never touches `BMesh`. Critically, each new 2D mouse segment is
   hit-tested not just against the *real, uncut* mesh but against **every
   `KnifeVert`/`KnifeEdge` the session has accumulated so far**
   (`editmesh_knife.cc:2902-3052`: `for (KnifeVert *v : kfvs)`,
   `for (KnifeEdge *kfe : kfes)`, both candidate sets drawn from a spatial
   query that includes the tool's own not-yet-realised cuts alongside real
   mesh geometry). That is where self-crossing and multi-cut junctions
   actually get resolved: a later segment that lands back on an earlier
   `KnifeVert` reuses it (`lh1->v = kfe->v1` etc., `:2010-2040`) instead of
   creating a new, disconnected one. This is graph-level deduplication in
   screen space, done incrementally, not a BMesh split happening early.
2. **Once, at confirm** -- `knife_make_cuts()` (`editmesh_knife.cc:2205-2292`,
   called from `knifetool_finish_single_pre` -> `knifetool_finish_ex`, the
   operator's finish path, never during interaction) converts the *entire*
   accumulated graph, across the whole session, into real `BMesh` geometry in
   one grouped-by-face pass (`fhash`, `:2216-2245`). Before splitting, each
   face's edge array is run through
   `BM_face_split_edgenet_connect_islands()` (`:2142-2159`, gated by
   `USE_NET_ISLAND_CONNECT`) -- which repairs a net that doesn't fully
   connect to the face boundary by inserting bridging edges -- and only then
   is `BM_face_split_edgenet()` called (`:2163`). Any edge still wire
   afterward is deleted (`:2166-2175`), same idea as this module's own
   dangling-edge cleanup.

So the real shape of the answer is: **build a fully-resolved abstract graph
first, incrementally, letting each new segment reuse or cross anything
already in the graph (from the same stroke or a different one) -- then
convert the finished, already-consistent graph to real mesh topology in one
batched pass per face.** Not "commit real geometry every step" (wrong, per
above), and not "batch a whole stroke's raw hits with no graph-level
resolution" (what this module does today, and the actual bug). The knife
never needs a special self-intersection test or a stroke-processing order
because its graph-building phase already prevents a malformed net from ever
reaching `BM_face_split_edgenet` in the first place -- by the time anything
is converted to `BMesh`, coincident points are already merged into one
graph vertex and genuine crossings are already graph edges meeting at a
shared vertex.

One piece is still not available from Python: `BM_face_split_edgenet_connect_islands`
is not exposed (checked again this session, matching the earlier note in
[DESIGN_grease_pencil_patterning.md](DESIGN_grease_pencil_patterning.md)).
That should matter less once the graph is built correctly in the first
place -- `connect_islands` is the knife's fallback repair for a net that
*still* doesn't reach the boundary despite the graph-building phase, not the
mechanism that resolves self-crossing or junctions. This project already has
its own fallback for the one case that plausibly still needs it (a closed
loop drawn wholly inside one face reaches no boundary at all): subdividing
the face until the stroke crosses an edge (`cut_stroke`'s `_all_inside`
refinement loop, `foam_gp_pattern.py:604-621`).

**Implication for this module:** `cut_stroke` currently classifies every
point of a whole stroke against one `BVHTree` snapshot
(`foam_gp_pattern.py:592-595`), bridges the whole stroke's hits
(`foam_gp_pattern.py:623`) by only checking whether *consecutive* points
share a face, and only then batches every face's net and splits all of them
(`foam_gp_pattern.py:628-630`) -- with no step anywhere that tests a new
segment against everything already placed, the way the knife's `kfvs`/`kfes`
query does. That missing step -- not batching *width* (whole stroke vs. one
segment) -- is the actual gap. Fixing it means giving this module its own
version of the `KnifeVert`/`KnifeEdge` graph: a structure that persists
across strokes (so junctions between separately-drawn strokes resolve by the
same mechanism as one stroke crossing itself), gets extended point by point
in drawn order, checks each new segment against the accumulated graph before
touching `BMesh`, and is only converted to real topology, batched per face,
once it is already consistent. This also reframes the "order-dependent
T-junction spur" (finding #1) and the "reorder chains" idea below: if
junctions are resolved at the graph level as strokes are added, order stops
being a per-chain scheduling problem the way it is in the current
architecture (or, worse, per stroke's *entire chained path*, since
`chain_strokes` merges several drawn strokes into one before `cut_stroke`
ever sees them).

**This one architecture change addresses both findings at once**, which the
earlier (incorrect) per-segment-BMesh-commit model did not: finding #2
(self-crossing) is a new segment reusing a `KnifeVert`-equivalent created
earlier by the *same* stroke; finding #1 (the order-dependent T-junction
spur) is a new segment reusing one created earlier by a *different* stroke.
Once every stroke feeds the same accumulated graph in sequence, "which
stroke should be cut first" stops being a scheduling question -- whichever
stroke's segment reaches a location *second* naturally attaches to whatever
the *first* stroke already placed there, exactly as it does by hand with the
interactive knife (confirmed directly by the user against their own
screenshots: both the self-crossing case and the T-junction case already
"just work" with the real tool). The density point still holds: the user
was explicit that a highly dense, near point-for-point graph is fine ("I
don't mind a highly dense 1:1 cut... The main issue was degen faces / edges
not density"), so the graph can be built at the same granularity the drawn
stroke was sampled at, same as the knife builds it at mouse-move
granularity.

## Implementation plan (agreed with the user; items 1-3 and 5 built and verified below, item 4 still open)

1. **A shared cut-graph, not a per-stroke one.** Give this module its own
   `KnifeVert`/`KnifeEdge`-equivalent: plain Python objects (or parallel
   dicts) holding a 3D position, the `BMFace`/`BMEdge`/`BMVert` it is
   ultimately anchored to once realised (mirroring `KnifeVert.v`), and
   adjacency. One graph, shared across every stroke in the whole bake -- not
   reset per stroke -- so two strokes drawn in either order reconcile at a
   shared point the same way one stroke's own return path does.
2. **Extend the graph in drawn order, testing against what's already there.**
   For each stroke's points, in order (interleaving across strokes only
   matters if it changes which segment "arrives second" at a shared point;
   simplest is one stroke fully at a time, since order stops mattering once
   junctions resolve at the graph level rather than by BMesh-cut order),
   test each new segment against the *accumulated graph* -- not just the raw
   uncut mesh -- for (a) landing on/near an existing graph vertex (reuse it,
   the way `knife_add_single_cut` reuses `lh1->v`/`lh2->v`), and (b) crossing
   an existing graph edge (insert a new shared vertex at the intersection,
   splitting both). This is the step currently missing entirely --
   `_bridge` only ever compares *consecutive* points of *one* stroke.
3. **Realise the finished graph into `BMesh`, batched per face, once.**
   Mirroring `knife_make_cuts`: group the graph's edges by which face they
   belong to, and call `bmesh.utils.face_split_edgenet` once per face, only
   after the whole graph is built and therefore already free of the
   malformed-net shapes documented above. `BM_face_split_edgenet_connect_islands`
   is still not exposed to Python -- expected to matter less now, since it is
   the knife's fallback for a net that *still* doesn't reach the boundary
   despite correct graph-building, not the mechanism that resolves
   self-crossing or junctions in the first place. If it turns out to still
   be needed for some shape, that is worth re-raising rather than silently
   working around.
4. **Near-miss junction judgement, as an extension of the same graph test.**
   The user's proposed heuristic -- for an open stroke's head (project
   backwards from point 0, using the direction from point 1 to point 0) and
   tail (project forwards from the last point, using the direction from the
   second-to-last point to the last point), extend that end along its own
   direction of travel by (this stroke's radius + the candidate target
   stroke's radius), and test whether the extension actually crosses a
   nearby stroke -- slots in as one more kind of test against the same
   accumulated graph, run at both ends of every open stroke, rather than a
   separate mechanism. If the extension crosses something, that crossing
   becomes a real graph vertex (same as any other) and the stroke's end
   point is replaced by it; if it crosses nothing, the end is left alone.
   This has no knife equivalent to match against -- the interactive knife
   only ever cuts one continuous mouse path and never reconciles two
   independently-drawn curves that end near each other -- so this piece is
   genuinely new logic the module has to own, not a case of matching an
   existing primitive. The user greenlit building it explicitly: "that is
   what we need to build... This is what will make the patterning process
   intuitive."
5. Make the silent-empty-result case of `face_split_edgenet` loud regardless
   of how complete the graph-building above turns out to be: log a warning
   in `_split_faces` whenever `live` is non-empty but `made` comes back
   empty, so this failure class shows up in `foam_craft.log` without needing
   an isolated `bmesh` session to discover it.

This is a substantial rewrite of `cut_stroke`/`cut_strokes`/`_bridge`/
`_net_edges`/`_split_faces`'s relationship to each other, not a patch. The
user has explicitly authorized this ("feel free to tear it apart to better
serve the end goal... commit regularly in case something goes wrong, but
risks can be taken to make it work").

## Implemented and verified (2026-08-16)

Items 1-3 and 5 above are built: `_CutGraph` (`foam_gp_pattern.py`, near
`_segment_crossing`), `_add_stroke_to_graph` (feeds one stroke's points into
a graph shared across the whole bake), `_realise_graph` (the single
batched-per-face conversion to real `BMesh`, mirroring `knife_make_cuts`),
and a warning logged whenever `face_split_edgenet` returns empty on a
non-empty net. `cut_stroke` (singular) still exists as a
single-stroke convenience wrapper (its own fresh graph, realised
immediately) for backward compatibility with existing single-stroke
callers/tests -- it fixes self-crossing but, since it does not share a graph
with any other call, cannot fix a cross-stroke junction on its own. Only
`cut_strokes` (plural) gets the full fix, by building one graph across every
stroke before realising anything.

Verified: [tests/test_gp_junctions.py](tests/test_gp_junctions.py) (the
self-crossing and order-independent-T-junction cases from earlier in this
document) now pass, as does the full pre-existing suite
(`test_gp_pattern.py`, `test_edgenet_cut.py`, `test_determinism.py`).

**Three further defects surfaced only once real multi-stroke data was run
through the new graph**, each caught by an existing test rather than
inferred, and each worth recording because the failure mode is easy to
reintroduce:

1. **Crossing search scoped to the wrong face.** The first version of
   `_CutGraph.connect` searched for a crossing only within the *home face*
   chosen for the current segment (`min(shared, key=index)`). Two strokes
   whose hits both touch several shared faces at once (e.g. both landing on
   an existing vertex) can each independently pick a *different* valid home
   face, and if the genuine crossing lives under the face the *other*
   stroke picked, it is never found. Fixed by scanning the *union* of both
   endpoints' faces (`scan_faces`) while still using the single `min`-index
   face as the new edge's own home. Caught by
   `test_theta_cuts_identically_every_time`
   (tests/test_determinism.py): a circle and a bar that visibly cross in
   the drawing came out with the bar never actually meeting the circle.
2. **Same-original-edge segments treated as a crossing candidate.** A
   stroke drawn along an existing edge loop (the theta test's bar, sitting
   exactly on a UV sphere's equator) produces almost entirely collinear
   segments -- and testing a collinear segment for a "crossing" against
   another stroke that happens to touch the same two faces produced a
   spurious near-parallel intersection, moving or losing the real crossing.
   Fixed by `_locally_trivial`: skip the crossing search outright when both
   hits already name the same real edge or vertex, or when one hit is a
   real vertex and the other stays inside that vertex's own face fan.
3. **The same-edge shortcut skipping over a third point.** Fixing #2 by
   connecting two same-edge hits *directly* introduced a new defect: if a
   *different* stroke's point sits between them on that same original
   edge, the direct edge skips straight over it, slicing a sliver triangle
   off the corner and orphaning whatever was attached to the skipped point
   (traced by hand to a single missing vertex in this session --
   `test_face_split_follows_a_polyline`-style isolated `bmesh` replay of one
   face's `face_split_edgenet` call showed the "north" side of a junction
   simply absent from either resulting face). Fixed by *not* connecting
   same-edge hits directly at all: `_realise_graph`'s per-edge realisation
   already builds one connected chain through every hit on an edge, from
   any stroke, correctly ordered by `fac` (`same_edge_chain`), so the
   graph-building step just needs to add nothing and let that handle it.
4. **Set-iteration order in `_CutGraph.add`/`connect`.** Both scan a set of
   `BMFace` objects (`_hit_faces` returns a `set`), and BMesh elements hash
   by identity -- the project's own recurring lesson (`CLAUDE.md`, "Rebuild
   scene state for every measurement"). When more than one candidate is
   within reach, *which one is found first* must not depend on a hash
   order that changes between runs or between differently-numbered
   meshes. Fixed by sorting both iterations by `face.index`. This resolved
   `test_theta_cuts_identically_every_time`'s "every_run_agrees" check
   completely (five identical runs now agree exactly).

## Near-miss junction extension (item 4) -- built 2026-08-16

Two strokes drawn to meet at a T routinely stop a whisker short of each
other: the ink overlaps and the eye reads a junction, but the curves never
touch, so no sideways tolerance small enough to be honest can join them.
`snap_stroke_ends` now handles this by *carrying the line on the way it was
going* and seeing whether it runs into anything, at both the head (direction
from point 1 back through point 0) and the tail (point -2 through point -1)
of every open stroke.

The pieces, and why each is there -- each of the last three was forced by a
case the previous version got wrong, so none of them is decoration:

- **Sideways and forward reach are different distances.** The sideways snap
  (a tip already within reach of another line, moved onto it) uses
  `snap_distance` alone. The forward extension uses the two pens' summed
  half-widths -- "would the ink have touched". Originally both used the
  pen widths, which made the extension dead code: for a stroke aimed
  squarely at another, the sideways distance and the forward distance to
  the same point are the *same number*, so the sideways test always fired
  first and the directional test never got a say. Splitting them is what
  makes the extension mean anything.
- **A cone, not a ray.** A drawn line's last segment carries whatever wobble
  the hand had as the pen lifted, so a single ray misses junctions the eye
  reads as obvious. `_extension_target` searches a 30-degree half-angle cone
  about the direction of travel, solving for where each candidate segment
  enters or leaves the cone (`_cone_candidates`) rather than sampling, since
  the nearest admissible point is usually exactly on the cone boundary and
  the closest approach never finds it.
- **The meeting must be at an angle.** A cone alone is not enough, for a
  geometric reason rather than a matter of taste: a line running *alongside*
  another a hair away is eventually clipped by any forward cone however
  narrow -- at a small enough offset there is always some distance ahead
  where the neighbour falls inside it. So a pure cone test called two
  side-by-side lines a junction, which is exactly what the directional test
  existed to reject. `_EXTENSION_MIN_MEETING_DEGREES` (20 degrees) requires
  the two to actually converge.
- **Extended, not dragged.** A tip that only reaches by being carried on
  *grows a new point* onto what it met; every drawn point stays where it was
  drawn. At a T that keeps the crossbar exactly as the maker drew it and
  lets the stem grow the last fraction, which is the behaviour the user
  asked for ("the T junction top should stay the same and the incoming line
  should be extended"). A tip already within sideways reach is still moved,
  as before.
- **Nothing is invented.** A stroke that reaches nothing either way is left
  completely alone -- no speculative vertices kept ("if not, don't keep
  those verts and keep going else where").

Covered by `test_near_miss_t_junction_extends_to_meet` (a 0.012 gap closed
from both the head and the tail, with the crossbar verified undragged) and
`test_near_miss_only_extends_along_direction_of_travel`, which pins the
three ways to *not* be a junction: running parallel, aimed past the end of
the target as actually drawn, and aimed but beyond the reach the pen widths
justify. That second test drives `snap_stroke_ends` directly rather than
through a full cut, because it is a pure function and a region count
downstream would confound "did not connect" with "connected but enclosed
nothing".

One case is deliberately absent from the negative tests: "tip near the
target but pointing away". For a straight stroke it is not constructible --
whichever end lies nearest the target necessarily has its outward direction
pointing toward it.

## Known remaining residual: seam positions vary slightly between runs

Not fixed, and worth stating precisely because the headline numbers look
stable and hide it.

Across five identical cuts of the theta pattern (a circle crossed by a bar
drawn exactly along a UV sphere's equator, `tests/test_determinism.py`):
region count, region sizes, vertex count, face count, degenerate-face count
and the *total* number of seamed mesh edges (200) are identical every run.
But a fingerprint of the seam edges' actual midpoint coordinates is **not**
identical, and the edge list `cut_strokes` returns varies between 174 and
176. So the same cut is landing in very slightly different places run to
run, without changing its structure.

`test_face_numbering_does_not_change_the_cut` shows the same thing under a
deliberately renumbered mesh, and `test_theta_cuts_identically_every_time`'s
`every_run_agrees` check fails on the `seam_edges` field for the same
reason.

What is already ruled out: classification is stable
(`test_points_on_an_edge_land_on_the_same_face_every_time` passes, so
`_classify` returns the same kind and location for every point every run),
and the sphere fixture is rebuilt identically
(`test_the_sphere_is_rebuilt_identically` passes). Both `_CutGraph.add` and
`_CutGraph.connect` were found iterating `set`s of `BMFace` and are now
sorted by face index, which fixed a larger version of this.

Prime remaining suspect, not yet confirmed: the `bmesh.ops.remove_doubles`
weld at the end of `cut_strokes`, which is seeded from
`sorted(touched, key=lambda v: v.index)`. Vertex *indices* are assigned by
the order geometry was created during the cut, so if anything upstream still
varies, the weld keeps a different survivor and moves the result by a
fraction. This is the exact failure mode CLAUDE.md already records twice
(the uvsphere primitive's `remove_doubles` and `uv_islands`' `set.pop()`).

Practical impact is small -- the pattern's structure, piece count and piece
sizes are all stable, and the drifting distance is far below the
`snap_distance` -- but this is a project that has been bitten by exactly
this class of bug before, so it should not be left implicit.

## Diagnostic script (the "17 dangling" trace above)

Kept here so it does not need to be rewritten.

```python
# Run via: python tools/blender_bridge.py --port 9885 < script.py
# Requires GPencil + Sphere objects in the live scene, FoamCraft settings
# pointing at them (pattern_source=Sphere, gp_object=GPencil).
import bpy, bmesh
import sys
sys.path.insert(0, r"O:\Work\BlenderPlugin\FoamCraft")
from foam_craft import foam_gp_pattern as gpm

gp = bpy.data.objects['GPencil']
source = bpy.data.objects['Sphere']
depsgraph = bpy.context.evaluated_depsgraph_get()
mesh = bpy.data.meshes.new_from_object(source.evaluated_get(depsgraph))
dummy = bpy.data.objects.new("DiagDummy", mesh)
dummy.matrix_world = source.matrix_world.copy()
bpy.context.collection.objects.link(dummy)

strokes = gpm.strokes_from_grease_pencil(gp, dummy, depsgraph=depsgraph)
snap_distance = bpy.context.scene.foam_craft.snap_distance
bm = bmesh.new()
bm.from_mesh(mesh)

pen = max((s[2] for s in strokes if len(s) > 2), default=0.0)
snapped = [(s[0], s[1]) for s in gpm.snap_stroke_ends(strokes, snap_distance)]
chain_tol = max(snap_distance, 2.0 * pen)
chained = gpm.chain_strokes(snapped, chain_tol)

dangling_locs = []
for index, (points, cyclic) in enumerate(chained):
    bm.faces.ensure_lookup_table(); bm.faces.index_update()
    bvh = gpm.BVHTree.FromBMesh(bm)
    hits = gpm._classify(bm, bvh, points)
    hits, missed = gpm._bridge(hits)
    verts = gpm._realise(bm, hits, snap_distance)
    net, reused = gpm._net_edges(bm, hits, verts, cyclic)
    gpm._split_faces(bm, net)
    new_edges, seen = [], set()
    for e, _f in net + [(r, None) for r in reused]:
        if not e.is_valid or e in seen:
            continue
        seen.add(e); new_edges.append(e)
    dangling = [e for e in new_edges if e.is_wire]
    for e in dangling:
        v0, v1 = e.verts
        dangling_locs.append({'chain': index, 'v0': tuple(v0.co), 'v1': tuple(v1.co)})
    if dangling:
        bmesh.ops.delete(bm, geom=dangling, context='EDGES')

result = {'n_dangling_total': len(dangling_locs), 'dangling_locs': dangling_locs}
bm.free()
bpy.data.objects.remove(dummy, do_unlink=True)
bpy.data.meshes.remove(mesh)
```
