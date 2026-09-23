# Research: why the theta cut marks 174 or 176 seam edges at random

## The symptom (2026-09-21)

`tests/test_determinism.py::test_theta_cuts_identically_every_time` fails
intermittently. It cuts the same theta (a circle crossed by a bar) into a
freshly built UV sphere five times in one process and compares every reported
field across the runs.

Everything about the resulting mesh agrees every time:

| field | value, every run |
|---|---|
| `regions` | 3 |
| `sizes` | `[4564, 40, 40]` |
| `verts` | 4702 |
| `faces` | 4644 |
| `degenerate` | 0 |

and exactly one field does not:

| field | values seen |
|---|---|
| `seam_edges` | **174** or **176** |

So the cut *geometry* is stable — identical vertex and face counts, identical
region sizes — and only the count of edges flagged as seam moves, by two.

Observed against `main` at `99d5fb0`, headless Blender 5.2.1 LTS on port 9897,
run as:

    python tools/launch_blender.py --port 9897
    python tools/blender_bridge.py --port 9897 < tests/test_determinism.py

Failure rate over six consecutive runs in one session:

| case | failed |
|---|---|
| `test_theta_cuts_identically_every_time` | 5 of 6 |
| `test_face_numbering_does_not_change_the_cut` | 3 of 6 |
| every other case in the file | 0 of 6 |

## It is not new, and not caused by the merge_fraction work

Worth stating plainly, because the obvious suspect was the
`join_touching_strokes` change landed in `5bdbe13` the same day.

Checked directly rather than argued. A worktree was cut at `f87affb` — the
commit before the `merge_fraction` work existed even as an uncommitted diff —
junctioned into Blender's extensions directory under its own name, and its own
copy of `test_determinism.py` run against it six times:

    git worktree add <scratch>/base f87affb
    mklink /J "...\extensions\user_default\foam_craft" "<scratch>\base\foam_craft"
    python tools/launch_blender.py --port 9896
    # six runs of <scratch>/base/tests/test_determinism.py

Result: **6 of 6 failed**, with the identical `seam_edges` values `[174, 176]`.

The defect predates the change. If anything the current code fails *less*
often (5 of 6 rather than 6 of 6), which is itself unexplained and is not
being claimed as an improvement — see "What is not known" below.

## What has been ruled out

`test_determinism.py` was already written to take this apart, and its other
cases all pass consistently. Each one is a hypothesis this session can now
report as eliminated:

- **The nearest-face lookup is stable.**
  `test_points_on_an_edge_land_on_the_same_face_every_time` passes 6 of 6. The
  bar runs along the equator of a sphere with an even ring count, so it lies
  exactly on an existing edge loop and every point on it is equidistant from
  the faces above and below — a genuine tie for `find_nearest` to break. The
  test asserts the tie is broken the *same way* every run, and it is. The
  foundation does not move.

- **Each stroke alone is stable.**
  `test_each_theta_stroke_alone_is_stable` passes 6 of 6. Cutting the circle by
  itself, and the bar by itself, both reproduce exactly. So neither stroke
  carries the instability on its own.

- **A plain closed loop is stable.**
  `test_circle_cuts_identically_every_time` passes 6 of 6.

The test file's own docstring sets out the three candidates — "the circle at
*this* radius, the bar, or only the combination of the two" — and says which
answer means what: *"a bar that is unstable alone is a plain-path bug, while
two stable strokes that vary together is a junction bug."*

Both strokes are stable alone. They vary together. **By the file's own
reasoning, this is in junction handling** — the two places where the bar
crosses the circle.

That the delta is exactly **two** seam edges, and the theta has exactly **two**
crossings, is consistent with one edge per crossing, though not yet proof of it.

## Where to look in the real pipeline

The crossing path is `_CutGraph` in
[foamcraft_drachis/foam_gp_pattern.py:653](foamcraft_drachis/foam_gp_pattern.py:653),
the shared cut-graph introduced to fix the seam gaps written up in
[RESEARCH_gp_pattern_seam_gaps.md](RESEARCH_gp_pattern_seam_gaps.md). Every
stroke is resolved into that graph before anything becomes real `BMesh`
topology, so whichever stroke reaches a shared point second reuses the graph
vertex the first one placed.

The marking, rather than the cutting, is what varies — so the suspicion is the
edges returned for seam-flagging at a reused junction vertex, not the split
itself. The split is provably stable: same verts, same faces, same regions,
same region sizes, every run.

## The edges are at the crossings (2026-09-22)

Measured by `test_which_edges_the_theta_disagrees_about`, added to
`tests/test_determinism.py`. It cuts the test's own theta `RUNS` times,
records the midpoint of every edge `cut_strokes` returns, and takes the
symmetric difference across runs. Midpoints rather than indices, because
BMesh renumbers and an index is not a name.

The theta's two crossings sit at phi = +/-0.28 on the equator. Three
consecutive sessions:

| run | counts per cut | stable | unstable | distance to nearer crossing |
|---|---|---|---|---|
| 1 | `[174, 176, 176, 174, 176]` | 169 | **10** | 0.0054 x2, 0.0073 x4, 0.0071 x2, then 0.208, 0.220 |
| 2 | `[176, 176, 176, 176, 176]` | 175 | **2** | 0.078, 0.090 |
| 3 | `[176, 176, 176, 176, 176]` | 175 | **2** | 0.078, 0.090 |

**Eight of the ten wandering edges on the failing run sit within 7 mm of a
crossing**, in symmetric pairs about both of them -- midpoints at
`y = +/-0.2760, z = +/-0.0072` and `y = +/-0.2813, z = 0`. That is a cluster at
the two points where the bar meets the circle, not a scatter. The junction
diagnosis above is confirmed.

**And the passing runs are not deterministic either.** Runs 2 and 3 returned
176 edges on every cut, so `seam_edges` agreed and
`test_theta_cuts_identically_every_time` went green -- while the edge *sets*
still differed by two. The existing check compares counts, so it catches only
the subset of the instability where the count happens to move. The defect is
wider than its failure rate suggests, and the test **understates** the problem
rather than overstating it. That is worth stating plainly, because the
tempting reading of an intermittent test is that the test is at fault.

Bounding the practical impact: `regions` is 3 and `sizes` is `[4564, 40, 40]`
on every run recorded in this file, failing ones included. The island
decomposition is stable, so the pieces a maker would cut appear unaffected by
which of these edges comes back. That is an observation about the fields this
test records, not a verification of the exported pattern.

## What is not known

- **Why the rate changed** from 6-of-6 at `f87affb` to 5-of-6 at `99d5fb0`. It
  may be noise across a small sample; it has not been measured enough times to
  say. No mechanism is proposed.
- **Whether `test_face_numbering_does_not_change_the_cut` is the same bug.** It
  fails less often (3 of 6) and always alongside the theta case, never alone,
  which is suggestive but not established. It asserts that renumbering a mesh's
  faces does not change the cut, so a shared cause in junction handling is
  plausible.
- **Whether it affects real output.** The user-visible consequence of two seam
  edges differing has not been assessed. Region count and sizes are stable, so
  the pieces themselves appear unaffected; whether the unwrap or the exported
  pattern differs has not been checked.

## Candidate next steps, none tried

- Compare `seam_edges` as a *set* rather than a count in
  `test_theta_cuts_identically_every_time`, so the runs that currently
  pass with two edges differing are caught too. More honest, and it would
  fail far more often -- that is the trade-off.
- Bake twice and diff the exported SVG, to settle whether any of this
  reaches the pattern a maker actually cuts from.
- (Done, 2026-09-22) Reproduce with the test's own fixture,
  recording midpoints of the marked edges per run and diffing them — cheap, and
  it names the two edges outright. Costs nothing but a measurement.
- Instrument `_CutGraph` to log the order junction vertices are created and
  reused in, and check whether that order moves between runs — more invasive,
  but goes at the mechanism rather than the symptom.
- Decide whether it blocks a release. The pieces are stable and only the seam
  flags move, so it may be cosmetic; that is a judgement about the shipped
  artefact, not something to settle from the test alone.

These are options to raise, not a plan to execute unasked.
