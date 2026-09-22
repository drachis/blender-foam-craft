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

## What is not known

- **Which two edges.** Not identified. An attempt to isolate them by cutting a
  theta 8 times and diffing the sets of marked-edge midpoints returned **0
  unstable edges** — the probe used its own fixture (radius 0.35, a 48×24
  sphere, `snap_distance` 0.002) rather than the test's (radius 0.28, and the
  test's own sphere and snap), and did not reproduce the symptom. A probe
  matching the test's fixture exactly is the obvious next measurement, and per
  this project's rules it belongs in `tests/test_determinism.py` rather than
  in a scratch file.
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

- Reproduce inside `tests/test_determinism.py` with the test's own fixture,
  recording midpoints of the marked edges per run and diffing them — cheap, and
  it names the two edges outright. Costs nothing but a measurement.
- Instrument `_CutGraph` to log the order junction vertices are created and
  reused in, and check whether that order moves between runs — more invasive,
  but goes at the mechanism rather than the symptom.
- Decide whether it blocks a release. The pieces are stable and only the seam
  flags move, so it may be cosmetic; that is a judgement about the shipped
  artefact, not something to settle from the test alone.

These are options to raise, not a plan to execute unasked.
