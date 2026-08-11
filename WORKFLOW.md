# Blender add-on development workflow

Process notes from building FoamCraft against a live Blender 5.2. Written to
be reusable on any Blender add-on project. Everything here was learned by
getting it wrong first; the failures are included because they are the point.

---

## 0. Ask before inventing

The cheapest tool available, and the most underused. A question costs nothing,
and the person you are working with usually already knows the answer — they
have context you do not, and they have often solved this before.

Order of preference, and inventing is last:

1. **Ask.** Zero cost, usually decisive.
2. **Use the built-in.** It is maintained and it already handles the cases you
   have not thought of.
3. **Read the documentation.**
4. **Read the source**, targeted at a specific question.
5. **Invent** — only with a stated reason why the four above did not serve.

Treat the thought *"I'll design something for this"* as the trigger to stop and
ask. Not as a courtesy: a question is genuinely cheaper than a wrong build plus
the debugging plus the removal.

Worked example from this project: the flat preview grew a bespoke shelf packer
with gaps and wrap-around, to arrange pieces in the viewport. It was invented,
it was wrong (offsets were added to UV positions rather than replacing them, so
pieces overlapped), and it was pointless — the answer was to use the UV layout
directly, u to x and v to y at true scale, because that layout had already been
arranged on purpose. The whole detour, and the bug inside it, cost more than
the question would have.

The same instinct produced the other expensive mistakes here: a hand-rolled
boundary walk when the Decimate modifier already had a UV-preserving Delimit
option, and a hand-rolled convex/concave sign test when
`BMEdge.calc_face_angle_signed` existed and was correct.

---

## 1. Drive a real Blender, do not guess at the API

Write nothing against remembered API. Blender's Python API changes between
releases and plausible-looking calls fail in ways that are invisible until
runtime — a draw handler that raises is often silently swallowed.

Keep a live instance and probe it:

```bash
python tools/launch_blender.py --port 9877          # headless
python tools/launch_blender.py --port 9878 --gui    # with a window
echo 'import bpy
result = {"v": bpy.app.version_string}' | python tools/blender_bridge.py --port 9877
```

Probe before designing, not after failing:

```python
result = {"enum": [i.identifier for i in
                   bpy.ops.uv.unwrap.get_rna_type().properties["method"].enum_items]}
```

Real examples this caught:

| Assumption | Reality on 5.2 |
|---|---|
| `gpu.shader.from_builtin('2D_UNIFORM_COLOR')` | Raises `ValueError`. The `2D_` names were removed; it is `UNIFORM_COLOR` |
| `blf.size(font, size, dpi)` | 3-arg form raises `TypeError`; 2-arg is correct |
| Decimate has no UV-safe mode | It has `delimit={'UV','SEAM',...}`, and the evaluated mesh keeps its UVMap |

**Rule: when an approach appears blocked, read the API or ask. Do not
substitute a hand-rolled design.** In this project a plan step was abandoned on
the belief that decimated geometry could not be mapped back to UVs. Both
premises were wrong, and a bespoke algorithm was written for a problem that did
not exist. That cost more than asking would have.

---

## 2. One Blender process per worker

Parallel agents sharing one instance corrupt each other through state that is
not obvious:

- **`sys.path`** — one worker inserting its worktree makes `import mypkg`
  resolve to *another worker's* copy.
- **The scene** — one worker deleting objects breaks another's test midway.
- **Registered classes** — `bl_idname` is global; re-registering replaces
  everyone's.
- **Edit mode** — several objects can be in edit mode at once, so an overlay
  can silently annotate a neighbour's mesh.

Sharing one instance here produced cross-contaminated results and a hard crash.
Give every worker its own port. Processes are cheap; debugging phantom
interference is not.

Headless suits anything that inspects data. Draw handlers and screenshots need
`--gui`, since background Blender has no window to render.

---

## 3. Never hold BMesh references beyond the call that fetched them

`BMFace`, `BMEdge`, `BMLoop` are valid only while their mesh is alive and
unmodified. Leaving edit mode, undo, or another object entering edit mode frees
them. Touching a freed element **crashes Blender outright** — it does not
raise, so `try/except` cannot save you.

This crashed Blender in this project: a draw handler cached objects holding
`BMLoop` references and reused them across frames.

Copy out plain numbers at the point of computation:

```python
class _RunMark(NamedTuple):
    ax: float; ay: float; bx: float; by: float   # never the BMLoop
```

---

## 4. Draw handler hygiene

Handlers accumulate. Registering without removing leaves the old callback
attached forever, each stale copy drawing from its own dead cache — the visible
symptom is doubled marks at slightly different positions.

Two rules:

```python
def register():
    _drop_existing_handler()          # idempotent
    handle = bpy.types.SpaceImageEditor.draw_handler_add(cb, (), 'WINDOW', 'POST_PIXEL')
    bpy.app.driver_namespace[KEY] = handle    # survives module reload
```

Park the handle somewhere that outlives the module. Reloading during
development replaces the module object, so the new copy's global is `None`
while Blender still holds the old copy's callback. A namespace that outlives
the module is the only place a successor can find it.

Blender has no API to enumerate draw handlers, so handlers leaked before this
was in place can only be cleared by restarting. Restart when in doubt.

---

## 4b. A renderer must not read the model

The costliest bug in this project. With the overlay active, dragging UVs
crashed Blender and corrupted geometry within about a frame. Two attempted
fixes made it worse before the cause was understood.

The rule is **not** "do less work in the draw callback". It is that the draw
path must not touch live data at all. Reading is enough to do damage: a modal
transform is mutating the same structures, and any call that rebuilds internal
state — `ensure_lookup_table()` on a BMesh sequence, for instance — mutates
while the operator is mid-write.

Blender's own UV overlay is built exactly this way, and it is worth reading
before inventing anything (`draw/engines/overlay/overlay_mesh.hh`,
`MeshUVs::edit_object_sync`): it only submits GPU batches extracted earlier,
and invalidation arrives as a dirty tag. Notably there is **no position-only
fast path** — a UV transform tags `BKE_MESH_BATCH_DIRTY_ALL` and everything is
re-extracted every frame. So the problem was never the volume of work. It was
doing any of it at the wrong moment.

Add-ons cannot hook that extraction pipeline, but the shape transfers:

| Piece | Does | Touches the model |
|---|---|---|
| Draw handler | renders cached plain data | **never** |
| `depsgraph_update_post` | raises a dirty flag, nothing else | **never** |
| `bpy.app.timers` tick | rebuilds when dirty, never mid-drag | yes, at a safe point |

A timer fires *between* operator executions rather than inside one, which is
the closest thing Python has to a sync point.

Two dead ends, so they are not retried: splitting the cache into cheap and
expensive halves still read the model from the draw handler and turned a freeze
into corruption; moving that read into a depsgraph handler was no better,
because during a modal transform it fires every frame, mid-mutation.

Three corollaries, each learned the same way:

- **Never draw known-stale data.** A flag set by the timer cannot prevent a
  stale frame — the draw beats the timer. Refuse to draw while dirty instead.
  One frame late and correct beats one frame early and wrong.
- **Detect the specific operator, not the general condition.** Testing for "any
  modal operator running" reported a drag permanently, because the UV editor
  keeps `IMAGE_OT_view_pan` registered as a modal handler at all times. The
  overlay silently never rebuilt and would never have drawn. Match
  `TRANSFORM_OT_`.
- **Hold no live-data handles across frames.** See section 3.

---

## 5. Rebuild the scene for every measurement

Inherited scene state produces numbers that move between supposedly identical
runs. In this project an object named `Torus` turned out to hold a 500-face
Suzanne left by an earlier test, and several measurements were quietly wrong
before anyone noticed.

Start each measurement from nothing:

```python
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
```

Assert what you built: `result['faces'] = len(ob.data.polygons)`. A 12x5 torus
has 60 faces. If it says 500, stop.

---

## 6. Check determinism explicitly, and check the right thing

Anything a user's work depends on — numbering, ordering, identity — must be
reproducible run to run.

Iterating a `set` of BMesh elements is **not** deterministic: ordering follows
identity hashing, which varies between runs on identical geometry. Seed
traversals from an index-ordered list instead.

The subtle part: an early determinism check here compared island *counts*,
which were stable, and passed. Island *order* was not stable, and order was
what piece numbering depended on. **Test the property you actually rely on.**

Use a worst case. A fully seamed cube makes every face equidistant from the
origin, so ties are guaranteed and any order-dependence shows immediately.

---

## 6b. Interactive features can only be verified interactively

This is why the project favours BDD over unit tests, and it is not a
preference — it is the only thing that catches this class of bug.

Every scripted check passed while the overlay was crashing Blender. None of
them entered a modal loop, because **scripted operators do not drag**:
`bpy.ops.transform.rotate(value=0.13)` executes immediately with an explicit
value and never enters modal. Nudging coordinates and calling
`bmesh.update_edit_mesh` is further still from a transform. Thirty-six real
transform operators ran clean in a script while a single hand-drag crashed
within a frame.

So for anything interactive:

- The scenario is *"given the overlay is on, when I drag a UV, then Blender
  keeps running"*, and only a human or a UI-driving harness can execute it.
- Until someone has done that, the fix is **not verified**. Say so plainly
  rather than reporting a pass — claiming a fix from tests that never touched
  the failing path is how a freeze became data corruption here.
- "Blocked on: needs a hand test" is a useful, finished-sounding result.

Scripted checks still earn their place for everything downstream of the
interaction — geometry, counts, exported files — where ground truth is
arithmetic. Just do not let them speak for the interaction itself.

---

## 6c. Verify the verification

Three separate checks misreported the same draw path in one session:
`inspect.getsource` served stale lines from `linecache` after a file edit, and
two string slices ran past the end of the function and swept in code that was
not being tested. Each returned a confident, wrong answer.

- Prefer reading state back out of the running application over introspecting
  source text.
- If you must inspect source, extract real function bounds — not a slice
  between two guessed markers.
- Distrust a check that has never failed. Make it fail on purpose once.
- When a result contradicts a previous measurement, suspect the measurement
  before the code. A 13% "failure" in outward normals turned out to be a
  centroid test that is simply invalid on concave shapes; the code was exact.

---

## 7. Verify visually, and verify the capture

For anything that draws, numbers are not enough — read the image back.

- Frame the subject: select only the generated objects and `view_selected()`,
  not `view_all()` on the whole scene.
- Flat layouts read best straight down and orthographic
  (`view_axis(type='TOP')` + `region_3d.view_perspective = 'ORTHO'`).
- Vertex colours need `shading.color_type = 'VERTEX'` or the work is invisible.
- Let the redraw settle. A screenshot grabbed mid-redraw shows doubled text
  that looks exactly like a real bug.
- Pick a subject that can actually show the feature. A cube demonstrates
  nothing about relief cuts, because flat panels need none.

Sequence-number captures so they sort chronologically; alphabetical names jump
around as topics change. Name them for *what changed*, never "final".

---

## 8. Separate the tool's constraints from the process's

A gate that looks technical is often a workshop assumption in disguise.

Cut runs here were split whenever they bowed away from a straight chord. That
silently imposed a **table saw** constraint — a band saw follows a curve at a
fixed blade tilt perfectly well. The straightness limit became opt-in and
defaults off; runs are defined by the angle holding steady, which is what
actually determines one pass.

Ask what the physical process requires before encoding a limit as a rule.

---

## 9. Commit per increment, with the measurements in the message

Every commit records what was verified and the numbers behind it, so a later
reader can tell a checked claim from an assumed one. Where something is
unverified, the message says so.

Agents get a worktree and a branch; collapse to main for linear history.

---

## 10. Drop a marker as each plan step completes

Tags are cheap and finding your way back is not. Drop one whenever a plan step
lands, so there is always a named point to return to when the next attempt goes
wrong — as the overlay work did, twice.

Two kinds, and keeping them distinct matters:

| Kind | Moves | For |
|---|---|---|
| `v0.1`, `v0.2` … | **never** | Fixed release markers. Meaningful milestones only. |
| `lastKnownGood` | every confirmation | The latest state confirmed working *by hand* |
| `step-<n>-<slug>` | never | One per completed plan step, for navigation |

Move `lastKnownGood` with `git tag -f`, and only after the state has actually
been confirmed — for interactive features, that means a human tried it. It is
the tag you reach for when something breaks and you need a floor to stand on.

**A version number is a claim, so do not attach it to an unverified state.**
`v0.1` was first cut at a commit that corrupted geometry on the first UV drag,
because the automated checks all passed. Naming the moving marker
`lastKnownGood` rather than a version keeps that distinction honest: one says
"this is a release", the other says "this is the furthest point someone has
actually confirmed".

Annotate them. A tag message costs nothing and is the natural place to record
what a state does and does not do — a good annotation lists what was verified,
with numbers, and states plainly what was not.

---

## Reusable pieces

- `tools/launch_blender.py` — start/stop an instance on a given port, headless
  or GUI. Handles the GUI case, where the socket must be started through the
  add-on's own operator so its timer is registered too; binding the socket
  alone accepts connections and then resets them.
- `tools/blender_bridge.py` — run Python inside a running instance. Note the
  **null-byte** framing: newline is not the delimiter, and the executed code
  must assign a JSON-serializable dict to `result`.
- `tools/shot.py` — sequence-numbered screenshot paths.
