# Blender add-on development workflow

Process notes from building FoamCraft against a live Blender 5.2. Written to
be reusable on any Blender add-on project. Everything here was learned by
getting it wrong first; the failures are included because they are the point.

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

## Reusable pieces

- `tools/launch_blender.py` — start/stop an instance on a given port, headless
  or GUI. Handles the GUI case, where the socket must be started through the
  add-on's own operator so its timer is registered too; binding the socket
  alone accepts connections and then resets them.
- `tools/blender_bridge.py` — run Python inside a running instance. Note the
  **null-byte** framing: newline is not the delimiter, and the executed code
  must assign a JSON-serializable dict to `result`.
- `tools/shot.py` — sequence-numbered screenshot paths.
