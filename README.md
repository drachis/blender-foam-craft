# FoamCraft

A Blender extension for building curved shapes out of flat foam sheet.

You have a sculpt — a horn, a helmet, a creature head — and you want to cover it
in individually-cut foam panels. Every panel needs an outline, a piece number,
and a **bevel angle**: the tilt the saw blade has to sit at so the panel's edge
meets its neighbour cleanly instead of leaving a wedge-shaped gap. That work is
traditionally done by hand on paper. FoamCraft computes it from the mesh.

![The FoamCraft overlay in the UV editor: eight numbered pieces from a torus,
each cut run bracketed, with the bevel angle printed at every
edge](docs/images/overlay-cut-plan.png)

*Eight panels unwrapped from a torus. Blue angles are convex folds, pink are
concave, and the brackets group edges into **cut runs** — one run is one pass at
one blade tilt.*

> **Status: 0.1, early.** The geometry has been checked against shapes with
> known answers (see [Known limitations](#known-limitations) for what hasn't).
> Expect to sanity-check the output against a test piece before cutting
> expensive stock.

## Requirements

- **Blender 5.2 or newer.** Earlier versions are untested and the extension
  declares 5.2 as its minimum.
- Optional: Blender's bundled **Export Paper Model** add-on, needed only for
  the companion paper-model export described below. Everything else works
  without it.

## Install

1. Download `foamcraft_drachis-<version>.zip` from the
   [Releases page](https://github.com/drachis/blender-foam-craft/releases).
2. In Blender: **Edit ▸ Preferences ▸ Add-ons ▸ ▾ ▸ Install from Disk…**
3. Pick the zip. It appears under the name *FoamCraft*.

The controls live in the **UV Editor sidebar** (press <kbd>N</kbd>), under the
**FoamCraft** tab.

## Workflow

### 1. Model at real-world scale

This matters more than it sounds. Several of FoamCraft's settings are
measurements of a physical tool — blade width, sheet thickness, the smallest
radius your saw can follow — and they are in metres. A 22 mm snap distance is a
coarse blade on a real horn and nonsense on a 2 m test sphere. If a tolerance
and the model seem to disagree, suspect the model's scale first.

### 2. Draw the panel boundaries

Create a Grease Pencil object and draw the piece seams directly onto the sculpt
with **stroke placement set to Surface**. Draw them where you want the real
seams — that is the whole input.

![Grease Pencil strokes drawn directly on a sphere, forming a closed loop
crossed by a bar](docs/images/bake-strokes-drawn.png)

### 3. Bake the pattern

In the **Pattern** panel:

| Field | What it is |
|---|---|
| **Sculpt** | The mesh to cut. Named explicitly, not taken from the selection — the active object right after drawing is the Grease Pencil. |
| **Strokes** | The Grease Pencil object holding your seams. |
| **Snap Distance** | The smallest feature your saw or hot wire can actually follow. |

Press **Bake Pattern**. It cuts the strokes into a *copy* of the sculpt (named
`<Sculpt>_Pattern`, leaving your original untouched) and unwraps the resulting
pieces. If the button is greyed out, the panel prints the reason directly above
it.

![The same strokes after baking, now real mesh edges following the drawn line
exactly, including the two crossings](docs/images/bake-strokes-cut.png)

*After baking: the strokes are real topology. The two crossings became shared
vertices rather than two edges passing through each other, which is what closes
the loop into a separate piece.*

**About Snap Distance:** set it from the tool, not to taste. Detail drawn finer
than this gets welded away, because a notch narrower than the blade cannot be
cut. The log reports a `welded=` count — that is how many of *your own drawn
points* were merged because they sat closer together than the blade is wide. A
high count is not a bug; it means the drawing is finer than the process can
make, and the fix is to redraw more coarsely or use a finer tool.

### 4. Set the material and the saw

In the **Foam Cut Plan** panel: foam grade and thickness (read the thickness off
the stock you actually have — grade and thickness are independent), max blade
tilt, and kerf. **Estimate From Foam** fills the flex tolerances from the grade.

### 5. Check craftability

**Check Craftability** (in Edit Mode) reports whether the pieces can actually be
built: seams needing more blade tilt than the saw has, pieces that won't lie
flat, pieces larger than your sheet stock, runs too fragmented to cut. Findings
are graded, and blockers are counted separately from warnings.

Saddle-shaped pieces block by default. **Allow Gores** permits them by adding
material — a wedge glued into a slit — but understand that this changes the
build rather than relaxing a check: every gore is another piece, another glue
line, and a seam running across a piece rather than around it.

### 6. Number the pieces

**Assign Foam Piece Numbers** numbers the islands, walking outward across the
surface from a chosen origin object (or the 3D cursor). Numbers persist, so
re-running keeps existing ids valid unless you ask for a full renumber.

### 7. Export

**File ▸ Export ▸ Foam Panel Pattern (.svg)** writes the flat cut patterns:
outlines, relief darts, piece numbers and bevel-angle labels. Print at scale,
transfer to foam, cut.

## Panels

All in the UV Editor sidebar (<kbd>N</kbd>):

- **FoamCraft ▸ Pattern** — Grease Pencil bake
- **FoamCraft ▸ Foam Cut Plan** — material, saw, darts, numbering
- **FoamCraft ▸ Foam Cut Plan ▸ Display** — what the overlay draws
- **FoamCraft ▸ Foam Cut Plan ▸ Craftability** — findings from the last check
- **Fold Angles** — the standalone live fold-angle overlay

## Paper model companion (optional)

**File ▸ Export ▸ Paper Model Fold Angles (.svg)** produces a companion overlay
for Blender's bundled Export Paper Model add-on, labelling the fold angle in
degrees at every edge — **including the cut edges between pieces**, which the
stock exporter deliberately leaves unlabelled because plain paper folding
doesn't need a bevel number at a cut.

This one operator requires *Export Paper Model* to be installed and enabled
(**Preferences ▸ Add-ons ▸** search "Paper Model"). If it isn't, the operator
says so and cancels; nothing else in FoamCraft is affected.

## Diagnostics

Every number FoamCraft computes is written to a rotating log, so a result can be
read out of a file rather than counted off the screen:

```
<Blender user data>/foamcraft_drachis/foamcraft_drachis.log
```

The exact path is shown in the FoamCraft sidebar. Set the level to `DEBUG` there
to include the overlay's per-rebuild detail — noisy by design, and off by
default.

## Known limitations

Honest list of what is unresolved as of 0.1:

- **Near-miss stroke junctions.** Where two drawn strokes come close to each
  other but never actually touch, they may not weld into a shared junction, and
  the seam can show a gap there. Strokes that cross, and strokes that meet in a
  T, are both handled. Workaround: draw junctions so the strokes visibly
  overlap.
- **Multiple UV maps.** FoamCraft reads the *active* UV layer. Whether that
  reliably matches what the UV Editor is displaying when a mesh has several UV
  maps has not been confirmed. With one UV map — the normal case — this cannot
  bite.
- **Two sidebar tabs.** The standalone fold-angle overlay sits under its own
  **Fold Angles** tab rather than with the rest of FoamCraft. Cosmetic.
- **Foam grade stiffness figures are rough.** The relative stiffness numbers
  behind the grade presets give a sensible *ordering*, not calibrated physical
  values. Calibrate against a bend test on your actual stock before trusting
  the dart spacing on a tight curve.
- **Bevel angles are computed, not proven.** The geometry is checked against
  shapes with arithmetically known answers (a sphere's curvature totals 4π, a
  cylinder's is zero, a cube's edges are all exactly 90°). That is a strong
  check on the maths and not a substitute for cutting one test panel first.

Found something else? [Open an
issue](https://github.com/drachis/blender-foam-craft/issues).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
