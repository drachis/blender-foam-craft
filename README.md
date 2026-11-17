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
outlines, relief darts, piece numbers and bevel-angle labels. The image export
is the recommended route to paper; the SVG is the alternative. See
[Printing the pattern](#printing-the-pattern).

## Printing the pattern

The recommended route is the **image export**. The image is sized from the
pattern's real dimensions, so printing it at the stated DPI gives a true-size
sheet.

Find it in the UV Editor sidebar (<kbd>N</kbd>) under **FoamCraft ▸ Foam Cut
Plan ▸ Print / Export**, and press **Export Image**. The same operator is also
at **File ▸ Export ▸ Foam Pattern Image (.png/.jpg)**. The panel's **Export
SVG** button is the SVG route, described below.

> **Tip:** apply the object's scale before exporting. Select the sculpt, press
> <kbd>Ctrl</kbd>+<kbd>A</kbd> and choose **Scale**, so the mesh's real size and
> the object's transform agree.

**Pages.** No big printer? Set **Pages** to *US Letter* (or *A4*) in the file
browser's side panel, in DPI mode only. Besides the whole sheet, the export
writes one file per page, each a full sheet of paper at the export DPI, with the
pattern inside a quarter-inch margin, corner marks at the printable area, and a
10 mm overlap with its neighbours to tape along. It is a plain crop of the
sheet, nothing is re-packed. Print at actual size, then check the scale ruler.

Pages are named like squares on a chess board: a column letter, then a row
number, with **row 1 at the top** (the sheet reads top-down), so `C1` is the
third page along the top row and `A1`, `B2`, `C3` run down a diagonal. The name
is printed in the top-left and bottom-right corners of the margin and is in the
file name (`name_C1.png`); the bottom margin also says how big the grid is
(`page C1 of A1-G6`). Pages are labelled the way pieces are:

- **Join numbers.** Every edge two pages share has a plain number, printed in
  the margin at the middle of that edge on *both* pages, turned square to the
  edge like a bevel label. Matching numbers go together, as matching seam
  numbers glue together.
- **Registration targets.** Two circle-and-cross targets per join sit in the
  10 mm overlap band (at a quarter and three quarters of the shared edge),
  labelled with the join number. They are at the same place on the pattern on
  both pages. They are mid grey, so they print on a black-and-white printer
  and are told from the black cut lines by tone and by their shape. A dashed
  light grey line marks the inner edge of each overlap band.

To assemble: lay the pages out by name, trim the margin off one page of a pair
along the side that overlaps, lay it over its neighbour with the same join
number, slide it until both targets sit exactly on top of each other, and tape.

**Format.** PNG is lossless and keeps lines and numbers sharpest. Use JPEG only
when a print service asks for it.

**Resolution.** Two modes:

- **DPI** (the default, 300 DPI). The image is sized from the pattern's real
  size, so printing at this DPI is true to scale. In the print dialog, turn
  scaling off: choose *Actual size* or 100%, and disable any *fit to page*.
- **Pixels.** Sets the longest edge of the image (default 4096 px). The DPI is
  whatever that works out to, and the report line shows it. If the request is
  too large for the pattern, the image is clamped and the report names the DPI
  to print at.

**Print / Export options**, all in the same panel:

- **Alignment Marks** (on by default). These are registration marks, not relief
  cuts. Each seam is marked at the same point on the shared edge and drawn on
  both pieces, so when two glued edges are lined up, the marks line up too.
  Marks are spread along each stretch of edge from one corner to the next:
  either a fixed number, evenly spaced (default 2), or one every set
  distance (default 25 mm), centred so none lands on a corner. Each mark is 6 mm long by default, and the numbering
  toggle is off. Turn the marks off if you do not want them.
- **Colour Pieces** (on by default). Each piece is filled with its own colour,
  so it can be matched to the model. Turn it off before printing on plain
  paper: outlines and numbers are what you want there.
- **Scale Ruler** (on by default). Prints a ruler and a one-inch square in a
  strip below the pattern. Measure the square after printing. If it is not
  exactly one inch, the print was scaled, and every piece on the sheet is the
  wrong size. Reprint with scaling off rather than cutting from it.

**SVG and tiled pages.** The SVG export still exists. It is vector, so it
scales cleanly in a drawing program. A sheet is often bigger than a home
printer's paper, and `tools/tile_svg.py` slices an SVG into pages:

```
python tools/tile_svg.py pattern.svg
python tools/tile_svg.py pattern.svg --page a4 --out pages/ --overlap 12
python tools/tile_svg.py pattern.svg --scale 0.5    # half-size test print
```

Pages are Letter by default and printed at 1:1, because the sheet is in real
millimetres and tiling must not change that. `--scale` is only for a deliberate
reduced proof print. Each page carries crop marks, its row and column, and a
shaded overlap band to align against its neighbour. Options are `--page`
(`letter`, `legal`, `a4`, `a3`), `--landscape`, `--margin`, and `--overlap`
(in mm, default 10). Measure the ruler on page 1 before trusting the rest.

## Panels

All in the UV Editor sidebar (<kbd>N</kbd>):

- **FoamCraft ▸ Pattern** — Grease Pencil bake
- **FoamCraft ▸ Foam Cut Plan** — material, saw, darts, numbering
- **FoamCraft ▸ Foam Cut Plan ▸ Display** — what the overlay draws
- **FoamCraft ▸ Foam Cut Plan ▸ Craftability** — findings from the last check
- **FoamCraft ▸ Foam Cut Plan ▸ Print / Export** — image and SVG export, with
  alignment marks, piece colour and the scale ruler
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
- **The seam flags are not perfectly repeatable.** Cutting the same pattern
  twice can mark a couple of edges differently as seams. The pieces
  themselves are stable -- same count, same sizes, same geometry every time --
  so this has not been seen to change a cut pattern, but it is a known open
  defect rather than a settled one.
- **Overlay and printed sheet label angles differently.** The UV-editor
  overlay shows the bevel per side of a seam on screen, while the printed sheet
  shows the full fold angle. Being resolved.
- **Image export is lightly tested by hand.** Image export has been exercised
  by automated tests; the file dialog has had little hands-on use. Check the
  printed scale before cutting.
- **Bevel angles are computed, not proven.** The geometry is checked against
  shapes with arithmetically known answers (a sphere's curvature totals 4π, a
  cylinder's is zero, a cube's edges are all exactly 90°). That is a strong
  check on the maths and not a substitute for cutting one test panel first.

Found something else? [Open an
issue](https://github.com/drachis/blender-foam-craft/issues).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
