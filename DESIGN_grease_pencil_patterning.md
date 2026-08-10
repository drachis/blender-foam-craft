# Drawing the pattern instead of unwrapping it

A design note, not a plan. Nothing here is built.

## The idea

Draw the piece boundaries directly onto the sculpt with Grease Pencil, cut the
mesh along those strokes, and take the resulting patches as the UV islands.

This mirrors what the workshop already does: tape over the 3D print, draw the
piece boundaries on the tape by eye, cut along the lines, peel off, flatten.
The maker decides where pieces go, because they are thinking about the saw, the
grain, where a seam will show, and how big a piece their hands and their foam
can manage.

## Why it is worth considering

Everything the craftability advisor currently does exists to detect damage
caused by unwrapping for the wrong purpose.

Smart UV Project, Angle Based, Lightmap Pack -- all of them optimise for
**texture packing**: minimal distortion, minimal wasted UV area, islands that
tile well. None of them know or care that each island becomes a physical piece
of foam that has to be sawn, handled and glued.

That mismatch produced every finding we added:

| Finding | What it is really detecting |
|---|---|
| `tight_notch` | a spike the packer left because it is free in UV space |
| `sliver_islands` | a strip that packs well and cuts badly |
| `exceeds_stock` | an island sized with no idea what stock exists |
| `non_developable` | a patch chosen without regard for curvature |
| `degenerate_unwrap` | unwrapping with no seams at all |

Measured on Suzanne, Smart UV Project gave 100 pieces, of which 76 were flagged
for heat forming and 4 were non-developable. That flag rate is so high it stops
being guidance. The tool is reporting, accurately, that an automatic unwrap is
not a pattern.

Drawing the pattern inverts the dependency: seams come from intent, and the
unwrap follows. The advisor then checks a pattern a person meant to draw,
rather than diagnosing one nobody chose.

## The pipeline, in existing Blender operators

All of these were confirmed present on Blender 5.2.0 in this session.

1. **Draw on the surface.**
   `tool_settings.gpencil_stroke_placement_view3d = 'SURFACE'` makes strokes
   land on the mesh rather than on a plane. `GREASEPENCIL` is a first-class
   object type in 5.2.

2. **Convert strokes to a curve.**
   `bpy.ops.object.convert(target='CURVE')`. Confirmed targets: `CURVE`,
   `MESH`, `POINTCLOUD`, `CURVES`, `GREASEPENCIL`.

3. **Cut the curve into the mesh.**
   `bpy.ops.mesh.knife_project(cut_through=False)`. This is the operator the
   whole idea depends on.

4. **Mark the new edges as seams.**
   `bpy.ops.mesh.mark_seam()` on the selection knife_project leaves behind.

5. **Unwrap.**
   Any method. With the seams already placed, the unwrap has no freedom to
   invent island boundaries -- it only flattens what was drawn.

Nothing here is reimplemented. The add-on's job would be sequencing these and
handling the parts the operators do not cover.

## The hard part

**`knife_project` cuts along the view direction, not along surface normals.**
It projects the curve from the current viewpoint, the way the knife tool works
interactively. For anything that wraps -- a horn, a head, the sides of a snout
-- one viewpoint cannot reach the whole surface, and `cut_through` will also
cut geometry behind the intended target.

That is the central problem to solve, not a detail. Options, none tested:

- **Cut per view.** Split the strokes by which viewpoint faces them, orbit, and
  knife_project once per view. Needs a rule for assigning a stroke to a view,
  and the seams must line up where the views meet.
- **Cut per stroke, along the stroke's own normal.** Use the average surface
  normal under each stroke as the projection direction. More cuts, each better
  aligned, but the joins between them are where it would fail.
- **Bypass knife_project.** Walk the stroke across the mesh directly, splitting
  edges where it crosses them, using a BVH tree for the surface queries. Full
  control, no view dependency -- but this is reimplementing a knife, which is
  exactly the kind of thing this project's working practice warns against.

The first two use the built-in and should be tried before the third is
considered.

## Testing this will be awkward, and that is the real cost

Everything verified so far had a ground truth. A 24-segment cylinder has a
15 degree dihedral, a cube has six islands with four neighbours each, a 2 m
cube exports at 2000 mm. Assertions could be written against arithmetic.

A drawn pattern has no such answer. It is a judgement about where seams should
go, and two competent makers would draw different ones, both fine.

What can still be tested automatically:

- **The pipeline runs.** Synthetic strokes, generated rather than drawn, still
  exercise convert, knife_project, mark_seam and unwrap end to end.
- **Cuts land on the strokes.** Sample points along each stroke and assert a
  seam edge lies within a tolerance of each. This catches the view-direction
  problem directly, which is the thing most likely to break.
- **Islands match the drawn regions.** Draw a closed loop on a sphere, assert
  it yields exactly two islands and that face counts either side match the
  region the loop encloses.
- **Nothing is cut that should not be.** With `cut_through=False`, assert no
  seam appears on faces pointing away from the projecting view.
- **Round trip.** `uv.seams_from_islands` recovers seams from islands, so
  seams -> unwrap -> seams should be stable.

What cannot be tested automatically is whether the pattern is *good*. That is a
question for the craftability advisor, which already exists and would apply
unchanged -- and a drawn pattern should trip far fewer of its findings than an
automatic unwrap does. That difference is itself a measurable result worth
recording: run the advisor on a drawn pattern and on Smart UV Project over the
same sculpt, and compare the counts.

## Where this leaves the current tool

Nothing built so far is wasted. The overlay, the bevel angles, the dart runs,
the piece numbering, the SVG export and the advisor all consume UV islands and
do not care how those islands came to exist. This would replace the step
*before* them, not any of them.

The one thing that shifts is emphasis: several advisor findings exist to warn
about damage an automatic unwrap causes. If patterns are drawn, those become
rare, and the advisor's centre of gravity moves toward genuine craft limits --
bevels beyond the saw, pieces beyond the stock, surfaces that will not lie
flat however they are cut.
