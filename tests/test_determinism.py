"""The same stroke on the same mesh must cut the same way every time.

    python tools/run_tests.py --only determinism

This is a test about the *other* tests. The shape sweep reported seam_edges=149
on one run and 150 on the next, from identical code against an identically
rebuilt sphere -- which means a change in that number is not evidence of
anything, and neither is a run that suddenly passes. Chasing a defect through a
measurement that moves on its own is how a session goes in circles: every fix
looks like it worked once, and every regression looks like bad luck.

So this pins the property directly. Cut the same pattern several times over,
each against a freshly built sphere in the same process, and assert every run
agrees down to the face counts. It fails loudly the moment an unordered
container gets back into the cutter.

Known source, already fixed once: BMesh elements hash by identity, so iterating
a *set* of them yields a different order per run. remove_doubles keeps the first
vertex of each cluster and welds the rest onto it, so a different order leaves a
different survivor and a different mesh. The same bug hit uv_islands, whose
flood fill seeded from set.pop().

The checks state both bounds. "All runs agree" is satisfied by a cutter that
does nothing at all -- every run would agree on zero -- so it is paired with an
assertion that the cut actually happened and produced the regions it should.
"""

import math
import sys
import importlib
import traceback

import bpy
import bmesh
from mathutils import Vector

_MOD = "bl_ext.user_default.foamcraft_drachis"
for _name in [n for n in list(sys.modules) if n.startswith(_MOD)]:
    del sys.modules[_name]

import bl_ext.user_default.foamcraft_drachis.foam_gp_pattern as gp    # noqa: E402

importlib.reload(gp)

SEGMENTS, RINGS = 96, 48
SNAP = 0.004
RUNS = 5


# Scene clearing goes through the guard, which refuses in a Blender this
# suite did not start. Imported here, self-contained, because these files set
# sys.path several different ways and the protection must not depend on which.
_GUARD_TOOLS = r"O:\Work\BlenderPlugin\FoamCraft\tools"
if _GUARD_TOOLS not in sys.path:
    sys.path.insert(0, _GUARD_TOOLS)
import sandbox_guard  # noqa: E402


def _clear():
    """Empty the scene -- but only in a Blender this suite started.

    Delegates to tools/sandbox_guard, which refuses when the session cannot
    prove it is disposable. On 2026-11-03 this function, called directly,
    deleted a user's unsaved sculpt after the launcher adopted their own
    Blender off a shared port. The guard is what stops that repeating.
    """
    sandbox_guard.clear_scene()
    for collection in (bpy.data.objects, bpy.data.meshes):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _sphere_bmesh():
    """A UV sphere whose faces are in a known order.

    ``primitive_uv_sphere_add`` does not give the same mesh twice. Measured
    across five rebuilds in one session: 4514 vertices and 4608 faces every
    time, every vertex coordinate identical to nine decimals -- and the face
    table different on every run. The sphere is the same shape with its faces
    numbered differently, most likely because the primitive is built by spinning
    a profile and welding the seam, and the weld does not order its output.

    That is enough to make the whole suite unreliable, because anything that has
    to break a tie between two equally-near faces breaks it by index. The stroke
    along the equator sits exactly on an edge for 54 of its 60 points, so it hits
    that tie constantly, and its cut came out different every run.

    Sorting by the vertices a face is made of fixes it, because the *vertices*
    are stable. This is fixture hygiene, not a workaround in the cutter: for a
    real user the mesh is whatever it is and its numbering does not change
    underneath them, so the cutter is deterministic where it counts. It was the
    test rig that moved.
    """
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=SEGMENTS,
                                         ring_count=RINGS)
    bm = bmesh.new()
    bm.from_mesh(bpy.context.object.data)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    # bmesh's sort takes a *number*, not a tuple, so the face's vertex indices
    # are packed into one. Four indices in base 4514 reach 4.2e14, well inside
    # what a double holds exactly (9e15), so no two faces collide by rounding.
    # Poles are triangles and pad with a zero, which orders them consistently
    # too -- all that is being asked for here is *an* order, the same one twice.
    base = len(bm.verts) + 1

    def _key(face):
        idx = sorted(v.index for v in face.verts)[:4]
        idx += [0] * (4 - len(idx))
        return ((idx[0] * base + idx[1]) * base + idx[2]) * base + idx[3]

    bm.faces.sort(key=_key)
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    return bm


def _point(theta, phi):
    return Vector((math.sin(theta) * math.cos(phi),
                   math.sin(theta) * math.sin(phi),
                   math.cos(theta)))


def _circle(centre_theta, centre_phi, radius, count=120):
    centre = _point(centre_theta, centre_phi)
    helper = Vector((0.0, 0.0, 1.0))
    if abs(centre.dot(helper)) > 0.9:
        helper = Vector((1.0, 0.0, 0.0))
    e1 = centre.cross(helper).normalized()
    e2 = centre.cross(e1).normalized()
    return [(centre * math.cos(radius)
             + (e1 * math.cos(2 * math.pi * i / count)
                + e2 * math.sin(2 * math.pi * i / count)) * math.sin(radius))
            for i in range(count)]


def _regions(bm):
    seen, sizes = set(), []
    for face in bm.faces:
        if face in seen:
            continue
        stack, size = [face], 0
        seen.add(face)
        while stack:
            current = stack.pop()
            size += 1
            for loop in current.loops:
                if loop.edge.seam:
                    continue
                for other in loop.edge.link_faces:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _cut_once(strokes):
    """Build a sphere from scratch, cut it, and report what came out."""
    _clear()
    bm = _sphere_bmesh()
    try:
        edges, open_ends = gp.cut_strokes(bm, strokes, snap_distance=SNAP)
        sizes = _regions(bm)
        return {
            "regions": len(sizes),
            "sizes": sizes[:6],
            "seam_edges": len(edges),
            "open_ends": open_ends,
            "verts": len(bm.verts),
            "faces": len(bm.faces),
            "degenerate": sum(1 for f in bm.faces if f.calc_area() < 1e-8),
        }
    finally:
        bm.free()


def _theta(radius):
    circle = _circle(1.5708, 0.0, radius)
    over = radius * 1.25
    bar = [_point(1.5708, -over + 2 * over * i / 59.0) for i in range(60)]
    return [(circle, True), (bar, False)]


def _repeat(strokes):
    runs = [_cut_once(strokes) for _ in range(RUNS)]
    first = runs[0]
    varying = sorted(k for k in first
                     if any(r[k] != first[k] for r in runs[1:]))
    return runs, varying


# --- 1. A theta, cut five times ----------------------------------------
def test_theta_cuts_identically_every_time():
    """Five identical cuts, five identical meshes.

    The theta is the pattern the sweep disagrees with itself about, so it is the
    one worth pinning. Every reported number is compared, not just the region
    count: vertex and face totals catch a weld landing on a different survivor
    even when the region count happens to survive it.
    """
    runs, varying = _repeat(_theta(0.28))
    return {
        "runs": runs,
        "fields_that_varied": varying,
        "checks": {
            "every_run_agrees": not varying,
            # Without this, a cutter that did nothing would pass the above.
            "the_cut_happened": runs[0]["seam_edges"] > 0,
            "three_regions": runs[0]["regions"] == 3,
        },
    }


# --- 2. A closed circle, cut five times --------------------------------
def test_circle_cuts_identically_every_time():
    """The simplest closed stroke, repeated.

    A single loop has no crossings, so if this varies the instability is in the
    plain path of the cutter rather than in junction handling -- which is a
    different bug in a different place, and worth telling apart.
    """
    runs, varying = _repeat([(_circle(1.5708, 0.0, 0.25), True)])
    return {
        "runs": runs,
        "fields_that_varied": varying,
        "checks": {
            "every_run_agrees": not varying,
            "the_cut_happened": runs[0]["seam_edges"] > 0,
            "two_regions": runs[0]["regions"] == 2,
        },
    }


# --- 3. Each half of the theta, on its own ------------------------------
def test_each_theta_stroke_alone_is_stable():
    """Which of the two strokes is the unstable one.

    The theta varies and a lone circle does not, which leaves three candidates:
    the circle at *this* radius, the bar, or only the combination of the two.
    Cutting each stroke by itself tells them apart, and that decides where to
    look -- a bar that is unstable alone is a plain-path bug, while two stable
    strokes that vary together is a junction bug.

    The bar is the suspicious one. It runs along the equator of a UV sphere with
    an even ring count, which puts it exactly on an existing edge loop, and a
    point sitting exactly on an edge is the same distance from the faces on
    either side of it -- so the nearest-face lookup has a tie to break.
    """
    radius = 0.28
    over = radius * 1.25
    circle = _circle(1.5708, 0.0, radius)
    bar = [_point(1.5708, -over + 2 * over * i / 59.0) for i in range(60)]

    circle_runs, circle_varied = _repeat([(circle, True)])
    bar_runs, bar_varied = _repeat([(bar, False)])

    return {
        "circle_alone": circle_runs,
        "circle_fields_that_varied": circle_varied,
        "bar_alone": bar_runs,
        "bar_fields_that_varied": bar_varied,
        "checks": {
            "circle_alone_agrees": not circle_varied,
            "bar_alone_agrees": not bar_varied,
            # Both must actually cut, or "agrees" is a statement about nothing.
            "circle_cut_something": circle_runs[0]["seam_edges"] > 0,
            "bar_cut_something": bar_runs[0]["seam_edges"] > 0,
        },
    }


# --- 4. The nearest-face lookup itself ----------------------------------
def test_points_on_an_edge_land_on_the_same_face_every_time():
    """A point exactly on an edge has two nearest faces. Which one is picked?

    This is the layer under the cutter. The bar runs along the equator of a
    sphere with an even ring count, so it lies exactly on an existing edge loop,
    and every one of its points is the same distance from the face above and the
    face below. find_nearest has to break that tie, and if it breaks it
    differently on different runs then everything built on top inherits the
    wobble -- no amount of care in the cutter can settle a foundation that moves.

    Measured separately from the cut so the answer is unambiguous: this reports
    the raw face indices, not their downstream consequences.

    Both bounds again. Identical assignments prove nothing unless the ties are
    real, so the count of genuinely tied points is reported and asserted
    non-zero -- otherwise this passes on a stroke that never sat on an edge.
    """
    radius = 0.28
    over = radius * 1.25
    bar = [_point(1.5708, -over + 2 * over * i / 59.0) for i in range(60)]

    assignments, tie_counts = [], []
    for _ in range(RUNS):
        _clear()
        bm = _sphere_bmesh()
        try:
            bvh = gp.BVHTree.FromBMesh(bm)
            # The cutter's own classification, not the raw tree. find_nearest is
            # entitled to break a tie however it likes -- what has to be stable
            # is the answer the cutter builds on top of it.
            #
            # Described geometrically, as a kind plus a rounded position, rather
            # than by element index. Indices are exactly what must not matter
            # here: welds and cuts renumber a mesh, so a check written against
            # numbering would pass while the real property failed.
            indices = [(h["kind"], tuple(round(c, 7) for c in h["loc"]))
                       for h in gp._classify(bm, bvh, bar)]
            ties = 0
            for point in bar:
                loc, _n, index, dist = bvh.find_nearest(point)
                # A tie is where the *landed* point sits on an edge shared by
                # two faces -- both are then exactly as near, and which one gets
                # named is arbitrary.
                #
                # An earlier version of this asked whether the point was on the
                # surface (`loc == point`) and counted zero, which is true and
                # irrelevant: a point on a sphere stands off the flat faces
                # approximating it. The tie is between faces, not between the
                # point and the mesh, and the check has to say so.
                face = bm.faces[index]
                for edge in face.edges:
                    a, b = edge.verts[0].co, edge.verts[1].co
                    span = b - a
                    if span.length_squared < 1e-18:
                        continue
                    t = max(0.0, min(1.0, (loc - a).dot(span)
                                     / span.length_squared))
                    if ((loc - (a + span * t)).length < 1e-6
                            and len(edge.link_faces) == 2):
                        ties += 1
                        break
            assignments.append(indices)
            tie_counts.append(ties)
        finally:
            bm.free()

    first = assignments[0]
    differing = [i for i in range(len(first))
                 if any(a[i] != first[i] for a in assignments[1:])]

    return {
        "tie_counts": tie_counts,
        # The indices themselves, not just whether they matched. A constant
        # offset between runs means something else entirely from a scatter.
        "first_eight_per_run": [a[:8] for a in assignments],
        "points_that_moved": differing[:20],
        "points_that_moved_total": len(differing),
        "checks": {
            "same_face_every_run": not differing,
            # Without this the check above is vacuous on a stroke off the edges.
            "the_points_really_are_on_edges": tie_counts[0] > 0,
        },
    }


# --- 5. Is it the lookup, or the mesh underneath it? --------------------
def test_the_sphere_is_rebuilt_identically():
    """Separate "find_nearest wobbles" from "the sphere is not the same sphere".

    Every stroke point changed face between runs, which is far too many for a
    tie-break on a handful of boundary points. That points at the input rather
    than the query: if the primitive comes back rotated, or with its vertices in
    a different order, then face index 900 is simply not the same face twice and
    nothing downstream can be stable.

    Two questions, kept apart:
      - queried twice against *one* tree, does find_nearest agree with itself?
      - built twice, is the mesh the same mesh?

    The second is checked on the actual coordinates, not the counts. A rotated
    sphere has identical vertex, edge and face totals, so counting would report
    all-clear on exactly the failure being looked for.
    """
    radius = 0.28
    over = radius * 1.25
    bar = [_point(1.5708, -over + 2 * over * i / 59.0) for i in range(60)]

    coords, repeat_query_stable = [], True
    for _ in range(RUNS):
        _clear()
        bm = _sphere_bmesh()
        try:
            # Every vertex and the whole face table, not a sample. Comparing the
            # first 200 vertices reported "same sphere" while the face numbering
            # was shifting underneath -- the vertices genuinely did match, and
            # the thing that mattered was not being looked at.
            coords.append((
                len(bm.verts), len(bm.faces),
                tuple(tuple(round(c, 9) for c in v.co) for v in bm.verts),
                tuple(tuple(v.index for v in f.verts) for f in bm.faces),
            ))
            bm.faces.ensure_lookup_table()
            bvh = gp.BVHTree.FromBMesh(bm)
            once = [bvh.find_nearest(p)[2] for p in bar]
            twice = [bvh.find_nearest(p)[2] for p in bar]
            repeat_query_stable = repeat_query_stable and once == twice
        finally:
            bm.free()

    same_mesh = all(c == coords[0] for c in coords[1:])
    return {
        "counts_per_run": [(c[0], c[1]) for c in coords],
        "verts_match": all(c[2] == coords[0][2] for c in coords[1:]),
        "faces_match": all(c[3] == coords[0][3] for c in coords[1:]),
        "checks": {
            # One tree, same question twice: this must never vary.
            "one_tree_answers_consistently": repeat_query_stable,
            "the_sphere_is_the_same_sphere": same_mesh,
        },
    }


# --- 6. Renumber the faces and cut again --------------------------------
def test_face_numbering_does_not_change_the_cut():
    """The same mesh, its faces numbered differently, must cut the same.

    This is the property that matters, and it is stronger than "two runs agree".
    Meshes get renumbered constantly in Blender -- every cut does it, and so does
    every weld, including the remove_doubles that ``bmo_create_uvsphere_exec``
    finishes the sphere primitive with. A cutter whose output depends on face
    numbering is reading something the mesh does not promise to keep, and it will
    read differently on a sculpt that has merely been edited since.

    Rather than hope no two faces ever tie, the tie is forced: the same geometry
    is presented in three different face orders -- as built, sorted, and reversed
    -- and all three must give the same cut. Sorting the fixture makes runs
    repeatable; this says the cutter did not need it.

    Two-sided: identical outputs would also be produced by a cutter that did
    nothing at all, so the cut is asserted to have happened and to have made the
    three regions a theta makes.
    """
    strokes = _theta(0.28)
    results = {}
    for label in ("as_built", "sorted", "reversed"):
        _clear()
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=SEGMENTS,
                                             ring_count=RINGS)
        bm = bmesh.new()
        bm.from_mesh(bpy.context.object.data)
        try:
            bm.verts.ensure_lookup_table()
            bm.verts.index_update()
            base = len(bm.verts) + 1

            def _key(face):
                idx = sorted(v.index for v in face.verts)[:4]
                idx += [0] * (4 - len(idx))
                return ((idx[0] * base + idx[1]) * base + idx[2]) * base + idx[3]

            if label == "sorted":
                bm.faces.sort(key=_key)
            elif label == "reversed":
                bm.faces.sort(key=_key, reverse=True)
            bm.faces.ensure_lookup_table()
            bm.faces.index_update()

            edges, _ends = gp.cut_strokes(bm, strokes, snap_distance=SNAP)
            sizes = _regions(bm)
            results[label] = {
                "regions": len(sizes),
                "sizes": sizes[:6],
                "seam_edges": len(edges),
                "verts": len(bm.verts),
                "faces": len(bm.faces),
            }
        finally:
            bm.free()

    first = results["sorted"]
    agree = all(r == first for r in results.values())
    return {
        "results": results,
        "checks": {
            "numbering_does_not_matter": agree,
            "the_cut_happened": first["seam_edges"] > 0,
            "three_regions": first["regions"] == 3,
        },
    }



def test_which_edges_the_theta_disagrees_about():
    """Name the edges that come and go, instead of only counting them.

    test_theta_cuts_identically_every_time says `seam_edges` is 174 or 176 and
    stops there, which is enough to fail a build and not enough to fix
    anything. A count cannot say whether the two extra edges are at the
    crossings, at the stroke ends, or scattered -- and those are three
    different bugs.

    So: cut the same theta repeatedly, record the midpoint of every edge the
    cutter returns, and take the symmetric difference across runs. Midpoints
    rather than indices, because BMesh renumbers and an index is not a name
    (CLAUDE.md, "Verify the fixture before chasing the code").

    The theta crosses itself exactly twice, at the two points where the bar
    meets the circle. Those sit at phi = +/- radius along the equator, and are
    reported alongside so the comparison is arithmetic rather than by eye.

    No expected value is asserted for *where* they are, because that is the
    open question. What is asserted is the shape of the answer: the unstable
    edges must be few, and they must be a strict subset of what a single run
    returns -- if every edge were unstable the cut would not be reproducible
    at all, and the diagnosis above would be wrong.
    """
    radius = 0.28
    strokes = _theta(radius)

    runs = []
    for _ in range(RUNS):
        _clear()
        bm = _sphere_bmesh()
        try:
            edges, _open = gp.cut_strokes(bm, strokes, snap_distance=SNAP)
            # Copy out floats immediately -- a BMEdge must not outlive the
            # call that produced it.
            runs.append({tuple(round(c, 5) for c in
                               ((e.verts[0].co + e.verts[1].co) / 2.0))
                         for e in edges})
        finally:
            bm.free()

    common = set.intersection(*runs)
    union = set.union(*runs)
    unstable = sorted(union - common)

    # Where the bar crosses the circle, by construction.
    crossings = [_point(1.5708, -radius), _point(1.5708, radius)]

    def _nearest_crossing(point):
        v = Vector(point)
        return round(min((v - c).length for c in crossings), 5)

    return {
        "counts_per_run": [len(r) for r in runs],
        "stable_edges": len(common),
        "unstable_edges": len(unstable),
        "unstable_midpoints": unstable,
        # How far each wandering edge sits from the nearer crossing. If the
        # junction theory is right these are small; if they are large the
        # theory is wrong and the write-up needs correcting.
        "distance_to_nearest_crossing":
            [_nearest_crossing(p) for p in unstable],
        "crossings": [[round(c, 5) for c in pt] for pt in crossings],
        "checks": {
            # The cut is mostly reproducible, or nothing above means anything.
            "most_edges_are_stable": len(common) > 0.9 * len(union),
            # Both bounds: a run that returned nothing would satisfy the above.
            "the_cut_happened": len(common) > 0,
        },
    }


TESTS = (
    test_theta_cuts_identically_every_time,
    test_which_edges_the_theta_disagrees_about,
    test_circle_cuts_identically_every_time,
    test_each_theta_stroke_alone_is_stable,
    test_points_on_an_edge_land_on_the_same_face_every_time,
    test_the_sphere_is_rebuilt_identically,
    test_face_numbering_does_not_change_the_cut,
)

result = {}
_all_pass = True
for _test in TESTS:
    try:
        _report = _test()
        _report["PASS"] = all(_report.get("checks", {}).values())
    except Exception:
        _report = {"ERROR": traceback.format_exc().splitlines()[-5:],
                   "PASS": False}
    result[_test.__name__] = _report
    _all_pass = _all_pass and _report["PASS"]
result["ALL_PASS"] = _all_pass
