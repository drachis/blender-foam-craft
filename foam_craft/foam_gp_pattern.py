"""Cut a drawn stroke into a mesh as real topology, then seam it.

The pattern for a foam piece is drawn on the sculpt and has to become UV island
boundaries. That means the drawn line must exist in the mesh as actual vertices
and edges, because Blender's Unwrap reads seams from real mesh data.

Blender has no single operator for this. What it does have, and what this module
uses, are the two primitives its own Knife tool is built from:

    bmesh.utils.edge_split(edge, vert, fac)          -> a vertex on an edge
    bmesh.utils.face_split(face, va, vb, coords=[])  -> a face split along a path

``face_split``'s ``coords`` is the important one: it splits a face from one
boundary vertex to another *through an arbitrary polyline*, so curvature inside
a single face is reproduced exactly. Measured on 5.2 against an S-curve with
seven inflections across one quad: two faces out, seven interior vertices, and
zero deviation from the requested points.

Routes that were tried and do not work, recorded so they are not retried:

- ``MeshBoolean``'s **Intersecting Edges** output is dead on 5.2. The socket is
  still declared but ``enabled=False`` under every operation, and a link forced
  by index reports ``is_valid=False``.
- ``bmesh.ops.bisect_plane`` cuts with an infinite plane, so it can only split
  faces it passes *through*. It cannot place a vertex inside a face, so an
  S-curve across a single quad comes out as straight chords.
- ``MESH_OT_knife_tool`` registers only ``invoke`` and ``modal``, so it cannot be
  handed a point list, and driving it with simulated events needs Blender started
  with ``--enable-event-simulate`` -- which no ordinary user does.
- ``bpy.ops.mesh.knife_project`` cuts along the *view* direction, so no single
  viewpoint can reach a surface that wraps.

What this module writes is only the bookkeeping: which face each stroke point
falls in, and where the stroke crosses from one face to the next. Every change to
the topology is made by Blender.
"""

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_line_line

from . import foam_angle_core as core
from . import foam_log

_log = foam_log.get_logger()

# How many times a face may be subdivided to let a stroke drawn inside it cross
# an edge. Each pass turns one face into 16, so this is a factor of 65536 -- far
# past the point where the mesh, not the cutter, is the problem.
_MAX_REFINE = 4


class StrokeCutError(Exception):
    """The stroke could not be walked across the mesh.

    Raised rather than guessing. Every case that reaches this is one where
    continuing would put the seam somewhere the user did not draw, and a seam in
    the wrong place is worse than no seam -- it becomes a cut in foam.
    """


def _densify(bm, bvh, points, cyclic, max_depth=6):
    """Add stroke samples, but only where the walk would otherwise fail.

    The walk works out where a stroke leaves one face for the next, so it needs
    successive points on faces that touch. A stroke drawn with long steps -- or
    a fine mesh under a coarse stroke -- skips faces, and the walk had no choice
    but to refuse it. Its own error said "resample the stroke more finely",
    which is something this can simply do: the line between two drawn points is
    a straight run on the surface, so filling it in adds no information the user
    did not give and loses none.

    Refined by bisection, and only on the segments that need it. Sampling the
    whole stroke to a fixed fraction of a face was tried first and is the wrong
    trade: every added sample becomes a vertex in the finished cut, so a blanket
    pass litters the seam with detail nobody drew and nothing needs. Here a
    segment is split only while its ends sit on faces that do not touch, so a
    stroke already fine enough passes through untouched and cost is paid only
    where the mesh is finer than the drawing.
    """
    if len(points) < 2:
        return points
    bm.faces.ensure_lookup_table()

    def face_at(point):
        _loc, _nrm, index, _dist = bvh.find_nearest(point)
        return bm.faces[index] if index is not None else None

    def touching(face_a, face_b):
        if face_a is None or face_b is None or face_a is face_b:
            return True
        return (_shared_edge(face_a, face_b) is not None
                or _shared_vert(face_a, face_b) is not None)

    def refine(p0, p1, depth, out):
        if depth >= max_depth or touching(face_at(p0), face_at(p1)):
            return
        middle = p0.lerp(p1, 0.5)
        refine(p0, middle, depth + 1, out)
        out.append(middle)
        refine(middle, p1, depth + 1, out)

    result = []
    count = len(points)
    last = count if cyclic else count - 1
    for i in range(last):
        a = points[i]
        b = points[(i + 1) % count]
        result.append(a)
        if (b - a).length < 1e-12:
            continue
        added = []
        refine(a, b, 0, added)
        result.extend(added)
    if not cyclic:
        result.append(points[-1])
    return result


def _assign_faces(bvh, bm, points):
    """Which face does each stroke point sit on, and where exactly.

    Returns [(face, surface_point)]. The stroke is drawn on the surface but its
    points are only approximately on it, so each is pulled to the real nearest
    point -- otherwise a point hovering microscopically above a face would land
    the cut on the wrong side of an edge.
    """
    bm.faces.ensure_lookup_table()
    out = []
    for p in points:
        loc, _nrm, index, _dist = bvh.find_nearest(p)
        if index is None:
            raise StrokeCutError(
                "a stroke point found no surface; is the stroke near the mesh?")
        out.append((bm.faces[index], loc.copy()))
    return out


def _shared_edge(face_a, face_b):
    for edge in face_a.edges:
        if face_b in edge.link_faces:
            return edge
    return None


def _shared_vert(face_a, face_b):
    """The corner two faces meet at, when they share no edge.

    On a quad grid this is the common case wherever a stroke passes close to a
    grid vertex: the point before and the point after sit on diagonal
    neighbours, which touch at one corner and share no edge at all. Treating
    that as an error rejected an ordinary circle drawn on a plane.

    The crossing is then the corner itself, and no edge needs splitting -- the
    stroke goes through a vertex that is already there.
    """
    verts_b = set(face_b.verts)
    for vert in face_a.verts:
        if vert in verts_b:
            return vert
    return None


def _crossing_on_edge(edge, p_from, p_to):
    """Where the stroke segment crosses this edge, as a factor along it.

    The segment and the edge are skew lines in 3D -- the stroke sits on a curved
    surface and the edge is straight -- so they rarely intersect exactly.
    ``intersect_line_line`` gives the closest approach on each, and the factor is
    taken from the point on the edge and clamped, because a crossing outside the
    edge's own span is not on this edge at all.
    """
    a, b = edge.verts[0].co, edge.verts[1].co
    hit = intersect_line_line(p_from, p_to, a, b)
    if hit is None:
        # Exactly parallel. Fall back to the midpoint of the stroke segment
        # projected onto the edge, which is the best available answer.
        target = (p_from + p_to) * 0.5
    else:
        target = hit[1]
    span = b - a
    denom = span.length_squared
    if denom < 1e-18:
        return 0.0
    return max(0.0, min(1.0, (target - a).dot(span) / denom))


def _vert_at(edge, fac, snap_distance):
    """Put a vertex on `edge` at `fac`, or reuse an endpoint if close enough.

    Reuse matters. A drawn line has the width of a marker tip while the curve
    derived from it is idealised to zero width, so a crossing that lands a
    fraction of a millimetre from an existing corner was, as drawn, *on* that
    corner. Splitting there anyway litters the mesh with near-duplicate vertices
    and zero-length edges, which then become unsawable slivers in the pattern.
    """
    a, b = edge.verts[0], edge.verts[1]
    length = (b.co - a.co).length
    if length > 1e-12 and snap_distance > 0.0:
        if fac * length <= snap_distance:
            return a
        if (1.0 - fac) * length <= snap_distance:
            return b
    _new_edge, new_vert = bmesh.utils.edge_split(edge, a, fac)
    return new_vert


def _endpoint_vert(face, point, snap_distance):
    """Resolve a stroke's loose end onto the boundary of the face it sits in.

    A pattern maker crosses strokes deliberately, so that the pieces they bound
    close up -- an end is normally drawn *onto* another line rather than left
    hanging. Strokes are cut one after another, so by the time this one is being
    cut, the line it ends on is already real mesh topology and the end is lying
    on an ordinary edge. Splitting that edge turns the end into a boundary
    vertex, which is exactly what ``face_split`` needs.

    Returns None when the end really is hanging in open space, further than the
    tolerance from anything. Nothing is invented to rescue it: a seam that
    dead-ends mid-face separates no regions anyway, so extending it to the
    nearest boundary would add a cut nobody drew and still achieve nothing.
    """
    if snap_distance <= 0.0:
        return None
    best_edge, best_fac, best_dist = None, 0.0, None
    for edge in face.edges:
        a, b = edge.verts[0].co, edge.verts[1].co
        span = b - a
        denom = span.length_squared
        fac = 0.0 if denom < 1e-18 else \
            max(0.0, min(1.0, (point - a).dot(span) / denom))
        dist = (point - (a + span * fac)).length
        if best_dist is None or dist < best_dist:
            best_edge, best_fac, best_dist = edge, fac, dist
    if best_edge is None or best_dist > snap_distance:
        return None
    return _vert_at(best_edge, best_fac, snap_distance)


def _run_follows_edge(interior, entry, exit_v, snap_distance):
    """Is this run actually lying along the edge joining its two ends?

    Two ends being neighbours is not enough on its own. A stroke can enter and
    leave a face through vertices that happen to be adjacent while bowing well
    away from the edge between them, and taking that edge instead would move the
    cut by up to half an edge length -- measured at 0.061 on a grid whose edges
    are 0.125, against 0.005 when the path is cut properly.

    So the run only counts as following the edge when every point of it stays
    within the snap tolerance of that edge. Same tolerance as everywhere else:
    the width of the drawn line.
    """
    if not interior:
        return True
    if snap_distance <= 0.0:
        return False
    a, b = entry.co, exit_v.co
    span = b - a
    denom = span.length_squared
    if denom < 1e-18:
        return True
    for point in interior:
        fac = max(0.0, min(1.0, (point - a).dot(span) / denom))
        if (point - (a + span * fac)).length > snap_distance:
            return False
    return True


def _cut_path_edges(face, entry, exit_v, n_interior):
    """The edges face_split just created, walked off the new face's boundary.

    The new face is bounded by the cut on one side and by part of the original
    outline on the other, and both run between the same two vertices. They are
    told apart by length: the cut carries exactly the interior points that were
    asked for, so it is ``n_interior + 1`` edges long.

    Reading the path off the face beats diffing edge sets before and after. An
    earlier version took ``entry.link_faces[0].edges``, which picks an arbitrary
    neighbouring face and silently missed parts of the cut -- the seam came out
    with loose ends and the sphere stayed a single island.
    """
    start = next((l for l in face.loops if l.vert is entry), None)
    if start is None:
        raise StrokeCutError("split face does not contain the entry vertex")

    want = n_interior + 1
    limit = len(face.loops)

    # loop.edge joins loop.vert to link_loop_next.vert, so walking forward the
    # edge is read before stepping, and walking back it is read after.
    forward, backward = [], []
    loop = start
    for _ in range(limit):
        forward.append(loop.edge)
        loop = loop.link_loop_next
        if loop.vert is exit_v:
            break
    if len(forward) == want:
        return forward

    loop = start
    for _ in range(limit):
        loop = loop.link_loop_prev
        backward.append(loop.edge)
        if loop.vert is exit_v:
            break
    if len(backward) == want:
        return backward

    raise StrokeCutError(
        "could not trace the cut across the split face; wanted %d edges, "
        "found %d forward and %d back" % (want, len(forward), len(backward)))


def on_surface_runs(bm, points, tolerance):
    """Split a stroke into the stretches that actually lie on the surface.

    A drawn line does not stop politely at the edge of the sculpt. It runs off
    into space, past a rim, or across a gap and back on again -- and the part
    that did land on the surface is still a cut the maker meant to make.

    Without this, a point that missed found its "nearest" surface somewhere
    arbitrary -- the far rim of a cylinder, the other side of a torus -- and the
    cut was dragged there with it. Silently, and nowhere near anything drawn.
    That is worse than not cutting: it is a cut in the wrong place, which in
    foam is a ruined piece.

    Returns a list of point lists, each at least two points long. A stroke that
    never leaves the surface comes back as a single run, unchanged.
    """
    bvh = BVHTree.FromBMesh(bm)
    runs, current = [], []
    for point in points:
        _loc, _nrm, index, distance = bvh.find_nearest(point)
        on = index is not None and (tolerance <= 0.0 or distance <= tolerance)
        if on:
            current.append(point)
            continue
        if len(current) >= 2:
            runs.append(current)
        current = []
    if len(current) >= 2:
        runs.append(current)
    return runs


def cut_stroke(bm, points, snap_distance=0.0, cyclic=False,
               project_tolerance=0.0):
    """Cut `points` into `bm` as real edges. Returns the new edge list.

    `points` are in the mesh's own space and are expected to lie on or near its
    surface. Nothing is deleted and no face is moved; the mesh gains vertices
    along the drawn line and the faces it crosses are split by it.

    `cyclic` closes the stroke, joining its last point back to its first. A
    closed pattern line is the normal case -- a piece boundary is a loop -- and
    it is also the only shape that leaves no loose ends, since a cut stopping
    inside a face cannot be made.
    """
    if len(points) < 2:
        raise StrokeCutError("a stroke needs at least two points")

    # Drop anything that missed the surface, and cut what is left. A stroke
    # broken into pieces is no longer closed, whatever it was drawn as: the part
    # that ran off took the closure with it.
    if project_tolerance > 0.0:
        runs_on = on_surface_runs(bm, points, project_tolerance)
        if not runs_on:
            raise StrokeCutError(
                "no part of the stroke lies within %.4g of the surface"
                % project_tolerance)
        if len(runs_on) != 1 or len(runs_on[0]) != len(points):
            collected, open_ends = [], 0
            for piece in runs_on:
                edges, ends = cut_stroke(bm, piece,
                                         snap_distance=snap_distance,
                                         cyclic=False)
                collected.extend(edges)
                open_ends += ends
            return collected, open_ends

    bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)
    points = _densify(bm, bvh, points, cyclic)
    assigned = _assign_faces(bvh, bm, points)

    # A closed stroke lying wholly inside one face crosses no edge, so it has no
    # entry or exit and face_split -- which joins two *boundary* vertices -- has
    # nothing to work with. Left alone this returns silently having cut nothing,
    # which is the worst outcome: the pattern looks applied and is not there.
    #
    # The mesh is simply too coarse for the piece being drawn, so the face is
    # subdivided until the stroke crosses it. Refining where a line was drawn is
    # what a modeller would do by hand, and it adds geometry only inside the
    # face the stroke is already in.
    refined = 0
    while refined < _MAX_REFINE and len({f for f, _ in assigned}) == 1:
        face = assigned[0][0]
        bmesh.ops.subdivide_edges(bm, edges=list(face.edges), cuts=3,
                                  use_grid_fill=True)
        bm.faces.ensure_lookup_table()
        bvh = BVHTree.FromBMesh(bm)
        assigned = _assign_faces(bvh, bm, points)
        refined += 1
    if len({f for f, _ in assigned}) == 1:
        raise StrokeCutError(
            "the stroke stays inside a single face after %d subdivisions; it "
            "is far smaller than the mesh it is drawn on" % refined)

    # --- Group the stroke into per-face runs, and note every crossing.
    #
    # A crossing is where the stroke leaves one face for the next. Each is
    # recorded against the *shared edge* rather than resolved immediately,
    # because the vertex made there belongs to both faces: it is the exit of one
    # run and the entry of the next, and that shared vertex is the only thing
    # keeping the finished seam continuous rather than a row of disconnected
    # cuts.
    runs = []           # [{face, interior:[Vector], entry_edge, exit_edge, ...}]
    current = {"face": assigned[0][0], "interior": [assigned[0][1]],
               "entry": None, "exit": None}
    crossings = []      # (edge, fac, run_index_before, run_index_after)

    order = list(range(1, len(assigned)))
    if cyclic:
        order.append(0)          # walk the closing segment back to the start
    for i in order:
        face, surf_pt = assigned[i]
        if face is current["face"]:
            if i != 0:
                current["interior"].append(surf_pt)
            continue
        edge = _shared_edge(current["face"], face)
        if edge is not None:
            fac = _crossing_on_edge(edge, assigned[i - 1][1], surf_pt)
            crossings.append(("edge", (edge, fac), len(runs)))
        else:
            corner = _shared_vert(current["face"], face)
            if corner is None:
                # The stroke jumped a face entirely. Resampling finer is the
                # fix; guessing a route across the faces in between would put a
                # cut where nothing was drawn.
                raise StrokeCutError(
                    "consecutive stroke points landed on faces that share "
                    "neither an edge nor a corner; resample the stroke more "
                    "finely")
            crossings.append(("vert", corner, len(runs)))
        runs.append(current)
        if i == 0:
            # The closing segment landed back on the run the walk began in, so
            # that run gains this crossing as its entry rather than starting a
            # new one. This is what makes a loop have no ends.
            current = None
            break
        current = {"face": face, "interior": [surf_pt],
                   "entry": None, "exit": None}
    if current is not None:
        if cyclic and runs and current["face"] is runs[0]["face"]:
            # The stroke closed without crossing an edge on its final segment:
            # first and last runs are the same face, so they are one run.
            runs[0]["interior"] = current["interior"] + runs[0]["interior"]
            runs[0]["entry"] = current["entry"]
        else:
            runs.append(current)

    # --- Make every crossing vertex first, before any face is split.
    #
    # Order matters. edge_split on a shared edge changes both adjacent faces, so
    # doing it up front means each run finds its entry vertex already present.
    # Splitting faces first would invalidate the edges these crossings refer to.
    for kind, payload, run_index in crossings:
        if kind == "edge":
            vert = _vert_at(payload[0], payload[1], snap_distance)
        else:
            # A corner the stroke passed through. It already exists, so there is
            # nothing to split -- and splitting anything here would move a
            # vertex the user never drew near.
            vert = payload
        runs[run_index]["exit"] = vert
        # Wraps for a closed stroke, where the last run's exit is the first
        # run's entry. On an open stroke this index is always in range.
        runs[(run_index + 1) % len(runs)]["entry"] = vert

    # An open stroke's two loose ends. Resolved after the crossings, so an end
    # landing on a line cut earlier in this same call finds it already there.
    if not cyclic and runs:
        if runs[0]["entry"] is None and runs[0]["interior"]:
            runs[0]["entry"] = _endpoint_vert(
                runs[0]["face"], runs[0]["interior"][0], snap_distance)
            if runs[0]["entry"] is not None:
                runs[0]["interior"] = runs[0]["interior"][1:]
        last = runs[-1]
        if last["exit"] is None and last["interior"]:
            last["exit"] = _endpoint_vert(
                last["face"], last["interior"][-1], snap_distance)
            if last["exit"] is not None:
                last["interior"] = last["interior"][:-1]

    # --- Split each face along its run.
    new_edges = []
    skipped_open = 0
    for run in runs:
        entry, exit_v = run["entry"], run["exit"]
        if entry is None or exit_v is None:
            # The first and last runs of an open stroke have a loose end sitting
            # inside a face. A face can only be split between two of its own
            # boundary vertices, so a cut that stops halfway across one has
            # nowhere to land. Counted and reported rather than fudged.
            skipped_open += 1
            continue
        if entry is exit_v:
            # Entered and left at the same vertex: the run is shorter than the
            # snap distance, which is the tolerance saying to treat it as one
            # point. Nothing to cut.
            continue

        # The stroke may be running along an edge that is already there. Drawing
        # down an obvious feature line is a normal thing to do, and on a UV
        # sphere an equatorial stroke lands exactly on a ring of the mesh.
        #
        # face_split cannot help here -- there is no face to divide, because the
        # two vertices are already neighbours -- so the earlier code fell through
        # and cut nothing, silently. The line the user drew simply did not
        # appear. Marking the existing edge is the whole job: the topology it
        # would have created is present already.
        shared = next((e for e in entry.link_edges if exit_v in e.verts), None)
        if shared is not None and _run_follows_edge(run["interior"], entry,
                                                    exit_v, snap_distance):
            new_edges.append(shared)
            continue
        # Thin out points that sit on top of one another, or on top of the
        # vertices the cut runs between. face_split places one vertex per coord
        # regardless, so a duplicate becomes a zero-length edge and the faces
        # around it collapse -- measured as 4 zero-length edges and 3 degenerate
        # faces on a bar drawn along a sphere's equator, showing up as a pinched
        # black wedge in the viewport.
        #
        # Same rule as everywhere else: two marks closer together than the width
        # of the drawn line were one mark.
        kept = []
        for point in run["interior"]:
            if snap_distance > 0.0:
                if (point - entry.co).length <= snap_distance:
                    continue
                if (point - exit_v.co).length <= snap_distance:
                    continue
                if kept and (point - kept[-1]).length <= snap_distance:
                    continue
            kept.append(point)
        interior = [tuple(p) for p in kept]
        try:
            new_face, _new_loop = bmesh.utils.face_split(
                run["face"], entry, exit_v, coords=interior)
        except ValueError as exc:
            raise StrokeCutError(
                "face_split refused a run: %s" % exc) from exc
        if new_face is None:
            raise StrokeCutError("face_split returned no face")
        new_edges.extend(_cut_path_edges(new_face, entry, exit_v,
                                         len(interior)))

    # Marked here rather than left to the caller, and this is not tidiness.
    #
    # Cutting a later stroke can edge_split an edge that this one produced --
    # which is exactly what happens wherever two strokes cross, since the second
    # stroke's crossing lands on the first stroke's line. Splitting an edge
    # leaves the returned list holding one half while the other half is a new
    # edge nobody has a reference to, so marking the accumulated list at the end
    # of the whole pattern leaves gaps in every stroke that a later one crossed.
    #
    # Marking now means the flag is already on the edge when it is split, and
    # both halves inherit it. Measured on a theta -- a circle crossed by a bar
    # -- the old order gave 1, 3 or 2 regions depending on how far the bar
    # overhung, where the answer is always 3.
    for edge in new_edges:
        edge.seam = True

    _log.debug(foam_log.summary(
        "cut_stroke", points=len(points), runs=len(runs),
        crossings=len(crossings), open_ends=skipped_open,
        snap=snap_distance))
    return new_edges, skipped_open


def chain_strokes(strokes, tolerance):
    """Join strokes that share an end into longer paths.

    A pattern is drawn as many short lines meeting at junctions, and a junction
    is where the cutter is weakest: a stroke that *ends* there has a loose end
    sitting mid-face, and a face can only be split between two of its own
    boundary vertices, so that end cannot be cut at all. Cut separately, three
    lines meeting at a point contribute three unusable ends.

    Chained, the junction becomes an interior point of a longer path -- a real
    vertex the moment that path is cut -- and any remaining line can then land
    on it. Measured on a spherical Voronoi over 24 seeds, where every vertex is
    a three-way junction: 1 region became 6 and loose ends fell from 20 to 5.

    Greedy and order-dependent, which is acceptable here because it only has to
    reduce the number of ends, not find the best decomposition. A stroke that
    joins nothing comes back unchanged.
    """
    if tolerance <= 0.0:
        return list(strokes)

    # Closed strokes are passed straight through. A loop has no ends, so there
    # is nothing to join -- and joining *to* one is actively destructive: an
    # open stroke ending on a ring would absorb the ring into itself, turning a
    # closed boundary into part of an open path. Measured on three rings crossed
    # by six meridians, which should leave 14 pieces: chaining without this
    # guard merged 9 strokes into 7 and left a single region.
    chains = [(list(points), True)
              for points, cyclic in strokes if cyclic and len(points) >= 2]
    pending = [list(points) for points, cyclic in strokes
               if not cyclic and len(points) >= 2]
    while pending:
        current = pending.pop(0)
        joined = True
        while joined:
            joined = False
            for index, other in enumerate(pending):
                if (current[-1] - other[0]).length <= tolerance:
                    current = current + other[1:]
                elif (current[-1] - other[-1]).length <= tolerance:
                    current = current + list(reversed(other))[1:]
                elif (current[0] - other[-1]).length <= tolerance:
                    current = other[:-1] + current
                elif (current[0] - other[0]).length <= tolerance:
                    current = list(reversed(other))[:-1] + current
                else:
                    continue
                pending.pop(index)
                joined = True
                break
        # A chain that returns to where it started is a closed boundary, and
        # saying so leaves it with no ends to lose.
        cyclic = len(current) > 3 and             (current[0] - current[-1]).length <= tolerance
        chains.append((current, cyclic))
    return chains


def cut_strokes(bm, strokes, snap_distance=0.0, project_tolerance=0.0,
                chain_tolerance=0.0):
    """Cut several strokes, one after another. Returns (edges, open_ends).

    Order is what makes intersections work. A pattern maker crosses strokes on
    purpose, because that is what closes a region into a piece -- and once the
    first stroke has been cut, its line *is* mesh topology, so the second one
    crossing it is an ordinary edge crossing with nothing special about it. The
    junction needs no special case at all, as long as the strokes are not cut
    simultaneously.

    `strokes` is a sequence of (points, cyclic) pairs.
    """
    if chain_tolerance > 0.0:
        strokes = chain_strokes(strokes, chain_tolerance)

    all_edges = []
    open_ends = 0
    for index, (points, cyclic) in enumerate(strokes):
        try:
            edges, ends = cut_stroke(bm, points, snap_distance=snap_distance,
                                     cyclic=cyclic,
                                     project_tolerance=project_tolerance)
        except StrokeCutError as exc:
            # One bad stroke should not discard the rest of the pattern. Which
            # one failed is worth knowing, so it is named rather than counted.
            _log.warning(foam_log.summary(
                "stroke_failed", index=index, points=len(points),
                cyclic=cyclic, reason=str(exc)[:120]))
            continue
        all_edges.extend(edges)
        open_ends += ends
    # Weld the leftovers. Splitting an edge at a point that already has a vertex
    # a fraction away leaves a zero-length edge, and the faces around it
    # collapse -- a pinched wedge in the viewport, a spike no saw can follow in
    # the pattern. Guarding every split against every case it could arise from
    # is a losing game; merging by distance afterwards is Blender's own answer
    # and it is one call.
    #
    # Restricted to the vertices the cut produced. Run over the whole mesh it
    # would weld the sculpt's own detail, which is far worse than the slivers
    # it came to remove.
    if snap_distance > 0.0 and all_edges:
        # The cut's own vertices only. Widening this to their neighbours was
        # tried and made things worse -- theta went from 3 collapsed faces to 5
        # -- because pulling in surrounding geometry drags faces together
        # rather than tidying the strays.
        touched = {v for e in all_edges if e.is_valid for v in e.verts}
        if touched:
            bmesh.ops.remove_doubles(bm, verts=list(touched),
                                     dist=snap_distance)
            all_edges = [e for e in all_edges if e.is_valid]

    _log.info(foam_log.summary(
        "cut_strokes", strokes=len(strokes), edges=len(all_edges),
        open_ends=open_ends, snap=snap_distance))
    return all_edges, open_ends


# --- Reading strokes off a Grease Pencil object -------------------------

def strokes_from_grease_pencil(gp_object, target_object, at_frame=None):
    """Every stroke on `gp_object`, in `target_object`'s local space.

    Returns ``[(points, cyclic)]``, ready for :func:`cut_strokes`.

    Both objects carry their own transform and the cutter works in the target
    mesh's local space, so each point goes out through the Grease Pencil's world
    matrix and back in through the target's. Skipping that puts the pattern in
    the wrong place the moment either object is moved -- a failure that looks
    like nothing at all until the sculpt is not at the origin.

    ``cyclic`` is a curve-domain *attribute* rather than a property on the
    stroke, following Blender's Curves model, and it only exists once something
    has set it. Absent means every stroke is open, which is the right default
    for a line drawn by hand.

    The Grease Pencil object is read and never written. Patterning must not
    consume the drawing: the entire point of re-baking is to adjust a stroke and
    try again.
    """
    if gp_object is None or gp_object.type != 'GREASEPENCIL':
        raise StrokeCutError("no Grease Pencil object to pattern from")

    to_local = target_object.matrix_world.inverted() @ gp_object.matrix_world
    out = []
    for layer in gp_object.data.layers:
        if layer.hide:
            continue
        frames = list(layer.frames)
        if not frames:
            continue
        if at_frame is None:
            drawing = frames[0].drawing
        else:
            usable = [f for f in frames if f.frame_number <= at_frame] or frames
            drawing = max(usable, key=lambda f: f.frame_number).drawing

        cyclic_attr = drawing.attributes.get("cyclic")
        for index, stroke in enumerate(drawing.strokes):
            points = [to_local @ p.position for p in stroke.points]
            if len(points) < 2:
                continue
            closed = False
            if cyclic_attr is not None and index < len(cyclic_attr.data):
                closed = bool(cyclic_attr.data[index].value)
            out.append((points, closed))
    if not out:
        raise StrokeCutError(
            "the Grease Pencil object has no strokes on a visible layer")
    return out


class OBJECT_OT_foam_bake_pattern(bpy.types.Operator):
    """Cut the drawn strokes into a copy of this sculpt and unwrap the pieces"""

    bl_idname = "object.foam_bake_pattern"
    bl_label = "Bake Pattern"
    bl_options = {'REGISTER', 'UNDO'}

    snap_distance: bpy.props.FloatProperty(
        name="Snap Distance",
        description="Crossings and stroke ends landing closer than this to an "
                    "existing vertex reuse it rather than splitting again. A "
                    "drawn line has the width of a marker tip while the curve "
                    "taken from it is idealised to zero width, so marks a "
                    "fraction of a millimetre apart were one line as drawn. "
                    "Measured on the sculpt, so it means the same at any scale",
        default=0.002, min=0.0, max=0.05,
        subtype='DISTANCE', unit='LENGTH', precision=4)
    unwrap_method: bpy.props.EnumProperty(
        name="Unwrap",
        items=[
            ('MINIMUM_STRETCH', "Minimum Stretch", "Least distortion"),
            ('ANGLE_BASED', "Angle Based", "Preserves angles"),
            ('CONFORMAL', "Conformal", "Faster, more stretch"),
        ],
        default='MINIMUM_STRETCH')
    unwrap_margin: bpy.props.FloatProperty(
        name="Island Margin", default=0.02, min=0.0, max=0.5)

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def invoke(self, context, event):
        settings = getattr(context.scene, "foam_craft", None)
        if settings is not None:
            self.snap_distance = settings.snap_distance
        return self.execute(context)

    def execute(self, context):
        settings = getattr(context.scene, "foam_craft", None)
        source = context.active_object
        gp_object = getattr(settings, "gp_object", None) if settings else None
        if gp_object is None:
            self.report({'ERROR'},
                        "Pick a Grease Pencil object in the Pattern panel")
            return {'CANCELLED'}
        if source.get("foam_pattern_source"):
            self.report({'ERROR'},
                        "That is already a pattern copy -- select the sculpt")
            return {'CANCELLED'}
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Counted before anything else so the promise that the inputs are left
        # alone can be checked rather than asserted.
        source_before = (len(source.data.vertices), len(source.data.edges),
                         len(source.data.polygons))

        # Built from the evaluated mesh, so the modifier stack is already
        # applied and the unwrap has finished geometry to work with.
        depsgraph = context.evaluated_depsgraph_get()
        mesh = bpy.data.meshes.new_from_object(source.evaluated_get(depsgraph))

        name = "%s_Pattern" % source.name
        existing = bpy.data.objects.get(name)
        if existing is not None and existing.get("foam_pattern_source"):
            # Re-bake replaces in place. Leaving a second copy behind on every
            # attempt would turn draw-bake-look-adjust into a cleanup chore, and
            # that loop is the whole point of baking to a copy.
            old_mesh = existing.data
            existing.data = mesh
            if old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)
            pattern = existing
        else:
            pattern = bpy.data.objects.new(name, mesh)
            context.collection.objects.link(pattern)
        pattern.matrix_world = source.matrix_world.copy()
        pattern["foam_pattern_source"] = source.name
        pattern["foam_pattern_gp"] = gp_object.name

        try:
            strokes = strokes_from_grease_pencil(gp_object, pattern)
        except StrokeCutError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            # cut_strokes marks each stroke's edges as it goes, which it has to
            # -- a later stroke splits edges an earlier one made, and only the
            # half still referenced would be marked if it were left until now.
            # Unwrapping is the step that waits for the end, once every cut is
            # in and the seams are complete.
            edges, open_ends = cut_strokes(bm, strokes,
                                           snap_distance=self.snap_distance)
            bm.to_mesh(mesh)
            mesh.update()
        except StrokeCutError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            bm.free()

        # Unwrapping happens on the copy, in edit mode, because Unwrap reads
        # seams from the original mesh rather than from evaluated modifier
        # output -- which is the reason this is baked at all instead of living
        # in a modifier.
        for obj in list(context.selected_objects):
            obj.select_set(False)
        pattern.select_set(True)
        context.view_layer.objects.active = pattern
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.unwrap(method=self.unwrap_method, margin=self.unwrap_margin)
        bm = bmesh.from_edit_mesh(mesh)
        uv_layer = bm.loops.layers.uv.active
        islands = len(core.uv_islands(bm, uv_layer)) if uv_layer else 0
        seams = sum(1 for e in bm.edges if e.seam)
        bpy.ops.object.mode_set(mode='OBJECT')

        source_after = (len(source.data.vertices), len(source.data.edges),
                        len(source.data.polygons))
        untouched = source_before == source_after

        _log.info(foam_log.summary(
            "bake_pattern", source=source.name, gp=gp_object.name,
            strokes=len(strokes), seam_edges=seams, islands=islands,
            open_ends=open_ends, snap=self.snap_distance,
            pattern_verts=len(mesh.vertices), source_untouched=untouched))

        if not untouched:
            # Should be impossible: nothing here writes to the sculpt. Reported
            # rather than trusted, because the damage would be silent and would
            # only surface as drift several bakes later.
            self.report({'WARNING'},
                        "the source mesh changed during the bake")

        message = "%d piece(s) from %d stroke(s), %d seam edge(s)" % (
            islands, len(strokes), seams)
        if open_ends:
            message += "; %d loose end(s) not cut" % open_ends
        self.report({'INFO'}, message)
        return {'FINISHED'}


_classes = (
    OBJECT_OT_foam_bake_pattern,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
