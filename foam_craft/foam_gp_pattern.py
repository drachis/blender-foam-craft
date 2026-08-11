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
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_line_line

from . import foam_log

_log = foam_log.get_logger()


class StrokeCutError(Exception):
    """The stroke could not be walked across the mesh.

    Raised rather than guessing. Every case that reaches this is one where
    continuing would put the seam somewhere the user did not draw, and a seam in
    the wrong place is worse than no seam -- it becomes a cut in foam.
    """


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


def cut_stroke(bm, points, snap_distance=0.0, cyclic=False):
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

    bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)
    assigned = _assign_faces(bvh, bm, points)

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
        if edge is None:
            # The stroke jumped a face. Resampling finer is the fix, and
            # guessing a route across the intervening faces would invent a line
            # the user did not draw.
            raise StrokeCutError(
                "consecutive stroke points landed on faces that share no edge; "
                "resample the stroke more finely")
        fac = _crossing_on_edge(edge, assigned[i - 1][1], surf_pt)
        crossings.append((edge, fac, len(runs)))
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
    for edge, fac, run_index in crossings:
        vert = _vert_at(edge, fac, snap_distance)
        runs[run_index]["exit"] = vert
        # Wraps for a closed stroke, where the last run's exit is the first
        # run's entry. On an open stroke this index is always in range.
        runs[(run_index + 1) % len(runs)]["entry"] = vert

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
        interior = [tuple(p) for p in run["interior"]]
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

    _log.debug(foam_log.summary(
        "cut_stroke", points=len(points), runs=len(runs),
        crossings=len(crossings), open_ends=skipped_open,
        snap=snap_distance))
    return new_edges, skipped_open
