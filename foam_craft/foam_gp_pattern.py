"""Cut a drawn stroke into a mesh as real topology, then seam it.

The pattern for a foam piece is drawn on the sculpt and has to become UV island
boundaries. That means the drawn line must exist in the mesh as actual vertices
and edges, because Blender's Unwrap reads seams from real mesh data.

Blender has no single operator for this, but it does expose the two primitives
its own Knife tool is built from, and this module uses both:

    bmesh.utils.edge_split(edge, vert, fac)      -> a vertex on an edge
    bmesh.utils.face_split_edgenet(face, edges)  -> a face split by an edge net

``face_split_edgenet`` is the one that matters. It takes an **unordered set of
edges** and works out for itself which regions they enclose, which is exactly
what the knife hands it (``BM_face_split_edgenet``, editmesh_knife.cc:2163).
Measured in tests/test_edgenet_cut.py: four edges meeting at a hub split one
quad into four faces in a single call, and a bent net keeps every interior point
at the coordinate given.

A correction worth keeping, because it cost more than any bug here. An earlier
version of this file recorded that ``BM_face_split_edgenet`` was C-only and
would need a patched Blender, and the cutter was built around that belief on
``bmesh.utils.face_split(face, va, vb, coords=[])`` -- which splits one face
between two of its own boundary vertices along one polyline. That signature
forces the caller to group the stroke into per-face runs, pair up entry and exit
vertices, walk the faces between non-adjacent points, and afterwards work out by
counting which edges the split had produced. Every defect this module suffered
lived in that bookkeeping rather than in the cutting, and none of it was ever
necessary. The binding was there the whole time.

The one genuine gap: a net that does not reach the face's boundary is silently
ignored ("Regions defined by edges need to connect to the face, otherwise
they're ignored as loose edges"). The knife bridges such islands with
``BM_face_split_edgenet_connect_islands``, which really is not exposed, so a
stroke drawn wholly inside one face is handled by refining that face instead.

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

What this module writes is only the bookkeeping: what each stroke point landed
on, and where the stroke crosses from one face to the next. Every change to the
topology is made by Blender.
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


def _exit_crossing(face, p_from, p_to, min_along):
    """The next boundary of `face` the segment meets, past `min_along`.

    Returns ``(edge, factor, along)`` or None when nothing lies ahead.

    The *next* crossing, not the furthest: a walk advances one face at a time,
    and taking the furthest skips the faces in between -- which also strands the
    walk when a point sits exactly on an edge, since the crossing it needs is at
    zero distance and the far one belongs to a face it never reaches.
    """
    direction = p_to - p_from
    span = direction.length
    if span < 1e-12:
        return None
    best = None
    for edge in face.edges:
        a, b = edge.verts[0].co, edge.verts[1].co
        hit = intersect_line_line(p_from, p_to, a, b)
        if hit is None:
            continue
        on_segment, on_edge = hit
        edge_span = b - a
        denom = edge_span.length_squared
        if denom < 1e-18:
            continue
        edge_fac = (on_edge - a).dot(edge_span) / denom
        if not (-1e-6 <= edge_fac <= 1.0 + 1e-6):
            continue
        along = (on_segment - p_from).dot(direction) / (span * span)
        if along <= min_along + 1e-7 or along > 1.0 + 1e-9:
            continue
        # Skew lines: reject a "crossing" where the two never really meet.
        if (on_segment - on_edge).length > span * 0.5:
            continue
        if best is None or along < best[0]:
            best = (along, edge, max(0.0, min(1.0, edge_fac)))
    if best is None:
        return None
    return best[1], best[2], best[0]


def _walk_faces(face_from, p_from, face_to, p_to, limit=64):
    """Every face boundary the segment crosses on its way from one face to another.

    Returns a list of ``(edge, factor)``, in order.

    This is the alternative to inventing extra stroke points. When two drawn
    points land on faces that do not touch, the stroke really did pass through
    the faces in between, and where it crossed each boundary is computable
    rather than guessable. Sampling the stroke more finely and hoping the extra
    points land conveniently was tried, and it is a workaround: the added points
    become vertices in the finished cut, so it writes detail nobody drew into
    the pattern. Walking the faces puts the cut exactly where the line goes and
    adds nothing.

    Returns None if the walk cannot reach the target, so the caller can refuse
    rather than lay down a cut along a route the stroke never took.
    """
    crossings = []
    current = face_from
    # Just below zero, so the *first* step may cross at the very start of the
    # segment. A stroke running along a crease begins each segment sitting on
    # the boundary it is about to cross, so its first exit is at zero distance;
    # requiring progress before the first crossing rejected it and stranded the
    # walk with nowhere to go. Measured: a circle of angular radius 0.9 on a
    # sphere failed at point 90 of 120 and the whole stroke was dropped.
    #
    # Only the first step is affected. Once a crossing is taken, `along` becomes
    # its own position and every later crossing must be strictly beyond it, so
    # the walk still cannot sit on one boundary or turn back on itself.
    along = -1e-6
    for _ in range(limit):
        if current is face_to:
            return crossings
        step = _exit_crossing(current, p_from, p_to, along)
        if step is None:
            return None
        edge, fac, along = step
        crossings.append((edge, fac))
        nxt = next((f for f in edge.link_faces if f is not current), None)
        if nxt is None:
            return None
        current = nxt
    return None


# How close a landed point has to be to an edge or a corner before it counts as
# being *on* it. This is a geometric epsilon for resolving an exact tie, not a
# tolerance for how wide a marker tip is -- that is `snap_distance`, and mixing
# the two would make a modelling decision out of a rounding question.
_ON_ELEMENT_EPS = 1e-6

# Two tolerances, doing two different jobs. Keeping them apart is the point.
#
#   snap_distance  -- the width of the marker tip. A user setting, in metres.
#                     It decides when two *separate* parts of the pattern were
#                     meant as one line: a stroke doubling back on itself, two
#                     strokes meeting at a piece corner, a line drawn down an
#                     edge that is already there.
#
#   the weld below  -- purely to stop a cut making geometry with no size. It is
#                     a fraction of the face being cut, so it scales with the
#                     mesh instead of meaning something different on a sculpt
#                     modelled in millimetres and one modelled in metres.
#
# Every drawn point cuts where it projects. Thinning a run with the marker width
# instead threw away real drawn detail -- on a unit sphere that discarded
# anything within 4mm of a neighbour, which is most of a finely drawn curve --
# and moved the cut off the line the user drew. The only points worth dropping
# are the ones that would otherwise produce a zero-length edge and the collapsed
# faces around it.
_WELD_FRACTION = 1e-3


def _weld_eps(face):
    """The smallest separation worth keeping inside `face`.

    Measured against the face's own shortest edge, so a dense sculpt and a
    coarse one both get a tolerance in proportion to what they are made of.
    """
    if not face.is_valid:
        return _ON_ELEMENT_EPS
    lengths = [edge.calc_length() for edge in face.edges]
    shortest = min(lengths) if lengths else 1.0
    return max(_ON_ELEMENT_EPS, shortest * _WELD_FRACTION)


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


# --- The cut, as Blender's knife makes it -------------------------------
#
# Five steps, matching editmesh_knife.cc:2056-2292:
#
#   1. classify every drawn point as a vertex, an edge, or a face
#   2. realise the edge points as real vertices, in order along each edge
#   3. realise the face points as loose vertices
#   4. join consecutive points into a net of edges, reusing any that exist
#   5. hand each face its share of the net and let Blender split it
#
# What is absent is the point: no per-face runs, no entry/exit pairing, no
# recovery of the cut afterwards, and no walk that has to succeed end to end or
# lose the whole stroke. face_split_edgenet takes an unordered *set* of edges
# and works out the regions itself.


VERT, EDGE, FACE = "VERT", "EDGE", "FACE"


def _classify(bm, bvh, points):
    """Land each drawn point on the surface and say what it landed on.

    Returns a list of dicts with `kind` (VERT/EDGE/FACE), `elem`, `fac` (along
    the edge, for EDGE), `loc` and `face`.

    This is the knife's own classification -- "Exactly one of kfe, v, or f
    should be non-null, saying whether cut line crosses an edge, is snapped to a
    vert, or is in the middle of some face" (editmesh_knife.cc:136-152). The
    tolerance is `_weld_eps`, a fraction of the face, so a point cuts where it
    projects and snaps only far enough to avoid geometry with no size.

    Note what is *not* decided here: an edge or vertex point is not given an
    owning face. It genuinely has none, and the old code's need to name one is
    what made the finished cut depend on face numbering.
    """
    hits = []
    for p in points:
        loc, _nrm, index, _dist = bvh.find_nearest(p)
        if index is None:
            raise StrokeCutError(
                "a stroke point found no surface; is the stroke near the mesh?")
        face = bm.faces[index]
        loc = loc.copy()
        weld = _weld_eps(face)

        hit = {"kind": FACE, "elem": face, "fac": 0.0, "loc": loc,
               "face": face}
        near_vert = min(face.verts, key=lambda v: (v.co - loc).length)
        if (near_vert.co - loc).length <= weld:
            hit["kind"], hit["elem"] = VERT, near_vert
        else:
            for edge in face.edges:
                a, b = edge.verts[0].co, edge.verts[1].co
                span = b - a
                denom = span.length_squared
                if denom < 1e-18:
                    continue
                t = max(0.0, min(1.0, (loc - a).dot(span) / denom))
                if (loc - (a + span * t)).length <= weld:
                    hit["kind"], hit["elem"], hit["fac"] = EDGE, edge, t
                    break
        hits.append(hit)
    return hits


def _hit_faces(hit):
    """Every face the hit could be said to lie in."""
    if hit["kind"] == FACE:
        return {hit["elem"]}
    return set(hit["elem"].link_faces)


def _bridge(hits):
    """Insert the crossings between points that share no face.

    Two consecutive drawn points can land on faces that do not touch, because
    the stroke passed straight over the faces between them. Those crossings are
    computable rather than guessable, so they are walked and inserted as EDGE
    hits -- after which every consecutive pair shares a face and the net can be
    built with no further special cases.

    A walk that fails no longer costs the stroke. The pair is left alone and
    counted; the net has a gap there and every other part of the line still
    cuts. Losing a whole drawn line because one segment was awkward was the
    worst behaviour of the previous design.
    """
    out, missed = [hits[0]], 0
    for previous, hit in zip(hits, hits[1:]):
        if _hit_faces(previous) & _hit_faces(hit):
            out.append(hit)
            continue

        # Neighbouring faces first, and by far the commonest case: the stroke
        # simply stepped over one edge. Where the two faces share an edge the
        # crossing is that edge and no walk is needed. Sending this through the
        # walk instead was measured to fail on four to nine segments of a single
        # circle -- the walk is for faces that genuinely do not touch, and using
        # it where the answer is already known just adds ways to be wrong.
        edge = _shared_edge(previous["face"], hit["face"])
        if edge is not None:
            fac = _crossing_on_edge(edge, previous["loc"], hit["loc"])
            a, b = edge.verts[0].co, edge.verts[1].co
            out.append({"kind": EDGE, "elem": edge, "fac": fac,
                        "loc": a + (b - a) * fac,
                        "face": previous["face"]})
            out.append(hit)
            continue

        # Diagonal neighbours, touching at one corner only. The crossing is the
        # corner itself and nothing needs splitting -- the stroke went through a
        # vertex that is already there.
        corner = _shared_vert(previous["face"], hit["face"])
        if corner is not None:
            out.append({"kind": VERT, "elem": corner, "fac": 0.0,
                        "loc": corner.co.copy(), "face": previous["face"]})
            out.append(hit)
            continue

        route = _walk_faces(previous["face"], previous["loc"],
                            hit["face"], hit["loc"])
        if not route:
            missed += 1
            out.append(hit)
            continue
        for edge, fac in route:
            a, b = edge.verts[0].co, edge.verts[1].co
            out.append({"kind": EDGE, "elem": edge, "fac": fac,
                        "loc": a + (b - a) * fac,
                        "face": edge.link_faces[0]})
        out.append(hit)
    return out, missed


def _realise_on_edge(edge, targets, weld):
    """Put every cut point on one edge, in order. Returns {index: BMVert}.

    `targets` is [(index, fac)] sorted by `fac`. Splitting in order along the
    edge is what the knife does (sort_verts_by_dist_cb then BM_edge_split,
    editmesh_knife.cc:2273-2282), and the order matters: each split shortens
    what is left, so a factor measured against the original edge has to be
    re-expressed against the remaining part.

    A point closer to either end than `weld` reuses that end instead of
    splitting, because splitting there leaves an edge with no length and the
    faces around it collapse.
    """
    found = {}
    segment = edge
    base, far = edge.verts[0], edge.verts[1]
    total = (far.co - base.co).length
    done = 0.0                      # where `base` now sits along the original
    for index, fac in targets:
        if total <= 1e-12:
            found[index] = base
            continue
        if (fac - done) * total <= weld:
            found[index] = base
            continue
        if (1.0 - fac) * total <= weld:
            found[index] = far
            continue
        remaining = (1.0 - done) * total
        _new_edge, new_vert = bmesh.utils.edge_split(
            segment, base, ((fac - done) * total) / remaining)
        found[index] = new_vert
        # Carry on down the part that still ends at the original far vertex.
        segment = next((e for e in new_vert.link_edges if far in e.verts),
                       segment)
        base, done = new_vert, fac
    return found


def _realise(bm, hits, snap_distance):
    """Turn every hit into a real BMVert. Returns a list aligned to `hits`."""
    by_edge = {}
    for index, hit in enumerate(hits):
        if hit["kind"] == EDGE:
            by_edge.setdefault(hit["elem"], []).append((index, hit["fac"]))

    verts = [None] * len(hits)
    for edge, targets in by_edge.items():
        if not edge.is_valid:
            continue
        natural = edge.calc_length() * _WELD_FRACTION
        weld = min(snap_distance, max(_ON_ELEMENT_EPS, natural)) \
            if snap_distance > 0.0 else natural
        for index, vert in _realise_on_edge(
                edge, sorted(targets, key=lambda t: t[1]), weld).items():
            verts[index] = vert

    for index, hit in enumerate(hits):
        if verts[index] is not None:
            continue
        if hit["kind"] == VERT:
            verts[index] = hit["elem"]
        else:
            verts[index] = bm.verts.new(hit["loc"])
    return verts


def _net_edges(bm, hits, verts, cyclic):
    """Join consecutive points into edges. Returns ([(edge, face)], reused).

    An existing edge between two consecutive points is reused rather than
    duplicated -- the knife does the same (BM_edge_exists,
    editmesh_knife.cc:2098) -- which is the whole of what `_existing_path` was
    written to do, for free, and without a greedy search that can go wrong.

    An edge already belonging to the face being split must not go into that
    face's net. bmesh_polygon_edgenet.cc:519-528 asserts against it, and the
    assert compiles out of a release build, so nothing would report a mistake.
    """
    pairs = list(zip(range(len(verts)), range(1, len(verts))))
    if cyclic and len(verts) > 2:
        pairs.append((len(verts) - 1, 0))

    net, reused, seen = [], [], set()
    for i, j in pairs:
        va, vb = verts[i], verts[j]
        if va is vb or not (va.is_valid and vb.is_valid):
            continue
        existing = bm.edges.get((va, vb))
        if existing is not None:
            # Already topology. It is part of the cut, but handing it to
            # face_split_edgenet as part of that face's net is forbidden.
            reused.append(existing)
            continue
        shared = _hit_faces(hits[i]) & _hit_faces(hits[j])
        if not shared:
            continue
        face = min(shared, key=lambda f: f.index)
        edge = bm.edges.new((va, vb))
        if edge in seen:
            continue
        seen.add(edge)
        net.append((edge, face))
    return net, reused


def _split_faces(bm, net):
    """Hand each face its share of the net and let Blender split it.

    One call per face with all of that face's edges at once, exactly as the
    knife groups them before splitting (the `fhash` pass,
    editmesh_knife.cc:2219). Splitting one face does not invalidate another, so
    no ordering between faces is needed.
    """
    by_face = {}
    for edge, face in net:
        by_face.setdefault(face, []).append(edge)

    made_total = 0
    for face, edges in by_face.items():
        if not face.is_valid:
            continue
        own = set(face.edges)
        live = [e for e in edges if e.is_valid and e not in own]
        if not live:
            continue
        try:
            made = bmesh.utils.face_split_edgenet(face=face, edgenet=live)
        except ValueError as exc:
            _log.warning(foam_log.summary(
                "face_split_edgenet_refused", edges=len(live),
                reason=str(exc)[:100]))
            continue
        made_total += len(made)
    return made_total


def cut_stroke(bm, points, snap_distance=0.0, cyclic=False,
               project_tolerance=0.0):
    """Cut `points` into `bm` as real edges. Returns (edges, open_ends).

    `points` are in the mesh's own space and are expected to lie on or near its
    surface. Nothing is deleted and no face is moved; the mesh gains vertices
    along the drawn line and the faces it crosses are split by it.

    `cyclic` closes the stroke, joining its last point back to its first.
    """
    if len(points) < 2:
        raise StrokeCutError("a stroke needs at least two points")

    # Drop anything that missed the surface, and cut what is left. A stroke
    # broken into pieces is no longer closed, whatever it was drawn as: the
    # part that ran off took the closure with it.
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
    bm.faces.index_update()
    bvh = BVHTree.FromBMesh(bm)
    hits = _classify(bm, bvh, points)

    # A closed stroke lying wholly inside one face reaches no boundary, and a
    # net that does not reach the boundary is silently ignored -- measured in
    # tests/test_edgenet_cut.py, and stated in the binding's own docs. The knife
    # bridges the island with BM_face_split_edgenet_connect_islands, which is
    # not exposed to Python, so the face is refined until the stroke crosses an
    # edge instead. Refining where a line was drawn is what a modeller would do
    # by hand, and it adds geometry only inside the face already involved.
    def _all_inside():
        return (not any(h["kind"] != FACE for h in hits)
                and len({h["face"] for h in hits}) == 1)

    refined = 0
    while refined < _MAX_REFINE and _all_inside():
        face = hits[0]["face"]
        bmesh.ops.subdivide_edges(bm, edges=list(face.edges), cuts=3,
                                  use_grid_fill=True)
        bm.faces.ensure_lookup_table()
        bm.faces.index_update()
        bvh = BVHTree.FromBMesh(bm)
        hits = _classify(bm, bvh, points)
        refined += 1
    if _all_inside():
        raise StrokeCutError(
            "the stroke stays inside a single face after %d subdivisions; it "
            "is far smaller than the mesh it is drawn on" % refined)

    hits, missed = _bridge(hits)
    if missed:
        _log.warning(foam_log.summary(
            "stroke_gaps", segments=missed, points=len(points)))

    verts = _realise(bm, hits, snap_distance)
    net, reused = _net_edges(bm, hits, verts, cyclic)
    _split_faces(bm, net)

    new_edges = [e for e, _f in net if e.is_valid] + \
                [e for e in reused if e.is_valid]

    # Anything still wire went nowhere -- a loose end sitting inside a face,
    # which cannot bound a piece. The knife removes these too: "Remove dangling
    # edges, not essential - but nice for users." (editmesh_knife.cc:2166)
    dangling = [e for e in new_edges if e.is_wire]
    if dangling:
        bmesh.ops.delete(bm, geom=dangling, context='EDGES')
        new_edges = [e for e in new_edges if e.is_valid]

    # Marked here rather than left to the caller. Cutting a later stroke can
    # split an edge this one produced -- which is what happens wherever two
    # strokes cross -- and marking now means both halves inherit the flag.
    for edge in new_edges:
        edge.seam = True

    _log.debug(foam_log.summary(
        "cut_stroke", points=len(points), net=len(net), reused=len(reused),
        dangling=len(dangling), snap=snap_distance))
    return new_edges, len(dangling)


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
            # Sorted, because the order decides the outcome and a set has none.
            #
            # BMesh elements hash by identity, so iterating a set of them comes
            # out in a different order on every run. remove_doubles keeps the
            # first vertex of each cluster it finds and welds the rest onto it,
            # so a different order welds different survivors and the finished
            # mesh differs run to run. Measured: three identical runs of the
            # same test gave three different results.
            #
            # This project has been here before -- uv_islands seeded its flood
            # fill from set.pop() and numbered the same mesh differently each
            # time it was processed.
            bm.verts.index_update()
            ordered = sorted(touched, key=lambda v: v.index)
            bmesh.ops.remove_doubles(bm, verts=ordered, dist=snap_distance)
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
    def blocked_because(cls, context):
        """Why the bake cannot run, in words, or None when it can.

        Kept apart from poll() so the panel can print the reason. A greyed
        button that does not say why is a puzzle: this one used to disable
        itself whenever the active object was not a mesh, which is the normal
        state right after drawing, since the Grease Pencil is what you had
        selected. Three unrelated causes all showed as the same silent grey.
        """
        settings = getattr(context.scene, "foam_craft", None)
        if settings is None:
            return "FoamCraft settings are unavailable"
        if settings.pattern_source is None:
            return "Pick the sculpt to cut from"
        if settings.pattern_source.type != 'MESH':
            return "The sculpt must be a mesh"
        if settings.pattern_source.get("foam_pattern_source"):
            return "That is a pattern copy, not the sculpt"
        if settings.gp_object is None:
            return "Pick the Grease Pencil object holding the strokes"
        return None

    @classmethod
    def poll(cls, context):
        # Deliberately not keyed to the active object. Baking reads two objects
        # named in the panel and writes a third, so what happens to be selected
        # is irrelevant -- and requiring a mesh there disabled the button in the
        # exact situation the feature exists for.
        return cls.blocked_because(context) is None

    def invoke(self, context, event):
        settings = getattr(context.scene, "foam_craft", None)
        if settings is not None:
            self.snap_distance = settings.snap_distance
        return self.execute(context)

    def execute(self, context):
        blocked = self.blocked_because(context)
        if blocked:
            self.report({'ERROR'}, blocked)
            return {'CANCELLED'}
        settings = context.scene.foam_craft
        source = settings.pattern_source
        gp_object = settings.gp_object
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


class VIEW3D_PT_foam_pattern(bpy.types.Panel):
    """Patterning lives in the 3D viewport, not the UV editor.

    Baking reads a sculpt and a set of drawn strokes and writes a new object.
    That is object-level work, and Blender users look for it where objects are
    -- it was in the UV editor's sidebar, which is where the *result* is
    inspected, not where the work is started.
    """

    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "FoamCraft"
    bl_label = "Pattern"

    def draw(self, context):
        layout = self.layout
        settings = getattr(context.scene, "foam_craft", None)
        if settings is None:
            layout.label(text="FoamCraft settings unavailable", icon='ERROR')
            return

        layout.prop(settings, "pattern_source", text="Sculpt")
        layout.prop(settings, "gp_object", text="Strokes")

        blocked = OBJECT_OT_foam_bake_pattern.blocked_because(context)
        if blocked:
            # Say it rather than only greying the button out.
            layout.label(text=blocked, icon='INFO')

        existing = None
        if settings.pattern_source is not None:
            existing = bpy.data.objects.get(
                "%s_Pattern" % settings.pattern_source.name)
        row = layout.row()
        row.scale_y = 1.4
        row.enabled = blocked is None
        row.operator("object.foam_bake_pattern",
                     text="Re-bake Pattern" if existing else "Bake Pattern",
                     icon='MOD_BOOLEAN')
        if existing is not None:
            layout.label(text="Replaces %s" % existing.name, icon='OBJECT_DATA')
        layout.prop(settings, "snap_distance")


_classes = (
    OBJECT_OT_foam_bake_pattern,
    VIEW3D_PT_foam_pattern,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
