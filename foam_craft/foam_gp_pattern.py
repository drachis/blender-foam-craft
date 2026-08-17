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

A second correction, recorded in full in RESEARCH_gp_pattern_seam_gaps.md.
``face_split_edgenet`` requires the edges handed to it for one face to form a
*valid* planar net -- every edge reaching the boundary at both ends, no two
edges crossing without a shared vertex -- and silently returns nothing **for
the whole face** if that is violated anywhere in it, not just for the
offending edge. Cutting one stroke's whole batch of hits at once, or a whole
face's accumulated hits at once, could hand it exactly that shape: a stroke
crossing its own earlier path, or a stroke's end snapped onto a point another
stroke had not been cut through yet. Checked against
``editmesh_knife.cc`` rather than invented a fix: the real knife never
constructs a malformed net in the first place. During interaction it only
ever extends its own lightweight ``KnifeVert``/``KnifeEdge`` graph -- no
``BMesh`` mutation at all -- testing each new segment against *everything the
graph already has*, from this stroke or a different one, so a later segment
landing back on an earlier point reuses it instead of creating a disconnected
one. Only once, at confirm, is the finished graph converted to real topology,
batched per face (``knife_make_cuts``). ``_CutGraph`` below is this module's
version of that graph: built up across every stroke in one bake, resolved
before anything touches ``bmesh``, realised once.
"""

import math

import bmesh
import bpy
from mathutils import Vector, kdtree
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


def _locally_trivial(hit_a, hit_b):
    """"same_edge", "vert_fan", or None: whether the segment between these
    two hits cannot need a crossing test, and why.

    Two shapes, handled differently by the caller because only one of them
    is safe to connect directly:

    "same_edge" -- both hits name the *same* existing edge, most commonly a
    stroke walking along an edge loop (drawn along an existing seam or
    crease). This does **not** mean "connect them directly": a *different*
    stroke can independently touch the very same edge at a point that lies
    *between* these two along it, and a direct edge here would skip straight
    over that point rather than through it, cutting a sliver off the corner.
    `_realise_graph`'s per-edge chain (`same_edge_chain`) already connects
    every hit on one edge in the correct order regardless of which stroke
    contributed which point, so the caller should add nothing here and let
    that handle it. Measured need for the "don't connect directly" half:
    `test_theta_cuts_identically_every_time` (tests/test_determinism.py) --
    a circle's own crossing point sat between two of a second stroke's
    points on that same edge, and connecting those two directly (skipping
    over the crossing point) sliced a sliver triangle off the corner and
    orphaned the circle's own approach into it.

    "vert_fan" -- one hit is an existing vertex, and the other stays inside
    that vertex's own edge/face fan: a curve passing close enough to an
    existing corner to snap onto it, with a drawn point on either side, not
    a crossing with anything else. This one *is* safe to connect directly --
    there is no shared "same original edge" for a different stroke to also
    be occupying a position along.

    Either way, the point is to skip the crossing search: two points that
    already sit on the same original edge have no well-defined "crossing" for
    `_segment_crossing` to find (collinear segments overlap, they do not
    cross), and testing every such trivial step against unrelated nearby
    strokes was found to produce spurious near-parallel "intersections" --
    moving a real crossing, or losing it, rather than finding nothing.
    """
    if hit_a["kind"] != FACE and hit_b["kind"] != FACE \
            and hit_a["kind"] == hit_b["kind"] and hit_a["elem"] is hit_b["elem"]:
        return "same_edge"
    vert_hit = hit_a if hit_a["kind"] == VERT else \
        hit_b if hit_b["kind"] == VERT else None
    if vert_hit is not None:
        other = hit_b if vert_hit is hit_a else hit_a
        if _hit_faces(other) <= set(vert_hit["elem"].link_faces):
            return "vert_fan"
    return None


def _faces_around(face_from, face_to, limit=64):
    """The patch of mesh the segment could plausibly cross, grown out from
    `face_from` by vertex adjacency until `face_to` is inside it.

    Vertex adjacency rather than edge adjacency on purpose: the whole point
    is to include faces that touch only at a corner, which is where the walk
    this replaces used to get stuck.

    `limit` is a safety stop, not a working bound, and it has to be well
    clear of what any real segment needs: the loop already stops the moment
    `face_to` is reached, so a generous cap costs nothing in the ordinary
    case of two points a face or two apart. It was 4, which quietly failed
    on long segments -- a sawtooth's near-vertical reset spans as many faces
    as its amplitude divided by the face size, and at 300 mm on an 83 mm
    grid that is seven, so the far end was never in the patch, its crossings
    were never collected, and the seam came out with a hole in it. Nothing
    reported: the cut simply did not close, and the whole sheet stayed one
    region.
    """
    seen = {face_from}
    frontier = [face_from]
    for _ in range(limit):
        if face_to in seen:
            break
        nxt = []
        for face in frontier:
            for vert in face.verts:
                for other in vert.link_faces:
                    if other not in seen:
                        seen.add(other)
                        nxt.append(other)
        if not nxt:
            break
        frontier = nxt
    return seen


def _crossings_along(face_from, face_to, p_from, p_to, weld):
    """Every existing edge or vertex the segment crosses, in order along it.

    This is the knife's own method, and it replaces a face-to-face walk that
    could not always find its way. `knife_find_line_hits` gathers every
    candidate element near the cut segment and tests each one, then
    `prepare_linehits_for_cut` sorts them by distance along the segment and
    discards edge hits that sit next to a near-coincident vertex hit
    (editmesh_knife.cc:1860-1933). Nothing steps from face to face, so there
    is no state to get stuck in.

    The walk it replaces failed exactly where a segment passed close to a
    corner: its next crossing was then at almost the same distance along as
    the last one, its "must make progress" epsilon rejected it, and the walk
    gave up with nowhere to go. Collecting and sorting has no such problem --
    two crossings at the same distance are simply two hits at the same
    distance, and if they share a vertex that is precisely the signal the
    line went through the corner.

    Returns a list of hit dicts, or None when nothing was found.
    """
    direction = p_to - p_from
    span = direction.length_squared
    if span < 1e-18:
        return None

    found = {}
    for face in _faces_around(face_from, face_to):
        for edge in face.edges:
            if edge in found:
                continue
            a, b = edge.verts[0].co, edge.verts[1].co
            meeting = intersect_line_line(p_from, p_to, a, b)
            if meeting is None:
                continue
            on_segment, on_edge = meeting
            # Skew lines: the stroke lies on a curved surface and the edge is
            # straight, so they rarely meet exactly. Same test the walk used.
            if (on_segment - on_edge).length > direction.length * 0.5:
                continue
            along = (on_segment - p_from).dot(direction) / span
            if not (1e-9 < along < 1.0 - 1e-9):
                continue
            edge_span = b - a
            denom = edge_span.length_squared
            if denom < 1e-18:
                continue
            fac = (on_edge - a).dot(edge_span) / denom
            if not (-1e-6 <= fac <= 1.0 + 1e-6):
                continue
            found[edge] = (along, max(0.0, min(1.0, fac)))

    if not found:
        return None

    ordered = sorted((along, fac, edge) for edge, (along, fac) in found.items())

    # A crossing that lands on an edge's own end is really a vertex hit, and
    # saying so keeps the cut off a zero-length sliver beside the corner.
    out, used_verts = [], set()
    for along, fac, edge in ordered:
        a, b = edge.verts[0], edge.verts[1]
        length = (b.co - a.co).length
        vert = None
        if fac * length <= weld:
            vert = a
        elif (1.0 - fac) * length <= weld:
            vert = b
        if vert is not None:
            # All but one of a series of hits on the same vertex is dropped,
            # as the knife does (editmesh_knife.cc:1899).
            if vert in used_verts:
                continue
            used_verts.add(vert)
            out.append((along, {"kind": VERT, "elem": vert, "fac": 0.0,
                                "loc": vert.co.copy(), "face": face_from}))
        else:
            out.append((along, {"kind": EDGE, "elem": edge, "fac": fac,
                                "loc": a.co + (b.co - a.co) * fac,
                                "face": edge.link_faces[0]
                                if edge.link_faces else face_from}))

    # Where the line passed through a corner, the two edges meeting there
    # were both crossed at the same distance; the vertex hit already stands
    # for that, so drop the edge hits it swallowed.
    kept = []
    for along, hit in out:
        if hit["kind"] == EDGE and any(
                other["kind"] == VERT
                and abs(other_along - along) <= 1e-6
                and other["elem"] in hit["elem"].verts
                for other_along, other in out):
            continue
        kept.append(hit)
    return kept or None


def _bridge(hits, stats=None):
    """Insert the crossings between points that share no face.

    `stats` is an optional counter dict, tallying which route each pair of
    consecutive points took. It is logged once per bake, because how a cut
    got from one drawn point to the next is the first thing worth knowing
    when a seam comes out somewhere the user did not draw -- and reading it
    out of the log beats writing a probe to rediscover it.

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
    if stats is None:
        stats = {}

    def tally(name, value=1):
        stats[name] = stats.get(name, 0) + value

    out, missed = [hits[0]], 0
    for previous, hit in zip(hits, hits[1:]):
        if _hit_faces(previous) & _hit_faces(hit):
            tally("same_face")
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
            tally("shared_edge")
            fac = _crossing_on_edge(edge, previous["loc"], hit["loc"])
            a, b = edge.verts[0].co, edge.verts[1].co
            out.append({"kind": EDGE, "elem": edge, "fac": fac,
                        "loc": a + (b - a) * fac,
                        "face": previous["face"]})
            out.append(hit)
            continue

        # Diagonal neighbours, touching at one corner only -- but only route
        # the cut *through* that corner when the drawn line really does pass
        # within a hair of it.
        #
        # Routing through it unconditionally is an old assumption from the
        # `face_split` design, where a cut had to run between vertices the
        # face already owned, and it is not what a knife does: the knife
        # registers a vertex hit only when the line passes within
        # KNIFE_FLT_EPS_PX_VERT of one (editmesh_knife.cc:2883-2935), and
        # otherwise takes the ordinary edge crossings either side of it.
        # Snapping regardless drags the seam sideways onto the corner and
        # back -- measured on the workshop file: 5 segments routed this way,
        # pulling the cut as much as 0.075 off the line the user drew, which
        # is the visible notch it leaves in the seam.
        corner = _shared_vert(previous["face"], hit["face"])
        if corner is not None:
            span = hit["loc"] - previous["loc"]
            denom = span.length_squared
            t = 0.0 if denom < 1e-18 else max(0.0, min(
                1.0, (corner.co - previous["loc"]).dot(span) / denom))
            off_line = (corner.co - (previous["loc"] + span * t)).length
            if off_line <= _weld_eps(previous["face"]):
                tally("through_corner")
                out.append({"kind": VERT, "elem": corner, "fac": 0.0,
                            "loc": corner.co.copy(), "face": previous["face"]})
                out.append(hit)
                continue

        # Collect and sort, the way the knife does -- see `_crossings_along`.
        crossed = _crossings_along(previous["face"], hit["face"],
                                   previous["loc"], hit["loc"],
                                   _weld_eps(previous["face"]))
        if crossed:
            tally("crossed")
            tally("crossings", len(crossed))
            out.extend(crossed)
            out.append(hit)
            continue

        route = _walk_faces(previous["face"], previous["loc"],
                            hit["face"], hit["loc"])
        if not route:
            # The walk could not find the way across. Falling back on the
            # shared corner keeps the line connected -- a seam bent to the
            # corner is worse than one that follows the drawing, but both
            # beat a hole in the piece boundary.
            if corner is not None:
                # Counted *and* measured: this is the one route that
                # knowingly puts the seam somewhere other than where the
                # line was drawn, so how far it bent it is worth having in
                # the log rather than needing a probe to find out.
                tally("corner_fallback")
                stats["corner_fallback_worst"] = max(
                    stats.get("corner_fallback_worst", 0.0), off_line)
                out.append({"kind": VERT, "elem": corner, "fac": 0.0,
                            "loc": corner.co.copy(),
                            "face": previous["face"]})
                out.append(hit)
                continue
            tally("no_route")
            missed += 1
            out.append(hit)
            continue
        tally("walked")
        tally("walk_crossings", len(route))
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


def _segment_crossing(p1, p2, q1, q2, tol):
    """Where segment p1-p2 crosses segment q1-q2, strictly in the interior of
    both -- an endpoint-to-endpoint touch is handled separately, by graph
    vertex reuse, so this only has to find a genuine crossing in the middle.

    The segments live on a curved surface, so ``intersect_line_line`` gives
    each its own closest-approach point rather than one shared point; they
    count as crossing when those two points land within `tol` of each other.
    Returns a location, or None.
    """
    hit = intersect_line_line(p1, p2, q1, q2)
    if hit is None:
        return None
    on_p, on_q = hit
    if (on_p - on_q).length > tol:
        return None
    span_p = p2 - p1
    denom_p = span_p.length_squared
    if denom_p < 1e-18:
        return None
    t_p = (on_p - p1).dot(span_p) / denom_p
    span_q = q2 - q1
    denom_q = span_q.length_squared
    if denom_q < 1e-18:
        return None
    t_q = (on_q - q1).dot(span_q) / denom_q
    margin = 1e-4
    if not (margin < t_p < 1.0 - margin):
        return None
    if not (margin < t_q < 1.0 - margin):
        return None
    return (on_p + on_q) * 0.5


class _CutGraph:
    """This module's own version of the knife's ``KnifeVert``/``KnifeEdge``
    graph (see the module docstring and RESEARCH_gp_pattern_seam_gaps.md).

    Accumulates cut points and the edges joining them across as many strokes
    as are fed into it via :func:`_add_stroke_to_graph`, resolving reuse (a
    later point landing on one already placed, whether from this stroke or a
    different one) and genuine crossings (two segments meeting without
    sharing a drawn point) at this abstract level, entirely in Python.
    Nothing here touches ``bmesh`` -- that happens once, in
    :func:`_realise_graph`, after every stroke has had its say.
    """

    def __init__(self):
        self.hits = []          # index -> hit dict (kind, elem, fac, loc, face)
        self.edges = []         # index -> [i, j] into self.hits (mutable pair)
        self.edge_face = []     # index -> the one face this edge belongs to
        self._by_key = {}       # (kind, id(elem), fac?) -> hit index, for reuse
        self.face_hits = {}     # BMFace -> [hit indices] landing in/on it
        self.face_edges = {}    # BMFace -> [edge indices] whose home is it
        # BMEdge -> [(lo, hi)] stretches of an *existing* mesh edge that a
        # drawn stroke actually ran along, as factors. See `cover`.
        self.edge_spans = {}

    def cover(self, edge, fac_a, fac_b):
        """Record that a stroke ran along `edge` between these two factors.

        Only these stretches may be joined up when the cut is realised. The
        distinction matters because several strokes can touch one mesh edge
        without any of them running along it -- a V's elbow and a C curving
        past outside it, both crossing the same edge at different heights.
        Joining every hit on an edge into one chain, which is what this
        replaces, laid a bar between those two down the mesh edge: a cut
        nobody drew, straight across the mouth of the V. Recording who
        actually travelled the edge keeps the join to the stroke that
        earned it.
        """
        lo, hi = (fac_a, fac_b) if fac_a <= fac_b else (fac_b, fac_a)
        self.edge_spans.setdefault(edge, []).append((lo, hi))

    def covered(self, edge, fac):
        spans = self.edge_spans.get(edge)
        if not spans:
            return False
        return any(lo - 1e-9 <= fac <= hi + 1e-9 for lo, hi in spans)

    @staticmethod
    def _key(hit):
        # VERT/EDGE hits are already anchored to a specific, identifiable
        # piece of the original mesh, so two hits naming the same one are the
        # same graph vertex by construction -- no distance test needed, and
        # this is the common case (most of a drawn line's points land on
        # ordinary edge crossings). FACE hits have no such anchor and fall
        # through to the distance-based scan in `add`.
        if hit["kind"] == VERT:
            return (VERT, id(hit["elem"]))
        if hit["kind"] == EDGE:
            return (EDGE, id(hit["elem"]), round(hit["fac"], 7))
        return None

    def add(self, hit, weld):
        """Return the graph index for `hit`, reusing an existing vertex that
        already names the same mesh element, or sits within `weld` of one
        placed earlier in a face `hit` also touches."""
        key = self._key(hit)
        if key is not None and key in self._by_key:
            return self._by_key[key]
        # Sorted, not iterated as the set `_hit_faces` returns: BMesh
        # elements hash by identity, so a set of them comes out in a
        # different order on every run, and when more than one existing
        # point is within `weld` the *first found* is the one reused --
        # letting that vary by run number renumbers the same mesh
        # differently each time (this project's own recurring lesson, see
        # CLAUDE.md's "Rebuild scene state for every measurement").
        for face in sorted(_hit_faces(hit), key=lambda f: f.index):
            for index in self.face_hits.get(face, ()):
                if (self.hits[index]["loc"] - hit["loc"]).length <= weld:
                    if key is not None:
                        self._by_key[key] = index
                    return index
        index = len(self.hits)
        self.hits.append(hit)
        if key is not None:
            self._by_key[key] = index
        for face in _hit_faces(hit):
            self.face_hits.setdefault(face, []).append(index)
        return index

    def connect(self, i, j, home_face, scan_faces, weld):
        """Join graph vertices `i` and `j`, splicing in a real crossing
        against anything already placed nearby first.

        `home_face` is where the plain edge `i`-`j` belongs if nothing
        crosses it. `scan_faces` is the *full* set of faces either endpoint
        touches, not just `home_face` -- searched for a crossing, because two
        strokes can each pick a different (but each individually valid) home
        face for hits that touch several faces at once, e.g. two hits that
        are both existing mesh vertices. Scanning only `home_face` missed a
        crossing whenever the two strokes disagreed on which shared face was
        "the" one, which is what `test_theta_cuts_identically_every_time`
        caught: a circle and a bar that visibly cross in the drawing came out
        with the bar's cut never actually meeting the circle's.

        This is the step that makes a stroke crossing itself, or a different
        stroke, resolve the same way regardless of which one reached this
        area first: the new segment is tested against the accumulated graph,
        not just against the raw mesh, exactly as the knife tests a new
        mouse segment against its own ``kfvs``/``kfes`` (editmesh_knife.cc,
        around line 2900) rather than only against real geometry.
        """
        current = i
        for _ in range(8):        # bounded: a real stroke crosses itself a
            if current == j:      # handful of times per face, not endlessly
                return
            found = None
            # Sorted for the same reason as in `add`: `scan_faces` is a set
            # of BMesh elements, and which candidate crossing is found
            # *first* among several must not depend on identity-hash order.
            for f in sorted(scan_faces, key=lambda face: face.index):
                for edge_index in self.face_edges.get(f, ()):
                    a, b = self.edges[edge_index]
                    if current in (a, b) or j in (a, b):
                        continue
                    loc = _segment_crossing(
                        self.hits[current]["loc"], self.hits[j]["loc"],
                        self.hits[a]["loc"], self.hits[b]["loc"], weld)
                    if loc is not None:
                        found = (edge_index, a, b, loc, self.edge_face[edge_index])
                        break
                if found is not None:
                    break
            if found is None:
                self._add_edge(current, j, home_face)
                return
            edge_index, a, b, loc, cross_face = found
            k = self.add({"kind": FACE, "elem": cross_face, "fac": 0.0,
                         "loc": loc, "face": cross_face}, weld)
            if k in (current, j, a, b):
                # Numerically landed on a point already there -- connect
                # straight through it rather than loop on the same crossing.
                self._add_edge(current, k, home_face)
                current = k
                continue
            self.edges[edge_index] = [a, k]           # splice the old edge...
            self._add_edge(k, b, cross_face)           # ...into two.
            self._add_edge(current, k, cross_face)
            current = k
        self._add_edge(current, j, home_face)

    def _add_edge(self, i, j, face):
        if i == j:
            return
        index = len(self.edges)
        self.edges.append([i, j])
        self.edge_face.append(face)
        self.face_edges.setdefault(face, []).append(index)


def _add_stroke_to_graph(bm, bvh, graph, points, cyclic, snap_distance,
                         stats=None):
    """Classify and bridge one stroke's points, then fold every hit and
    segment into `graph` -- reusing or crossing whatever earlier strokes (or
    this stroke's own earlier points) already placed there. `bm` is read but
    never written; nothing here creates real topology.
    """
    hits = _classify(bm, bvh, points)
    hits, missed = _bridge(hits, stats)
    if missed:
        _log.warning(foam_log.summary(
            "stroke_gaps", segments=missed, points=len(points)))

    def weld_for(hit):
        face = hit["face"] if hit["kind"] != FACE else hit["elem"]
        natural = _weld_eps(face)
        return min(snap_distance, natural) if snap_distance > 0.0 else natural

    indices = [graph.add(hit, weld_for(hit)) for hit in hits]

    pairs = list(zip(range(len(hits)), range(1, len(hits))))
    if cyclic and len(hits) > 2:
        pairs.append((len(hits) - 1, 0))
    for p, q in pairs:
        a, b = indices[p], indices[q]
        if a == b:
            continue
        shared = _hit_faces(hits[p]) & _hit_faces(hits[q])
        if not shared:
            continue
        home_face = min(shared, key=lambda f: f.index)

        # See `_locally_trivial`: both ends already anchored to real
        # topology, and neither reaches beyond what the other already owns,
        # so there is nothing a crossing search could find that is not
        # already accounted for -- and running it anyway is how this went
        # wrong (`test_theta_...` in tests/test_determinism.py).
        trivial = _locally_trivial(hits[p], hits[q])
        if trivial == "vert_fan":
            graph._add_edge(a, b, home_face)
            continue
        if trivial == "same_edge":
            # Deliberately no edge added here -- see `_locally_trivial`.
            # `_realise_graph` joins up this stretch of the mesh edge for
            # us, in the right order and including any point a *different*
            # stroke placed between these two along it. Adding a direct edge
            # here would skip over such a point. But it only joins stretches
            # a stroke really travelled, which is what this records.
            graph.cover(hits[p]["elem"], hits[p]["fac"], hits[q]["fac"])
            continue

        scan_faces = _hit_faces(hits[p]) | _hit_faces(hits[q])
        graph.connect(a, b, home_face, scan_faces, weld_for(hits[p]))
    return indices


def _refine_if_wholly_inside(bm, bvh, points):
    """If `points` never leaves a single face, subdivide until it does (or
    give up after `_MAX_REFINE` passes). Returns the current `bvh` -- rebuilt
    if a subdivision happened, unchanged otherwise.

    A closed stroke lying wholly inside one face reaches no boundary, and a
    net that does not reach the boundary is silently ignored -- measured in
    tests/test_edgenet_cut.py, and stated in the binding's own docs. The
    knife bridges the island with `BM_face_split_edgenet_connect_islands`,
    which is not exposed to Python, so the face is refined until the stroke
    crosses an edge instead. Refining where a line was drawn is what a
    modeller would do by hand, and it adds geometry only inside the face
    already involved.
    """
    hits = _classify(bm, bvh, points)

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
    return bvh


def _realise_graph(bm, graph, snap_distance):
    """Convert the finished graph into real `BMesh` topology, once, batched
    per face -- mirroring `knife_make_cuts`. Returns (new_edges, open_ends).
    """
    # Realise every hit into a BMVert. Hits on the same original edge are
    # grouped and split in order along it in one pass, via `_realise_on_edge`
    # -- unchanged from before this rewrite, just now working over every
    # stroke's edge hits at once instead of one stroke's.
    by_edge = {}
    for index, hit in enumerate(graph.hits):
        if hit["kind"] == EDGE:
            by_edge.setdefault(hit["elem"], []).append((index, hit["fac"]))

    verts = [None] * len(graph.hits)
    # Consecutive real edges between two touches on the *same* original edge
    # -- see below.
    same_edge_chain = []
    for edge, targets in by_edge.items():
        if not edge.is_valid:
            continue
        natural = edge.calc_length() * _WELD_FRACTION
        weld = min(snap_distance, max(_ON_ELEMENT_EPS, natural)) \
            if snap_distance > 0.0 else natural
        ordered = sorted(targets, key=lambda t: t[1])
        realised = _realise_on_edge(edge, ordered, weld)
        for index, vert in realised.items():
            verts[index] = vert

        # Whatever stroke each of these came from, they all sit on the same
        # original edge, and `_realise_on_edge` splits it into one connected
        # chain to hold them, in order -- regardless of which stroke
        # contributed which point. The stretch between two touches belongs
        # to the seam the same way an ordinary crossing does; it just could
        # not be *found* as one, because two points on the very same
        # straight edge have no well-defined crossing for
        # `_segment_crossing` to locate (collinear segments do not cross,
        # they overlap). Two different strokes independently landing on the
        # same original edge -- a stroke crossing another almost exactly
        # along an existing edge loop, e.g. a pattern drawn near a UV
        # sphere's equator -- hits exactly this. Measured in
        # `test_theta_cuts_identically_every_time`
        # (tests/test_determinism.py): the span between a circle's own
        # crossing point and a second stroke's nearest point on that same
        # edge was never marked seam by either stroke's own connectivity on
        # its own, leaving one unsealed edge exactly where the two were
        # supposed to meet -- which is all it takes to leak an otherwise
        # fully enclosed region into its neighbour.
        for (index_a, fac_a), (index_b, fac_b) in zip(ordered, ordered[1:]):
            # Only the stretches a stroke actually ran along -- see
            # `_CutGraph.cover`. Judged at the middle of the stretch, since
            # its two ends are exactly where hits sit and a hit belonging to
            # some other stroke is precisely what must not drag a bar across.
            if not graph.covered(edge, (fac_a + fac_b) * 0.5):
                continue
            va, vb = realised[index_a], realised[index_b]
            if va is vb or not (va.is_valid and vb.is_valid):
                continue
            chain_edge = bm.edges.get((va, vb))
            if chain_edge is not None:
                same_edge_chain.append(chain_edge)

    for index, hit in enumerate(graph.hits):
        if verts[index] is not None:
            continue
        if hit["kind"] == VERT:
            verts[index] = hit["elem"]
        else:
            verts[index] = bm.verts.new(hit["loc"])

    # Join every graph edge, reusing existing bmesh topology where it
    # already connects two of these verts -- the knife does the same
    # (BM_edge_exists, editmesh_knife.cc:2098). Grouped by the face each
    # edge was registered under while the graph was built, which is the
    # `fhash` pass the knife itself does (editmesh_knife.cc:2219): one call
    # per face with all of that face's edges at once.
    by_face, reused, seen_pairs = {}, [], set()
    for face, edge_indices in graph.face_edges.items():
        made = []
        for edge_index in edge_indices:
            i, j = graph.edges[edge_index]
            va, vb = verts[i], verts[j]
            if va is vb or not (va.is_valid and vb.is_valid):
                continue
            pair = (id(va), id(vb)) if id(va) < id(vb) else (id(vb), id(va))
            if pair in seen_pairs:
                continue
            existing = bm.edges.get((va, vb))
            if existing is not None:
                seen_pairs.add(pair)
                reused.append(existing)
                continue
            seen_pairs.add(pair)
            made.append(bm.edges.new((va, vb)))
        if made:
            by_face[face] = made

    spurs = []
    for face, edges in by_face.items():
        if not face.is_valid:
            continue
        own = set(face.edges)
        live = [e for e in edges if e.is_valid and e not in own]
        if not live:
            continue

        # Prune the spurs out of this face's net before handing it over.
        #
        # A drawn line that simply stops in the middle of a face -- the end
        # of a stroke, which every open stroke has two of -- leaves an edge
        # reaching the boundary at one end only. `face_split_edgenet`
        # refuses a net containing one, and refuses it *wholly*: it returns
        # nothing for the entire face, so the perfectly good crossing that
        # happened to share that face is destroyed along with the stub. That
        # is the "open end leaves an uncut area" defect, and it is the
        # original sin of this module surviving in the one shape a cut-graph
        # cannot resolve away, since no amount of tidying makes a dangling
        # end reach a boundary.
        #
        # Pruning first lets the rest of the net split the face as it
        # should. The spurs are not thrown away -- see below.
        boundary = set(face.verts)
        pruned, dropped = list(live), True
        while dropped:
            dropped = False
            degree = {}
            for edge in pruned:
                for vert in edge.verts:
                    degree[vert] = degree.get(vert, 0) + 1
            keep = []
            for edge in pruned:
                a, b = edge.verts
                if ((degree[a] == 1 and a not in boundary)
                        or (degree[b] == 1 and b not in boundary)):
                    spurs.append(edge)
                    dropped = True
                else:
                    keep.append(edge)
            pruned = keep
        if not pruned:
            continue
        live = pruned
        try:
            made = bmesh.utils.face_split_edgenet(face=face, edgenet=live)
        except ValueError as exc:
            _log.warning(foam_log.summary(
                "face_split_edgenet_refused", edges=len(live),
                reason=str(exc)[:100]))
            continue
        if not made:
            # The empty-and-silent case, not the ValueError above: the net
            # handed to this face did not form a valid planar graph (a spur
            # reaching the boundary at only one end, or two edges crossing
            # without a shared vertex), and the call dropped the *whole*
            # face's split rather than the offending edge alone. Should be
            # rare once `_CutGraph` has resolved crossings ahead of time --
            # if this fires, that resolution missed something.
            _log.warning(foam_log.summary(
                "face_split_edgenet_empty", edges=len(live),
                face=face.index if face.is_valid else -1))

    # Deduplicated, and this is not tidiness. A stroke that doubles back over
    # its own path resolves several segments onto the same existing edge, so
    # `reused` can hold it more than once. bmesh.ops.delete refuses a
    # sequence containing any element twice.
    new_edges, seen_edges = [], set()
    for edge in ([e for edges in by_face.values() for e in edges]
                + reused + same_edge_chain):
        if not edge.is_valid or edge in seen_edges:
            continue
        seen_edges.add(edge)
        new_edges.append(edge)

    # Anything still wire went nowhere -- a loose end sitting inside a face,
    # which cannot bound a piece on its own.
    #
    # The knife deletes these ("Remove dangling edges, not essential - but
    # nice for users", editmesh_knife.cc:2166) and this module used to
    # follow it. It should not, and the difference is what the two tools are
    # for. The knife is interactive: the person cutting watched the stroke
    # happen and a tidy result serves them. Here the drawn line *is* the
    # deliverable -- it is the maker's pattern, and the promise is that
    # every point they drew has a vertex of the cut near it. Deleting the
    # stub takes its vertices with it (`context='EDGES'` removes vertices
    # left isolated), so the last stretch of every open stroke simply
    # vanished: measured on two crossing curves, the final three points of a
    # stroke ended up as much as 0.09 from the nearest cut vertex.
    #
    # So they are kept, counted, and reported. A loose end is a fact about
    # the pattern the maker should see -- that piece does not close there --
    # not a mess to sweep up.
    dangling = [e for e in new_edges if e.is_wire]

    # Marked here, once, now that every stroke that might split one of these
    # edges further has already had its say -- unlike the old per-stroke
    # design, nothing downstream can still cut through an edge marked here.
    for edge in new_edges:
        edge.seam = True

    return new_edges, len(dangling)


def cut_stroke(bm, points, snap_distance=0.0, cyclic=False,
               project_tolerance=0.0):
    """Cut `points` into `bm` as real edges. Returns (edges, open_ends).

    `points` are in the mesh's own space and are expected to lie on or near its
    surface. Nothing is deleted and no face is moved; the mesh gains vertices
    along the drawn line and the faces it crosses are split by it.

    `cyclic` closes the stroke, joining its last point back to its first.

    A single-stroke convenience wrapper around `_CutGraph`: builds a graph
    from just this stroke and realises it immediately. That is enough to
    resolve this stroke crossing *itself*, but -- unlike `cut_strokes`, which
    shares one graph across every stroke in a bake -- it cannot resolve a
    junction against a *different* stroke cut in a separate call, since that
    stroke's points were never added to this graph.
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
    bvh = _refine_if_wholly_inside(bm, bvh, points)

    graph = _CutGraph()
    _add_stroke_to_graph(bm, bvh, graph, points, cyclic, snap_distance)
    new_edges, dangling = _realise_graph(bm, graph, snap_distance)

    _log.debug(foam_log.summary(
        "cut_stroke", points=len(points), edges=len(new_edges),
        dangling=dangling, snap=snap_distance))
    return new_edges, dangling


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


def _nearest_on_polyline(point, points, cyclic):
    """Nearest point of a polyline to `point`.

    Returns ``(distance, segment_index, factor, location)`` or None.
    """
    best = None
    count = len(points) if cyclic else len(points) - 1
    if count < 1:
        return None
    for j in range(count):
        a = points[j]
        b = points[(j + 1) % len(points)]
        span = b - a
        denom = span.length_squared
        t = 0.0 if denom < 1e-18 \
            else max(0.0, min(1.0, (point - a).dot(span) / denom))
        q = a + span * t
        d = (point - q).length
        if best is None or d < best[0]:
            best = (d, j, t, q)
    return best


def _segment_closest(a0, a1, b0, b1):
    """Closest approach between two segments.

    Returns ``(distance, fac_a, point_a, fac_b, point_b)``.
    """
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    uu, uv, vv = u.dot(u), u.dot(v), v.dot(v)
    uw, vw = u.dot(w), v.dot(w)
    denom = uu * vv - uv * uv
    if denom < 1e-18:                      # parallel, or a degenerate segment
        s = 0.0
        t = (vw / vv) if vv > 1e-18 else 0.0
    else:
        s = (uv * vw - vv * uw) / denom
        t = (uu * vw - uv * uw) / denom
    s = max(0.0, min(1.0, s))
    t = max(0.0, min(1.0, t))
    qa, qb = a0 + u * s, b0 + v * t
    return (qa - qb).length, s, qa, t, qb


def join_touching_strokes(strokes, tolerance):
    """Join strokes whose ink overlaps but whose centrelines never meet.

    A hand draws with a nib that has width; the curve taken from it is
    idealised to none. So two marks can plainly touch on the page -- their
    ink shares the same patch of paper -- while the polylines through their
    middles pass by each other with a gap. Read as curves they do not
    cross, and nothing joins them; read as marks they are obviously one
    junction, and a maker looking at the drawing would cut it as one.

    The case that named this: a C drawn just under the corner of a V, close
    enough that the two marks merge into a single blot at the elbow. Neither
    is an end meeting anything -- `snap_stroke_ends` handles ends and this
    is mid-stroke on both -- and they do not intersect, so the ordinary
    crossing machinery never sees them. The corner simply did not cut.

    Where the two run within a nib's width of each other, this puts a point
    on the **midline between them** and inserts it into both strokes, so
    both pass through the same place and the graph joins them there like any
    other crossing. The midline rather than one line or the other because
    neither mark is the junction on its own: the blot is, and its middle is
    the honest reading of where the maker would put the knife.

    Only the closest point of each overlapping stretch is taken, not every
    pair within reach. Two strokes running alongside each other for a while
    overlap at hundreds of sampled points and mean one meeting, not
    hundreds; taking the local minimum of each stretch is what tells "these
    two touch here" from "these two run together".
    """
    strokes = [(list(s[0]), s[1], s[2] if len(s) > 2 else 0.0)
               for s in strokes]
    inserts = [{} for _ in strokes]
    joins = 0
    for i in range(len(strokes)):
        points_a, _cyclic_a, width_a = strokes[i]
        for j in range(i + 1, len(strokes)):
            points_b, _cyclic_b, width_b = strokes[j]
            reach = max(tolerance, width_a + width_b)

            # How near the other mark runs, all along this one. Walking one
            # stroke and asking "how far is the other from here" turns the
            # question into a single profile, and the meetings are simply
            # the places that profile dips inside the ink.
            profile = []
            for x in range(len(points_a) - 1):
                best = None
                for y in range(len(points_b) - 1):
                    found = _segment_closest(
                        points_a[x], points_a[x + 1],
                        points_b[y], points_b[y + 1])
                    if best is None or found[0] < best[0]:
                        best = (found[0], found[1], found[2],
                                y, found[3], found[4])
                if best is not None:
                    profile.append((x,) + best)

            # One junction per *stretch* of overlap, taken at its closest
            # point. Not one per sample, and not one per fixed distance
            # along the run either -- which is what this used to do, and it
            # scattered a join every reach-width down two marks that ran
            # close for a long way, walling off a sliver between each pair.
            # Measured on a sine and a sawtooth 7.5 mm apart over 1.7 m: 78
            # junctions and 78 slivers, where the drawing has three
            # meetings. A stretch is where the ink genuinely overlaps, so
            # its ends are where the profile crosses out of reach, and the
            # meeting is its lowest point.
            run = []
            for entry in profile + [None]:
                if entry is not None and entry[1] <= reach:
                    run.append(entry)
                    continue
                if run:
                    x, distance, fac_a, on_a, y, fac_b, on_b = min(
                        run, key=lambda item: item[1])
                    middle = (on_a + on_b) * 0.5
                    inserts[i].setdefault(x, []).append((fac_a, middle))
                    inserts[j].setdefault(y, []).append((fac_b, middle))
                    joins += 1
                    run = []

    out = []
    for index, (points, cyclic, width) in enumerate(strokes):
        if not inserts[index]:
            out.append((points, cyclic, width))
            continue
        grown = []
        for position, point in enumerate(points):
            grown.append(point)
            for _fac, middle in sorted(inserts[index].get(position, []),
                                       key=lambda item: item[0]):
                if (middle - grown[-1]).length > 1e-9:
                    grown.append(middle.copy())
        out.append((grown, cyclic, width))

    if joins:
        _log.info(foam_log.summary(
            "strokes_joined_by_ink", junctions=joins, tolerance=tolerance))
    return out


def _end_direction(points, end_index):
    """Unit vector pointing outward along the stroke at one of its ends.

    "Outward" means the direction the hand was travelling as it left the
    line: at the tail that is the last drawn step continued forwards, and at
    the head it is the first step continued backwards. A stroke sampled from
    a real drawing routinely carries repeated or near-repeated points at its
    ends -- the pen resting before it moves -- so this walks inward until it
    finds a step with real length rather than trusting the outermost pair.

    Returns None when the whole stroke is degenerate, which is a stroke with
    no direction to extend along.
    """
    count = len(points)
    for step in range(1, count):
        if end_index == 0:
            outward = points[0] - points[step]
        else:
            outward = points[-1] - points[-1 - step]
        if outward.length > 1e-9:
            return outward.normalized()
    return None


# How far off its own heading a stroke's end may look for the line it was
# drawn to meet. A single ray is too brittle for a drawn line: the last
# sampled segment carries whatever wobble the hand had at the moment the pen
# lifted, so a ray can miss by a few degrees a junction the eye reads as
# obvious. A cone forgives that without becoming an omnidirectional snap --
# at 90 degrees this would be "anything nearby", which is the sideways test
# it exists to be different from.
_EXTENSION_CONE_DEGREES = 30.0

# How square the meeting has to be to count as a junction. A cone on its own
# is not enough, and the reason is geometric rather than a matter of taste:
# a line running *alongside* another, a hair away, is eventually clipped by
# any forward cone however narrow -- at a small enough offset there is always
# some distance ahead where the neighbour falls inside it. So a pure cone
# test reports a junction for two lines that run side by side and never
# converge, which is exactly the case the directional test was supposed to
# reject. Requiring the two to actually meet at an angle separates "drawn
# into" from "drawn alongside", and it is what makes the extension mean
# "this line was heading into that one" rather than "that one is nearby".
_EXTENSION_MIN_MEETING_DEGREES = 20.0


def _cone_candidates(tip, direction, half_angle_cos, reach, a, b):
    """Parameters along segment `a`-`b` worth testing as the nearest point
    inside the cone: its closest approach to `tip`, and where the segment
    enters or leaves the cone.

    Both are needed. When the segment's closest point is already inside the
    cone that is the answer, but when the cone clips the segment part-way --
    the usual case for a line running past at an angle -- the nearest
    *admissible* point is exactly on the cone's boundary, which the closest
    approach never finds.

    The boundary case solves ``(v . d)^2 = cos^2(a) * |v|^2`` along the
    segment, which is a quadratic in the segment parameter.
    """
    span = b - a
    denom = span.length_squared
    if denom < 1e-18:
        return ()
    w = a - tip
    out = [max(0.0, min(1.0, -w.dot(span) / denom))]

    p = w.dot(direction)
    q = span.dot(direction)
    A = w.length_squared
    B = w.dot(span)
    C = denom
    k = half_angle_cos * half_angle_cos
    qa = q * q - k * C
    qb = 2.0 * (p * q - k * B)
    qc = p * p - k * A
    if abs(qa) < 1e-18:
        if abs(qb) > 1e-18:
            out.append(-qc / qb)
    else:
        disc = qb * qb - 4.0 * qa * qc
        if disc >= 0.0:
            root = disc ** 0.5
            out.append((-qb + root) / (2.0 * qa))
            out.append((-qb - root) / (2.0 * qa))
    return tuple(t for t in out if -1e-9 <= t <= 1.0 + 1e-9)


def _extension_target(tip, direction, reach, other_points, other_cyclic,
                      half_angle_cos, meeting_cos):
    """The nearest point of another stroke lying ahead of this one's end,
    inside a cone about the direction it was travelling. Returns
    ``(distance, segment_index, factor, location)`` or None.

    This is the near-miss junction case, and the reason a plain
    nearest-point test is not enough for it: two strokes drawn to meet at a
    T can stop a whisker short of each other, so the incoming line's tip has
    no seam within any honest sideways tolerance, yet the drawing plainly
    shows a junction because the ink overlaps. Asking "how close is the tip
    to that line" answers the wrong question -- it says yes to a line the
    stroke was merely passing near, and no to the line it was visibly
    heading into. Asking "if this line carried on the way it was going,
    would it run into anything" matches what the maker drew.

    `reach` is how far the line is carried on -- the two pens' summed
    half-widths, so the test is "would the ink have touched", the same
    judgement the rest of this module already makes about width.
    Deliberately short: this closes the gap a hand leaves at a junction, it
    does not hunt for distant lines to attach to.
    """
    best = None
    count = len(other_points) if other_cyclic else len(other_points) - 1
    for j in range(max(0, count)):
        a = other_points[j]
        b = other_points[(j + 1) % len(other_points)]
        span = b - a
        if span.length_squared < 1e-18:
            continue
        # Running alongside is not meeting -- see _EXTENSION_MIN_MEETING_DEGREES.
        if abs(direction.dot(span.normalized())) > meeting_cos:
            continue
        for t in _cone_candidates(tip, direction, half_angle_cos, reach, a, b):
            t = max(0.0, min(1.0, t))
            q = a + span * t
            v = q - tip
            distance = v.length
            if distance > reach or distance < 1e-12:
                continue
            # Inside the cone, and ahead rather than behind: the squared
            # form above is blind to sign on its own.
            if v.dot(direction) < half_angle_cos * distance - 1e-9:
                continue
            if best is None or distance < best[0]:
                best = (distance, j, t, q)
    return best


def snap_stroke_ends(strokes, tolerance):
    """Pin each open stroke's ends onto the line they were drawn to meet.

    This copies the knife's division of labour, which it took two wrong designs
    to appreciate. The knife treats a cut's *intermediate* crossings exactly --
    a fraction-of-a-pixel tolerance -- and its *endpoints* generously, snapping
    them onto nearby geometry with a user-scale threshold and moving the point
    onto whatever it snapped to (editmesh_knife.cc: KNIFE_FLT_EPS_PX_EDGE for
    hits, `knife_snap_size` for endpoints, and the snapped vert rewriting the
    mouse position). Ends are where the hand is imprecise; the middle of a line
    is wherever it was drawn.

    Applying one pen-width tolerance to *every* segment pair instead was tried,
    and it failed in both directions at once: mid-line meetings wiggled the
    stroke through each near-approach, trapping single-face slivers at real
    crossings, while the deduplication needed to control that kept unmaking the
    T junctions it existed to serve.

    So: only ends move. An open stroke's end within `tolerance` of another
    stroke is moved onto it, and the landing point is inserted into the other
    stroke too, so both lines pass through the meeting whichever is cut first.

    Sideways reach is deliberately the *small* one -- `tolerance`, the snap
    distance -- while the pen widths buy reach only along the direction the
    line was travelling (see below). Sideways and forwards are not the same
    judgement: a line that stopped a whisker short of the one it was heading
    into was meant to meet it, whereas a line that merely passes within a
    pen width of another, going its own way, was not. Giving both the full
    ink-touching-ink distance made every near-approach a junction.

    Genuine mid-line crossings need none of this: strokes are cut one after
    another, so a later stroke crossing an earlier one meets real topology and
    the crossing is an ordinary edge crossing.

    A stroke's end near its own middle (a P shape) is not handled -- the end is
    trivially at zero distance from its own adjacent segments, and telling
    "near my own middle" from "near my own tail" needs arc-length bookkeeping
    this does not yet have. Ends meeting *ends* are `chain_strokes`.

    Where the tip does not reach at all, the line is carried on in the
    direction it was travelling to see whether it *would* have met something
    -- see `_extension_crossing`. That covers the junction a hand leaves a
    whisker short, which is the common case in a real drawing and the one a
    nearest-point test cannot tell from a line merely passing nearby. The
    two behave differently on purpose: a tip already within reach is *moved*
    onto what it met, while a tip that only reaches by being carried on is
    *extended* to it, growing a new point and leaving every drawn point where
    it was. At a T that keeps the crossbar exactly as drawn and lets the stem
    grow the last fraction to meet it, rather than dragging the drawn tip
    sideways.

    A stroke that reaches nothing either way is left completely alone --
    nothing invented, no vertices kept from a speculative extension.
    """
    strokes = [(list(s[0]), s[1], s[2] if len(s) > 2 else 0.0)
               for s in strokes]
    inserts = [{} for _ in strokes]
    # Extensions are applied after every end has been judged, because
    # growing a stroke mid-pass would change what a later end measures
    # against and make the result depend on stroke order.
    extensions = [{} for _ in strokes]
    moved = 0
    extended = 0
    for i, (points, cyclic, width) in enumerate(strokes):
        if cyclic or len(points) < 2:
            continue
        for end_index in (0, len(points) - 1):
            tip = points[end_index]
            best = None
            for j, (other, other_cyclic, other_width) in enumerate(strokes):
                if j == i:
                    continue
                found = _nearest_on_polyline(tip, other, other_cyclic)
                if found is None or found[0] > tolerance:
                    continue
                if best is None or found[0] < best[1][0]:
                    best = (j, found)
            if best is not None:
                j, (d, seg, fac, q) = best
                if d > 1e-9:
                    points[end_index] = q.copy()
                    moved += 1
                inserts[j].setdefault(seg, []).append((fac, q))
                continue

            # Nothing within reach of the tip itself. Carry the line on the
            # way it was going and see whether it would have met anything.
            direction = _end_direction(points, end_index)
            if direction is None:
                continue
            reached = None
            cone_cos = math.cos(math.radians(_EXTENSION_CONE_DEGREES))
            meeting_cos = math.cos(
                math.radians(_EXTENSION_MIN_MEETING_DEGREES))
            for j, (other, other_cyclic, other_width) in enumerate(strokes):
                if j == i or len(other) < 2:
                    continue
                reach = max(tolerance, width + other_width)
                found = _extension_target(tip, direction, reach, other,
                                          other_cyclic, cone_cos, meeting_cos)
                if found is None:
                    continue
                if reached is None or found[0] < reached[1][0]:
                    reached = (j, found)
            if reached is None:
                continue
            j, (_along, seg, fac, q) = reached
            extensions[i][end_index] = q.copy()
            inserts[j].setdefault(seg, []).append((fac, q))
            extended += 1

    out = []
    for index, (points, cyclic, width) in enumerate(strokes):
        grown = []
        head = extensions[index].get(0)
        if head is not None and (head - points[0]).length > 1e-9:
            grown.append(head)
        for i, point in enumerate(points):
            grown.append(point)
            for _fac, q in sorted(inserts[index].get(i, []),
                                  key=lambda item: item[0]):
                # Two stems landing on the same spot insert it once.
                if (q - grown[-1]).length > 1e-9:
                    grown.append(q.copy())
        tail = extensions[index].get(len(points) - 1)
        if tail is not None and (tail - grown[-1]).length > 1e-9:
            grown.append(tail)
        out.append((grown, cyclic, width))

    if moved or extended:
        _log.info(foam_log.summary(
            "stroke_ends_snapped", ends=moved, extended=extended,
            tolerance=tolerance))
    return out


def cut_strokes(bm, strokes, snap_distance=0.0, project_tolerance=0.0,
                chain_tolerance=0.0):
    """Cut every stroke into `bm` through one shared cut-graph, then realise
    the whole thing at once. Returns (edges, open_ends).

    A pattern maker crosses strokes on purpose, because that is what closes a
    region into a piece -- and a stroke's end is routinely snapped onto a
    point *another* stroke passes through, at a T-junction. Both are
    junctions in exactly the same sense: two lines meeting at a point neither
    would place alone. Cutting strokes one at a time into an evolving mesh,
    the way this function used to, made the first case work for free (the
    first stroke cut is already real topology by the time the second crosses
    it) but not the second: whichever stroke's snapped end happened to be cut
    *before* its target left a spur with nothing there yet to attach to, and
    `face_split_edgenet` silently drops the whole face's split around a spur
    like that -- see the module docstring and RESEARCH_gp_pattern_seam_gaps.md.

    Feeding every stroke into one `_CutGraph` before any of it becomes real
    `BMesh` topology removes the ordering question rather than scheduling
    around it: whichever stroke's segment reaches a shared point *second*
    simply reuses the graph vertex the first one already placed there,
    exactly as the interactive knife's own graph does for a mouse segment
    tested against its accumulated `KnifeVert`/`KnifeEdge` pool.

    `strokes` is a sequence of (points, cyclic) pairs, or (points, cyclic,
    width) triples as `strokes_from_grease_pencil` returns them -- `width` is
    the drawn line's half-width and feeds the crossing tolerance.
    """
    # Ends first, while each stroke still knows its own drawn width -- chaining
    # merges strokes and the widths would not survive it.
    pen = max((s[2] for s in strokes if len(s) > 2), default=0.0)
    strokes = snap_stroke_ends(strokes, snap_distance)
    # Then the marks that overlap without their centrelines meeting -- see
    # `join_touching_strokes`. After the ends, because an end that reaches
    # its target properly should do so on its own terms rather than being
    # read as a blot.
    strokes = join_touching_strokes(strokes, snap_distance)
    strokes = [(s[0], s[1]) for s in strokes]
    # Ends meeting *ends* is the same judgement at a corner: two marks within a
    # pen width of each other were one mark.
    chain_tolerance = max(chain_tolerance, 2.0 * pen)
    if chain_tolerance > 0.0:
        strokes = chain_strokes(strokes, chain_tolerance)

    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    bvh = BVHTree.FromBMesh(bm)

    graph = _CutGraph()
    route_stats = {}
    drawn = []
    for index, (points, cyclic) in enumerate(strokes):
        if len(points) < 2:
            continue
        pieces = [(points, cyclic)]
        if project_tolerance > 0.0:
            # Drop anything that missed the surface, and cut what is left. A
            # stroke broken into pieces is no longer closed, whatever it was
            # drawn as: the part that ran off took the closure with it.
            runs_on = on_surface_runs(bm, points, project_tolerance)
            if not runs_on:
                _log.warning(foam_log.summary(
                    "stroke_off_surface", index=index, points=len(points)))
                continue
            pieces = [(points, cyclic)] if (
                len(runs_on) == 1 and len(runs_on[0]) == len(points)) \
                else [(piece, False) for piece in runs_on]

        for piece_points, piece_cyclic in pieces:
            try:
                bvh = _refine_if_wholly_inside(bm, bvh, piece_points)
                _add_stroke_to_graph(bm, bvh, graph, piece_points,
                                     piece_cyclic, snap_distance, route_stats)
                drawn.extend(piece_points)
            except StrokeCutError as exc:
                # One bad stroke should not discard the rest of the pattern.
                # Which one failed is worth knowing, so it is named rather
                # than counted.
                _log.warning(foam_log.summary(
                    "stroke_failed", index=index, points=len(piece_points),
                    cyclic=piece_cyclic, reason=str(exc)[:120]))

    all_edges, open_ends = _realise_graph(bm, graph, snap_distance)

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
    welded = 0
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
            before = len(bm.verts)
            bmesh.ops.remove_doubles(bm, verts=ordered, dist=snap_distance)
            welded = before - len(bm.verts)
            all_edges = [e for e in all_edges if e.is_valid]

    # How much of the cut went which way. Read this first when a seam comes
    # out somewhere the drawing does not go: `corner_fallback` above zero is
    # the cut knowingly bending to a face corner, and `no_route` is a segment
    # that could not be crossed at all.
    _log.info(foam_log.summary("cut_routes", **route_stats))

    # And how faithfully the finished cut followed the line, which is the
    # question this module exists to answer -- so it belongs in the log
    # rather than in a probe written afresh each time somebody doubts it.
    #
    # Measured against the drawn points *projected onto the surface*, not the
    # raw ones: strokes drawn with a Shrinkwrap sit a fixed hover above the
    # mesh, and comparing against that hover reports a constant error that is
    # nothing to do with the cut. Nearest cut *vertex* rather than nearest
    # seam edge, via a KD-tree, because every drawn point should have become
    # one and the tree keeps this cheap on a dense sculpt.
    if drawn and all_edges:
        cut_verts = [v for e in all_edges if e.is_valid for v in e.verts]
        if cut_verts:
            unique = {v.index: v for v in cut_verts}
            tree = kdtree.KDTree(len(unique))
            for i, v in enumerate(unique.values()):
                tree.insert(v.co, i)
            tree.balance()
            surface = BVHTree.FromBMesh(bm)
            offs = []
            for point in drawn:
                landed = surface.find_nearest(point)[0]
                if landed is None:
                    continue
                offs.append(tree.find(landed)[2])
            if offs:
                offs.sort()
                _log.info(foam_log.summary(
                    "cut_accuracy", points=len(offs),
                    median=offs[len(offs) // 2],
                    p90=offs[min(len(offs) - 1, int(len(offs) * 0.9))],
                    worst=offs[-1], snap=snap_distance,
                    welded=welded))

    _log.info(foam_log.summary(
        "cut_strokes", strokes=len(strokes), edges=len(all_edges),
        open_ends=open_ends, snap=snap_distance))
    return all_edges, open_ends


# --- Reading strokes off a Grease Pencil object -------------------------

def strokes_from_grease_pencil(gp_object, target_object, at_frame=None,
                               depsgraph=None):
    """Every stroke on `gp_object`, in `target_object`'s local space.

    Returns ``[(points, cyclic, width)]``, ready for :func:`cut_strokes` --
    `width` is the stroke's mean drawn radius, read off its own points.

    Pass `depsgraph` to read the *evaluated* strokes, and the bake always does.
    The workshop setup keeps a Shrinkwrap modifier on the Grease Pencil object
    to pin strokes to the sculpt's surface after drawing -- and modifier output
    only exists on the evaluated object. Reading the original data reads the
    strokes where the hand left them, hovering off the mesh, and the Shrinkwrap
    silently does nothing. The same mistake was made once already with
    Decimate, where the answer was likewise "the evaluated mesh has what you
    need" (see CLAUDE.md).

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
    if depsgraph is not None:
        gp_object = gp_object.evaluated_get(depsgraph)

    to_local = target_object.matrix_world.inverted() @ gp_object.matrix_world
    # Radii are lengths in the Grease Pencil's own space and must cross into the
    # target's with the points, or a scaled object changes what "the width the
    # user drew with" means. median_scale is exact for uniform scaling, which is
    # the practical case.
    radius_scale = to_local.median_scale
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
            # The drawn line's real half-width, off the points themselves --
            # each carries a `radius` in object units. This is what decides
            # whether two strokes that pass close were meant as one line: the
            # marker has width even though the polyline does not, so the right
            # tolerance is the width the user actually drew with, not a scene
            # setting standing in for it.
            radii = [p.radius for p in stroke.points]
            width = (sum(radii) / len(radii)) * radius_scale if radii else 0.0
            out.append((points, closed, width))
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
        description="The smallest feature the saw or hot wire can actually "
                    "follow. Set it from the tool, not to taste: detail "
                    "drawn finer than this is welded away, because a notch "
                    "narrower than the blade cannot be cut. Crossings and "
                    "stroke ends landing closer than this to an existing "
                    "vertex reuse it rather than splitting again. Measured "
                    "on the sculpt, so the sculpt must be at real-world "
                    "scale for it to mean anything",
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
            # Through the depsgraph, so the Shrinkwrap that pins strokes to the
            # sculpt is included. The original data is where the hand drew.
            strokes = strokes_from_grease_pencil(gp_object, pattern,
                                                 depsgraph=depsgraph)
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
            # Chaining was written and then never switched on: the bake passed
            # only snap_distance, so strokes drawn end-to-end at a piece corner
            # were cut as separate lines with a gap between them and the piece
            # never closed. The marker width is the right tolerance for it --
            # it is the same judgement, that two marks this close were one mark.
            edges, open_ends = cut_strokes(bm, strokes,
                                           snap_distance=self.snap_distance,
                                           chain_tolerance=self.snap_distance)
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


class IMAGE_PT_foam_pattern(bpy.types.Panel):
    """One panel, in the UV editor, alongside the rest of FoamCraft.

    A modifier would be the natural home -- this is mesh modification with
    iteration, which is exactly what modifiers are for, and it would carry the
    stack, the toggles and the non-destructive re-evaluation for free.

    It cannot be one. Modifiers are not registrable from Python; the attempt
    fails outright with

        ValueError: register_class(...): expected a subclass of a registerable
        RNA type (Modifier does not support registration)

    The single exception is a Geometry Nodes group, which *is* a modifier and
    *is* reachable from Python -- but Geometry Nodes has no mesh-cutting
    primitive to build this on, which is what ruled that route out earlier.
    Writing a real one means C, and a patched Blender cannot be distributed as
    an extension, so the feature would not reach anyone.

    Failing that, it belongs with the other FoamCraft panels rather than in a
    second place in the 3D viewport. It sat there briefly on the argument that
    baking is object-level work; one sidebar the user already has open beats a
    tidier taxonomy split across two editors.
    """

    bl_space_type = 'IMAGE_EDITOR'
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
    IMAGE_PT_foam_pattern,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
