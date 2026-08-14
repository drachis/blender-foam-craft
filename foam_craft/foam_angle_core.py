"""Shared geometry and math for the FoamCraft add-on.

Pure functions only -- no bpy.types subclasses, no register()/unregister().
Both the paper workflow (uv_fold_angle_overlay, paper_model_export) and the
foam workflow (foam_panel_tabs) import from here.
"""

import sys
from dataclasses import dataclass, field
from math import acos, degrees, tan, cos, radians

from mathutils import Vector

# UV coordinates closer than this are treated as the same point when deciding
# whether two faces share a UV island boundary.
UV_EPSILON = 1e-6

# Turn beyond which a boundary vertex counts as a corner rather than part of a
# curve. Corners break a run: one angle guide should belong to one edge, and
# running it through a corner produces a mark that describes neither side.
CORNER_ANGLE_DEG = 35.0

# Below this dihedral the foam flexes and no cut is made. A property of the
# material, not of the model: a subdivided sculpt reports a degree or two across
# nearly every edge, and cutting all of those is meaningless work on a sheet
# that bends to shape by itself. The reference patterns bottom out around 5.
MIN_BEVEL_ANGLE_DEG = 5.0

# How much outline turning one relief dart absorbs, in degrees. Dart count is
# the run's total turning divided by this, so a straight run gets none however
# long it is and a tight curve gets several however short. Darts once landed on
# every vertex, which made their density a function of how finely the mesh was
# subdivided rather than of anything physical.
DART_RELIEF_DEG = 12.0


def dihedral_angle_deg(edge):
    """Signed fold angle between the two 3D faces sharing this edge.

    Positive = convex/mountain, negative = concave/valley. None if the edge
    doesn't have exactly two connected faces (mesh boundary).

    Uses BMEdge.calc_face_angle_signed rather than a hand-rolled cross-product
    sign test. The hand-rolled version this replaces derived its sign from
    ``edge.verts[1].co - edge.verts[0].co``, but BMEdge vertex order carries no
    fixed relationship to the winding of the two adjacent faces, so its sign was
    effectively arbitrary -- measured live on Blender 5.2 it disagreed on exactly
    half of a torus's 192 interior edges, and was inverted on all 12 edges of a
    cube (every one of which is convex). The convention below was pinned against
    three unambiguous cases on 5.2.0: cube = +90 (all convex), normals-flipped
    cube = -90, explicit valley crease = -90.
    """
    if len(edge.link_faces) != 2:
        return None
    return degrees(edge.calc_face_angle_signed(0.0))


def iter_uv_edges(bm, uv_layer, only_selected=False):
    """Yield (edge, uv_a, uv_b) once for each distinct UV-space occurrence of
    every 3D edge, so a seam between two islands yields two entries -- one per
    island, at their respective UV positions."""
    seen = set()
    for face in bm.faces:
        if only_selected and not face.select:
            continue
        loops = face.loops
        n = len(loops)
        for i in range(n):
            loop = loops[i]
            next_loop = loops[(i + 1) % n]
            uv_a = loop[uv_layer].uv.copy()
            uv_b = next_loop[uv_layer].uv.copy()
            key = (loop.edge.index, round(uv_a.x, 6), round(uv_a.y, 6),
                   round(uv_b.x, 6), round(uv_b.y, 6))
            key_rev = (loop.edge.index, round(uv_b.x, 6), round(uv_b.y, 6),
                       round(uv_a.x, 6), round(uv_a.y, 6))
            if key in seen or key_rev in seen:
                continue
            seen.add(key)
            yield loop.edge, uv_a, uv_b


def find_paper_model_module():
    """Locate the already-loaded Export Paper Model add-on module, whatever
    internal name Blender registered it under.

    Duck-typing is warranted here (unlike for this package's own modules)
    because Paper Model is a third-party add-on whose module name varies
    between legacy installs and Blender 4.2+'s Extensions Platform.
    """
    for mod in list(sys.modules.values()):
        if (mod is not None and hasattr(mod, "Unfolder")
                and hasattr(mod, "Mesh") and hasattr(mod, "UnfoldError")):
            return mod
    raise ImportError(
        "Could not find the 'Export Paper Model' add-on. "
        "Enable it first: Preferences > Add-ons > search 'Paper Model'."
    )


# --- Foam bevel geometry -------------------------------------------------

FOAM_GRADES = (
    ('L200', "L200 (soft PE)", "Low-density closed-cell polyethylene"),
    ('L400', "L400 (medium PE)", "Medium-density closed-cell polyethylene"),
    ('L600', "L600 (firm PE)", "High-density closed-cell polyethylene"),
    ('EVA', "EVA", "Ethylene-vinyl acetate craft foam"),
    ('CUSTOM', "Custom", "Set flex tolerances manually"),
)

# Relative bending stiffness per grade, normalised so L400 == 1.0. Rough
# ordering only -- calibrate against a physical bend test, see
# estimate_flex_tolerance().
_GRADE_STIFFNESS = {'L200': 0.6, 'L400': 1.0, 'L600': 1.6, 'EVA': 1.2,
                    'CUSTOM': 1.0}


@dataclass
class BevelResult:
    """Saw setup for one dart/seam. Angles in degrees, lengths in metres."""
    dihedral_deg: float
    per_side_deg: float
    achievable: bool
    bevel_depth: float
    bevel_face_width: float
    needs_heat_form: bool


def compute_bevel(dihedral_deg, thickness_m, max_tilt_deg):
    """Saw bevel for one seam of total dihedral angle `dihedral_deg`.

    Each of the two mating panels is cut at half the total angle, so the pair
    reconstructs the full dihedral when glued -- the same reason a 90-degree
    box corner takes two 45-degree miters. The split is symmetric only because
    both panels share one global thickness; differing thicknesses would need an
    asymmetric split.
    """
    per_side = dihedral_deg / 2.0
    tilt = abs(per_side)
    achievable = tilt <= max_tilt_deg
    tilt_rad = radians(min(tilt, 89.9))
    return BevelResult(
        dihedral_deg=dihedral_deg,
        per_side_deg=per_side,
        achievable=achievable,
        bevel_depth=thickness_m * tan(tilt_rad),
        bevel_face_width=thickness_m / cos(tilt_rad),
        needs_heat_form=not achievable,
    )


# --- UV islands and boundaries ------------------------------------------

def uv_boundary_edge(edge, uv_layer):
    """True if `edge` bounds a UV island: an open mesh edge, or a seam where the
    two adjacent faces place the edge at different UV coordinates."""
    if len(edge.link_faces) != 2:
        return True
    try:
        la = next(l for l in edge.link_faces[0].loops if l.edge is edge)
        lb = next(l for l in edge.link_faces[1].loops if l.edge is edge)
    except StopIteration:
        return True
    a0, a1 = la[uv_layer].uv, la.link_loop_next[uv_layer].uv
    b0, b1 = lb[uv_layer].uv, lb.link_loop_next[uv_layer].uv
    if la.vert is lb.vert:
        joined = (a0 - b0).length < UV_EPSILON and (a1 - b1).length < UV_EPSILON
    else:
        # The two loops traverse the edge in opposite directions.
        joined = (a0 - b1).length < UV_EPSILON and (a1 - b0).length < UV_EPSILON
    return not joined


def uv_islands(bm, uv_layer, only_selected=False):
    """Partition faces into UV islands -- connected components of faces joined
    across non-boundary edges. Returns a list of sets of BMFace.

    Island order is deterministic: components come out ordered by their lowest
    face index. This matters because the list index is an island's identity for
    everything downstream -- piece numbers, overlay labels, export order. Seeding
    from ``set.pop()`` instead made the order depend on BMFace identity hashing,
    which varies between runs on identical geometry, so the same mesh could be
    numbered differently each time it was processed.
    """
    bm.faces.ensure_lookup_table()
    ordered = [f for f in bm.faces if not only_selected or f.select]
    pool = set(ordered)
    islands = []
    for seed in ordered:
        if seed not in pool:
            continue
        pool.discard(seed)
        component = {seed}
        stack = [seed]
        while stack:
            face = stack.pop()
            for loop in face.loops:
                if uv_boundary_edge(loop.edge, uv_layer):
                    continue
                for other in loop.edge.link_faces:
                    if other is not face and other in pool:
                        pool.discard(other)
                        component.add(other)
                        stack.append(other)
        islands.append(component)
    return islands


# A boundary vertex turning inward by more than this is a notch: the outline
# doubles back into the piece, leaving a spike of material. Texture packers
# produce these happily because they cost nothing in UV space, but they are
# awkward to saw, fragile at the point and hard to glue.
NOTCH_ANGLE_DEG = 75.0


def find_tight_notches(island_faces, uv_layer, notch_angle_deg=NOTCH_ANGLE_DEG):
    """UV positions where this island's outline cuts sharply back into itself.

    Returns a list of (Vector uv, turn_degrees). Only concave vertices count --
    a sharp convex point is just a corner of the piece and cuts fine.
    """
    found = []
    for cycle in island_boundary_cycles(island_faces, uv_layer):
        if len(cycle) < 3:
            continue
        pts = [l[uv_layer].uv.copy() for l in cycle]
        pts.append(cycle[-1].link_loop_next[uv_layer].uv.copy())

        shoelace = 0.0
        for k in range(len(pts) - 1):
            shoelace += pts[k].x * pts[k + 1].y - pts[k + 1].x * pts[k].y
        winding = 1.0 if shoelace >= 0.0 else -1.0

        for k in range(1, len(pts) - 1):
            v1 = pts[k] - pts[k - 1]
            v2 = pts[k + 1] - pts[k]
            if v1.length < 1e-12 or v2.length < 1e-12:
                continue
            cosine = max(-1.0, min(1.0, v1.normalized().dot(v2.normalized())))
            turn = degrees(acos(cosine))
            concave = (v1.x * v2.y - v1.y * v2.x) * winding < 0.0
            if concave and turn >= notch_angle_deg:
                found.append((pts[k].copy(), turn))
    return found


def island_sort_key(faces):
    """Stable identity for an island, independent of traversal order."""
    return min(f.index for f in faces)


def _next_boundary_loop(loop, uv_layer, island_faces):
    """Walk to the next boundary loop around the island, by rotating about the
    shared vertex until another boundary edge is found."""
    current = loop.link_loop_next
    for _ in range(128):  # guard against non-manifold fans
        if uv_boundary_edge(current.edge, uv_layer):
            return current
        radial = current.link_loop_radial_next
        if radial is current or radial.face not in island_faces:
            return None
        current = radial.link_loop_next
    return None


def island_boundary_cycles(island_faces, uv_layer):
    """Ordered boundary loops for one island, as a list of BMLoop lists.

    Each returned cycle walks the island's outline in order; a loop's edge runs
    from `loop.vert` to `loop.link_loop_next.vert`. An island with holes yields
    more than one cycle.
    """
    boundary = [l for f in island_faces for l in f.loops
                if uv_boundary_edge(l.edge, uv_layer)]
    unvisited = set(boundary)
    cycles = []
    while unvisited:
        start = unvisited.pop()
        cycle = [start]
        current = start
        while True:
            nxt = _next_boundary_loop(current, uv_layer, island_faces)
            if nxt is None or nxt is start or nxt not in unvisited:
                break
            unvisited.discard(nxt)
            cycle.append(nxt)
            current = nxt
        cycles.append(cycle)
    return cycles


# --- Dart run segmentation ----------------------------------------------

@dataclass
class DartRun:
    """One continuous saw pass along an island boundary.

    A run is a stretch of boundary that can be cut in a single straight pass at
    a single blade tilt, so it is bounded by two constraints: it must stay
    within `max_chord_deviation` of its own straight chord, and its dihedral
    angle must not vary by more than `max_angle_variation_deg` along its length.
    """
    island_index: int
    loops: list = field(repr=False)
    uv_start: object = None
    uv_end: object = None
    mean_angle_deg: float = 0.0
    min_angle_deg: float = 0.0
    max_angle_deg: float = 0.0
    length_3d: float = 0.0
    chord_deviation: float = 0.0
    # Unit vector in UV space pointing out of the piece, perpendicular to the
    # run. Derived from the boundary's winding rather than from the island
    # centroid: "away from the centre" only coincides with "outside" on convex
    # outlines, and real pieces are frequently concave, where the centroid test
    # silently puts annotations and darts on the wrong side.
    uv_outward: object = None
    # The run's own outline in UV space: every vertex it passes through, plus a
    # matching outward normal at each. A run can span several mesh edges along a
    # curve, so a straight line between its endpoints would not sit against the
    # edge it annotates. Offsetting these points along their normals traces the
    # real shape instead.
    uv_points: list = field(default_factory=list, repr=False)
    uv_normals: list = field(default_factory=list, repr=False)
    # Indices into uv_points where a relief dart belongs. Interior vertices
    # only, and only concave ones -- a convex vertex already has its material
    # falling away and needs nothing released.
    dart_indices: list = field(default_factory=list, repr=False)
    # False when the run lies on an open mesh boundary (the rim of the sculpt)
    # rather than a seam against another panel. Such a run has no facing piece,
    # so there is no dihedral to split and no saw bevel to set -- distinct from
    # a seam that genuinely measures 0 degrees.
    has_dihedral: bool = True


def _point_segment_distance(p, a, b):
    ab = b - a
    denom = ab.length_squared
    if denom < 1e-12:
        return (p - a).length
    t = max(0.0, min(1.0, (p - a).dot(ab) / denom))
    return (p - (a + ab * t)).length


def _run_deviation(points, i, j):
    """Max perpendicular distance of points[i+1:j] from the chord points[i]->points[j]."""
    a, b = points[i], points[j]
    worst = 0.0
    for k in range(i + 1, j):
        worst = max(worst, _point_segment_distance(points[k], a, b))
    return worst


def segment_boundary_cycle(cycle, uv_layer, max_chord_deviation,
                           max_angle_variation_deg, island_index=0,
                           corner_angle_deg=CORNER_ANGLE_DEG,
                           matrix=None, dart_relief_deg=DART_RELIEF_DEG,
                           max_darts_per_run=0):
    """Split one boundary cycle into DartRuns.

    A run is one continuous pass at one blade tilt, so the angle staying put is
    what defines it. Straightness is a separate, optional concern: a band saw
    happily follows a curve at a fixed tilt, and only a table saw needs the cut
    to be straight. Pass ``max_chord_deviation <= 0`` to drop the straightness
    constraint entirely, which is the default -- otherwise a curved edge held at
    a constant angle gets chopped into several runs carrying the same number,
    which is exactly what the hand-drafted sheets do not do: they draw one
    bracket across the whole stretch with the angle written once.

    Greedy forward scan: extend a run while it stays inside the active
    tolerances, close it when one is exceeded, then start the next. Greedy
    (rather than a global optimum like Douglas-Peucker) matches the physical
    process -- runs are cut one after another along the edge, each as long as
    the material and the saw allow.

    The scan starts at the sharpest corner of the cycle so a run never straddles
    an obvious physical break.
    """
    n = len(cycle)
    if n == 0:
        return []

    angles = []
    measured = []
    for loop in cycle:
        ang = dihedral_angle_deg(loop.edge)
        measured.append(ang is not None)
        angles.append(0.0 if ang is None else ang)

    # Rotate so index 0 sits at the most obvious physical corner of the piece.
    # Scored on how sharply the outline turns there, plus any step in the seam
    # angle. Turn matters most: on a flat panel every dihedral is zero, so a
    # dihedral-only score would start the scan mid-edge and split one straight
    # side into two runs.
    if n > 2:
        directions = []
        for loop in cycle:
            d = loop.link_loop_next.vert.co - loop.vert.co
            directions.append(d.normalized() if d.length > 1e-12 else d)

        def corner_score(i):
            prev, cur = directions[i - 1], directions[i]
            turn = degrees(prev.angle(cur)) if prev.length and cur.length else 0.0
            return turn + abs(angles[i] - angles[i - 1])

        pivot = max(range(n), key=corner_score)
        if pivot:
            cycle = cycle[pivot:] + cycle[:pivot]
            angles = angles[pivot:] + angles[:pivot]
            measured = measured[pivot:] + measured[:pivot]

    # points[k] is the start vertex of cycle[k]; the final entry closes the loop.
    # Taken through the object's world matrix when one is given, because every
    # length derived from them -- run length, chord deviation -- is compared
    # against a real-world tolerance in metres. Vertex coordinates are
    # object-local, so on an object scaled 10x they understate the piece by a
    # factor of ten and the tolerances silently mean something else.
    points = [l.vert.co.copy() for l in cycle]
    points.append(cycle[-1].link_loop_next.vert.co.copy())
    if matrix is not None:
        points = [matrix @ p for p in points]
    uvs = [l[uv_layer].uv.copy() for l in cycle]
    uvs.append(cycle[-1].link_loop_next[uv_layer].uv.copy())

    # Winding of this boundary loop in UV space, via the shoelace formula.
    # Positive means counter-clockwise. Every run in the loop shares it, so the
    # outward perpendicular is a fixed rotation of the run direction -- correct
    # for concave outlines too, unlike a centroid test. Holes wind the opposite
    # way to the outer boundary and therefore get the opposite rotation, which
    # is exactly right: "outside the piece" for a hole points into the hole.
    shoelace = 0.0
    for k in range(len(uvs) - 1):
        shoelace += uvs[k].x * uvs[k + 1].y - uvs[k + 1].x * uvs[k].y
    winding = 1.0 if shoelace >= 0.0 else -1.0

    # Outward normal at every vertex of the cycle, computed once for the whole
    # boundary rather than per run. Runs meet at shared vertices, and if each
    # derived its own end normal from only its own last segment, the two
    # brackets would part company there -- one of them ending up on the wrong
    # side of the edge. Sharing the vertex normal makes neighbouring brackets
    # meet exactly, at a single junction.
    cycle_seg_normals = []
    for k in range(len(uvs) - 1):
        d = uvs[k + 1] - uvs[k]
        dl = d.length
        cycle_seg_normals.append(Vector((d.y, -d.x)) * (winding / dl)
                                 if dl > 1e-12 else Vector((0.0, 0.0)))

    closed = (uvs[0] - uvs[-1]).length < 1e-9
    # Two normals per vertex: the one approaching it and the one leaving it.
    # On a gentle bend they are the same blended vector, so brackets either
    # side meet seamlessly. At a genuine corner, blending would give a bisector
    # that suits neither edge and visually welds two separate brackets into
    # one, so each side keeps its own normal and the brackets stay distinct.
    normals_in, normals_out, corner_flags = [], [], []
    # Turn at each boundary vertex, kept so relief darts can be distributed by
    # how much the outline bends rather than by distance or by vertex count.
    turns = []
    for k in range(len(uvs)):
        before = cycle_seg_normals[k - 1] if k > 0 else (
            cycle_seg_normals[-1] if closed else None)
        after = cycle_seg_normals[k] if k < len(cycle_seg_normals) else (
            cycle_seg_normals[0] if closed else None)
        if before is None or after is None:
            single = (after if before is None else before).copy()
            normals_in.append(single)
            normals_out.append(single.copy())
            corner_flags.append(False)
            turns.append(0.0)
            continue
        cosine = max(-1.0, min(1.0, before.dot(after)))
        turn = degrees(acos(cosine))
        turns.append(turn)
        if turn >= corner_angle_deg:
            normals_in.append(before.copy())
            normals_out.append(after.copy())
            corner_flags.append(True)
        else:
            blended = before + after
            blended = (blended.normalized() if blended.length > 1e-12
                       else after.copy())
            normals_in.append(blended)
            normals_out.append(blended.copy())
            corner_flags.append(False)

    # Convexity of each boundary vertex, as seen from the material. Sign of the
    # turn combined with the loop's winding: a convex vertex is one where the
    # outline bulges outward.
    convex_flags = []
    for k in range(len(uvs)):
        if k == 0 or k >= len(uvs) - 1:
            convex_flags.append(not closed)
            continue
        v1 = uvs[k] - uvs[k - 1]
        v2 = uvs[k + 1] - uvs[k]
        convex_flags.append((v1.x * v2.y - v1.y * v2.x) * winding > 0.0)

    runs = []
    i = 0
    while i < n:
        j = i + 1  # run covers cycle[i:j]; always at least one edge
        while j < n:
            # A corner is a physical break in the outline. Running one bracket
            # through it welds two separate edges into a single mark that
            # belongs to neither.
            if corner_flags[j]:
                break
            # A seam and an open rim are different cuts -- never merge them.
            if measured[j] != measured[i]:
                break
            span = angles[i:j + 1]
            if (max(span) - min(span)) > max_angle_variation_deg:
                break
            if max_chord_deviation > 0.0 and \
                    _run_deviation(points, i, j + 1) > max_chord_deviation:
                break
            j += 1
        span = angles[i:j]
        seg = cycle[i:j]
        length = sum((points[k + 1] - points[k]).length for k in range(i, j))

        run_uv = uvs[j] - uvs[i]
        run_len = run_uv.length
        if run_len > 1e-12:
            # Rotate the run direction a quarter turn, the way the winding says.
            outward = Vector((run_uv.y, -run_uv.x)) * (winding / run_len)
        else:
            outward = Vector((0.0, 0.0))

        # This run's slice of the shared outline. Its first vertex takes the
        # leaving normal and its last the approaching one, so at a corner each
        # side keeps the normal belonging to its own edge.
        pts = [uvs[k].copy() for k in range(i, j + 1)]
        normals = []
        for k in range(i, j + 1):
            if k == i:
                normals.append(normals_out[k].copy())
            elif k == j:
                normals.append(normals_in[k].copy())
            else:
                normals.append(normals_in[k].copy())
        # Dart marks belong at the interior vertices of the run, but not at
        # corners: a corner of the piece already gets its shape from the bevel
        # of the cuts meeting there, so a relief cut on top of that is
        # redundant. Convexity deliberately does not filter here -- a dart can
        # remove material to let a face close up, or open to give it room to
        # stretch, so it is wanted on either kind of bend.
        #
        # How many, and where, are two separate questions.
        #
        # How many comes from stress. A dart exists because flat foam cannot
        # follow a curve without the material somewhere having to give, and how
        # much it has to give is how far the outline turns -- so the number of
        # darts is the run's total turning divided by how much bend one dart
        # relieves. A straight run gets none however long it is, and a tight
        # curve gets several however short.
        #
        # Where is measured in turning too, not in distance. A spline's ticks
        # are spaced by parameter rather than arc length, so they bunch where
        # the curve is tight and spread where it is slack -- and that is exactly
        # where the material needs relief. Spacing them evenly by distance would
        # put just as many darts along the slack part of a run as along the
        # tight part, which is even-looking and physically wrong.
        #
        # Placing one at every vertex, which is what this did before, made dart
        # density a function of how finely the mesh was subdivided -- on a dense
        # sculpt a relief cut every millimetre, which perforates the piece
        # instead of patterning it. Neither the count nor the placement should
        # know anything about topology.
        span_turn = sum(turns[k] for k in range(i + 1, j))
        n_darts = 0
        if dart_relief_deg > 0.0:
            n_darts = int(round(span_turn / dart_relief_deg))
        if max_darts_per_run > 0:
            n_darts = min(n_darts, max_darts_per_run)

        darts = []
        if n_darts > 0:
            # Cumulative *turning* along the run. cumulative[k - i] is how far
            # the outline has bent by the time it reaches vertex k, so stepping
            # through it in equal shares puts a dart wherever an equal share of
            # the bending has been used up.
            cumulative = [0.0]
            for k in range(i, j):
                cumulative.append(cumulative[-1] +
                                  (turns[k + 1] if k + 1 < len(turns) else 0.0))
            total = cumulative[-1]
            if total > 1e-9:
                for m in range(n_darts):
                    target = total * (m + 1) / float(n_darts + 1)
                    # Nearest interior vertex to that point on the run. A dart
                    # has to sit on a vertex to be cut, but which vertex is a
                    # detail of the mesh, not of the pattern.
                    best, best_gap = None, None
                    for k in range(i + 1, j):
                        # Corners are skipped: a corner already gets its shape
                        # from the bevels meeting there, so relief on top of it
                        # is redundant.
                        if corner_flags[k]:
                            continue
                        gap = abs(cumulative[k - i] - target)
                        if best_gap is None or gap < best_gap:
                            best, best_gap = k - i, gap
                    if best is not None and best not in darts:
                        darts.append(best)
                darts.sort()
        runs.append(DartRun(
            island_index=island_index,
            loops=seg,
            uv_start=uvs[i],
            uv_end=uvs[j],
            mean_angle_deg=sum(span) / len(span),
            min_angle_deg=min(span),
            max_angle_deg=max(span),
            length_3d=length,
            chord_deviation=_run_deviation(points, i, j),
            has_dihedral=measured[i],
            uv_outward=outward,
            uv_points=pts,
            uv_normals=normals,
            dart_indices=darts,
        ))
        i = j
    return runs


def iter_boundary_dart_segments(bm, uv_layer, max_chord_deviation,
                                max_angle_variation_deg, only_selected=False,
                                islands=None,
                                corner_angle_deg=CORNER_ANGLE_DEG,
                                matrix=None, dart_relief_deg=DART_RELIEF_DEG,
                           max_darts_per_run=0):
    """Yield a DartRun for every saw pass along every UV island boundary.

    This replaces per-mesh-edge iteration: foam flexes between darts, so a dart
    belongs where the boundary stops being cuttable in one pass, not at every
    polygon edge.
    """
    if islands is None:
        islands = uv_islands(bm, uv_layer, only_selected)
    for index, faces in enumerate(islands):
        for cycle in island_boundary_cycles(faces, uv_layer):
            for run in segment_boundary_cycle(
                    cycle, uv_layer, max_chord_deviation,
                    max_angle_variation_deg, island_index=index,
                    corner_angle_deg=corner_angle_deg, matrix=matrix,
                    dart_relief_deg=dart_relief_deg,
                    max_darts_per_run=max_darts_per_run):
                yield run


def build_island_adjacency(bm, uv_layer, islands):
    """Map island index -> set of island indices sharing a 3D boundary edge.

    Used to place a dart symmetrically on both mating pieces, and to number
    islands by walking the surface rather than the UV layout.
    """
    owner = {}
    for index, faces in enumerate(islands):
        for face in faces:
            owner[face] = index
    adjacency = {i: set() for i in range(len(islands))}
    for edge in bm.edges:
        if len(edge.link_faces) != 2:
            continue
        a = owner.get(edge.link_faces[0])
        b = owner.get(edge.link_faces[1])
        if a is None or b is None or a == b:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def _uv_area(face, uv_layer):
    uvs = [l[uv_layer].uv for l in face.loops]
    area = 0.0
    for i in range(1, len(uvs) - 1):
        u, v = uvs[i] - uvs[0], uvs[i + 1] - uvs[0]
        area += abs(u.x * v.y - u.y * v.x) * 0.5
    return area


def uv_stretch_map(bm, uv_layer, faces=None):
    """Relative areal distortion per face: {face: stretch}.

    A face's raw UV-area-to-3D-area ratio is meaningless on its own -- UV space
    is normalised while the mesh is in metres, so the raw ratio just reports the
    unwrap's overall scale. What matters is how far a face deviates from the
    *rest of the mesh*, so ratios are normalised against their median: 0.0 means
    the face carries the same scale as the bulk of the unwrap, 0.5 means it is
    stretched or compressed by half again as much.

    This is the signal that a panel needs heat forming through its interior
    rather than just a dart at its edge. It is the same quantity the UV editor's
    Area stretch overlay displays, which is draw-only and unreadable from Python.
    """
    faces = list(bm.faces if faces is None else faces)
    ratios = {}
    for face in faces:
        area_3d = face.calc_area()
        if area_3d <= 1e-12:
            continue
        ratios[face] = _uv_area(face, uv_layer) / area_3d
    if not ratios:
        return {}
    ordered = sorted(ratios.values())
    median = ordered[len(ordered) // 2]
    if median <= 1e-12:
        return {f: 0.0 for f in ratios}
    return {f: abs(r / median - 1.0) for f, r in ratios.items()}


# --- Craftability assessment --------------------------------------------

# Rules that hold regardless of what the model looks like. Shown verbatim when
# the tool cannot confidently classify the shape, so the user always gets
# something actionable rather than a shrug.
FOAM_CRAFTABILITY_RULES = (
    "Each UV island becomes one flat piece of foam, so every island has to be "
    "something you could cut from a sheet and bend into place.",
    "Foam bends in one direction at a time. An island spanning a dome or saddle "
    "(curving two ways at once) cannot lie flat -- split it with more seams, or "
    "plan to heat-form it.",
    "A seam's bevel is half its dihedral angle, and a saw can only tilt so far. "
    "Past that limit the joint has to be heat-formed or carved instead.",
    "Put seams where the sculpt already changes direction. Seams along creases "
    "and ridges disappear; seams across a smooth panel always show.",
    "Islands must fit the foam stock, and want a consistent orientation if the "
    "foam has a grain or a skin side.",
    "Very small or sliver-thin islands are hard to cut accurately and harder to "
    "glue. Merge them into a neighbour where you can.",
)


# Mean areal distortion above this is not curvature -- it means the unwrap
# collapsed or overlapped. Measured on Suzanne: a sound Smart UV Project sits
# near 0.10 and the worst island near 0.30, while unwrapping the same mesh with
# no seams at all reached a median of 17 and a worst of 120.
DEGENERATE_STRETCH = 3.0


@dataclass
class Finding:
    """One craftability observation, with the fix attached."""
    severity: str          # 'ok' | 'info' | 'warn' | 'blocker'
    code: str
    title: str
    detail: str            # what was measured
    advice: str            # what to do about it
    islands: list = field(default_factory=list)


def _island_metrics(faces, uv_layer):
    area_3d = sum(f.calc_area() for f in faces)
    area_uv = sum(_uv_area(f, uv_layer) for f in faces)
    us = [l[uv_layer].uv for f in faces for l in f.loops]
    if not us:
        return None
    min_u = min(p.x for p in us); max_u = max(p.x for p in us)
    min_v = min(p.y for p in us); max_v = max(p.y for p in us)
    # UV space is unitless, so convert island extents to real lengths using the
    # island's own area ratio rather than assuming any global scale.
    scale = (area_3d / area_uv) ** 0.5 if area_uv > 1e-12 else 0.0
    width = (max_u - min_u) * scale
    height = (max_v - min_v) * scale
    return {"area_3d": area_3d, "width": max(width, height),
            "height": min(width, height)}


def classify_surface(bm):
    """Describe the model as 'faceted', 'smooth', or 'mixed'.

    Decided from the spread of interior dihedral angles: a faceted build is
    mostly flat panels punctuated by hard edges, a smooth sculpt bends a little
    at nearly every edge. Both are legitimate -- this only picks which advice
    applies.
    """
    angles = [abs(a) for a in (dihedral_angle_deg(e) for e in bm.edges)
              if a is not None]
    if not angles:
        return "unknown", {}
    total = len(angles)
    flat = sum(1 for a in angles if a < 1.0)
    gentle = sum(1 for a in angles if 1.0 <= a < 20.0)
    sharp = sum(1 for a in angles if a >= 30.0)
    flat_ratio = flat / total
    gentle_ratio = gentle / total
    sharp_ratio = sharp / total
    # A faceted build is bimodal: flat panels and hard creases, little between.
    # Testing flatness alone misreads a plain cube, whose every edge is a 90°
    # crease and which has no interior edges at all.
    bimodal_ratio = flat_ratio + sharp_ratio
    stats = {"edges": total, "flat_ratio": round(flat_ratio, 3),
             "gentle_ratio": round(gentle_ratio, 3),
             "sharp_ratio": round(sharp_ratio, 3),
             "bimodal_ratio": round(bimodal_ratio, 3),
             "max_angle": round(max(angles), 2)}
    if bimodal_ratio > 0.7 and gentle_ratio < 0.25:
        return "faceted", stats
    if gentle_ratio > 0.6:
        return "smooth", stats
    return "mixed", stats


def assess_craftability(bm, uv_layer, *, max_tilt_deg=45.0,
                        max_chord_deviation=0.003,
                        max_angle_variation_deg=10.0,
                        sheet_width=None, sheet_height=None,
                        stretch_tolerance=0.25, min_run_length=0.006,
                        islands=None, matrix=None):
    """Check whether this mesh and unwrap can actually be built from foam.

    Returns a list of Findings, most severe first. Every warning carries the
    change that would fix it; when the shape does not match a known profile the
    general rules are returned instead of a guess.
    """
    findings = []

    if uv_layer is None:
        return [Finding('blocker', 'no_uv', "No UV map",
                        "This mesh has no active UV layer.",
                        "Unwrap the mesh first: mark seams where the pieces "
                        "should separate, then Unwrap. Each UV island becomes "
                        "one piece of foam.")]

    if islands is None:
        islands = uv_islands(bm, uv_layer)
    if not islands:
        return [Finding('blocker', 'no_islands', "No faces to build",
                        "No UV islands were found.",
                        "Select the geometry you want to build and unwrap it.")]

    profile, stats = classify_surface(bm)

    # --- Seams that no saw can cut.
    steep = []
    worst = 0.0
    for edge in bm.edges:
        ang = dihedral_angle_deg(edge)
        if ang is None:
            continue
        per_side = abs(ang) / 2.0
        worst = max(worst, per_side)
        if per_side > max_tilt_deg:
            steep.append(edge)
    if steep:
        findings.append(Finding(
            'warn', 'bevel_beyond_saw', "Some seams are too sharp to saw",
            f"{len(steep)} edge(s) need a bevel over {max_tilt_deg:.0f}° per "
            f"side (worst {worst:.0f}°).",
            "These can still be built, just not with a straight saw cut -- "
            "heat-form or hand-carve them. To make them sawable instead, move "
            "the seam off the hard crease so the angle is shared between two "
            "gentler joints, or raise the max blade tilt if your saw allows."))

    # --- Islands that cannot lie flat.
    stretch = uv_stretch_map(bm, uv_layer)
    bad_islands = []
    island_mean = {}
    for index, faces in enumerate(islands):
        vals = [stretch.get(f, 0.0) for f in faces]
        if not vals:
            continue
        island_mean[index] = sum(vals) / len(vals)
        if island_mean[index] > stretch_tolerance:
            bad_islands.append(index)

    # Distortion this extreme is not a curvature problem -- it means the unwrap
    # itself collapsed or overlapped, usually from unwrapping without seams.
    # Worth separating, because the fix is completely different.
    wrecked = [i for i, v in island_mean.items() if v > DEGENERATE_STRETCH]
    if wrecked:
        worst_val = max(island_mean.values())
        findings.append(Finding(
            'blocker', 'degenerate_unwrap', "The unwrap is collapsed or overlapping",
            f"{len(wrecked)} island(s) distort by {worst_val:.0f}x -- far beyond "
            f"anything curvature alone can cause.",
            "Almost always this is an unwrap with no seams: the mesh has "
            "nowhere to open, so faces pile on top of each other. Mark seams "
            "where the pieces should separate and unwrap again. Smart UV "
            "Project is a good starting point if you have no seams yet.",
            islands=wrecked))

    if bad_islands:
        findings.append(Finding(
            'warn', 'non_developable', "Some pieces will not lie flat",
            f"{len(bad_islands)} island(s) distort more than "
            f"{stretch_tolerance:.0%} when flattened.",
            "This is a dome or saddle being forced onto one flat piece. Add a "
            "seam across the curved area to split it into two shallower pieces, "
            "or keep it and plan to heat-form. Darts along the edge help at the "
            "border but will not fix curvature through the middle.",
            islands=bad_islands))

    # --- Pieces that outgrow the stock.
    if sheet_width and sheet_height:
        oversized = []
        for index, faces in enumerate(islands):
            m = _island_metrics(faces, uv_layer)
            if m and (m["width"] > max(sheet_width, sheet_height)
                      or m["height"] > min(sheet_width, sheet_height)):
                oversized.append(index)
        if oversized:
            findings.append(Finding(
                'warn', 'exceeds_stock', "Some pieces are bigger than the foam",
                f"{len(oversized)} island(s) exceed "
                f"{sheet_width:.3g} x {sheet_height:.3g} m of stock.",
                "Add a seam to split each oversized piece. A seam along a "
                "natural line costs nothing visually and is easier than piecing "
                "an overhang later.",
                islands=oversized))

    # --- Notches: the outline doubling back into the piece.
    notched = []
    worst_turn = 0.0
    for index, faces in enumerate(islands):
        notches = find_tight_notches(faces, uv_layer)
        if notches:
            notched.append(index)
            worst_turn = max(worst_turn, max(t for _, t in notches))
    if notched:
        findings.append(Finding(
            'warn', 'tight_notch', "Some pieces have a sharp inward notch",
            f"{len(notched)} island(s) have an outline that cuts back into "
            f"itself, turning up to {worst_turn:.0f} degrees.",
            "This is a patterning problem, not a drawing one. An unwrap packed "
            "for texturing will happily leave a needle-thin spike, because it "
            "costs nothing in UV space -- but that piece is awkward to saw, "
            "fragile at the point and hard to glue, and no one drafting by hand "
            "would draw it. Move the seam so the notch becomes a piece "
            "boundary, or split the island there.",
            islands=notched))

    # --- Slivers that are painful to cut and glue.
    metrics = [(i, _island_metrics(f, uv_layer)) for i, f in enumerate(islands)]
    slivers = [i for i, m in metrics
               if m and m["height"] > 0 and (m["width"] / max(m["height"], 1e-9)) > 12.0]
    if slivers:
        findings.append(Finding(
            'info', 'sliver_islands', "Some pieces are very long and thin",
            f"{len(slivers)} island(s) are over 12x longer than they are wide.",
            "Thin strips are hard to cut accurately and tend to curl when "
            "glued. Merge them into a neighbouring island by clearing the seam "
            "between them, unless the strip is deliberate.",
            islands=slivers))

    # --- Cuts too short to make cleanly.
    runs = list(iter_boundary_dart_segments(
        bm, uv_layer, max_chord_deviation, max_angle_variation_deg,
        islands=islands, matrix=matrix))
    if runs:
        stubby = [r for r in runs if r.has_dihedral and r.length_3d < min_run_length]
        if len(stubby) > len(runs) * 0.25:
            findings.append(Finding(
                'info', 'fragmented_runs', "Cut runs are heavily fragmented",
                f"{len(stubby)} of {len(runs)} runs are shorter than "
                f"{min_run_length * 1000:.0f} mm.",
                "Very short runs mean re-setting the blade constantly. Raise "
                "the chord deviation or the angle variation tolerance to merge "
                "neighbouring runs into single passes -- foam bends enough to "
                "absorb the difference."))
        findings.append(Finding(
            'ok', 'runs_ready', "Cut runs resolved",
            f"{len(runs)} run(s) across {len(islands)} island(s); "
            f"{sum(1 for r in runs if r.has_dihedral)} need a bevel.",
            "Each run is one straight pass at one blade tilt."))

    # --- Open rims are worth stating so they are not mistaken for a bug.
    open_edges = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    if open_edges:
        findings.append(Finding(
            'info', 'open_boundary', "The model has open edges",
            f"{open_edges} edge(s) border no second face.",
            "These are the rim of the build. They get no bevel because nothing "
            "meets them -- expected for a shell, worth checking if you meant "
            "the model to be closed."))

    non_manifold = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    if non_manifold:
        findings.append(Finding(
            'warn', 'non_manifold', "Non-manifold geometry",
            f"{non_manifold} edge(s) join more than two faces.",
            "Three faces meeting at one edge has no physical equivalent in "
            "sheet foam. Clean it up (Mesh > Clean Up > Merge by Distance, or "
            "delete the interior face) before planning cuts."))

    # --- Profile-specific guidance, or the general rules when unsure.
    if profile == 'faceted':
        findings.append(Finding(
            'ok', 'profile_faceted', "Chunky, faceted build",
            f"{stats['flat_ratio']:.0%} of edges lie flat and "
            f"{stats['sharp_ratio']:.0%} are hard creases "
            f"(sharpest {stats['max_angle']:.0f}°).",
            "This suits foam well: flat panels meeting at defined edges. Seam "
            "every hard edge so each flat region becomes its own piece, and the "
            "bevels follow directly."))
    elif profile == 'smooth':
        findings.append(Finding(
            'info', 'profile_smooth', "Smooth, curved sculpt",
            f"{stats['gentle_ratio']:.0%} of edges bend gently, "
            f"sharpest {stats['max_angle']:.0f}°.",
            "Curved sculpts need the surface broken into near-flat patches. "
            "Run seams along the directions the form does not curve (down a "
            "horn, around a dome) so each piece bends one way only."))
    else:
        findings.append(Finding(
            'info', 'profile_unknown', "Shape did not match a known profile",
            f"Mixed curvature: {stats.get('flat_ratio', 0):.0%} flat, "
            f"{stats.get('gentle_ratio', 0):.0%} gently curved.",
            "Not confident which pattern applies here, so judge it against the "
            "general rules:\n- " + "\n- ".join(FOAM_CRAFTABILITY_RULES)))

    order = {'blocker': 0, 'warn': 1, 'info': 2, 'ok': 3}
    findings.sort(key=lambda f: order.get(f.severity, 9))
    return findings


def estimate_flex_tolerance(foam_grade, thickness_m):
    """Suggested (max_chord_deviation, dissolve_angle_limit_deg) for a foam type.

    A starting point, not a measurement. Bending stiffness of a flat sheet goes
    roughly with thickness cubed and with material stiffness, so thicker/denser
    foam holds less unaided curve and wants darts sooner. Calibrate by bending a
    sample strip of the real stock and measuring the sagitta it holds before it
    needs a relief cut.
    """
    stiffness = _GRADE_STIFFNESS.get(foam_grade, 1.0)
    thickness_ratio = thickness_m / 0.0127  # relative to 1/2" reference stock
    resistance = stiffness * (thickness_ratio ** 3)
    deviation = max(0.0005, min(0.010, 0.003 / max(resistance, 0.05)))
    angle_limit = max(1.0, min(20.0, 5.0 / max(resistance ** 0.5, 0.2)))
    return deviation, angle_limit


# --- Why a piece will not lie flat: Gaussian curvature -------------------

def vertex_angle_defect(vert, faces=None):
    """Angle defect at one vertex, in radians. Discrete Gaussian curvature.

    ``2*pi`` minus the corner angles meeting at the vertex, which is literally
    the gap that opens when those faces are flattened out onto a table. That gap
    is the material a dart has to remove.

    This is the quantity darts are actually about, and it is not the same as how
    much the surface bends. A cylinder bends as hard as you like and its defect
    is exactly zero, because a flat sheet wraps a cylinder without stretching
    anywhere -- measured on a 64-sided open cylinder, 0.0 degrees total. A cone
    is also flat everywhere except its tip, where the whole requirement piles up
    into one wedge: measured 199.05 degrees on a cone of radius 0.5 and depth
    1.0, against ``2*pi*(1 - r/slant)`` = 199.05. That wedge is precisely the pie
    slice you cut from a paper disc to make a cone.

    On a boundary vertex the reference is ``pi`` rather than ``2*pi``, since only
    a half-turn of material is present.
    """
    linked = [f for f in vert.link_faces if faces is None or f in faces]
    if not linked:
        return 0.0
    total = 0.0
    for face in linked:
        for loop in face.loops:
            if loop.vert is vert:
                total += loop.calc_angle()
    interior = (not vert.is_boundary) and len(linked) == len(vert.link_faces)
    from math import pi
    return (2.0 * pi - total) if interior else (pi - total)


def island_angle_defect(faces):
    """Total interior angle defect of a patch, in radians, and per vertex.

    Returns ``(total, {vert: defect})`` over the vertices strictly inside the
    patch. By Gauss-Bonnet this total is how much material the patch's darts
    have to remove between them -- a whole sphere comes to 4*pi, measured
    12.5664 against 12.5664 at three densities.

    Boundary vertices are reported but excluded from the total: their defect is
    a statement about the outline's shape, not about the piece refusing to lie
    flat, and counting it would double-charge a seam that the bevel already
    handles.
    """
    faces = set(faces)
    total = 0.0
    per_vertex = {}
    seen = set()
    for face in faces:
        for vert in face.verts:
            if vert in seen:
                continue
            seen.add(vert)
            linked = [f for f in vert.link_faces if f in faces]
            interior = ((not vert.is_boundary)
                        and len(linked) == len(vert.link_faces))
            defect = vertex_angle_defect(vert, faces)
            per_vertex[vert] = defect
            if interior:
                total += defect
    return total, per_vertex
