"""Persistent, spatially-sequential piece numbering for foam islands.

Each UV island becomes one physical piece of foam, and the whole point of
numbering is reassembly: a cut piece has to be matched back to its place on
the model by eye, from a number written on it. That only works if numbers
read as a walk across the surface -- neighbours in 3D get neighbouring
numbers -- which is why the reference hand-made patterns cluster e.g.
58,59,60,71,72 around one eye. Numbering by UV bounding-box position would
scatter physical neighbours randomly across the ID range, since UV layout is
free to pack islands however is most compact on the page.

Storage: a per-face INT attribute named ``foam_island_id`` (sentinel -1 =
unassigned), verified working via ``mesh.attributes.new(..., type='INT',
domain='FACE')`` on live Blender 5.2. In edit mode this is read/written
through the matching bmesh int layer (``bm.faces.layers.int``), which stays
synced with the Attribute API for a named layer -- no separate sync step
needed.

IDs must survive small edits (moving the 3D cursor, tweaking unrelated seams)
so a half-finished numbering pass isn't thrown away by accident. But an ID is
a promise that "this exact set of faces is one UV island" -- if a reseam
splits or merges an island, the old ID no longer names a single cuttable
piece and must not be trusted. See ``_validate_existing_ids``.
"""

import bmesh
import bpy
from mathutils import Vector

from . import foam_angle_core as core

ATTR_NAME = "foam_island_id"
UNASSIGNED = -1


# --- Read-only helpers, usable outside the operator (e.g. by the overlay) --

def island_id_map(bm, uv_layer, islands=None):
    """Map island index -> foam_island_id, or -1 if unassigned.

    An island whose faces don't all agree on one id also reads as -1 -- that
    disagreement means the id layer is stale relative to the current UV
    layout (a split/merge happened since it was written) and should be
    treated as "needs assignment" by any caller, not trusted at face value.
    Does not create the attribute layer; a mesh with none simply maps every
    island to -1.
    """
    if islands is None:
        islands = core.uv_islands(bm, uv_layer)
    id_layer = bm.faces.layers.int.get(ATTR_NAME)
    mapping = {}
    for index, faces in enumerate(islands):
        if id_layer is None:
            mapping[index] = UNASSIGNED
            continue
        ids = {f[id_layer] for f in faces}
        mapping[index] = ids.pop() if len(ids) == 1 else UNASSIGNED
    return mapping


def island_centroid_uv(faces, uv_layer):
    """UV-space centroid of an island's faces, for label placement.

    Averaged per-face (each face's own loop-average) rather than per-loop
    across the whole island, so an island with one large n-gon and several
    triangles isn't dragged toward the n-gon just because it contributes more
    loops.
    """
    faces = list(faces)
    if not faces:
        return Vector((0.0, 0.0))
    total = Vector((0.0, 0.0))
    for face in faces:
        face_total = Vector((0.0, 0.0))
        for loop in face.loops:
            face_total += loop[uv_layer].uv
        total += face_total / len(face.loops)
    return total / len(faces)


def _island_centroid_3d(faces):
    """Local-space 3D centroid of an island, for adjacency-walk distances."""
    faces = list(faces)
    if not faces:
        return Vector((0.0, 0.0, 0.0))
    total = Vector((0.0, 0.0, 0.0))
    for face in faces:
        total += face.calc_center_median()
    return total / len(faces)


# --- Invalidation ----------------------------------------------------------

def _validate_existing_ids(bm, islands, id_layer):
    """Clear any id whose face set is no longer exactly one current UV island.

    An id is trustworthy only if: every face carrying it belongs to the same
    current island, AND that island has no other faces without it (i.e. the
    id's face set and the island's face set are identical). That single check
    catches both failure modes of a re-unwrap:
    - split: the old island's faces now land in two+ current islands, so the
      id's faces don't all share one owner.
    - merge: two old islands' faces now land in one current island, so that
      island's faces disagree on id (caught here) or, if it also swallowed a
      never-numbered island, the id's face set is smaller than the island's.

    Mutates the id layer in place (clearing to -1) and prints one warning per
    invalidated id, naming it, so the change is visible on the console rather
    than silently discarded. Returns the set of current island indices that
    were touched, for the operator's "reassigned" count.
    """
    owner = {}
    for index, faces in enumerate(islands):
        for face in faces:
            owner[face] = index

    by_id = {}
    for face in bm.faces:
        fid = face[id_layer]
        if fid != UNASSIGNED:
            by_id.setdefault(fid, set()).add(face)

    invalidated_islands = set()
    for fid, faces in sorted(by_id.items()):
        owners = {owner[face] for face in faces}
        island_index = next(iter(owners)) if len(owners) == 1 else None
        matches_island = (island_index is not None
                          and len(faces) == len(islands[island_index]))
        if matches_island:
            continue
        print(f"FoamCraft: foam_island_id {fid} no longer matches a single "
              f"UV island (seam changed since it was assigned) -- clearing "
              f"{len(faces)} face(s) for reassignment.")
        invalidated_islands |= owners
        for face in faces:
            face[id_layer] = UNASSIGNED
    return invalidated_islands


# --- Spatial ordering --------------------------------------------------

def _walk_order(adjacency, centroids, origin):
    """Visit every island once, in an order that reads as a walk across the
    surface: consecutive IDs are always 3D neighbours whenever the adjacency
    graph makes that possible.

    Single-current greedy walk, matching the numbering scheme by hand: start
    at the island nearest `origin`; from the current island, step to its
    nearest not-yet-visited neighbour and make that the new current. Only
    when the current island has no unvisited neighbour left (its local
    neighbourhood is exhausted, or it was never connected to the rest --
    same situation from this function's point of view) does it jump to the
    nearest remaining island by raw centroid distance and continue from
    there. Consecutive IDs can fail to be adjacent only across such a jump.

    Distance ties are broken on the centroid's own coordinates (not on
    island index) so the result depends only on geometry. `islands` -- and
    therefore these `centroids` -- come from ``core.uv_islands``, which
    builds its result from a Python set keyed by BMFace identity; on a
    symmetric mesh like a seamed cube, where every face is exactly as close
    to `origin` as several others, breaking ties on index let that
    incidental, run-to-run-varying iteration order flip which of several
    equal-distance islands `min()` picked -- measured live: the same cube,
    same origin, same seams still produced a different walk (and a
    different non-adjacent consecutive pair) from one call to the next.
    """
    n = len(centroids)
    if n == 0:
        return []

    def tie_key(i, ref):
        c = centroids[i]
        return ((c - ref).length, round(c.x, 6), round(c.y, 6), round(c.z, 6))

    unvisited = set(range(n))
    current = min(unvisited, key=lambda i: tie_key(i, origin))
    order = [current]
    unvisited.discard(current)

    while unvisited:
        neighbours = [i for i in adjacency.get(current, ()) if i in unvisited]
        if neighbours:
            nxt = min(neighbours, key=lambda i: tie_key(i, centroids[current]))
        else:
            nxt = min(unvisited, key=lambda i: tie_key(i, centroids[current]))
        order.append(nxt)
        unvisited.discard(nxt)
        current = nxt
    return order


# --- Operator ----------------------------------------------------------

class FOAMCRAFT_DRACHIS_OT_assign_island_ids(bpy.types.Operator):
    """Assign persistent foam_island_id numbers to UV islands, walking 3D
    adjacency outward from a chosen origin"""

    bl_idname = "foamcraft_drachis.assign_island_ids"
    bl_label = "Assign Foam Piece Numbers"
    bl_options = {'REGISTER', 'UNDO'}

    renumber_all: bpy.props.BoolProperty(
        name="Renumber All",
        description="Discard every existing id and renumber every island "
                    "from scratch. Off: keep ids that are still valid and "
                    "only number unassigned or invalidated islands",
        default=False)

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        recall_mode = obj.mode
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        try:
            return self._execute(context, obj)
        finally:
            if obj.mode != recall_mode:
                bpy.ops.object.mode_set(mode=recall_mode)

    def _execute(self, context, obj):
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            self.report({'ERROR'}, "Mesh has no active UV map to derive islands from.")
            return {'CANCELLED'}

        id_layer = bm.faces.layers.int.get(ATTR_NAME)
        if id_layer is None:
            id_layer = bm.faces.layers.int.new(ATTR_NAME)
            for face in bm.faces:
                face[id_layer] = UNASSIGNED

        islands = core.uv_islands(bm, uv_layer)
        if not islands:
            self.report({'INFO'}, "No faces found.")
            return {'CANCELLED'}

        settings = context.scene.foamcraft_drachis
        if settings.origin_object is not None:
            origin = settings.origin_object.matrix_world.translation.copy()
        else:
            origin = context.scene.cursor.location.copy()
        # Island centroids are in the mesh's local space; go to world space
        # so the distance-to-origin comparison is fair regardless of the
        # object's own transform.
        centroids = [obj.matrix_world @ _island_centroid_3d(faces) for faces in islands]

        assigned = reassigned = kept = 0

        if self.renumber_all:
            for face in bm.faces:
                face[id_layer] = UNASSIGNED
            adjacency = core.build_island_adjacency(bm, uv_layer, islands)
            order = _walk_order(adjacency, centroids, origin)
            for new_id, index in enumerate(order, start=1):
                for face in islands[index]:
                    face[id_layer] = new_id
            assigned = len(order)
        else:
            invalidated = _validate_existing_ids(bm, islands, id_layer)
            existing_map = island_id_map(bm, uv_layer, islands)
            existing_ids = [v for v in existing_map.values() if v != UNASSIGNED]
            next_id = max(existing_ids) + 1 if existing_ids else 1
            for index, faces in enumerate(islands):
                if existing_map[index] != UNASSIGNED:
                    kept += 1
                    continue
                for face in faces:
                    face[id_layer] = next_id
                next_id += 1
                if index in invalidated:
                    reassigned += 1
                else:
                    assigned += 1

        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'},
                    f"Foam piece numbers: {assigned} assigned, "
                    f"{reassigned} reassigned, {kept} kept.")
        return {'FINISHED'}


_classes = (FOAMCRAFT_DRACHIS_OT_assign_island_ids,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
