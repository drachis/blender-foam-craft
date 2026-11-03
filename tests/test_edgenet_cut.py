"""What bmesh.utils.face_split_edgenet actually does.

    python tools/run_tests.py --only edgenet

This characterises the primitive **before** anything is built on it, because the
last time a primitive's behaviour was assumed rather than measured it cost the
whole architecture. An earlier session recorded that BM_face_split_edgenet was
C-only and would need a patched Blender; it is in fact exposed, and the cutter
was built the hard way around a function that was there all along. The lesson
is not "read more carefully" -- it is that a claim about an API is worth a test.

So: nothing here uses FoamCraft's cutter. Every case is a bare quad, a net of
edges made by hand, and one call. What comes back is asserted exactly.

What the source says it should do, and what each case checks:

    a net crossing the face          splits it, one region either side
    a polyline net                   follows the drawn points, not a chord
    a net meeting at a middle vertex four regions from one call, no ordering
    a net that misses the boundary   *ignored*, silently, not an error
    a net edge already in the face   forbidden, and what actually happens

Reference, read from the local checkout:

    blender/source/blender/bmesh/intern/bmesh_polygon_edgenet.cc:477   the split
    blender/source/blender/bmesh/intern/bmesh_polygon_edgenet.cc:519   the asserts
    blender/source/blender/python/bmesh/bmesh_py_utils.cc:589          the binding
    blender/source/blender/editors/mesh/editmesh_knife.cc:2070         the knife's use

The binding's own note is the one that governs the awkward case:

    "Regions defined by edges need to connect to the face, otherwise they're
     ignored as loose edges."

Every check states both bounds. "It made some faces" is satisfied by a splitter
that shatters the quad, so counts are asserted exactly, and the cases that
should do *nothing* assert that the face count is unchanged -- which a working
splitter would fail.
"""

import sys
import traceback

import bpy
import bmesh
from mathutils import Vector

_TOOLS = r"O:/Work/BlenderPlugin/FoamCraft/tools"


def _shot(label, mesh_obj):
    """Capture the result, framed on the seams. See tools/shot.capture_cut."""
    if bpy.app.background:
        return None
    if _TOOLS not in sys.path:
        sys.path.insert(0, _TOOLS)
    try:
        from shot import capture_cut
    except ImportError:
        print("FoamCraft tests: cannot import shot from %r" % _TOOLS)
        return None
    return capture_cut(mesh_obj, "test_%s" % label)


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


def _quad():
    """One quad, corners at (+-1, +-1, 0). Returns (object, bmesh, face)."""
    _clear()
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    obj = bpy.context.object
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    return obj, bm, bm.faces[0]


def _edge_at(face, axis, value):
    """The face's edge whose midpoint has `axis` closest to `value`.

    Picking edges by geometry rather than by index, so the test does not depend
    on the order the plane primitive happens to hand them back -- the same
    numbering assumption that made the sphere tests unrepeatable.
    """
    def key(edge):
        mid = (edge.verts[0].co + edge.verts[1].co) * 0.5
        return abs(mid[axis] - value)
    return min(face.edges, key=key)


def _finish(obj, bm, net, label):
    """Write the result back, seam the net, photograph it, and report."""
    for edge in net:
        if edge.is_valid:
            edge.seam = True
    faces = len(bm.faces)
    verts = len(bm.verts)
    wire = sum(1 for e in bm.edges if e.is_wire)
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    shot = _shot(label, obj)
    return {"faces": faces, "verts": verts, "wire_edges": wire,
            "shot": bool(shot)}


# --- 1. A single edge straight across ------------------------------------
def test_one_edge_splits_the_quad_in_two():
    """The simplest possible net: two boundary vertices joined by one edge.

    Exactly 2 faces. A splitter that did nothing leaves 1, and one that treated
    the wire edge as a separate region could leave more.
    """
    obj, bm, face = _quad()
    left = _edge_at(face, 0, -1.0)
    right = _edge_at(face, 0, 1.0)
    _e, v_left = bmesh.utils.edge_split(left, left.verts[0], 0.5)
    _e, v_right = bmesh.utils.edge_split(right, right.verts[0], 0.5)
    bm.faces.ensure_lookup_table()
    face = bm.faces[0]

    net = [bm.edges.new((v_left, v_right))]
    made = bmesh.utils.face_split_edgenet(face=face, edgenet=net)

    report = _finish(obj, bm, net, "edgenet_one_edge")
    report["faces_returned"] = len(made)
    report["checks"] = {
        "two_faces": report["faces"] == 2,
        # The binding returns what it created; both halves come back.
        "returned_both": len(made) == 2,
        "no_wire_left": report["wire_edges"] == 0,
    }
    return report


# --- 2. A polyline, not a chord ------------------------------------------
def test_a_polyline_net_keeps_its_interior_points():
    """A net bent through interior vertices follows them exactly.

    This is the case that killed bisect_plane: a plane cuts a straight chord and
    cannot place a vertex inside a face. Three interior vertices are placed off
    the straight line between the two boundary points, and all three must
    survive into the finished mesh at the coordinates given.
    """
    obj, bm, face = _quad()
    left = _edge_at(face, 0, -1.0)
    right = _edge_at(face, 0, 1.0)
    _e, v_left = bmesh.utils.edge_split(left, left.verts[0], 0.5)
    _e, v_right = bmesh.utils.edge_split(right, right.verts[0], 0.5)
    bm.faces.ensure_lookup_table()
    face = bm.faces[0]

    # A zig-zag, well clear of the straight line joining the two ends.
    wanted = [Vector((-0.5, 0.6, 0.0)),
              Vector((0.0, -0.6, 0.0)),
              Vector((0.5, 0.6, 0.0))]
    inner = [bm.verts.new(co) for co in wanted]
    chain = [v_left] + inner + [v_right]
    net = [bm.edges.new((chain[i], chain[i + 1]))
           for i in range(len(chain) - 1)]
    made = bmesh.utils.face_split_edgenet(face=face, edgenet=net)

    placed = [any((v.co - co).length < 1e-6 for v in bm.verts)
              for co in wanted]
    report = _finish(obj, bm, net, "edgenet_polyline")
    report["faces_returned"] = len(made)
    report["interior_points_placed"] = placed
    report["checks"] = {
        "two_faces": report["faces"] == 2,
        # 4 original corners + 2 on edges + 3 interior.
        "nine_verts": report["verts"] == 9,
        "every_interior_point_survived": all(placed),
    }
    return report


# --- 3. Four regions from one call ---------------------------------------
def test_a_net_meeting_at_a_centre_makes_four_regions():
    """One call, four faces, from a net that is not a single path.

    This is the property the whole rebuild rests on: the net is an unordered
    *set* of edges, and the splitter works out the regions itself. There is no
    start, no end and no traversal order to get right -- the thing that all the
    run/entry/exit bookkeeping in the old cutter existed to manage.
    """
    obj, bm, face = _quad()
    mids = {}
    for axis, value, name in ((0, -1.0, "left"), (0, 1.0, "right"),
                              (1, -1.0, "bottom"), (1, 1.0, "top")):
        edge = _edge_at(face, axis, value)
        _e, vert = bmesh.utils.edge_split(edge, edge.verts[0], 0.5)
        mids[name] = vert
        bm.faces.ensure_lookup_table()
        face = bm.faces[0]

    centre = bm.verts.new(Vector((0.0, 0.0, 0.0)))
    net = [bm.edges.new((mids[name], centre)) for name in mids]
    made = bmesh.utils.face_split_edgenet(face=face, edgenet=net)

    report = _finish(obj, bm, net, "edgenet_cross")
    report["faces_returned"] = len(made)
    report["checks"] = {
        "four_faces": report["faces"] == 4,
        "returned_four": len(made) == 4,
        "no_wire_left": report["wire_edges"] == 0,
    }
    return report


# --- 4. A net that never reaches the boundary ----------------------------
def test_a_net_inside_the_face_is_ignored():
    """A closed loop floating inside the face does nothing at all.

    The documented behaviour, and the one real gap in the primitive: the walk is
    seeded from a vertex of the face's own boundary, so a component that never
    touches it is never reached. The knife works around this by calling
    BM_face_split_edgenet_connect_islands first, which is *not* exposed to
    Python -- so this case has to be handled before the call, not after.

    Asserted as "nothing happened", both in the face count and in the loop's
    edges still being wire. An implementation that quietly made a face here
    would fail, and so would one that raised.
    """
    obj, bm, face = _quad()
    ring = [bm.verts.new(Vector(co)) for co in
            ((-0.4, -0.4, 0.0), (0.4, -0.4, 0.0),
             (0.4, 0.4, 0.0), (-0.4, 0.4, 0.0))]
    net = [bm.edges.new((ring[i], ring[(i + 1) % 4])) for i in range(4)]

    raised = None
    try:
        made = bmesh.utils.face_split_edgenet(face=face, edgenet=net)
    except ValueError as exc:
        made, raised = (), str(exc)[:120]

    report = _finish(obj, bm, net, "edgenet_island")
    report["faces_returned"] = len(made)
    report["raised"] = raised
    report["checks"] = {
        "face_count_unchanged": report["faces"] == 1,
        "nothing_returned": len(made) == 0,
        "it_did_not_raise": raised is None,
        # All four stay wire, confirming they were skipped rather than consumed.
        "loop_still_wire": report["wire_edges"] == 4,
    }
    return report


# --- 5. A net edge that already belongs to the face ----------------------
def test_a_net_edge_already_in_the_face():
    """Passing one of the face's own edges is forbidden. Record what it does.

    bmesh_polygon_edgenet.cc:519-528 asserts BM_edge_in_face(edge_net[i], f) ==
    false, but BLI_assert compiles out of a release build and the Python binding
    does not check it -- so a caller that gets this wrong gets no warning. The
    knife guards it explicitly ("Shouldn't happen, but in this case just
    ignore.", editmesh_knife.cc:2105) and so must we.

    This measures the consequence rather than asserting a particular one, so
    that the guard we write is justified by what actually happens. The check is
    only that Blender survives it and the mesh is not left in pieces.
    """
    obj, bm, face = _quad()
    left = _edge_at(face, 0, -1.0)
    right = _edge_at(face, 0, 1.0)
    _e, v_left = bmesh.utils.edge_split(left, left.verts[0], 0.5)
    _e, v_right = bmesh.utils.edge_split(right, right.verts[0], 0.5)
    bm.faces.ensure_lookup_table()
    face = bm.faces[0]

    crossing = bm.edges.new((v_left, v_right))
    # ...plus an edge that is already part of the face's own outline.
    offender = next(e for e in face.edges if e is not crossing)
    net = [crossing, offender]

    raised = None
    try:
        made = bmesh.utils.face_split_edgenet(face=face, edgenet=net)
    except ValueError as exc:
        made, raised = (), str(exc)[:120]

    report = _finish(obj, bm, [crossing], "edgenet_edge_in_face")
    report["faces_returned"] = len(made)
    report["raised"] = raised
    report["checks"] = {
        # Whatever it does, it must not destroy the mesh.
        "mesh_survived": report["faces"] >= 1 and report["verts"] == 6,
    }
    return report


TESTS = (
    test_one_edge_splits_the_quad_in_two,
    test_a_polyline_net_keeps_its_interior_points,
    test_a_net_meeting_at_a_centre_makes_four_regions,
    test_a_net_inside_the_face_is_ignored,
    test_a_net_edge_already_in_the_face,
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
