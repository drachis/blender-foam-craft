"""3D viewport preview and debugging for foam panel layouts.

Two operators:
- FOAMCRAFT_DRACHIS_OT_preview_thickness: Show the sculpt with real foam thickness.
- FOAMCRAFT_DRACHIS_OT_layout_preview: Generate mesh geometry for each UV island
  (FLAT, BENT, or DART_CUT modes) so pieces can be inspected in the viewport.

Collections: generated objects placed in 'FoamCraft Preview', cleared on each run.
"""

import bpy
import bmesh
from mathutils import Vector, Matrix
from . import foam_angle_core as core


# --- Thickness preview operator -----------------------------------------------

class FOAMCRAFT_DRACHIS_OT_preview_thickness(bpy.types.Operator):
    """Show the sculpt with real foam thickness using a Solidify modifier.

    Per-source replacement: only the existing preview for this source object
    is deleted; previews belonging to other objects are left untouched.
    """

    bl_idname = "foamcraft_drachis.preview_thickness"
    bl_label = "Preview Foam Thickness"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}

        settings = context.scene.foamcraft_drachis
        thickness = settings.foam_thickness

        # Delete any existing preview for this source object.
        _delete_preview_for_source(obj.name)

        # Duplicate mesh data.
        mesh_copy = obj.data.copy()
        preview_obj = bpy.data.objects.new(f"{obj.name}_foam_preview", mesh_copy)
        context.collection.objects.link(preview_obj)

        # Tag it as a foam preview.
        preview_obj["is_foam_preview"] = True
        preview_obj["foam_preview_source"] = obj.name

        # Offset in X so it doesn't z-fight the original.
        bounds_x = max((v.co.x for v in mesh_copy.vertices), default=0.0)
        preview_obj.location.x = obj.location.x + bounds_x + thickness + 0.01

        # Add Solidify modifier for thickness.
        mod = preview_obj.modifiers.new(name="Foam Thickness", type='SOLIDIFY')
        mod.thickness = thickness
        mod.offset = 0.0
        mod.use_even_offset = True

        self.report({'INFO'},
                    f"Foam preview created: {preview_obj.name}, "
                    f"thickness {thickness * 1000:.1f} mm")
        return {'FINISHED'}


def _delete_preview_for_source(source_name):
    """Delete the foam preview whose foam_preview_source equals source_name."""
    for obj in bpy.data.objects:
        if (obj.get("is_foam_preview") and
                obj.get("foam_preview_source") == source_name):
            bpy.data.objects.remove(obj, do_unlink=True)


# --- Layout preview operator ---------------------------------------------------

class FOAMCRAFT_DRACHIS_OT_layout_preview(bpy.types.Operator):
    """Generate real mesh geometry for foam pieces in the 3D viewport.

    Modes:
    - FLAT: flat pieces laid out on XY plane in grid packing, true size.
    - BENT: pieces at original 3D position/orientation on the sculpt.
    - DART_CUT: FLAT mode plus marker geometry for relief darts.
    """

    bl_idname = "foamcraft_drachis.layout_preview"
    bl_label = "Preview Foam Layout"
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(
        name="Layout Mode",
        items=[
            ('FLAT', "Flat (Shelf)", "Flat pieces laid out on XY plane"),
            ('BENT', "Bent (On Sculpt)", "Pieces at original 3D position"),
            ('DART_CUT', "Dart Cut (Flat + Relief)", "Flat mode plus dart markers"),
        ],
        default='FLAT')

    use_boolean: bpy.props.BoolProperty(
        name="Experimental: Boolean Cut",
        description="Apply boolean subtraction for darts (experimental, may be unstable)",
        default=False)

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}

        settings = context.scene.foamcraft_drachis

        # Get or create the FoamCraft Preview collection.
        collection = _get_or_create_collection("FoamCraft Preview", context)

        # Clear old preview geometry.
        for o in list(collection.objects):
            bpy.data.objects.remove(o, do_unlink=True)

        # Enter edit mode and get bmesh.
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            self.report({'ERROR'}, "Mesh has no UV map")
            bm.free()
            return {'CANCELLED'}

        # Partition into UV islands.
        islands = core.uv_islands(bm, uv_layer)
        if not islands:
            self.report({'ERROR'}, "No UV islands found")
            bm.free()
            return {'CANCELLED'}

        # Compute the uv-to-world scale for sizing flat pieces.
        scale_uv_to_world = _compute_uv_to_world_scale(bm, islands, uv_layer)

        # Create geometry for each island.
        if self.mode == 'FLAT':
            _layout_flat(obj, bm, islands, uv_layer, collection, settings,
                         scale_uv_to_world)
        elif self.mode == 'BENT':
            _layout_bent(obj, bm, islands, uv_layer, collection, settings)
        elif self.mode == 'DART_CUT':
            _layout_dart_cut(obj, bm, islands, uv_layer, collection, settings,
                             scale_uv_to_world, self.use_boolean)

        bm.free()

        self.report({'INFO'}, f"Layout preview created: {len(islands)} piece(s) "
                              f"in mode '{self.mode}'")
        return {'FINISHED'}


def _get_or_create_collection(name, context):
    """Get or create a collection in the scene."""
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    context.scene.collection.children.link(col)
    return col


def _compute_uv_to_world_scale(bm, islands, uv_layer):
    """Compute metres-per-UV-unit so flat pieces are true size.

    Ratio is sqrt(sum(face.calc_area()) / sum(uv_area)).
    """
    total_area_3d = 0.0
    total_area_uv = 0.0
    for island in islands:
        for face in island:
            total_area_3d += face.calc_area()
            uvs = [l[uv_layer].uv for l in face.loops]
            area_uv = 0.0
            for i in range(1, len(uvs) - 1):
                u, v = uvs[i] - uvs[0], uvs[i + 1] - uvs[0]
                area_uv += abs(u.x * v.y - u.y * v.x) * 0.5
            total_area_uv += area_uv

    if total_area_uv <= 1e-12:
        return 1.0
    return (total_area_3d / total_area_uv) ** 0.5


def _layout_flat(src_obj, bm, islands, uv_layer, collection, settings,
                 scale):
    """Create one flat piece per island, positioned by its UVs at true size."""

    for island_index, island_faces in enumerate(islands):
        # Create one flat mesh per island.
        mesh = _extract_island_mesh_flat(bm, island_faces, uv_layer, scale)
        if mesh.vertices.__len__() == 0:
            continue

        obj = bpy.data.objects.new(f"Piece_{island_index:02d}", mesh)
        collection.objects.link(obj)

        # No packing. The mesh is built straight from the UVs, so the pieces
        # already lie exactly where the unwrap put them, at true size -- u to
        # x, v to y, scaled to match the sculpt. Re-packing them onto shelves
        # threw away a layout the user had arranged deliberately, and got it
        # wrong besides: the offsets were added to UV positions rather than
        # replacing them, so pieces overlapped.

        # Color by angle and heat-form status.
        _apply_vertex_colors(obj, mesh, bm, island_faces, uv_layer, settings)


def _extract_island_mesh_flat(bm, island_faces, uv_layer, scale):
    """Extract one island as a flat mesh using UV coordinates as XY positions."""
    mesh = bpy.data.meshes.new("island")
    verts = []
    vert_map = {}
    faces_out = []

    # Collect unique vertices from the island.
    vert_index = 0
    for face in island_faces:
        face_verts = []
        for loop in face.loops:
            uv = loop[uv_layer].uv
            v_key = (round(uv.x, 8), round(uv.y, 8))

            if v_key not in vert_map:
                # Create a new vertex at the UV position (XY), Z=0.
                vert_map[v_key] = vert_index
                verts.append((uv.x * scale, uv.y * scale, 0.0))
                vert_index += 1

            face_verts.append(vert_map[v_key])
        faces_out.append(tuple(face_verts))

    mesh.from_pydata(verts, [], faces_out)
    mesh.update()
    return mesh


def _layout_bent(src_obj, bm, islands, uv_layer, collection, settings):
    """Create pieces at original 3D position/orientation on the sculpt."""
    for island_index, island_faces in enumerate(islands):
        # Create one mesh per island, keeping 3D positions.
        mesh = _extract_island_mesh_bent(bm, island_faces, uv_layer)
        if mesh.vertices.__len__() == 0:
            continue

        obj = bpy.data.objects.new(f"Piece_{island_index:02d}", mesh)
        collection.objects.link(obj)

        # Color by angle and heat-form status.
        _apply_vertex_colors(obj, mesh, bm, island_faces, uv_layer, settings)


def _extract_island_mesh_bent(bm, island_faces, uv_layer):
    """Extract one island as a mesh, keeping 3D positions."""
    mesh = bpy.data.meshes.new("island")
    verts = []
    vert_map = {}
    faces_out = []

    vert_index = 0
    for face in island_faces:
        face_verts = []
        for loop in face.loops:
            v_key = loop.vert.index

            if v_key not in vert_map:
                vert_map[v_key] = vert_index
                co = loop.vert.co
                verts.append((co.x, co.y, co.z))
                vert_index += 1

            face_verts.append(vert_map[v_key])
        faces_out.append(tuple(face_verts))

    mesh.from_pydata(verts, [], faces_out)
    mesh.update()
    return mesh


def _layout_dart_cut(src_obj, bm, islands, uv_layer, collection, settings,
                     scale, use_boolean):
    """Create flat pieces plus marker geometry for relief darts.

    Flat mode first, then add marker polygons (or boolean subtractions) for darts.
    """
    piece_objects = []

    for island_index, island_faces in enumerate(islands):
        mesh = _extract_island_mesh_flat(bm, island_faces, uv_layer, scale)
        if mesh.vertices.__len__() == 0:
            continue

        obj = bpy.data.objects.new(f"Piece_{island_index:02d}", mesh)
        collection.objects.link(obj)
        piece_objects.append((obj, island_index, island_faces))

        # Left where the UVs put it -- see _layout_flat. The dart markers are
        # built in the same UV space, so they land on their pieces with no
        # further bookkeeping.

        # Color by angle and heat-form status.
        _apply_vertex_colors(obj, mesh, bm, island_faces, uv_layer, settings)

    # Now add dart markers (or boolean subtractions).
    if use_boolean:
        # Experimental: apply boolean subtractions.
        for obj, island_index, island_faces in piece_objects:
            _add_dart_booleans(obj, bm, island_faces, uv_layer, settings, scale)
    else:
        # Default: add red marker geometry at dart footprints.
        dart_mesh = _build_dart_markers_flat(bm, islands, uv_layer, scale)
        if dart_mesh.vertices.__len__() > 0:
            dart_obj = bpy.data.objects.new("Dart_Markers", dart_mesh)
            collection.objects.link(dart_obj)

            # Apply red material.
            mat = bpy.data.materials.new(name="Dart_Red")
            mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)
            dart_obj.data.materials.append(mat)
            for face in dart_mesh.polygons:
                face.material_index = 0


def _build_dart_markers_flat(bm, islands, uv_layer, scale):
    """Build marker geometry for relief darts at island boundaries."""
    mesh = bpy.data.meshes.new("dart_markers")
    verts = []
    faces = []

    for island_index, island_faces in enumerate(islands):
        # Get dart runs for this island via all boundary cycles.
        cycles = core.island_boundary_cycles(island_faces, uv_layer)
        if not cycles:
            continue

        for cycle in cycles:
            dart_runs = core.segment_boundary_cycle(
                cycle, uv_layer, 0.003, 10.0, island_index)

            for dart_run in dart_runs:
                if not dart_run.has_dihedral:
                    continue

                # Create a small marker polygon at the dart start point.
                marker_size = 0.002  # 2mm marker
                uv_start = dart_run.uv_start

                # Convert UV to flat XY.
                x0 = uv_start.x * scale
                y0 = uv_start.y * scale

                v_idx = len(verts)
                verts.append((x0 - marker_size, y0 - marker_size, 0.0))
                verts.append((x0 + marker_size, y0 - marker_size, 0.0))
                verts.append((x0 + marker_size, y0 + marker_size, 0.0))
                verts.append((x0 - marker_size, y0 + marker_size, 0.0))
                faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))

    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def _add_dart_booleans(obj, bm, island_faces, uv_layer, settings, scale):
    """Add experimental boolean modifiers for dart subtractions."""
    # This is marked experimental and intentionally simple.
    # Production use would need more robust boolean geometry.
    pass  # Stub for now; full implementation requires careful geometry handling.


def _apply_vertex_colors(obj, mesh, bm, island_faces, uv_layer, settings):
    """Color vertices by dihedral angle and heat-form status."""
    # Create a vertex colour layer.
    color_layer = mesh.vertex_colors.new(name="foam_colors")

    # Build a map: vert.index -> mean angle of adjacent boundary edges.
    # For FLAT mode, all edges are potential boundaries, so we compute angle per vert.
    angle_per_vert = {}
    for face in island_faces:
        for loop in face.loops:
            vert_idx = loop.vert.index
            if core.uv_boundary_edge(loop.edge, uv_layer):
                angle = core.dihedral_angle_deg(loop.edge)
                if angle is not None:
                    if vert_idx not in angle_per_vert:
                        angle_per_vert[vert_idx] = []
                    angle_per_vert[vert_idx].append(angle)

    # Compute mean angle per vert.
    mean_angles = {}
    for vert_idx, angles in angle_per_vert.items():
        mean_angles[vert_idx] = sum(angles) / len(angles) if angles else 0.0

    # Assign colors to vertex color loops.
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            loop = mesh.loops[loop_idx]
            vert_idx = loop.vertex_index

            # Determine color based on mean angle.
            angle = mean_angles.get(vert_idx, 0.0)

            if angle >= 0:
                # Convex: blue.
                color = (0.15, 0.45, 1.0, 1.0)
            else:
                # Concave: pink.
                color = (1.0, 0.2, 0.55, 1.0)

            color_layer.data[loop_idx].color = color


# --- Registration ---------------------------------------------------------------

_classes = (
    FOAMCRAFT_DRACHIS_OT_preview_thickness,
    FOAMCRAFT_DRACHIS_OT_layout_preview,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
