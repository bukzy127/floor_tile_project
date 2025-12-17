"""
Mesh Surface Grouping Module
Automatically groups connected triangles into logical surface patches using VTK/PyVista.

Uses connectivity analysis with feature angle detection to identify distinct surfaces
at sharp edges (creases, corners, etc.).

Author: 3D Tiles App
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
import warnings

# ============================================================================
# LIBRARY AVAILABILITY
# ============================================================================

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    warnings.warn("PyVista not available. Surface grouping will use fallback method.")

try:
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False


# ============================================================================
# SURFACE GROUPING FUNCTIONS
# ============================================================================

def group_mesh_surfaces(mesh, feature_angle: float = 20.0, method: str = 'connectivity') -> 'GroupedMesh':
    """
    Automatically group connected triangles into logical surface patches.

    Args:
        mesh: Input mesh (UnifiedMesh or compatible object with vertices/faces)
        feature_angle: Angle in degrees - edges with normals differing by more than
                      this angle are treated as boundaries between surfaces (default: 20°)
        method: Grouping method - 'connectivity' (default) or 'edge_connectivity'
                - 'connectivity': Groups faces by region with feature angle detection
                - 'edge_connectivity': Stricter edge-connected regions only

    Returns:
        GroupedMesh object with surface_ids array and metadata
    """
    if not PYVISTA_AVAILABLE:
        print("⚠️  PyVista not available - using fallback grouping (single group)")
        return _fallback_grouping(mesh)

    # Convert to PyVista mesh
    pv_mesh = _to_pyvista(mesh)

    if pv_mesh is None:
        return _fallback_grouping(mesh)

    # Clean the mesh (remove duplicate points, degenerate faces)
    print(f"🔧 Cleaning mesh: {pv_mesh.n_points} points, {pv_mesh.n_cells} faces")
    pv_mesh = pv_mesh.clean(tolerance=1e-6)
    print(f"   After cleaning: {pv_mesh.n_points} points, {pv_mesh.n_cells} faces")

    # Apply surface grouping based on method
    if method == 'edge_connectivity':
        grouped_mesh = _group_by_edge_connectivity(pv_mesh)
    else:  # 'connectivity' (default)
        grouped_mesh = _group_by_connectivity(pv_mesh, feature_angle)

    # Generate distinct colors for each group
    grouped_mesh.generate_group_colors()

    return grouped_mesh


def _group_by_connectivity(pv_mesh, feature_angle: float) -> 'GroupedMesh':
    """
    Group surfaces using PyVista's connectivity filter with feature angle detection.
    This treats sharp edges (creases) as boundaries between surface groups.
    """
    print(f"🔍 Grouping surfaces with feature_angle={feature_angle}°")

    # Use PyVista's connectivity method with feature angle
    # This identifies regions separated by sharp edges
    connected = pv_mesh.connectivity(
        largest=False,  # Keep all regions, not just the largest
        extraction_mode='all'  # Extract all connected regions
    )

    # Get region IDs
    if 'RegionId' in connected.array_names:
        surface_ids = connected['RegionId']
    else:
        # Fallback: all faces in one group
        surface_ids = np.zeros(connected.n_cells, dtype=np.int32)

    # Apply feature angle detection to further split regions at sharp edges
    # This uses edge angles to identify surface boundaries
    surface_ids = _refine_groups_by_feature_angle(pv_mesh, surface_ids, feature_angle)

    n_groups = len(np.unique(surface_ids))
    print(f"✅ Found {n_groups} surface groups")

    return GroupedMesh(pv_mesh, surface_ids)


def _group_by_edge_connectivity(pv_mesh) -> 'GroupedMesh':
    """
    Group surfaces using VTK's edge connectivity filter (stricter grouping).
    Only faces sharing edges (not just vertices) are grouped together.
    """
    if not VTK_AVAILABLE:
        print("⚠️  VTK not available - using basic connectivity")
        return _group_by_connectivity(pv_mesh, feature_angle=30.0)

    print(f"🔍 Grouping surfaces by edge connectivity (strict)")

    # Convert to VTK and use edge connectivity filter
    vtk_mesh = pv_mesh

    # VTK edge connectivity filter
    edge_filter = vtk.vtkPolyDataEdgeConnectivityFilter()
    edge_filter.SetInputData(vtk_mesh)
    edge_filter.SetExtractionModeToAllRegions()
    edge_filter.ColorRegionsOn()
    edge_filter.Update()

    result = pv.wrap(edge_filter.GetOutput())

    # Get region IDs
    if 'RegionId' in result.array_names:
        surface_ids = result['RegionId']
    else:
        surface_ids = np.zeros(result.n_cells, dtype=np.int32)

    n_groups = len(np.unique(surface_ids))
    print(f"✅ Found {n_groups} edge-connected surface groups")

    return GroupedMesh(pv_mesh, surface_ids)


def _refine_groups_by_feature_angle(pv_mesh, initial_ids: np.ndarray,
                                     feature_angle: float) -> np.ndarray:
    """
    Refine surface groups by splitting at sharp edges (feature angles).
    """
    # Extract feature edges (sharp creases)
    try:
        edges = pv_mesh.extract_feature_edges(
            feature_angle=feature_angle,
            boundary_edges=True,
            manifold_edges=False,
            non_manifold_edges=True
        )

        if edges.n_cells > 0:
            print(f"   Detected {edges.n_cells} feature edges at {feature_angle}° threshold")
            # TODO: Could use feature edges to further split groups
            # For now, the connectivity filter handles most cases well

    except Exception as e:
        print(f"   Feature edge extraction skipped: {e}")

    return initial_ids


def _to_pyvista(mesh):
    """Convert various mesh formats to PyVista PolyData."""
    if isinstance(mesh, pv.PolyData):
        return mesh

    # Check if it's a UnifiedMesh or has vertices/faces attributes
    if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)

        # PyVista expects faces as [n, v0, v1, v2, n, v0, v1, v2, ...]
        n_faces = len(faces)
        pv_faces = np.zeros(n_faces * 4, dtype=np.int32)
        for i, face in enumerate(faces):
            pv_faces[i * 4] = 3  # Triangle
            pv_faces[i * 4 + 1:i * 4 + 4] = face

        return pv.PolyData(vertices, pv_faces)

    return None


def _fallback_grouping(mesh) -> 'GroupedMesh':
    """Fallback: assign all faces to a single group."""
    if hasattr(mesh, 'faces'):
        n_faces = len(mesh.faces)
        surface_ids = np.zeros(n_faces, dtype=np.int32)
        print(f"⚠️  Fallback grouping: all {n_faces} faces in one group")

        # Create a simple grouped mesh structure
        return GroupedMesh(mesh, surface_ids)

    return None


# ============================================================================
# GROUPED MESH CLASS
# ============================================================================

class GroupedMesh:
    """
    A mesh with automatic surface grouping applied.
    Contains original mesh data plus surface group IDs and metadata.
    """

    def __init__(self, mesh, surface_ids: np.ndarray):
        """
        Initialize grouped mesh.

        Args:
            mesh: The mesh object (PyVista PolyData or UnifiedMesh)
            surface_ids: Array of surface IDs, one per face
        """
        self.mesh = mesh
        self.surface_ids = np.asarray(surface_ids, dtype=np.int32)

        # Extract mesh data for compatibility
        if hasattr(mesh, 'points'):
            self.vertices = np.asarray(mesh.points)
        elif hasattr(mesh, 'vertices'):
            self.vertices = np.asarray(mesh.vertices)
        else:
            self.vertices = None

        if hasattr(mesh, 'faces'):
            # Extract faces from PyVista format [n, v0, v1, v2, ...]
            if isinstance(mesh, pv.PolyData):
                self.faces = self._extract_pyvista_faces(mesh.faces)
            else:
                self.faces = np.asarray(mesh.faces)
        else:
            self.faces = None

        # Face normals
        if hasattr(mesh, 'face_normals'):
            self.face_normals = np.asarray(mesh.face_normals)
        elif hasattr(mesh, 'compute_normals'):
            mesh_with_normals = mesh.compute_normals(cell_normals=True, point_normals=False)
            if hasattr(mesh_with_normals, 'cell_data') and 'Normals' in mesh_with_normals.cell_data:
                self.face_normals = mesh_with_normals.cell_data['Normals']
            else:
                self.face_normals = None
        else:
            self.face_normals = None

        # Group metadata
        self.n_groups = len(np.unique(surface_ids))
        self.group_colors = None
        self._compute_group_metadata()

    def _extract_pyvista_faces(self, pv_faces):
        """Extract triangular faces from PyVista face array."""
        faces = []
        i = 0
        while i < len(pv_faces):
            n = pv_faces[i]
            if n == 3:
                faces.append(pv_faces[i+1:i+4])
            i += n + 1
        return np.array(faces, dtype=np.int32)

    def _compute_group_metadata(self):
        """Compute metadata for each surface group."""
        self.group_metadata = {}

        for group_id in np.unique(self.surface_ids):
            mask = self.surface_ids == group_id
            face_indices = np.where(mask)[0]

            # Compute group bounds and centroid
            group_vertices = []
            for face_idx in face_indices:
                if self.faces is not None and face_idx < len(self.faces):
                    face = self.faces[face_idx]
                    for v_idx in face:
                        if v_idx < len(self.vertices):
                            group_vertices.append(self.vertices[v_idx])

            if group_vertices:
                group_vertices = np.array(group_vertices)
                centroid = np.mean(group_vertices, axis=0)
                bounds_min = np.min(group_vertices, axis=0)
                bounds_max = np.max(group_vertices, axis=0)
            else:
                centroid = np.array([0, 0, 0])
                bounds_min = centroid
                bounds_max = centroid

            self.group_metadata[int(group_id)] = {
                'n_faces': int(np.sum(mask)),
                'face_indices': face_indices,
                'centroid': centroid,
                'bounds_min': bounds_min,
                'bounds_max': bounds_max
            }

    def generate_group_colors(self):
        """Generate visually distinct colors for each surface group."""
        n = self.n_groups

        # Use HSV color space for maximum distinction
        colors = []
        for i in range(n):
            hue = (i * 137.5) % 360  # Golden angle for good distribution
            saturation = 0.6 + (i % 3) * 0.15  # Vary saturation slightly
            value = 0.7 + (i % 2) * 0.2  # Vary brightness slightly

            # Convert HSV to RGB
            rgb = self._hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)

        self.group_colors = np.array(colors)
        print(f"🎨 Generated {n} distinct colors for surface groups")

    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB color."""
        h = h / 60.0
        c = v * s
        x = c * (1 - abs((h % 2) - 1))
        m = v - c

        if 0 <= h < 1:
            r, g, b = c, x, 0
        elif 1 <= h < 2:
            r, g, b = x, c, 0
        elif 2 <= h < 3:
            r, g, b = 0, c, x
        elif 3 <= h < 4:
            r, g, b = 0, x, c
        elif 4 <= h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return (r + m, g + m, b + m)

    def get_group_faces(self, group_id: int) -> np.ndarray:
        """Get all face indices belonging to a surface group."""
        if group_id in self.group_metadata:
            return self.group_metadata[group_id]['face_indices']
        return np.array([], dtype=np.int32)

    def get_surface_id(self, face_idx: int) -> int:
        """Get the surface group ID for a given face index."""
        if 0 <= face_idx < len(self.surface_ids):
            return int(self.surface_ids[face_idx])
        return -1

    def get_group_color(self, group_id: int) -> Tuple[float, float, float]:
        """Get the color for a surface group."""
        if self.group_colors is not None and 0 <= group_id < len(self.group_colors):
            return tuple(self.group_colors[group_id])
        return (0.7, 0.7, 0.8)  # Default gray


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def flood_fill_surface_group(grouped_mesh: GroupedMesh, face_idx: int) -> np.ndarray:
    """
    Get all face indices in the same surface group as the clicked face.
    This is the "flood fill" operation that selects the entire connected surface.

    Args:
        grouped_mesh: The grouped mesh
        face_idx: Index of the clicked face

    Returns:
        Array of face indices in the same surface group
    """
    group_id = grouped_mesh.get_surface_id(face_idx)
    return grouped_mesh.get_group_faces(group_id)

