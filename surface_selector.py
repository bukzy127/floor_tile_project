"""
Surface Selector for 3D Model Import
Handles surface selection from imported 3D models by grouping connected, coplanar triangles
into logical surface groups. Provides raycast helpers and surface metadata used by the
main application.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Set

# Attempt to import PyVista for faster connected-component pre-processing.
# PyVista is optional; behavior falls back to pure-numpy/trimesh logic when not present.
try:
    import pyvista as pv
    _HAS_PYVISTA = True
except Exception:
    pv = None
    _HAS_PYVISTA = False


class ValidationResult:
    def __init__(self, is_valid: bool, is_planar: bool, message: str = ""):
        self.is_valid = is_valid
        self.is_planar = is_planar
        self.message = message


class SelectedSurface:
    """Represents a selected surface from a 3D model (single face or aggregated group)."""

    def __init__(self, face_index: int, vertices: np.ndarray, normal: np.ndarray):
        """
        Initialize a selected surface.

        Args:
            face_index: Index of the face in the mesh
            vertices: Array of vertex positions for this face
            normal: Normal vector of the face
        """
        self.face_index = face_index
        self.vertices = vertices
        self.normal = normal

        # Calculate surface properties
        self.center = np.mean(vertices, axis=0)
        self.bounds = self._calculate_bounds()
        self.dimensions = self._calculate_dimensions()

    def _calculate_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate bounding box of the surface."""
        min_bounds = np.min(self.vertices, axis=0)
        max_bounds = np.max(self.vertices, axis=0)
        return min_bounds, max_bounds

    def _calculate_dimensions(self) -> Tuple[float, float]:
        """Calculate width and length of the surface."""
        min_bounds, max_bounds = self.bounds
        dimensions = max_bounds - min_bounds
        # Return width (max of x/y) and length (min of x/y)
        width = max(dimensions[0], dimensions[1])
        length = min(dimensions[0], dimensions[1])
        return width, length

    def get_width(self) -> float:
        """Get the width of the surface."""
        return self.dimensions[0]

    def get_length(self) -> float:
        """Get the length of the surface."""
        return self.dimensions[1]

    def get_area(self) -> float:
        """Calculate approximate area of the surface."""
        return self.dimensions[0] * self.dimensions[1]


class SurfaceSelector:
    """Helper class for selecting surfaces from a 3D mesh.

    Primary features added:
    - Automatic grouping of connected triangles that are approximately coplanar
      (based on face adjacency and normal angle threshold).
    - Raycast wrapper using trimesh.ray (falls back gracefully).
    - Query methods to get group faces, group id, projected bounds and validation.
    """

    def __init__(self, mesh):
        """
        Initialize the surface selector with a mesh.

        Args:
            mesh: A trimesh.Trimesh object
        """
        self.mesh = mesh
        self.selected_surface: Optional[SelectedSurface] = None

        # Computed structures
        self.face_normals = None
        self.edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
        self.adjacency: Dict[int, Set[int]] = {}
        self.face_to_group: List[int] = []  # mapping face_idx -> group_id
        self.groups: Dict[int, Set[int]] = {}  # group_id -> set(face_idx)

        # whether to attempt PyVista-accelerated pre-processing
        self._use_pyvista = _HAS_PYVISTA

        # Build groups immediately
        try:
            self.build_surface_groups(angle_threshold_deg=5.0)
        except Exception as e:
            # Don't raise on construction - the main app will still function with per-face selection
            print(f"SurfaceSelector: failed to precompute groups: {e}")

    # ---------------------- Grouping / adjacency --------------------------------
    def build_surface_groups(self, angle_threshold_deg: float = 5.0):
        """Group faces into connected regions where adjacent faces are within
        angle_threshold_deg of each other's normals.

        This groups by shared edges (mesh connectivity) and coplanarity.

        Implementation notes:
        - If PyVista is available it is used to compute connected components quickly
          (by pure adjacency). We then perform a local BFS inside each component to
          split by normal-angle, which is robust for large or non-manifold meshes.
        - If PyVista is not available we fall back to a pure numpy adjacency + BFS
          algorithm (previous behavior).
        """
        import math

        face_count = len(self.mesh.faces)
        verts = np.asarray(self.mesh.vertices)

        # Compute face normals if available or compute them
        if hasattr(self.mesh, 'face_normals') and self.mesh.face_normals is not None:
            self.face_normals = np.asarray(self.mesh.face_normals)
        else:
            self.face_normals = np.zeros((face_count, 3), dtype=float)
            for i, f in enumerate(self.mesh.faces):
                v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
                n = np.cross(v1 - v0, v2 - v0)
                ln = np.linalg.norm(n)
                if ln > 1e-12:
                    self.face_normals[i] = n / ln
                else:
                    self.face_normals[i] = np.array([0.0, 0.0, 1.0])

        # Build a quick edge->faces map which we will use for adjacency in either path
        self.edge_to_faces = {}
        for fi, f in enumerate(self.mesh.faces):
            for k in range(len(f)):
                a = int(f[k]); b = int(f[(k + 1) % len(f)])
                e = (a, b) if a < b else (b, a)
                self.edge_to_faces.setdefault(e, []).append(fi)

        # Build adjacency list from edge->faces (shared-edge adjacency)
        self.adjacency = {i: set() for i in range(face_count)}
        for e, faces in self.edge_to_faces.items():
            if len(faces) < 2:
                continue
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    a = faces[i]; b = faces[j]
                    self.adjacency[a].add(b); self.adjacency[b].add(a)

        # Precompute cosine threshold for the allowed angle between normals
        angle_thresh_rad = math.radians(max(0.1, angle_threshold_deg))
        cos_thresh = math.cos(angle_thresh_rad)

        # Prepare grouping containers
        self.face_to_group = [-1] * face_count
        self.groups = {}
        group_id = 0

        if self._use_pyvista:
            # Use PyVista connectivity to partition mesh into adjacency-connected components
            # This is generally much faster on large meshes and helps avoid crossing
            # non-manifold bridges during the per-face BFS.
            try:
                # Build pyvista PolyData - pyvista expects faces in the format: [n, v0, v1, v2, n, v0, v1, v2, ...]
                faces_flat = np.hstack([np.array([len(f), *f], dtype=np.int64) for f in self.mesh.faces])
                pv_mesh = pv.PolyData(verts, faces_flat)
                # compute cell normals so connectivity won't change, but store them in case pyvista created them
                pv_mesh = pv_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
                conn = pv_mesh.connectivity()
                # RegionId array labels each cell with an integer region id
                if 'RegionId' in conn.cell_arrays:
                    labels = np.asarray(conn.cell_arrays['RegionId'], dtype=int)
                else:
                    # fallback - treat whole mesh as one region
                    labels = np.zeros(face_count, dtype=int)

                unique_regions = np.unique(labels)
                # Iterate each pyvista region and run local BFS using our normal-based threshold
                for reg in unique_regions:
                    region_faces = np.where(labels == reg)[0].tolist()
                    for start in region_faces:
                        if self.face_to_group[start] != -1:
                            continue
                        # BFS within this region splitting by normal angle
                        stack = [start]
                        self.face_to_group[start] = group_id
                        self.groups[group_id] = {start}
                        while stack:
                            cur = stack.pop()
                            ncur = self.face_normals[cur]
                            # Only consider neighbors that are in the same pyvista region
                            for nb in self.adjacency.get(cur, []):
                                if labels[nb] != reg:
                                    continue
                                if self.face_to_group[nb] != -1:
                                    continue
                                nd = self.face_normals[nb]
                                if np.dot(ncur, nd) >= cos_thresh:
                                    self.face_to_group[nb] = group_id
                                    self.groups[group_id].add(nb)
                                    stack.append(nb)
                        group_id += 1
                return self.groups
            except Exception as e:
                # If anything goes wrong with pyvista path, gracefully fall back to numpy approach
                print(f"SurfaceSelector: PyVista path failed ({e}), falling back to numpy algorithm")
                self._use_pyvista = False

        # Fallback: perform BFS over faces to form groups based on coplanarity using adjacency map
        for start in range(face_count):
            if self.face_to_group[start] != -1:
                continue
            stack = [start]
            self.face_to_group[start] = group_id
            self.groups[group_id] = {start}
            while stack:
                cur = stack.pop()
                ncur = self.face_normals[cur]
                for nb in self.adjacency.get(cur, []):
                    if self.face_to_group[nb] != -1:
                        continue
                    nd = self.face_normals[nb]
                    # Compare normals - use dot product for angle
                    if np.dot(ncur, nd) >= cos_thresh:
                        self.face_to_group[nb] = group_id
                        self.groups[group_id].add(nb)
                        stack.append(nb)
            group_id += 1

        return self.groups

    # ---------------------- Raycast helper -------------------------------------
    def raycast(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[int]:
        """Raycast against the mesh and return the hit face index (closest), or None.

        Uses trimesh.ray if available, otherwise performs a naive loop.
        """
        try:
            # Prefer trimesh.ray which is optimized
            locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                ray_origins=[ray_origin],
                ray_directions=[ray_direction]
            )
            if len(index_tri) > 0:
                # pick closest
                dists = np.linalg.norm(locations - ray_origin, axis=1)
                k = np.argmin(dists)
                return int(index_tri[k])
        except Exception:
            # fallback to brute-force Moller-Trumbore (safe, slower)
            try:
                closest_face = None
                min_t = float('inf')
                verts = np.asarray(self.mesh.vertices)
                for face_idx, face in enumerate(self.mesh.faces):
                    v0 = verts[face[0]]
                    v1 = verts[face[1]]
                    v2 = verts[face[2]]
                    # Moller-Trumbore
                    eps = 1e-9
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    h = np.cross(ray_direction, edge2)
                    a = np.dot(edge1, h)
                    if -eps < a < eps:
                        continue
                    f = 1.0 / a
                    s = ray_origin - v0
                    u = f * np.dot(s, h)
                    if u < 0.0 or u > 1.0:
                        continue
                    q = np.cross(s, edge1)
                    v = f * np.dot(ray_direction, q)
                    if v < 0.0 or u + v > 1.0:
                        continue
                    t = f * np.dot(edge2, q)
                    if t > 1e-6 and t < min_t:
                        min_t = t; closest_face = face_idx
                if closest_face is not None:
                    return int(closest_face)
            except Exception:
                pass

        return None

    # ---------------------- Group queries -------------------------------------
    def get_group_id_for_face(self, face_index: int) -> Optional[int]:
        if face_index < 0 or face_index >= len(self.face_to_group):
            return None
        return int(self.face_to_group[face_index])

    def get_group_faces(self, face_index: int) -> List[int]:
        gid = self.get_group_id_for_face(face_index)
        if gid is None:
            return [face_index]
        return sorted(list(self.groups.get(gid, {face_index})))

    def get_group_count(self) -> int:
        return len(self.groups)

    # ---------------------- Selection helpers ---------------------------------
    def select_face(self, face_index: int) -> SelectedSurface:
        """Select a single face (returns SelectedSurface for compatibility)."""
        if face_index < 0 or face_index >= len(self.mesh.faces):
            raise IndexError(f"Face index {face_index} out of bounds (mesh has {len(self.mesh.faces)} faces)")

        face = self.mesh.faces[face_index]
        vertices = np.asarray([self.mesh.vertices[idx] for idx in face])

        if hasattr(self.mesh, 'face_normals') and self.mesh.face_normals is not None:
            normal = np.asarray(self.mesh.face_normals[face_index])
        else:
            v0, v1, v2 = vertices[0], vertices[1], vertices[2]
            normal = np.cross(v1 - v0, v2 - v0)
            ln = np.linalg.norm(normal)
            if ln > 1e-12:
                normal = normal / ln
            else:
                normal = np.array([0.0, 0.0, 1.0])

        self.selected_surface = SelectedSurface(face_index, vertices, normal)
        return self.selected_surface

    def select_coplanar_region(self, face_index: int, angle_threshold: float = 5.0) -> List[int]:
        """Return list of faces in the same precomputed group as face_index.

        angle_threshold is ignored here because grouping was precomputed; keep it
        for backward compatibility.
        """
        return self.get_group_faces(face_index)

    def validate_surface(self, faces: List[int]) -> ValidationResult:
        """Validate the group of faces: check planarity and reasonable area.

        Returns ValidationResult with helpful message if problematic.
        """
        if not faces:
            return ValidationResult(False, False, "No faces in selection")

        verts = np.asarray(self.mesh.vertices)
        # gather all vertices used by the faces
        idxs = np.unique(self.mesh.faces[faces].flatten())
        pts = verts[idxs]

        # Fit plane by PCA
        centroid = pts.mean(axis=0)
        cov = np.cov((pts - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # smallest eigenvalue corresponds to normal
        normal = eigvecs[:, np.argmin(eigvals)]

        # Measure deviation from plane
        dists = np.abs(np.dot(pts - centroid, normal))
        max_dev = float(np.max(dists))
        rms = float(np.sqrt(np.mean(dists * dists)))

        # heuristics for planarity (tolerance in meters)
        planar_tol = max(1e-4, np.linalg.norm(pts.ptp(axis=0)) * 1e-3)
        is_planar = (max_dev <= planar_tol)

        area = 0.0
        try:
            # approximate area: sum triangle areas
            area = float(np.sum(self.mesh.area_faces[faces])) if hasattr(self.mesh, 'area_faces') else 0.0
        except Exception:
            area = 0.0

        msg = f"Faces: {len(faces)}, Area: {area:.4f} m^2, Max deviation from best-fit plane: {max_dev:.6f} m"
        if not is_planar:
            msg = "Surface not perfectly planar. " + msg

        return ValidationResult(True, is_planar, msg)

    def get_surface_bounds_projected(self, faces: List[int]) -> Optional[Tuple[float, float]]:
        """Return (width, length) projected into the surface plane for the given faces.

        Uses PCA to build a local frame and projects vertices into that frame to compute extents.
        """
        if not faces:
            return None

        verts = np.asarray(self.mesh.vertices)
        face_array = np.asarray(self.mesh.faces)
        idxs = np.unique(face_array[faces].flatten())
        pts = verts[idxs]

        centroid = pts.mean(axis=0)
        cov = np.cov((pts - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # sort eigenvectors by eigenvalue descending
        order = np.argsort(eigvals)[::-1]
        u = eigvecs[:, order[0]]
        v = eigvecs[:, order[1]]

        # project
        coords = np.dot(pts - centroid, np.vstack([u, v]).T)
        min_uv = coords.min(axis=0)
        max_uv = coords.max(axis=0)
        width = float(max_uv[0] - min_uv[0])
        length = float(max_uv[1] - min_uv[1])
        return (width, length)

    def get_surface_info(self, faces: List[int]) -> dict:
        """Return dictionary with surface metadata for provided faces or empty dict."""
        if not faces:
            return {}

        verts = np.asarray(self.mesh.vertices)
        face_array = np.asarray(self.mesh.faces)
        idxs = np.unique(face_array[faces].flatten())
        pts = verts[idxs]

        centroid = pts.mean(axis=0)
        # average normal
        normals = self.face_normals[faces]
        avg_normal = np.mean(normals, axis=0)
        nlen = np.linalg.norm(avg_normal)
        if nlen > 1e-12:
            avg_normal = avg_normal / nlen

        bounds_min = pts.min(axis=0)
        bounds_max = pts.max(axis=0)

        projected = self.get_surface_bounds_projected(faces)
        width, length = (None, None)
        if projected is not None:
            width, length = projected

        area = 0.0
        try:
            area = float(np.sum(self.mesh.area_faces[faces])) if hasattr(self.mesh, 'area_faces') else 0.0
        except Exception:
            area = 0.0

        return {
            'faces': list(faces),
            'center': centroid.tolist(),
            'normal': avg_normal.tolist(),
            'bounds': [bounds_min.tolist(), bounds_max.tolist()],
            'projected_width_length': (width, length),
            'area': area
        }


# End of surface_selector.py
