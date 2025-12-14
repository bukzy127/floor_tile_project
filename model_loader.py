"""
Advanced 3D Model Loader with Multiple Library Support

Supports: .obj, .dxf, .dwg, .skp, .stl, .ply, .off, .3ds, .gltf, .glb
Libraries: Open3D, PyVista, numpy-stl, trimesh, ezdxf

Author: 3D Tiles App
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import warnings

# ============================================================================
# CUSTOM EXCEPTION
# ============================================================================

class ModelLoadError(Exception):
    """Custom exception for model loading errors with helpful messages."""
    pass


# ============================================================================
# LIBRARY AVAILABILITY CHECKS
# ============================================================================

def check_library_availability() -> Dict[str, bool]:
    """Check which 3D libraries are available."""
    libraries = {}

    try:
        import open3d
        libraries['open3d'] = True
    except ImportError:
        libraries['open3d'] = False

    try:
        import pyvista
        libraries['pyvista'] = True
    except ImportError:
        libraries['pyvista'] = False

    try:
        import trimesh
        libraries['trimesh'] = True
    except ImportError:
        libraries['trimesh'] = False

    try:
        import stl
        libraries['numpy-stl'] = True
    except ImportError:
        libraries['numpy-stl'] = False

    try:
        import ezdxf
        libraries['ezdxf'] = True
    except ImportError:
        libraries['ezdxf'] = False

    return libraries


AVAILABLE_LIBS = check_library_availability()


# ============================================================================
# UNIFIED MESH CLASS (Library-agnostic)
# ============================================================================

class UnifiedMesh:
    """
    A unified mesh representation that works with any 3D library.
    Provides consistent interface for vertices, faces, and normals.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray,
                 normals: Optional[np.ndarray] = None,
                 face_normals: Optional[np.ndarray] = None):
        """
        Initialize UnifiedMesh.

        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices (triangles)
            normals: Optional Nx3 array of vertex normals
            face_normals: Optional Mx3 array of face normals
        """
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int32)
        self._normals = normals
        self._face_normals = face_normals
        self._bounds = None
        self._centroid = None


    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_coords, max_coords) bounding box."""
        if self._bounds is None:
            self._bounds = (self.vertices.min(axis=0), self.vertices.max(axis=0))
        return self._bounds

    @property
    def centroid(self) -> np.ndarray:
        """Return center of bounding box."""
        if self._centroid is None:
            min_c, max_c = self.bounds
            self._centroid = (min_c + max_c) / 2.0
        return self._centroid

    @property
    def extents(self) -> np.ndarray:
        """Return size in each dimension."""
        min_c, max_c = self.bounds
        return max_c - min_c

    @property
    def face_normals(self) -> np.ndarray:
        """Compute or return face normals."""
        if self._face_normals is None:
            self._face_normals = self._compute_face_normals()
        return self._face_normals

    def _compute_face_normals(self) -> np.ndarray:
        """Compute face normals from vertices and faces."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return normals / norms

    def to_trimesh(self):
        """Convert to trimesh.Trimesh for compatibility with existing code."""
        try:
            import trimesh
            return trimesh.Trimesh(
                vertices=self.vertices,
                faces=self.faces,
                face_normals=self._face_normals,
                process=False
            )
        except ImportError:
            raise ModelLoadError(
                "trimesh is required for compatibility mode.\n"
                "Install with: pip install trimesh"
            )


# ============================================================================
# OBJ LOADER - Best libraries: Open3D, PyVista, trimesh
# ============================================================================

def load_obj_open3d(filepath: str) -> UnifiedMesh:
    """
    Load OBJ file using Open3D.

    PROS:
    - Excellent for point clouds and meshes
    - Fast C++ backend
    - Good memory efficiency
    - Supports colors, normals, textures

    CONS:
    - Large dependency (~100MB)
    - May have issues with complex materials
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ModelLoadError(
            "Open3D not installed.\n"
            "Install with: pip install open3d"
        )

    try:
        mesh = o3d.io.read_triangle_mesh(filepath)

        if not mesh.has_vertices():
            raise ModelLoadError(f"OBJ file has no vertices: {filepath}")

        vertices = np.asarray(mesh.vertices)

        if not mesh.has_triangles():
            # Try to convert to triangles if possible
            raise ModelLoadError(
                f"OBJ file has no triangulated faces.\n"
                f"Vertices found: {len(vertices)}\n"
                "Try opening in Blender and export with 'Triangulate Faces' enabled."
            )

        faces = np.asarray(mesh.triangles)

        # Get normals if available
        normals = None
        if mesh.has_vertex_normals():
            normals = np.asarray(mesh.vertex_normals)

        return UnifiedMesh(vertices, faces, normals=normals)

    except Exception as e:
        if "ModelLoadError" in str(type(e)):
            raise
        raise ModelLoadError(f"Open3D failed to load OBJ: {str(e)}")


def load_obj_pyvista(filepath: str) -> UnifiedMesh:
    """
    Load OBJ file using PyVista.

    PROS:
    - Excellent visualization capabilities
    - Based on VTK (robust, well-tested)
    - Handles large meshes well
    - Good for scientific visualization

    CONS:
    - Requires VTK (~200MB dependency)
    - Slower startup time
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ModelLoadError(
            "PyVista not installed.\n"
            "Install with: pip install pyvista"
        )

    try:
        mesh = pv.read(filepath)

        vertices = np.array(mesh.points)

        if mesh.n_cells == 0:
            raise ModelLoadError(
                f"OBJ file has no faces.\n"
                f"Vertices found: {len(vertices)}"
            )

        # Extract triangular faces
        faces = []
        if hasattr(mesh, 'faces'):
            # PyVista stores faces as [n, v0, v1, v2, n, v0, v1, v2, ...]
            face_array = mesh.faces
            i = 0
            while i < len(face_array):
                n_verts = face_array[i]
                if n_verts == 3:
                    faces.append(face_array[i+1:i+4])
                elif n_verts == 4:
                    # Triangulate quad
                    v = face_array[i+1:i+5]
                    faces.append([v[0], v[1], v[2]])
                    faces.append([v[0], v[2], v[3]])
                elif n_verts > 4:
                    # Fan triangulation for polygons
                    v = face_array[i+1:i+1+n_verts]
                    for j in range(1, n_verts - 1):
                        faces.append([v[0], v[j], v[j+1]])
                i += n_verts + 1

        if not faces:
            # Try triangulate method
            mesh = mesh.triangulate()
            face_array = mesh.faces
            i = 0
            while i < len(face_array):
                n_verts = face_array[i]
                faces.append(face_array[i+1:i+1+n_verts].tolist())
                i += n_verts + 1

        return UnifiedMesh(vertices, np.array(faces))

    except Exception as e:
        if "ModelLoadError" in str(type(e)):
            raise
        raise ModelLoadError(f"PyVista failed to load OBJ: {str(e)}")


def load_obj_trimesh(filepath: str) -> UnifiedMesh:
    """
    Load OBJ file using trimesh.

    PROS:
    - Lightweight and fast
    - Good OBJ support with materials
    - Many utility functions (boolean ops, etc.)
    - Active development

    CONS:
    - May struggle with very complex meshes
    - Some edge cases with malformed files
    """
    try:
        import trimesh
    except ImportError:
        raise ModelLoadError(
            "trimesh not installed.\n"
            "Install with: pip install trimesh"
        )

    try:
        mesh = trimesh.load(filepath, force='mesh')

        if isinstance(mesh, trimesh.Scene):
            # Combine all meshes in scene
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ModelLoadError("OBJ file contains no mesh geometry.")
            mesh = trimesh.util.concatenate(meshes)

        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            raise ModelLoadError("OBJ file has no vertices.")

        if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            raise ModelLoadError("OBJ file has no faces.")

        return UnifiedMesh(
            mesh.vertices,
            mesh.faces,
            face_normals=mesh.face_normals if hasattr(mesh, 'face_normals') else None
        )

    except Exception as e:
        if "ModelLoadError" in str(type(e)):
            raise
        raise ModelLoadError(f"trimesh failed to load OBJ: {str(e)}")


def load_obj(filepath: str, preferred_lib: str = 'auto') -> UnifiedMesh:
    """
    Load OBJ file using the best available library.

    Args:
        filepath: Path to OBJ file
        preferred_lib: 'auto', 'open3d', 'pyvista', or 'trimesh'

    Returns:
        UnifiedMesh object
    """
    loaders = {
        'open3d': (load_obj_open3d, AVAILABLE_LIBS.get('open3d', False)),
        'pyvista': (load_obj_pyvista, AVAILABLE_LIBS.get('pyvista', False)),
        'trimesh': (load_obj_trimesh, AVAILABLE_LIBS.get('trimesh', False)),
    }

    if preferred_lib != 'auto' and preferred_lib in loaders:
        loader, available = loaders[preferred_lib]
        if available:
            return loader(filepath)
        else:
            raise ModelLoadError(f"{preferred_lib} is not installed.")

    # Try loaders in order of preference
    errors = []
    for lib_name, (loader, available) in loaders.items():
        if available:
            try:
                return loader(filepath)
            except ModelLoadError as e:
                errors.append(f"{lib_name}: {str(e)}")

    if errors:
        raise ModelLoadError(
            f"All available loaders failed for OBJ file:\n" +
            "\n".join(errors)
        )

    raise ModelLoadError(
        "No OBJ loaders available.\n"
        "Install one of: pip install open3d pyvista trimesh"
    )


# ============================================================================
# STL LOADER - Best libraries: numpy-stl, Open3D, trimesh
# ============================================================================

def load_stl_numpy(filepath: str) -> UnifiedMesh:
    """
    Load STL file using numpy-stl.

    PROS:
    - Extremely fast for STL files
    - Minimal dependencies (just numpy)
    - Handles binary and ASCII STL
    - Low memory footprint

    CONS:
    - STL-only (no other formats)
    - No color/texture support
    """
    try:
        from stl import mesh as stl_mesh
    except ImportError:
        raise ModelLoadError(
            "numpy-stl not installed.\n"
            "Install with: pip install numpy-stl"
        )

    try:
        stl_data = stl_mesh.Mesh.from_file(filepath)

        # STL stores triangles directly - each face has 3 vertices
        # Shape is (n_triangles, 3, 3) for vectors
        n_triangles = len(stl_data.vectors)

        if n_triangles == 0:
            raise ModelLoadError("STL file has no triangles.")

        # Flatten vertices and create faces
        vertices = stl_data.vectors.reshape(-1, 3)
        faces = np.arange(n_triangles * 3).reshape(-1, 3)

        # Get normals
        normals = stl_data.normals

        # Optionally deduplicate vertices for smaller memory
        # (STL format duplicates vertices for each face)
        vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)

        return UnifiedMesh(vertices, faces, face_normals=normals)

    except Exception as e:
        if "ModelLoadError" in str(type(e)):
            raise
        raise ModelLoadError(f"numpy-stl failed to load STL: {str(e)}")


def load_stl_open3d(filepath: str) -> UnifiedMesh:
    """Load STL using Open3D."""
    try:
        import open3d as o3d
    except ImportError:
        raise ModelLoadError("Open3D not installed.")

    try:
        mesh = o3d.io.read_triangle_mesh(filepath)
        if not mesh.has_vertices():
            raise ModelLoadError("STL file has no vertices.")

        return UnifiedMesh(
            np.asarray(mesh.vertices),
            np.asarray(mesh.triangles)
        )
    except Exception as e:
        raise ModelLoadError(f"Open3D failed to load STL: {str(e)}")


def load_stl(filepath: str, preferred_lib: str = 'auto') -> UnifiedMesh:
    """Load STL file using best available library."""
    loaders = [
        ('numpy-stl', load_stl_numpy, AVAILABLE_LIBS.get('numpy-stl', False)),
        ('open3d', load_stl_open3d, AVAILABLE_LIBS.get('open3d', False)),
        ('trimesh', lambda f: load_obj_trimesh(f), AVAILABLE_LIBS.get('trimesh', False)),
    ]

    errors = []
    for lib_name, loader, available in loaders:
        if available:
            try:
                return loader(filepath)
            except ModelLoadError as e:
                errors.append(f"{lib_name}: {str(e)}")

    raise ModelLoadError(
        "No STL loaders available.\n"
        "Install: pip install numpy-stl"
    )


# ============================================================================
# DXF/DWG LOADER - Best library: ezdxf
# ============================================================================

def load_dxf_ezdxf(filepath: str) -> UnifiedMesh:
    """
    Load DXF file using ezdxf.

    PROS:
    - Best DXF/DWG support in Python
    - Handles 2D and 3D entities
    - Active development
    - Pure Python (no binary dependencies)

    CONS:
    - DWG support is read-only
    - Some complex DWG features may not be supported
    - 2D drawings need extrusion for 3D
    """
    try:
        import ezdxf
    except ImportError:
        raise ModelLoadError(
            "ezdxf not installed.\n"
            "Install with: pip install ezdxf"
        )

    try:
        doc = ezdxf.readfile(filepath)
    except Exception as e:
        raise ModelLoadError(f"Cannot read DXF/DWG file: {str(e)}")

    msp = doc.modelspace()

    vertices = []
    faces = []
    vertex_map = {}

    def get_vertex_index(v):
        """Add vertex to list if not exists, return index."""
        key = (round(v[0], 6), round(v[1], 6), round(v[2] if len(v) > 2 else 0, 6))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append([key[0], key[1], key[2]])
        return vertex_map[key]

    entity_count = {'3DFACE': 0, 'MESH': 0, 'POLYFACE': 0, '3DSOLID': 0, 'LINE': 0, 'POLYLINE': 0}

    # ---- 3DFACE entities (most common for 3D CAD exports) ----
    for entity in msp.query('3DFACE'):
        entity_count['3DFACE'] += 1
        try:
            pts = [
                (entity.dxf.vtx0.x, entity.dxf.vtx0.y, entity.dxf.vtx0.z),
                (entity.dxf.vtx1.x, entity.dxf.vtx1.y, entity.dxf.vtx1.z),
                (entity.dxf.vtx2.x, entity.dxf.vtx2.y, entity.dxf.vtx2.z),
            ]

            # Check for quad (4th vertex)
            vtx3 = (entity.dxf.vtx3.x, entity.dxf.vtx3.y, entity.dxf.vtx3.z)
            is_quad = vtx3 != pts[2]

            # First triangle
            indices = [get_vertex_index(p) for p in pts]
            faces.append(indices)

            # Second triangle if quad
            if is_quad:
                indices2 = [
                    get_vertex_index(pts[0]),
                    get_vertex_index(pts[2]),
                    get_vertex_index(vtx3)
                ]
                faces.append(indices2)

        except Exception:
            continue

    # ---- MESH entities ----
    for entity in msp.query('MESH'):
        entity_count['MESH'] += 1
        try:
            if hasattr(entity, 'vertices') and hasattr(entity, 'faces'):
                vert_start = len(vertices)
                for v in entity.vertices:
                    vertices.append([v.x, v.y, v.z])
                for face in entity.faces:
                    if len(face) >= 3:
                        # Triangulate if needed
                        for i in range(1, len(face) - 1):
                            faces.append([
                                vert_start + face[0],
                                vert_start + face[i],
                                vert_start + face[i + 1]
                            ])
        except Exception:
            continue

    # ---- POLYFACE meshes ----
    for entity in msp.query('POLYLINE'):
        if hasattr(entity, 'is_poly_face_mesh') and entity.is_poly_face_mesh:
            entity_count['POLYFACE'] += 1
            try:
                mesh_vertices = list(entity.points())
                vert_start = len(vertices)
                for v in mesh_vertices:
                    z = v[2] if len(v) > 2 else 0
                    vertices.append([v[0], v[1], z])

                if hasattr(entity, 'faces'):
                    for face in entity.faces():
                        # Face indices in polyface are 1-based and may be negative
                        face_indices = [abs(i) - 1 for i in face if i != 0]
                        if len(face_indices) >= 3:
                            for i in range(1, len(face_indices) - 1):
                                faces.append([
                                    vert_start + face_indices[0],
                                    vert_start + face_indices[i],
                                    vert_start + face_indices[i + 1]
                                ])
            except Exception:
                continue
        else:
            entity_count['POLYLINE'] += 1

    # ---- 3DSOLID entities (ACIS solids - limited support) ----
    for entity in msp.query('3DSOLID'):
        entity_count['3DSOLID'] += 1
        # Note: ACIS solids require special handling, often need conversion

    if not faces:
        # Build helpful error message
        error_parts = ["No 3D faces found in DXF/DWG file.\n"]

        entities_found = [(k, v) for k, v in entity_count.items() if v > 0]
        if entities_found:
            error_parts.append("Entities found:")
            for ent_type, count in entities_found:
                error_parts.append(f"  • {ent_type}: {count}")

        error_parts.append("\n💡 Conversion Tips:")
        error_parts.append("• In AutoCAD: Use FACETRES command to increase mesh density")
        error_parts.append("• Export as 3DFACE entities or use MESH command")
        error_parts.append("• For 2D drawings: Use EXTRUDE to create 3D geometry")
        error_parts.append("• Try exporting as OBJ format for better compatibility")
        error_parts.append("• In AutoCAD: EXPORT → Select OBJ format")

        raise ModelLoadError("\n".join(error_parts))

    print(f"Loaded DXF/DWG: {len(vertices)} vertices, {len(faces)} faces")
    return UnifiedMesh(np.array(vertices), np.array(faces))


def load_dwg(filepath: str) -> UnifiedMesh:
    """Load DWG file (uses ezdxf which has limited DWG support)."""
    return load_dxf_ezdxf(filepath)


def load_dxf(filepath: str) -> UnifiedMesh:
    """Load DXF file."""
    return load_dxf_ezdxf(filepath)


# ============================================================================
# SKP (SketchUp) LOADER - Best approach: trimesh with assimp, or convert
# ============================================================================

def load_skp_trimesh(filepath: str) -> UnifiedMesh:
    """
    Load SketchUp file using trimesh (requires assimp backend).

    PROS:
    - Works if pyassimp/assimp is installed
    - Handles many SketchUp features

    CONS:
    - Requires assimp library (complex to install)
    - May not support newest SKP versions
    - Component instances may not import correctly
    """
    try:
        import trimesh
    except ImportError:
        raise ModelLoadError("trimesh not installed.")

    try:
        # trimesh can load SKP if assimp is available
        mesh = trimesh.load(filepath, force='mesh')

        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ModelLoadError("SKP file contains no mesh geometry.")
            mesh = trimesh.util.concatenate(meshes)

        return UnifiedMesh(mesh.vertices, mesh.faces)

    except Exception as e:
        error_msg = str(e)
        if 'assimp' in error_msg.lower() or 'pyassimp' in error_msg.lower():
            raise ModelLoadError(
                "SketchUp files require the Assimp library.\n\n"
                "Installation options:\n"
                "1. pip install pyassimp (requires assimp system library)\n"
                "2. conda install -c conda-forge assimp\n\n"
                "Alternative: Export from SketchUp as:\n"
                "• OBJ format (File → Export → 3D Model → OBJ)\n"
                "• DAE/Collada format\n"
                "• STL format"
            )
        raise ModelLoadError(f"Failed to load SKP: {str(e)}")


def load_skp_pyvista(filepath: str) -> UnifiedMesh:
    """Attempt to load SKP with PyVista (limited support)."""
    try:
        import pyvista as pv
    except ImportError:
        raise ModelLoadError("PyVista not installed.")

    try:
        mesh = pv.read(filepath)
        vertices = np.array(mesh.points)

        # Extract faces
        faces = []
        face_array = mesh.faces
        i = 0
        while i < len(face_array):
            n = face_array[i]
            if n >= 3:
                v = face_array[i+1:i+1+n]
                for j in range(1, n - 1):
                    faces.append([v[0], v[j], v[j+1]])
            i += n + 1

        return UnifiedMesh(vertices, np.array(faces))

    except Exception as e:
        raise ModelLoadError(
            f"PyVista cannot load SKP files directly.\n"
            f"Please export from SketchUp as OBJ or STL format."
        )


def load_skp(filepath: str) -> UnifiedMesh:
    """Load SketchUp file."""
    errors = []

    if AVAILABLE_LIBS.get('trimesh', False):
        try:
            return load_skp_trimesh(filepath)
        except ModelLoadError as e:
            errors.append(str(e))

    # Provide helpful conversion guidance
    raise ModelLoadError(
        "SketchUp (.skp) files have limited Python support.\n\n"
        "🔧 Best Solutions:\n\n"
        "1. Export from SketchUp:\n"
        "   • File → Export → 3D Model → OBJ (recommended)\n"
        "   • File → Export → 3D Model → DAE (Collada)\n"
        "   • File → Export → 3D Model → STL\n\n"
        "2. Install Assimp (advanced):\n"
        "   • macOS: brew install assimp && pip install pyassimp\n"
        "   • Linux: apt install libassimp-dev && pip install pyassimp\n"
        "   • Windows: Use conda install -c conda-forge assimp\n\n"
        "3. Use online converter:\n"
        "   • https://www.convertio.co/skp-obj/\n"
        "   • https://cloudconvert.com/skp-to-obj"
    )


# ============================================================================
# PLY LOADER - Best libraries: Open3D, PyVista, trimesh
# ============================================================================

def load_ply(filepath: str) -> UnifiedMesh:
    """Load PLY file using best available library."""
    if AVAILABLE_LIBS.get('open3d', False):
        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(filepath)
            if mesh.has_vertices() and mesh.has_triangles():
                return UnifiedMesh(
                    np.asarray(mesh.vertices),
                    np.asarray(mesh.triangles)
                )
        except Exception:
            pass

    if AVAILABLE_LIBS.get('pyvista', False):
        try:
            return load_obj_pyvista(filepath)  # PyVista handles PLY too
        except Exception:
            pass

    if AVAILABLE_LIBS.get('trimesh', False):
        try:
            return load_obj_trimesh(filepath)  # trimesh handles PLY too
        except Exception:
            pass

    raise ModelLoadError(
        "No PLY loaders available.\n"
        "Install: pip install open3d pyvista trimesh"
    )


# ============================================================================
# UNIFIED LOADER
# ============================================================================

def load_model(filepath: str, preferred_lib: str = 'auto') -> 'UnifiedMesh':
    """
    Load any supported 3D model file.

    Supported formats:
    - OBJ: Open3D, PyVista, trimesh
    - STL: numpy-stl, Open3D, trimesh
    - DXF/DWG: ezdxf
    - SKP: trimesh+assimp (or convert to OBJ)
    - PLY: Open3D, PyVista, trimesh

    Args:
        filepath: Path to 3D model file
        preferred_lib: 'auto' or specific library name

    Returns:
        UnifiedMesh object (can be converted to trimesh via .to_trimesh())
    """
    path = Path(filepath)

    if not path.exists():
        raise ModelLoadError(f"File not found: {filepath}")

    ext = path.suffix.lower()

    loaders = {
        '.obj': load_obj,
        '.stl': load_stl,
        '.dxf': load_dxf,
        '.dwg': load_dwg,
        '.skp': load_skp,
        '.ply': load_ply,
    }

    if ext not in loaders:
        supported = ', '.join(loaders.keys())
        raise ModelLoadError(
            f"Unsupported format: {ext}\n"
            f"Supported formats: {supported}"
        )

    return loaders[ext](filepath)


def normalize_mesh_units(mesh: UnifiedMesh, target_unit: str = 'meters',
                         source_unit: str = 'auto') -> UnifiedMesh:
    """
    Normalize mesh units to target unit.

    Auto-detection heuristics:
    - If max dimension > 100: assume millimeters
    - If max dimension > 10: assume centimeters
    - Otherwise: assume meters

    Args:
        mesh: UnifiedMesh object
        target_unit: 'meters', 'centimeters', 'millimeters'
        source_unit: 'auto', 'meters', 'centimeters', 'millimeters'

    Returns:
        New UnifiedMesh with scaled vertices
    """
    scale_to_meters = {
        'meters': 1.0,
        'centimeters': 0.01,
        'millimeters': 0.001,
        'inches': 0.0254,
        'feet': 0.3048,
    }

    max_dim = mesh.extents.max()

    if source_unit == 'auto':
        if max_dim > 1000:
            source_unit = 'millimeters'
            print(f"Auto-detected units: millimeters (max dimension: {max_dim:.1f})")
        elif max_dim > 100:
            source_unit = 'centimeters'
            print(f"Auto-detected units: centimeters (max dimension: {max_dim:.1f})")
        else:
            source_unit = 'meters'
            print(f"Assuming units: meters (max dimension: {max_dim:.2f})")

    if source_unit not in scale_to_meters:
        raise ModelLoadError(f"Unknown source unit: {source_unit}")
    if target_unit not in scale_to_meters:
        raise ModelLoadError(f"Unknown target unit: {target_unit}")

    scale = scale_to_meters[source_unit] / scale_to_meters[target_unit]

    if abs(scale - 1.0) < 1e-6:
        return mesh  # No scaling needed

    scaled_vertices = mesh.vertices * scale
    print(f"Scaled from {source_unit} to {target_unit} (factor: {scale})")

    return UnifiedMesh(
        scaled_vertices,
        mesh.faces,
        normals=mesh._normals,
        face_normals=mesh._face_normals
    )


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def plot_mesh_matplotlib(mesh: UnifiedMesh, mode: str = 'wireframe',
                         title: str = '3D Mesh', figsize: tuple = (10, 8)):
    """
    Plot mesh using Matplotlib's mplot3d.

    Args:
        mesh: UnifiedMesh object
        mode: 'wireframe', 'surface', or 'both'
        title: Plot title
        figsize: Figure size

    PROS:
    - No extra dependencies (matplotlib is common)
    - Good for quick visualization
    - Easy to customize

    CONS:
    - Slow for large meshes (>10k faces)
    - Limited interactivity
    - No proper depth sorting
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    except ImportError:
        raise ModelLoadError("matplotlib not installed.")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    vertices = mesh.vertices
    faces = mesh.faces

    # Limit faces for performance
    max_faces = 5000
    if len(faces) > max_faces:
        print(f"Warning: Showing only {max_faces} of {len(faces)} faces for performance")
        indices = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[indices]

    # Create face vertices
    face_verts = vertices[faces]

    if mode in ('surface', 'both'):
        # Surface plot
        collection = Poly3DCollection(
            face_verts,
            alpha=0.7,
            facecolor='cyan',
            edgecolor='darkblue' if mode == 'both' else 'none',
            linewidth=0.3
        )
        ax.add_collection3d(collection)

    if mode == 'wireframe':
        # Wireframe plot
        edges = []
        for face in faces:
            for i in range(3):
                edges.append([vertices[face[i]], vertices[face[(i+1) % 3]]])

        edge_collection = Line3DCollection(edges, colors='blue', linewidths=0.5)
        ax.add_collection3d(edge_collection)

    # Set axis limits
    min_coords, max_coords = mesh.bounds
    max_range = (max_coords - min_coords).max() / 2
    center = mesh.centroid

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_mesh_pyvista(mesh: UnifiedMesh, mode: str = 'surface'):
    """
    Plot mesh using PyVista (interactive, GPU-accelerated).

    PROS:
    - Fast, GPU-accelerated rendering
    - Interactive rotation/zoom
    - Handles large meshes well

    CONS:
    - Requires VTK/PyVista installation
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ModelLoadError("PyVista not installed. Install with: pip install pyvista")

    # Create PyVista mesh
    n_faces = len(mesh.faces)
    pv_faces = np.column_stack([
        np.full(n_faces, 3),  # Each face has 3 vertices
        mesh.faces
    ]).flatten()

    pv_mesh = pv.PolyData(mesh.vertices, pv_faces)

    plotter = pv.Plotter()

    if mode == 'wireframe':
        plotter.add_mesh(pv_mesh, style='wireframe', color='blue', line_width=1)
    elif mode == 'surface':
        plotter.add_mesh(pv_mesh, color='cyan', show_edges=True, edge_color='darkblue')
    else:  # both
        plotter.add_mesh(pv_mesh, color='cyan', opacity=0.7)
        plotter.add_mesh(pv_mesh, style='wireframe', color='darkblue', line_width=0.5)

    plotter.add_axes()
    plotter.show()


def plot_mesh_open3d(mesh: UnifiedMesh, mode: str = 'surface'):
    """
    Plot mesh using Open3D (interactive, fast).

    PROS:
    - Very fast rendering
    - Good for large point clouds/meshes
    - Clean interface

    CONS:
    - Requires Open3D installation
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ModelLoadError("Open3D not installed. Install with: pip install open3d")

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()

    if mode == 'wireframe':
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
        wireframe.paint_uniform_color([0, 0, 1])  # Blue
        o3d.visualization.draw_geometries([wireframe])
    else:
        o3d_mesh.paint_uniform_color([0, 0.8, 0.8])  # Cyan
        o3d.visualization.draw_geometries([o3d_mesh])


# ============================================================================
# COMPATIBILITY: Make UnifiedMesh work like trimesh.Trimesh
# ============================================================================

# Add trimesh-like interface to UnifiedMesh for backwards compatibility
UnifiedMesh.face_normals = property(lambda self: self._compute_face_normals()
                                    if self._face_normals is None
                                    else self._face_normals)


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("3D Model Loader - Library Availability")
    print("=" * 60)

    for lib, available in AVAILABLE_LIBS.items():
        status = "✅ Available" if available else "❌ Not installed"
        print(f"  {lib}: {status}")

    print("\n" + "=" * 60)
    print("Format Support Summary")
    print("=" * 60)

    format_info = {
        '.obj': {
            'libs': ['Open3D', 'PyVista', 'trimesh'],
            'best': 'Open3D (fast) or trimesh (lightweight)',
        },
        '.stl': {
            'libs': ['numpy-stl', 'Open3D', 'trimesh'],
            'best': 'numpy-stl (fastest, minimal deps)',
        },
        '.dxf/.dwg': {
            'libs': ['ezdxf'],
            'best': 'ezdxf (only option, works well)',
        },
        '.skp': {
            'libs': ['trimesh+assimp'],
            'best': 'Convert to OBJ in SketchUp',
        },
        '.ply': {
            'libs': ['Open3D', 'PyVista', 'trimesh'],
            'best': 'Open3D (handles colors well)',
        },
    }

    for fmt, info in format_info.items():
        print(f"\n{fmt}:")
        print(f"  Libraries: {', '.join(info['libs'])}")
        print(f"  Recommendation: {info['best']}")

    print("\n" + "=" * 60)
    print("Installation Commands")
    print("=" * 60)
    print("""
# Minimal (OBJ, STL, DXF):
pip install trimesh numpy-stl ezdxf

# Full support:
pip install open3d pyvista trimesh numpy-stl ezdxf

# For SketchUp files (complex):
# macOS: brew install assimp && pip install pyassimp
# Linux: apt install libassimp-dev && pip install pyassimp
# Or: export from SketchUp as OBJ (recommended)
""")

