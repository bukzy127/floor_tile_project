# 3D Model Format Support Guide

This document explains the best Python libraries for loading each 3D format, with pros/cons and code examples.

## Current Library Status

Run this to check your installation:
```python
from model_loader import check_library_availability
print(check_library_availability())
```

## Quick Installation

```bash
# Minimal (OBJ, STL, DXF):
pip install trimesh numpy-stl ezdxf

# Full support (recommended):
pip install open3d pyvista trimesh numpy-stl ezdxf

# For SketchUp files:
# macOS: brew install assimp && pip install pyassimp
# Linux: apt install libassimp-dev && pip install pyassimp
```

---

## Format: OBJ (Wavefront)

### Best Libraries

| Library | Pros | Cons |
|---------|------|------|
| **Open3D** ⭐ | Fast C++ backend, handles colors/normals/textures, memory efficient | Large dependency (~100MB) |
| **trimesh** ⭐ | Lightweight, good OBJ support, many utilities | May struggle with very complex meshes |
| **PyVista** | VTK-based (robust), handles large meshes, great visualization | Large dependency (~200MB) |

### Code Example: Load OBJ with trimesh

```python
import trimesh
import numpy as np

def load_obj_trimesh(filepath):
    """Load OBJ with trimesh - lightweight and reliable."""
    try:
        mesh = trimesh.load(filepath, force='mesh')
        
        # Handle scenes (multiple objects)
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() 
                      if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(meshes)
        
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        normals = mesh.face_normals
        
        print(f"Loaded: {len(vertices)} verts, {len(faces)} faces")
        return vertices, faces, normals
        
    except Exception as e:
        raise ValueError(f"Failed to load OBJ: {e}")

# Usage
verts, faces, normals = load_obj_trimesh("model.obj")
```

### Code Example: Load OBJ with Open3D

```python
import open3d as o3d
import numpy as np

def load_obj_open3d(filepath):
    """Load OBJ with Open3D - fast and feature-rich."""
    mesh = o3d.io.read_triangle_mesh(filepath)
    
    if not mesh.has_vertices():
        raise ValueError("OBJ has no vertices")
    
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # Compute normals if missing
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    
    return vertices, faces, normals

# Usage
verts, faces, normals = load_obj_open3d("model.obj")
```

### Visualization: Matplotlib (mplot3d)

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_mesh_wireframe(vertices, faces, title="3D Mesh"):
    """Plot mesh as wireframe using matplotlib."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create face vertices
    face_verts = vertices[faces]
    
    # Wireframe
    collection = Poly3DCollection(
        face_verts,
        alpha=0.3,
        facecolor='cyan',
        edgecolor='blue',
        linewidth=0.5
    )
    ax.add_collection3d(collection)
    
    # Auto-scale
    ax.auto_scale_xyz(vertices[:,0], vertices[:,1], vertices[:,2])
    ax.set_title(title)
    plt.show()

# Usage
plot_mesh_wireframe(verts, faces, "My OBJ Model")
```

### Error Handling

```python
def load_obj_safe(filepath):
    """Load OBJ with comprehensive error handling."""
    from pathlib import Path
    
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if path.suffix.lower() != '.obj':
        raise ValueError(f"Expected .obj file, got {path.suffix}")
    
    try:
        import trimesh
        mesh = trimesh.load(filepath, force='mesh')
    except ImportError:
        raise ImportError("trimesh not installed. Run: pip install trimesh")
    except Exception as e:
        # Common OBJ issues
        if "empty" in str(e).lower():
            raise ValueError("OBJ file is empty or has no geometry")
        if "parse" in str(e).lower():
            raise ValueError("OBJ file is malformed. Check file format.")
        raise
    
    if len(mesh.vertices) == 0:
        raise ValueError("OBJ loaded but contains no vertices")
    if len(mesh.faces) == 0:
        raise ValueError("OBJ loaded but contains no faces (points only?)")
    
    return mesh.vertices, mesh.faces
```

---

## Format: STL (Stereolithography)

### Best Libraries

| Library | Pros | Cons |
|---------|------|------|
| **numpy-stl** ⭐ | Extremely fast, minimal deps, handles binary/ASCII | STL-only, no colors |
| **Open3D** | Fast, part of larger toolkit | Overkill for just STL |
| **trimesh** | Works well, many utilities | Slightly slower than numpy-stl |

### Code Example: Load STL with numpy-stl

```python
from stl import mesh as stl_mesh
import numpy as np

def load_stl_numpy(filepath):
    """Load STL with numpy-stl - fastest option."""
    stl_data = stl_mesh.Mesh.from_file(filepath)
    
    # STL stores triangles directly
    n_triangles = len(stl_data.vectors)
    
    # Flatten and deduplicate vertices
    vertices = stl_data.vectors.reshape(-1, 3)
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    
    # Normals are stored per-face in STL
    normals = stl_data.normals
    
    print(f"Loaded STL: {len(vertices)} verts, {len(faces)} faces")
    return vertices, faces, normals

# Usage
verts, faces, normals = load_stl_numpy("model.stl")
```

### Conversion Tips

```python
# Convert STL to OBJ using trimesh
import trimesh

mesh = trimesh.load("model.stl")
mesh.export("model.obj")
print("Converted STL to OBJ")
```

---

## Format: DXF/DWG (AutoCAD)

### Best Library

| Library | Pros | Cons |
|---------|------|------|
| **ezdxf** ⭐ | Best DXF support, pure Python, active development | DWG read-only, limited ACIS solid support |

### Code Example: Load DXF with ezdxf

```python
import ezdxf
import numpy as np

def load_dxf(filepath):
    """Load DXF/DWG 3D faces using ezdxf."""
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()
    
    vertices = []
    faces = []
    vertex_map = {}
    
    def add_vertex(v):
        key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append(list(key))
        return vertex_map[key]
    
    # Extract 3DFACE entities
    for entity in msp.query('3DFACE'):
        try:
            pts = [
                (entity.dxf.vtx0.x, entity.dxf.vtx0.y, entity.dxf.vtx0.z),
                (entity.dxf.vtx1.x, entity.dxf.vtx1.y, entity.dxf.vtx1.z),
                (entity.dxf.vtx2.x, entity.dxf.vtx2.y, entity.dxf.vtx2.z),
            ]
            vtx3 = (entity.dxf.vtx3.x, entity.dxf.vtx3.y, entity.dxf.vtx3.z)
            
            # Triangle
            indices = [add_vertex(p) for p in pts]
            faces.append(indices)
            
            # Quad -> second triangle
            if vtx3 != pts[2]:
                faces.append([add_vertex(pts[0]), add_vertex(pts[2]), add_vertex(vtx3)])
        except:
            continue
    
    # Also check MESH entities
    for entity in msp.query('MESH'):
        try:
            vert_start = len(vertices)
            for v in entity.vertices:
                vertices.append([v.x, v.y, v.z])
            for face in entity.faces:
                if len(face) >= 3:
                    for i in range(1, len(face) - 1):
                        faces.append([vert_start + face[0], 
                                      vert_start + face[i], 
                                      vert_start + face[i+1]])
        except:
            continue
    
    if not faces:
        raise ValueError(
            "No 3D faces found in DXF.\n"
            "Tips:\n"
            "• Use 3DFACE or MESH entities in AutoCAD\n"
            "• For 2D: use EXTRUDE to create 3D\n"
            "• Try exporting as OBJ instead"
        )
    
    return np.array(vertices), np.array(faces)

# Usage
verts, faces = load_dxf("drawing.dxf")
```

### Conversion Tips (AutoCAD to OBJ)

In AutoCAD:
1. File → Export → Other Formats
2. Select "OBJ (*.obj)" as format
3. Check "Triangulate faces" option

Or use FreeCAD:
1. Open DWG/DXF file
2. File → Export → Select OBJ

---

## Format: SKP (SketchUp)

### Challenge
SketchUp files have **limited Python support**. The best approach is to convert to OBJ.

### Options

| Approach | Pros | Cons |
|----------|------|------|
| **Export from SketchUp** ⭐ | Best compatibility, full control | Requires SketchUp |
| **trimesh + Assimp** | Direct Python loading | Complex installation |
| **Online converters** | No software needed | Privacy concerns, file size limits |

### Solution 1: Export from SketchUp (Recommended)

In SketchUp:
1. File → Export → 3D Model
2. Select format: **OBJ** (best) or **DAE** or **STL**
3. Options: Check "Triangulate all faces"
4. Export

### Solution 2: Install Assimp (Advanced)

```bash
# macOS
brew install assimp
pip install pyassimp

# Linux
sudo apt install libassimp-dev
pip install pyassimp

# Windows (use conda)
conda install -c conda-forge assimp
```

Then load with trimesh:
```python
import trimesh
mesh = trimesh.load("model.skp")  # Works if Assimp is installed
```

### Solution 3: Online Converters

- https://www.convertio.co/skp-obj/
- https://cloudconvert.com/skp-to-obj
- https://www.online-convert.com/

---

## Visualization Comparison

### Matplotlib (mplot3d)
```python
# Pros: No extra deps, simple, customizable
# Cons: Slow for >5k faces, no depth sorting

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_matplotlib(verts, faces, mode='surface'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    face_verts = verts[faces]
    
    if mode == 'wireframe':
        coll = Poly3DCollection(face_verts, facecolor='none', 
                                 edgecolor='blue', linewidth=0.5)
    else:  # surface
        coll = Poly3DCollection(face_verts, alpha=0.7, 
                                 facecolor='cyan', edgecolor='darkblue')
    
    ax.add_collection3d(coll)
    ax.auto_scale_xyz(verts[:,0], verts[:,1], verts[:,2])
    plt.show()
```

### PyVista (GPU-accelerated)
```python
# Pros: Fast, interactive, handles large meshes
# Cons: Requires VTK (~200MB)

import pyvista as pv
import numpy as np

def plot_pyvista(verts, faces, mode='surface'):
    n_faces = len(faces)
    pv_faces = np.column_stack([np.full(n_faces, 3), faces]).flatten()
    mesh = pv.PolyData(verts, pv_faces)
    
    plotter = pv.Plotter()
    if mode == 'wireframe':
        plotter.add_mesh(mesh, style='wireframe', color='blue')
    else:
        plotter.add_mesh(mesh, color='cyan', show_edges=True)
    plotter.show()
```

### Open3D (Clean and fast)
```python
# Pros: Fast, clean UI, good for point clouds
# Cons: Requires Open3D (~100MB)

import open3d as o3d

def plot_open3d(verts, faces, mode='surface'):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    
    if mode == 'wireframe':
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        o3d.visualization.draw_geometries([wireframe])
    else:
        o3d.visualization.draw_geometries([mesh])
```

---

## Unit Normalization

3D files often use different units. Auto-detect and convert:

```python
def normalize_units(vertices, target='meters'):
    """Auto-detect and normalize mesh units."""
    max_dim = np.ptp(vertices, axis=0).max()
    
    # Heuristic detection
    if max_dim > 1000:
        source = 'millimeters'
        scale = 0.001
    elif max_dim > 100:
        source = 'centimeters'
        scale = 0.01
    else:
        source = 'meters'
        scale = 1.0
    
    print(f"Detected {source}, scaling to {target}")
    return vertices * scale
```

---

## Summary: Recommended Setup

```bash
# Install these for full format support:
pip install trimesh numpy-stl ezdxf

# Optional for better visualization:
pip install open3d  # or pyvista
```

| Format | Best Library | Fallback |
|--------|--------------|----------|
| .obj | trimesh or Open3D | PyVista |
| .stl | numpy-stl | trimesh |
| .dxf/.dwg | ezdxf | - |
| .skp | Convert to OBJ | trimesh+assimp |
| .ply | Open3D | trimesh |
| .gltf/.glb | trimesh | PyVista |

The `model_loader.py` in this project automatically tries the best available library for each format.

