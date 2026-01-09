"""Test script to simulate the import flow and debug tile generation"""
import sys
import trimesh
import numpy as np

def test_obj_import():
    """Test loading TEST.obj and analyzing its structure"""
    
    print("="*60)
    print("TEST.OBJ IMPORT FLOW ANALYSIS")
    print("="*60)
    
    # Load the mesh
    mesh = trimesh.load('TEST.obj', force='mesh')
    print(f"\n1. MESH LOADING:")
    print(f"   - Vertices: {len(mesh.vertices)}")
    print(f"   - Faces: {len(mesh.faces)}")
    
    # Analyze coordinate ranges
    vertices = mesh.vertices
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    z_coords = vertices[:, 2]
    
    print(f"\n2. COORDINATE RANGES:")
    print(f"   - X: [{x_coords.min():.3f}, {x_coords.max():.3f}] (span: {x_coords.max()-x_coords.min():.3f})")
    print(f"   - Y: [{y_coords.min():.3f}, {y_coords.max():.3f}] (span: {y_coords.max()-y_coords.min():.3f})")
    print(f"   - Z: [{z_coords.min():.3f}, {z_coords.max():.3f}] (span: {z_coords.max()-z_coords.min():.3f})")
    
    # Check scale
    width = x_coords.max() - x_coords.min()
    print(f"\n3. SCALE ANALYSIS:")
    if width > 100:
        print(f"   ⚠ WARNING: Model appears to be in MILLIMETERS (width={width:.1f})")
        print(f"   ⚠ App expects METERS - this will cause issues!")
        print(f"   ⚠ Should be ~{width/1000:.2f}m, not {width:.1f}mm")
    else:
        print(f"   ✓ Model appears to be in meters (width={width:.2f}m)")
    
    # Analyze triangles for floor detection
    print(f"\n4. FLOOR DETECTION ANALYSIS:")
    triangles = []
    for face_idx, face in enumerate(mesh.faces):
        if len(face) < 3:
            continue
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        centroid = (v0 + v1 + v2) / 3.0
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        area = np.linalg.norm(normal) / 2.0
        if area < 1e-9:
            continue
        normal = normal / (2.0 * area)
        triangles.append({
            'centroid': centroid,
            'normal': normal,
            'area': area,
            'z': centroid[2]
        })
    
    print(f"   - Total triangles: {len(triangles)}")
    
    # Find floor candidates (Z near minimum, normal facing +Z or -Z)
    z_threshold = 0.85
    all_z = [t['z'] for t in triangles]
    min_z, max_z = min(all_z), max(all_z)
    z_range = max_z - min_z
    z_band_threshold = min_z + 0.35 * z_range if z_range > 1e-6 else min_z + 1.0
    
    upward_floor = [t for t in triangles if t['normal'][2] >= z_threshold and t['z'] <= z_band_threshold]
    downward_floor = [t for t in triangles if t['normal'][2] <= -z_threshold and t['z'] <= z_band_threshold]
    abs_floor = [t for t in triangles if abs(t['normal'][2]) >= z_threshold and t['z'] <= z_band_threshold]
    
    print(f"   - Z range: [{min_z:.3f}, {max_z:.3f}], band threshold: {z_band_threshold:.3f}")
    print(f"   - Upward-facing floor triangles (normal[2] >= 0.85): {len(upward_floor)}")
    print(f"   - Downward-facing floor triangles (normal[2] <= -0.85): {len(downward_floor)}")
    print(f"   - Either direction (abs(normal[2]) >= 0.85): {len(abs_floor)}")
    
    if len(downward_floor) > len(upward_floor):
        print(f"   ⚠ WARNING: Floor normals are INVERTED (facing downward)!")
        print(f"   ⚠ Original code only accepts upward normals - 0 floor triangles detected!")
        print(f"   ✓ Fixed code accepts both directions - {len(abs_floor)} floor triangles detected!")
    
    # Sample a few triangles
    if triangles:
        print(f"\n5. SAMPLE TRIANGLES (first 5):")
        for i, t in enumerate(triangles[:5]):
            print(f"   Triangle {i}: normal=({t['normal'][0]:.3f}, {t['normal'][1]:.3f}, {t['normal'][2]:.3f}), z={t['z']:.3f}")
    
    # Check tile size vs model size
    print(f"\n6. TILE GENERATION ESTIMATE:")
    tile_width = 0.6  # Default tile size in app
    tile_length = 0.6
    
    if width > 100:  # Likely millimeters
        print(f"   ⚠ With model in millimeters and tile={tile_width}m:")
        print(f"   ⚠ Tile step will be 0.6 units (interpreted as 0.6mm)")
        print(f"   ⚠ Grid will be {int(width/tile_width)} x {int((y_coords.max()-y_coords.min())/tile_length)} tiles")
        print(f"   ⚠ Result: {int(width/tile_width) * int((y_coords.max()-y_coords.min())/tile_length)} tiles (WAY TOO MANY!)")
    else:
        print(f"   ✓ With model in meters and tile={tile_width}m:")
        print(f"   ✓ Grid will be {int(width/tile_width)} x {int((y_coords.max()-y_coords.min())/tile_length)} tiles")
        print(f"   ✓ Result: ~{int(width/tile_width) * int((y_coords.max()-y_coords.min())/tile_length)} tiles")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    
    issues = []
    if width > 100:
        issues.append("❌ SCALE MISMATCH: Model is in millimeters, app expects meters")
    if len(downward_floor) > len(upward_floor):
        issues.append("❌ INVERTED NORMALS: Floor faces downward (now fixed with abs())")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No major issues detected!")
    
    print("\n")

if __name__ == '__main__':
    test_obj_import()
