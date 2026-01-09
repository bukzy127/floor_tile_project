"""Test the updated scale detection logic"""
import sys
sys.path.insert(0, '.')

from model_loader import load_model, normalize_mesh_units

def test_scale_detection():
    print("="*60)
    print("TESTING UPDATED SCALE DETECTION")
    print("="*60)
    
    # Load the model
    print("\n1. Loading TEST.obj...")
    mesh = load_model('TEST.obj')
    
    print(f"\n2. Original mesh extents: {mesh.extents}")
    print(f"   Max dimension: {mesh.extents.max():.3f}")
    
    print("\n3. Normalizing to meters...")
    mesh_normalized = normalize_mesh_units(mesh, target_unit='meters', source_unit='auto')
    
    print(f"\n4. After normalization:")
    print(f"   Extents: {mesh_normalized.extents}")
    print(f"   Max dimension: {mesh_normalized.extents.max():.3f}m")
    print(f"   Bounding box: {mesh_normalized.extents[0]:.3f}m x {mesh_normalized.extents[1]:.3f}m x {mesh_normalized.extents[2]:.3f}m")
    
    # Check if reasonable
    max_dim = mesh_normalized.extents.max()
    if 0.3 < max_dim < 10:
        print(f"\n✓ SUCCESS: Dimensions are reasonable for a room ({max_dim:.3f}m)")
    else:
        print(f"\n❌ WARNING: Dimensions seem unusual ({max_dim:.3f}m)")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    test_scale_detection()
