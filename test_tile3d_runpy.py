# Minimal standalone test that loads your module via runpy and exercises Tile3D
import runpy
import sys
import os

MODULE_PATH = r"c:\Users\USER\OneDrive - Technische Hochschule Deggendorf\project work\tile_project\3D Tiles App V3 .py"
if not os.path.isfile(MODULE_PATH):
    print(f"Module file not found: {MODULE_PATH}")
    sys.exit(2)

ns = runpy.run_path(MODULE_PATH)
Tile3D = ns.get('Tile3D')
if Tile3D is None:
    print('Tile3D class not found in module')
    sys.exit(3)

# Create a simple square footprint tile and compute corners
footprint = [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)]
try:
    tile = Tile3D(origin_xy=(0.0,0.0), xtile=1.0, ytile=1.0, thickness=0.1, is_cut=False, cut_polygon_xy=footprint, qr_data='T-0-0')
    tile.compute_3d_corners(0.0)
except Exception as e:
    print('Exception during compute_3d_corners:', e)
    raise

if len(tile.corners_top_xyz) != 4:
    print('Unexpected number of top corners:', len(tile.corners_top_xyz))
    sys.exit(4)

if tile.pick_origin is None:
    print('pick_origin was not set')
    sys.exit(5)

print('TEST PASSED')
