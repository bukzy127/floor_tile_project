# Quick Start Guide: 3D Model Import Feature

## Installation

Dependencies have been installed. If you need to reinstall:

```bash
pip install trimesh ezdxf
```

## Quick Usage Guide

### For Beginners: 5 Simple Steps

1. **Launch the Application**
   ```bash
   python3 "3D Tiles App V3 .py"
   ```

2. **Switch to 3D Model Mode**
   - In the left panel, find "Room Input" section
   - Change dropdown from "Manual (W×L)" to **"Import 3D Model"**

3. **Import Your Model**
   - Click **"Browse 3D Model (.obj/.dwg/.skp)"**
   - Select your 3D file (room.obj, building.dwg, etc.)
   - Wait for model to load in viewer

4. **Select a Surface**
   - Click **"Select Surface for Tiling"**
   - Rotate the 3D view to see desired surface (drag with left mouse button)
   - Click on the surface you want to tile (floor, wall, etc.)
   - Surface turns green when selected
   - Dimensions auto-fill

5. **Generate Tiles**
   - Configure tile size if needed (default: 0.6m × 0.6m)
   - Click **"Compute and Visualize Layout"**
   - Done! Your tiles appear on the selected surface

## Example Files You Can Use

### Test with .OBJ files
- Most 3D modeling software exports to .obj
- Blender, SketchUp, 3ds Max, Maya all support .obj

### Test with .DWG/.DXF files
- AutoCAD files work directly
- Must contain 3DFACE entities for proper import

### Test with .SKP files
- SketchUp native format
- Requires SketchUp file format support in trimesh

## Tips and Tricks

### Camera Controls
- **Rotate**: Left-click and drag
- **Pan**: Middle-click and drag
- **Zoom**: Mouse wheel

### Surface Selection Tips
- Zoom in close to the surface before selecting
- Large flat surfaces work best
- Minimum surface size: 10cm × 10cm

### Common Issues

**Q: Model loads but appears very small/large?**
- Camera automatically adjusts, but you may need to zoom in/out

**Q: Can't click on surface?**
- Make sure "Select Surface for Tiling" button was clicked first
- Try clicking center of surface, not edges

**Q: Selected wrong surface?**
- Click "Select Different Surface" button (appears after selection)
- Re-select the correct surface

**Q: Tiles don't appear?**
- Check that surface dimensions are reasonable (not too small)
- Ensure tile size fits on surface (adjust tile dimensions)

## Advanced Usage

### Multiple Tile Configurations
After selecting a surface, you can:
- Change tile dimensions
- Adjust tile thickness
- Change material properties
- Apply material textures
- Click "Compute" again to regenerate

### Export Options
Once tiles are generated on imported model:
- **Export to .OBJ**: Full scene with model + tiles
- **Export to .DXF**: For AutoCAD with 3D tiles and QR codes
- **Export Report**: Text file with all tile/pedestal data

### Material Textures on Imported Models
1. Generate tiles on imported model surface
2. Click "Import Material Texture" 
3. Select seamless texture image (PNG/JPG)
4. Texture applies to all tiles on surface

## Keyboard Shortcuts (Standard)

- **Left Mouse**: Select/Rotate
- **Middle Mouse**: Pan camera
- **Mouse Wheel**: Zoom in/out

## File Format Support Details

| Format | Extension | Notes |
|--------|-----------|-------|
| Wavefront OBJ | .obj | Best compatibility, widely supported |
| AutoCAD | .dwg, .dxf | Requires 3DFACE entities |
| SketchUp | .skp | May require additional libraries |

## What Gets Generated

On selected surface:
- ✓ Tiles (full and cut tiles as needed)
- ✓ Pedestals under each tile
- ✓ QR codes on each tile
- ✓ Material textures (if applied)
- ✓ Proper elevation handling

## Next Steps

1. Try importing one of your existing 3D models
2. Experiment with different surfaces (floors, walls, ceilings)
3. Configure different tile materials and sizes
4. Export the complete scene for use in other software

## Need Help?

Check the full documentation: `3D_MODEL_IMPORT_FEATURE.md`

