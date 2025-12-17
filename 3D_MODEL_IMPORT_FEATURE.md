# 3D Model Import Feature Documentation

## Overview
Your tile layout application now supports importing 3D models (.obj, .dwg, .skp) with interactive surface selection for tile generation. This allows you to:
1. Import existing 3D building/room models
2. Select any surface (floor, wall, etc.) for tiling
3. Automatically extract dimensions and generate tiles on that surface

## New Features Added

### 1. 3D Model Import Support
- **Formats Supported**: .obj, .dwg, .dxf, .skp (SketchUp)
- **Library Used**: trimesh for mesh processing
- **DWG/DXF**: Uses ezdxf to extract 3D faces and convert to mesh

### 2. Surface Selection Mode
- Click-to-select any planar surface from the imported model
- Visual feedback with highlighting (yellow on hover, green when selected)
- Validation checks for surface planarity and minimum size

### 3. Automatic Dimension Extraction
- Extracts width and length from selected surface bounds
- Auto-populates "Room Dimensions" fields
- Detects surface orientation (horizontal/vertical)

### 4. Surface-Mapped Tile Generation
- Tiles are generated directly on the selected surface
- Uses existing tile generation logic with minimal modifications
- Supports all existing features (pedestals, QR codes, materials, etc.)

## How to Use

### Step 1: Switch to 3D Model Mode
1. In the **"Room Input"** panel, change the dropdown to **"Import 3D Model"**
2. New controls will appear:
   - "Browse 3D Model" button
   - Model status indicator
   - "Select Surface for Tiling" button
   - Surface information display

### Step 2: Import a 3D Model
1. Click **"Browse 3D Model (.obj/.dwg/.skp)"**
2. Select your 3D model file
3. The model will load into the 3D viewer
4. You'll see a success message with instructions

### Step 3: Select a Surface
1. Click **"Select Surface for Tiling"** button
2. The button will show "Selection Mode Active..."
3. Click on any surface in the 3D viewer (use mouse to rotate/zoom first if needed)
4. Selected surface will turn **green** with a yellow wireframe outline
5. Surface dimensions auto-populate in "Room Dimensions" fields

### Step 4: Generate Tiles
1. Configure tile dimensions, materials, and other settings as usual
2. Click **"Compute and Visualize Layout"**
3. Tiles and pedestals will be generated on the selected surface
4. All existing features work: QR codes, material textures, export, etc.

## Technical Implementation

### New Functions in GLWidget Class

#### `import_model(file_path)`
- Loads 3D model from supported formats
- Returns: `(success: bool, error_message: str | None)`
- Stores model in `self.imported_model` (trimesh object)
- Prepares face data for rendering and selection

#### `select_surface(event)`
- Handles surface selection from mouse click
- Uses ray casting to find clicked face
- Validates surface (planar, minimum size)
- Returns: `(success: bool, error_message: str | None)`

#### `_pick_model_face(x_screen, y_screen)`
- Ray-triangle intersection using Möller-Trumbore algorithm
- Returns face index of clicked surface

#### `_validate_surface(face_idx)`
- Checks if surface is suitable for tiling
- Validates minimum size (10cm)
- Ensures surface is reasonably planar

#### `_extract_surface_bounds(face_idx)`
- Extracts 2D bounds from 3D surface
- Detects orientation (horizontal/vertical)
- Stores in `self.selected_surface_bounds`

#### `map_generation_to_surface(surface_bounds)`
- Maps tile generation to selected surface coordinates
- Creates polygon matching surface bounds
- Reuses existing polygon-based generation

#### `draw_imported_model()`
- Renders 3D model faces with OpenGL
- Highlights selected/hovered surfaces
- Integrated into existing paintGL() method

### Modified Methods

#### `paintGL()`
- Now renders imported model if available
- Existing floor/tile rendering still works

#### `mousePressEvent()`
- Checks for surface selection mode first
- Falls back to tile/pedestal picking for normal mode

#### `update_visualization()`
- Detects 3D model mode
- Validates model and surface selection
- Uses surface bounds for tile generation

### New UI Controls (MainWindow)

- **Room Mode Dropdown**: Added "Import 3D Model" option
- **Browse 3D Model Button**: Opens file dialog for .obj/.dwg/.skp
- **Model Status Label**: Shows loaded model filename
- **Select Surface Button**: Enables surface selection mode
- **Surface Info Label**: Displays selected surface dimensions

### Helper Methods

- `on_room_mode_changed()`: Shows/hides 3D model controls
- `import_3d_model()`: Handles model file import dialog
- `enable_surface_selection()`: Activates selection mode
- `check_surface_selection()`: Timer-based check for selection completion

## Dependencies Added

Updated `requirements.txt`:
```
ezdxf>=1.0.0      # DXF/DWG file support
trimesh>=3.0.0     # 3D mesh processing and format conversion
```

## Error Handling

- Invalid file formats: Clear error message
- Library not installed: Helpful installation message
- Invalid surface selection: Validation with user feedback
- Small surfaces: Minimum 10cm requirement with warning

## Fallback Behavior

- If no model imported: Shows warning when clicking "Compute"
- If no surface selected: Shows warning to select surface first
- Manual/Polygon modes: Unchanged, work as before

## Code Structure (Modular Design)

All new functionality is isolated in new methods:
- **Import**: `import_model()`, `_load_dxf_as_mesh()`
- **Selection**: `select_surface()`, `_pick_model_face()`, `_validate_surface()`
- **Extraction**: `_extract_surface_bounds()`, `map_generation_to_surface()`
- **Rendering**: `draw_imported_model()`

Existing generation logic (`compute_and_build_layout()`) is **reused** with minimal adaptation:
- No changes to tile generation algorithms
- Polygon mode handles surface bounds automatically
- All existing features (pedestals, materials, QR codes, export) work on imported models

## Example Workflow

```
1. User imports room.obj (a 3D building model)
2. Model appears in viewer showing complete building
3. User clicks "Select Surface" and clicks on a floor surface
4. Floor surface highlights in green
5. Dimensions auto-populate: Width: 5.0m × Length: 4.0m
6. User configures tiles: 0.6m × 0.6m × 0.02m thickness
7. User clicks "Compute and Visualize Layout"
8. Tiles with pedestals appear on the selected floor surface
9. User can export to .obj/.dxf with model + tiles together
```

## Future Enhancements (Optional)

- Multi-surface selection (tile multiple surfaces at once)
- Non-planar surface support (curved surfaces)
- SketchUp API integration for better .skp support
- Blend file (.blend) support via Blender API
- Automatic floor detection (find horizontal surfaces)

## Notes

- Surface selection uses ray casting for accurate picking
- Selected surface bounds are mapped to room polygon coordinates
- Existing elevation model and slope features work on selected surfaces
- Material textures apply to tiles on imported model surfaces
- QR codes are generated and positioned correctly on all tiles

