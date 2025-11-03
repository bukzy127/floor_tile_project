The program should parse these points, create a polygon, and visualize it in 3D.

TILE ESTIMATOR PROJECT — AI DEVELOPMENT PROMPT
Project Purpose:
This application estimates tile placement and pedestal heights in 3D for a given room using  Python, PyQt6, and OpenGL.  
It computes how tiles are positioned based on room and tile dimensions and visualizes the layout in 3D.
GOAL — NEW FEATURES TO IMPLEMENT
You are extending the existing project with the following features:

1. Flexible Room Dimension Input Options
Implement multiple ways for users to define the room’s geometry:

Manual Input
Allow users to input room width and length or coordinates directly in the UI.

Coordinate Points Input
Enable users to manually define an irregular room by specifying a list of 2D points (x, y) that form a polygon outline.

File Upload
Add support for importing room geometry from a `.txt` file:

2. Support for Regular and Irregular Room Shapes

 - Extend the tile placement logic to handle both rectangular and irregular polygonal rooms.  
- Use point_in_polygon()  to determine if a tile’s center lies within the valid room boundary.  
- Ensure that tiles near irregular edges are trimmed or excluded properly.  
- Modify the rendering so that the polygonal boundary is visible in the OpenGL view.

3. Uneven Floor Support (Elevation Mapping)
Enable modeling of non-flat floors with  local elevation variations:
- Allow manual entry of Z-values for corner points.
- Option to import elevation data from `.txt` or `.csv` files.
- Provide an “Elevation Mode” selector:
  - Flat
  - Planar Slope
  - Irregular Map
- Add visualization of the uneven surface (mesh or color gradient) in GLWidget.

 4. Variable Support Heights
For uneven floors, supports (pedestals) should have variable heights to keep the top surface of all tiles perfectly level.

Implementation details:
- In compute_and_build_layout() :
  - Compute each tile’s pedestal height as:  
    support_height = tile_top_z - local_floor_z
- For flat floors, keep all support heights uniform.

5. User Interface Updates (PyQt6)
Extend the main window with new UI components:

-  Dropdowns:
  - Room Input Mode → `[Rectangle | Polygon | Import from File]`
  - Elevation Mode → `[Flat | Planar Slope | Irregular Map]`
- Buttons:
  - “Import Room File”
  - “Import Elevation Map”
- Dialogs:
  - Use `QFileDialog` for importing `.txt` or `.csv` files.
  - Show error messages using `QMessageBox` for invalid formats or bad geometry.
- **Visualization toggles:**
  - Floor wireframe view
  - Elevation heatmap
  - 3D tiles view

# 6. Validation and Error Handling

- Validate polygon input for:
  - Proper closure
  - Non-intersecting edges
  - Minimum of 3 points
- Provide clear UI feedback when invalid input is detected.
- Catch file parsing exceptions gracefully.

 7. Testing and Verification

Add testing or sample data for:
- Rectangular rooms (basic)
- Irregular polygons (L-shapes, trapezoids)
- Elevation maps (flat vs. uneven)
- Ensure visual correctness and consistent tile alignment.


# IMPLEMENTATION GUIDANCE

*Existing Core Methods:*
- GLWidget.compute_and_build_layout() → main algorithm for tile and support placement.
- Tile3D → represents individual tiles.
- point_in_polygon() and polygon_area_2d() → for geometry operations.
- draw_original_floor() → extend to visualize uneven surfaces.

*Recommended New Modules/Classes:
- RoomInputHandler: parses manual, polygon, or file-based room definitions.
- ElevationModel: handles elevation data, interpolation, and slope generation.
- UIRoomInputDialog: PyQt6 form for entering or importing room data


EXPECTED RESULT
After implementation, the software should:

- Accept multiple room input methods (manual, polygon, or file import).
- Support irregular and regular room shapes.
- Model uneven floors with variable elevations.
- Automatically adjust  support heights for a perfectly flat tile surface.
- Visually display these configurations in 3D.
- Provide clear error handling and validation.

# Example AI Command
> “Extend the GLWidget  class to support irregular polygonal rooms and uneven floors.  
> Update the UI to allow room definition via manual entry or file upload, and compute pedestal heights dynamically based on local elevation. Ensure all tiles maintain a uniform top Z level.”
