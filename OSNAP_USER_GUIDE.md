# OSNAP System - Quick User Guide

## What is OSNAP?

OSNAP (Object Snap) is an AutoCAD-inspired feature that automatically snaps your measurement tool cursor to precise points on geometry (floor vertices, walls, imported models) within a 12-pixel radius.

## How to Use

### Step 1: Enable Measurement Tool
- Click the **Ruler** button in the floating toolbar (or press keyboard shortcut)
- Cursor changes to crosshair

### Step 2: Click First Point
- Click anywhere on the floor to set the starting point
- If you click near a vertex, it automatically snaps to it
- A measurement point appears

### Step 3: Move to Second Point
As you move your cursor:
- **Green spheres** appear near vertices (snap candidates)
- **Yellow square marker** appears when within 12px of a vertex
- **"SNAP: Endpoint"** label shows snap type
- Ruler line previews your measurement in real-time

### Step 4: Confirm Measurement
- Click second point → snaps exactly to nearest vertex
- **Distance displayed** at line midpoint
- **Constraint hints** show if vertical [VERTICAL] or horizontal [HORIZONTAL]

### Step 5: Continue or Clear
- **Click again** to extend measurement to 3rd point
- **Right-click** to clear all measurements and start over

## Visual Feedback

### Snap Indicators

| Indicator | Meaning |
|-----------|---------|
| **Green Sphere** | Snap candidate nearby (8-12px away) |
| **Bright Green Sphere** | Snap candidate very close (<8px away) |
| **Yellow Square** | Active snap - exactly on vertex |
| **"SNAP: Endpoint"** | Label showing snap type |

### Ruler Display

```
First point ————————•———— Second point
                   |
              Distance: 2.345m SNAP: Endpoint [VERTICAL]
```

## Snap Types

- **Endpoint**: Snap to vertex or corner
- **Midpoint**: Snap to edge or face center (if enabled)
- **Intersection**: Snap to crossing edges (future)

## Tips & Tricks

### Precise Corner Measurements
1. Activate Measurement tool
2. Click one room corner (snaps to vertex)
3. Move cursor slowly to opposite corner
4. When yellow marker appears, click to snap

### Height Measurements
- Use OSNAP to measure floor-to-ceiling height
- Start at floor corner → measure to ceiling vertex
- Constraint shows [VERTICAL] for Z-dominant measurements

### Disable Snapping (if needed)
- Move cursor >12px away from any vertex → no snap marker
- Click point freely without snapping

### Snap Threshold
- **12 pixels** screen-space tolerance
- Automatically scales for high-DPI displays (Retina, 4K)
- Larger for distant objects (perspective scaling)

## Keyboard Shortcuts (if enabled)

- **Ruler**: Activate measurement tool
- **R + Drag**: Measure and hold (live update)
- **Right-Click**: Clear all measurements
- **ESC**: Exit measurement mode

## Common Workflows

### Floor Area Measurement
1. Click corner A → snaps to vertex
2. Click corner B → snaps to vertex
3. Click corner C → snaps to vertex
4. Right-click → clear, measurements display room area

### Ceiling Height Check
1. Click floor point → snaps to corner
2. Move up → click ceiling → snaps to ceiling vertex
3. Distance shows exact height
4. Display shows [VERTICAL] constraint

### Wall Length Measurement
1. Click wall start corner → snaps
2. Click wall end corner → snaps
3. Distance shows exact wall length
4. Display shows [HORIZONTAL] constraint

### 3D Model Measurements
1. Load 3D model (OBJ, STL, FBX)
2. Click model vertex A → snaps to model geometry
3. Click model vertex B → snaps to other model geometry
4. Exact distance between model parts measured

## Troubleshooting

### Yellow Marker Not Appearing
- Check that cursor is within 12 pixels of a vertex
- Move cursor closer to floor/wall corners
- Try zooming in for easier targeting

### Snap Pointing to Wrong Vertex
- Only the **closest** vertex within 12px snaps
- Move cursor away slightly, then back for different vertex
- Use constraint hints to avoid vertical/horizontal snaps when unwanted

### Performance Slow with Large Models
- OSNAP intelligently decimates large meshes
- Max 500 snap candidates loaded simultaneously
- Performance is optimized for typical use

### Snap Not Working After Import
- Wait a moment for cache to rebuild
- Move mouse over model → green spheres should appear
- Try refreshing view (pan/zoom)

## Pro Tips for Architects/Designers

### BIM-like Workflow
1. Import floor plan as 3D model
2. Use OSNAP to measure key dimensions
3. Compare to design drawings
4. Identify discrepancies via measurement

### Quality Control
1. Measure each corner of floor area
2. Compare diagonal lengths
3. OSNAP ensures precision to vertex locations
4. Document measurements with screenshots

### Site Verification
1. Place reference human model
2. Use OSNAP ruler to verify furniture heights
3. Check door/window dimensions
4. Ensure accessibility clearances

---

**Version**: 1.0  
**Last Updated**: 2025-01-20  
**Status**: Production Ready
