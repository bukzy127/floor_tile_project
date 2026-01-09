# PATCH SUMMARY: Clearance/Headroom Logic for Imported 3D Models

## Files Modified
- `3D Tiles App V3 .py` - Main application file

## Key Changes

### 1. UI INITIALIZATION (Lines ~4150-4160)

```python
# BEFORE:
self.headroom_in = QDoubleSpinBox(..., value=2.00, ...)
self.min_pedestal_in = QDoubleSpinBox(..., value=0.10, ...)

# AFTER:
self.headroom_in = QDoubleSpinBox(..., value=0.0, ...)  # 0 at start, set on 3D import
self.min_pedestal_in = QDoubleSpinBox(..., value=0.0, ...)  # 0 at start, set on 3D import
# Updated tooltips note: "(Set after 3D model import)"
```

**Rationale:** Avoid confusion; clearance fields only meaningful for imported 3D models.

---

### 2. 3D MODEL IMPORT SUCCESS HANDLER (Lines ~4406-4430)

```python
# NEW CODE ADDED after "self.showtiles_cb.setChecked(False)"

# ===================================================================
# Set clearance defaults for 3D imported model (only on success)
# ===================================================================
# Compute imported floor elevation statistics
floor_vertices = mesh.vertices
floor_z_values = floor_vertices[:, 2] if floor_vertices.shape[1] > 2 else np.zeros(len(floor_vertices))
imported_floor_min_z = float(np.min(floor_z_values))
imported_floor_max_z = float(np.max(floor_z_values))
imported_floor_range = imported_floor_max_z - imported_floor_min_z

# Store for internal use in layout generation
self.gl_widget.imported_floor_min_z = imported_floor_min_z
self.gl_widget.imported_floor_max_z = imported_floor_max_z
self.gl_widget.imported_floor_range = imported_floor_range

print(f"[IMPORT-3D] Floor elevation: min={imported_floor_min_z:.4f}, max={imported_floor_max_z:.4f}, range={imported_floor_range:.4f}")

# Set clearance defaults (NOT the legacy "Original subfloor" fields - those stay 0)
with QtCore.QSignalBlocker(self.headroom_in):
    self.headroom_in.setValue(2.0)
with QtCore.QSignalBlocker(self.min_pedestal_in):
    self.min_pedestal_in.setValue(0.10)

print(f"[IMPORT-3D] Set clearance defaults: headroom=2.0m, min_pedestal=0.10m")
```

**Key Points:**
- Only runs on successful import
- Computes floor min/max/range internally
- Sets clearance field defaults to 2.0m and 0.10m
- Original subfloor fields NOT touched (stay 0 unless user/file sets)
- Adds logging for debugging

---

### 3. LAYOUT GENERATION FUNCTION SIGNATURE (Line ~1520)

```python
# BEFORE:
def compute_layout_on_selected_surfaces(self, tile_params, pedestal_height=0.1016):

# AFTER:
def compute_layout_on_selected_surfaces(self, tile_params, pedestal_height=0.1016, 
                                       headroom_m=2.0, min_pedestal_total=0.10):
```

**New Parameters:**
- `headroom_m`: User-specified headroom (meters)
- `min_pedestal_total`: Minimum pedestal height constraint (meters)

---

### 4. VALIDATION LOGIC IN LAYOUT GENERATION (Lines ~1540-1570)

```python
# NEW CODE ADDED after "mesh = self.imported_mesh"

# ===================================================================
# VALIDATION: Imported 3D floor elevation constraints
# ===================================================================
# Get imported floor elevation stats (computed at import time)
floor_max_z = getattr(self, 'imported_floor_max_z', 0.0)

# Derive ceiling_z and tile planes from headroom
ceiling_z = headroom_m  # Simple case: ceiling at headroom height
tile_top_z = ceiling_z - headroom_m  # = 0 in simple case
tile_bottom_z = tile_top_z - tile_thickness

# Validate: (tile_bottom_z - floor_max_z) >= min_pedestal_total
clearance_at_max_floor = tile_bottom_z - floor_max_z
if clearance_at_max_floor < min_pedestal_total - 1e-6:
    # FAIL: Show error dialog with actionable info
    headroom_max = ceiling_z - tile_thickness - floor_max_z - min_pedestal_total
    error_msg = (
        "❌ Layout generation FAILED: Insufficient clearance\n\n"
        "The specified headroom is too large to satisfy the minimum pedestal height\n"
        "at the highest floor point.\n\n"
        f"Current constraints:\n"
        f"  • Ceiling Z: {ceiling_z:.4f}m\n"
        f"  • Tile thickness: {tile_thickness:.4f}m\n"
        f"  • Tile bottom Z: {tile_bottom_z:.4f}m\n"
        f"  • Floor highest point (max Z): {floor_max_z:.4f}m\n"
        f"  • Clearance at highest floor: {clearance_at_max_floor:.4f}m\n"
        f"  • Minimum required: {min_pedestal_total:.4f}m\n\n"
        f"Recommended action:\n"
        f"  • Max allowed headroom: {max(0.0, headroom_max):.4f}m\n"
        f"  • Reduce headroom, OR\n"
        f"  • Reduce minimum pedestal height, OR\n"
        f"  • Increase ceiling height"
    )
    self._last_validation_error = error_msg
    print(f"[VALIDATION-FAIL] {error_msg}")
    return False

print(f"[VALIDATION-PASS] Clearance at max floor: {clearance_at_max_floor:.4f}m >= {min_pedestal_total:.4f}m (required)")
```

**Key Points:**
- Gets floor_max_z computed at import time
- Validates: clearance at highest floor >= min_pedestal_total
- **DOES NOT auto-clamp** - shows error and returns False
- Stores error message for UI dialog
- Adds logging for each case

---

### 5. LAYOUT GENERATION CALL (Lines ~5000-5020)

```python
# BEFORE:
success = self.gl_widget.compute_layout_on_selected_surfaces(
    tile_params,
    pedestal_height=pedestal_height
)

# AFTER:
# Pass headroom and min_pedestal from UI (for imported 3D floor validation)
headroom_m = params.get('headroom_m', 2.0)
min_pedestal_total = params.get('min_pedestal_height_m', 0.10)

success = self.gl_widget.compute_layout_on_selected_surfaces(
    tile_params,
    pedestal_height=pedestal_height,
    headroom_m=headroom_m,
    min_pedestal_total=min_pedestal_total
)

# If validation failed, show error dialog
if not success and hasattr(self.gl_widget, '_last_validation_error'):
    QMessageBox.critical(
        self,
        "Layout Generation Failed",
        self.gl_widget._last_validation_error
    )
```

**Key Points:**
- Extracts headroom and min_pedestal from params dict
- Passes to validation function
- Shows error dialog if validation failed
- User can then adjust values manually

---

## Design Goals Achieved

✅ **Two Independent UI Systems**
- "Original subfloor slope/elevation" (manual/legacy) - untouched, stays at 0
- "Clearance & Heights" (3D import) - starts at 0, set on successful import

✅ **Strict Validation**
- No automatic clamping
- Clear error messages with actionable recommendations
- User controls final decision

✅ **Clean Integration**
- Imported floor stats computed once at import
- Reused in layout generation
- Minimal changes to existing code

✅ **Backward Compatible**
- Original subfloor system unchanged
- pedestal_height parameter kept for compatibility
- Existing code paths not modified

---

## Testing Checklist

- [ ] App starts: clearance fields at 0.0
- [ ] Import 3D model: floor stats logged, fields set to 2.0 and 0.10
- [ ] Original subfloor fields stay 0 after import
- [ ] Layout generation (sufficient clearance): validates and generates
- [ ] Layout generation (insufficient clearance): shows error dialog with recommendations
- [ ] Error dialog provides actionable max headroom value
- [ ] No auto-clamping occurs
