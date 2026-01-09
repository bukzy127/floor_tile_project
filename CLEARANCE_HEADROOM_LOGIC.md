# Clearance / Headroom Logic Implementation for Imported 3D Models

## Overview
Implemented clearance/headroom logic for imported 3D models with uneven floor elevation while keeping the "Original subfloor slope/elevation" UI section independent (manual/legacy).

## Key Changes

### 1. UI Initialization (Clearance & Heights)

**Location:** Lines ~4150-4160 in `3D Tiles App V3 .py`

**Changes:**
- `headroom_in` spinbox: **Default value changed from 2.00 to 0.0** at app start
- `min_pedestal_in` spinbox: **Default value changed from 0.10 to 0.0** at app start
- Both fields now have updated tooltips noting they are "Set after 3D model import"
- **Purpose:** Avoids confusion when no 3D model is loaded; values are only meaningful for imported 3D models

**Constraint:** Original subfloor slope/elevation fields (`sbz_in`, `sx_in`, `sy_in`) are NOT modified - they remain at their legacy defaults (independent manual/legacy system).

### 2. 3D Model Import Success Handler

**Location:** Lines ~4406-4430 in `3D Tiles App V3 .py`

**New Logic:** When a 3D model is successfully imported:
1. **Compute imported floor elevation stats:**
   - Extract Z coordinates from mesh vertices
   - Calculate: `imported_floor_min_z`, `imported_floor_max_z`, `imported_floor_range`
   - Store as attributes on `gl_widget` for use in layout generation

2. **Set clearance defaults:**
   - `headroom_in.setValue(2.0)` - set to 2.0 meters
   - `min_pedestal_in.setValue(0.10)` - set to 0.10 meters
   - Use `QSignalBlocker` to prevent triggering change handlers
   - **DO NOT modify original subfloor fields** - they stay 0 unless user/file sets them

3. **Logging:**
   - `[IMPORT-3D] Floor elevation: min=..., max=..., range=...`
   - `[IMPORT-3D] Set clearance defaults: headroom=2.0m, min_pedestal=0.10m`

### 3. Layout Generation with Validation

**Location:** Lines ~1520-1570 in `3D Tiles App V3 .py` (compute_layout_on_selected_surfaces)

**Function Signature Change:**
```python
def compute_layout_on_selected_surfaces(self, tile_params, pedestal_height=0.1016, 
                                       headroom_m=2.0, min_pedestal_total=0.10):
```

**New Parameters:**
- `headroom_m`: Headroom from tile TOP to ceiling (meters)
- `min_pedestal_total`: Minimum pedestal height at highest floor point (meters)

**Validation Logic:**

1. **Get imported floor max Z:**
   ```python
   floor_max_z = getattr(self, 'imported_floor_max_z', 0.0)
   ```

2. **Compute tile planes from headroom:**
   - `ceiling_z = headroom_m` (simplified: assumes tile_top_z = 0)
   - `tile_top_z = ceiling_z - headroom_m` (= 0)
   - `tile_bottom_z = tile_top_z - tile_thickness` (= -tile_thickness)

3. **Validate clearance at highest floor point:**
   ```
   clearance_at_max_floor = tile_bottom_z - floor_max_z
   if clearance_at_max_floor < min_pedestal_total - 1e-6:
       FAIL ❌
   ```

4. **On Validation FAILURE:**
   - **DO NOT auto-clamp values**
   - Store error message in `self._last_validation_error`
   - Return `False` (failure signal)
   - Error message includes:
     - Explanation of the problem
     - All computed values (ceiling_z, tile_thickness, tile_bottom_z, floor_max_z, etc.)
     - Clearance at highest floor point
     - Recommended max allowed headroom
     - Actionable suggestions (reduce headroom, reduce min pedestal, increase ceiling)

5. **On Validation PASS:**
   - Log: `[VALIDATION-PASS] Clearance at max floor: X.XXXX >= Y.YYYY (required)`
   - Continue to generate tiles and pedestals

### 4. Error Dialog in UI Handler

**Location:** Lines ~5001-5020 in `3D Tiles App V3 .py` (on_compute_clicked)

**Logic:**
```python
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
- Only shows error if validation explicitly failed (not for other reasons)
- UI values remain unchanged - user must manually adjust
- Dialog provides actionable information for the user

## Design Rationale

### 1. Two Independent UI Systems
- **"Original Subfloor"** (legacy): Manual/user-controlled plane definition
  - Always initialized to 0
  - Only changes if user manually inputs or file provides values
  - NOT affected by 3D model import

- **"Clearance & Heights"** (3D import): Headroom and pedestal constraints
  - Initialized to 0 (disabled until import)
  - Set to meaningful defaults (2.0m, 0.10m) on successful 3D import
  - Used for imported floor validation

### 2. Validation at Import Time
- Compute floor elevation stats when model is imported
- Store internally for later use in layout generation
- Separate from the legacy manual elevation system

### 3. Strict Validation (No Auto-Clamp)
- Validation checks: `(tile_bottom_z - floor_max_z) >= min_pedestal_total`
- If fails: **STOP, show error, do NOT auto-clamp**
- If passes: proceed with layout generation
- This ensures user is aware of constraint violations and can make informed decisions

### 4. Pedestal Height Computation
- Uses `floor_z_at(x,y)` sampled from imported floor mesh
- `needed_total = tile_bottom_z - base_z`
- Global constraint enforced via validation, not per-pedestal inflation

## Testing

### Scenario 1: No 3D Model
- Clearance fields stay at 0.0
- Original subfloor fields stay at their defaults
- Both systems independent

### Scenario 2: 3D Model Import Success
- Clearance fields auto-set to 2.0 (headroom) and 0.10 (min pedestal)
- Original subfloor fields unchanged (still 0 unless user sets)
- Floor elevation stats logged

### Scenario 3: Layout Generation - Sufficient Clearance
- Validation passes
- Layout generates successfully
- Message: `[VALIDATION-PASS] Clearance at max floor: ...`

### Scenario 4: Layout Generation - Insufficient Clearance
- Validation fails
- Error dialog shows with actionable info
- No layout generated
- No UI values changed automatically

## Implementation Summary

| Component | Initial State | After 3D Import | At Layout Generation |
|-----------|---------------|-----------------|----------------------|
| **headroom_in** | 0.0 | 2.0 | Read from UI |
| **min_pedestal_in** | 0.0 | 0.10 | Read from UI |
| **Original subfloor** | 0.0 | 0.0 (unchanged) | Not used for 3D |
| **imported_floor_max_z** | N/A | Computed | Used for validation |
| **Validation** | N/A | N/A | Strict (no auto-clamp) |
