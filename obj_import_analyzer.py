"""
OBJ Import Analysis Pipeline
=============================

Safe, optional analyzer for OBJ files imported from BIM software
(Revit, SketchUp, Rhino, AutoCAD, etc.).

Provides METADATA ONLY about:
- Unit scale inference (suggested scale factor, NOT applied)
- Origin placement suggestions (rebasing metadata, NOT applied)
- Confidence levels and reasoning

IMPORTANT SAFETY GUARANTEES:
- This module is completely OPTIONAL and ISOLATED
- Zero geometry mutations
- Zero side effects on the main application
- No automatic transformations
- All results are READ-ONLY metadata
- Future application requires explicit user confirmation
  (to be implemented separately in the main UI)

Usage:
    from obj_import_analyzer import analyze_obj_file, report_text
    
    result = analyze_obj_file("model.obj")
    print(report_text(result))
    print(result.to_metadata_dict())

"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BoundingBox:
    """Axis-aligned bounding box from raw vertex data."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    
    @property
    def size_x(self) -> float:
        """Size in X dimension."""
        return self.max_x - self.min_x
    
    @property
    def size_y(self) -> float:
        """Size in Y dimension."""
        return self.max_y - self.min_y
    
    @property
    def size_z(self) -> float:
        """Size in Z dimension."""
        return self.max_z - self.min_z
    
    @property
    def min_corner(self) -> Tuple[float, float, float]:
        """Minimum corner coordinates."""
        return (self.min_x, self.min_y, self.min_z)
    
    @property
    def max_corner(self) -> Tuple[float, float, float]:
        """Maximum corner coordinates."""
        return (self.max_x, self.max_y, self.max_z)


@dataclass
class UnitInferenceResult:
    """Results from unit scale inference heuristic."""
    suggested_scale_factor: float
    """Multiply vertices by this to convert to meters (NOT applied automatically)."""
    
    suggested_unit_label: str
    """Human-readable unit name (e.g., 'meters', 'centimeters', 'millimeters', 'feet', 'inches')."""
    
    confidence: float
    """0.0 (very uncertain) to 1.0 (very confident)."""
    
    reasoning: str
    """Explanation of why this unit was inferred."""


@dataclass
class ObjImportAnalysisResult:
    """Complete analysis of an imported OBJ file."""
    
    filepath: str
    """Path to the analyzed OBJ file."""
    
    vertex_count: int
    """Number of vertices parsed from the file."""
    
    face_count: int
    """Number of faces parsed from the file."""
    
    bounding_box: BoundingBox
    """Axis-aligned bounding box of raw vertices."""
    
    unit_inference: UnitInferenceResult
    """Suggested unit scale (metadata only - NOT applied)."""
    
    suggested_origin_offset: Tuple[float, float, float]
    """Offset to move min corner to (0,0,0) - metadata only, NOT applied.
    This is the vector you would SUBTRACT from all vertices to rebase.
    """
    
    notes: str
    """Additional notes or warnings."""
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        """Export analysis as a dictionary for programmatic access."""
        return {
            'filepath': self.filepath,
            'vertex_count': self.vertex_count,
            'face_count': self.face_count,
            'bbox': {
                'min_x': self.bounding_box.min_x,
                'max_x': self.bounding_box.max_x,
                'min_y': self.bounding_box.min_y,
                'max_y': self.bounding_box.max_y,
                'min_z': self.bounding_box.min_z,
                'max_z': self.bounding_box.max_z,
                'size_x': self.bounding_box.size_x,
                'size_y': self.bounding_box.size_y,
                'size_z': self.bounding_box.size_z,
            },
            'unit_inference': {
                'suggested_scale_factor': self.unit_inference.suggested_scale_factor,
                'suggested_unit_label': self.unit_inference.suggested_unit_label,
                'confidence': self.unit_inference.confidence,
                'reasoning': self.unit_inference.reasoning,
            },
            'suggested_origin_offset': self.suggested_origin_offset,
            'notes': self.notes,
        }


# ============================================================================
# UNIT INFERENCE HEURISTIC
# ============================================================================

def infer_unit_scale(size_x: float, size_y: float, size_z: float = 0.0) -> UnitInferenceResult:
    """
    Heuristic function to infer the likely unit scale of raw OBJ vertices.
    
    IMPORTANT: This function ONLY PROVIDES METADATA. It does NOT modify any geometry.
    
    Args:
        size_x: Bounding box extent in X dimension
        size_y: Bounding box extent in Y dimension
        size_z: Bounding box extent in Z dimension (optional)
    
    Returns:
        UnitInferenceResult with suggested scale factor and unit label.
        The scale factor is NOT applied automatically.
    
    Heuristic Logic:
    ----------------
    BIM software typical output ranges:
    - Revit: Feet (0.3-3000 units) or Millimeters (300-3,000,000 units)
    - SketchUp: Inches (12-120,000 units) or Feet (1-10,000 units)
    - Rhino: Millimeters (10-100,000 units) or Inches (1-10,000 units)
    - AutoCAD: Feet (1-50,000 units) or Millimeters (300-5,000,000 units)
    
    Real-world floor dimensions:
    - Single tile: 0.3-1.0 meters (12-40 inches)
    - Room: 3-100 meters (10-300 feet)
    - Building floor: 10-500 meters (30-1500 feet)
    
    The heuristic uses typical horizontal dimensions (size_x, size_y) as primary
    indicators since vertical variation is often compressed in BIM models.
    """
    # Use dominant horizontal dimension
    horizontal_dims = [d for d in [size_x, size_y] if d > 0]
    if not horizontal_dims:
        # Degenerate case: assume meters
        return UnitInferenceResult(
            suggested_scale_factor=1.0,
            suggested_unit_label="meters (default - degenerate geometry)",
            confidence=0.2,
            reasoning="No valid horizontal dimensions detected. Defaulting to meters as fallback."
        )
    
    avg_horizontal = sum(horizontal_dims) / len(horizontal_dims)
    max_horizontal = max(horizontal_dims)
    
    # Conservative thresholds based on typical room sizes in different units
    if avg_horizontal < 0.05:
        # Tiny model: possibly scaled-down representation
        return UnitInferenceResult(
            suggested_scale_factor=1.0,
            suggested_unit_label="meters",
            confidence=0.5,
            reasoning=(
                f"Detected tiny geometry (avg horizontal: {avg_horizontal:.6f} units). "
                f"Could be a scaled-down model or unusual coordinate system. "
                f"Defaulting to meters. Verify manually."
            )
        )
    
    elif avg_horizontal < 0.5:
        # Small model: likely centimeters or decimeters
        return UnitInferenceResult(
            suggested_scale_factor=0.01,
            suggested_unit_label="centimeters",
            confidence=0.75,
            reasoning=(
                f"Detected small geometry (avg horizontal: {avg_horizontal:.4f} units). "
                f"Typical of centimeter-scaled BIM exports. "
                f"Scale factor: 0.01× (1 cm = 0.01 m). "
                f"Confidence: 75%."
            )
        )
    
    elif avg_horizontal < 2.0:
        # Small-to-medium: likely meters or decimeters
        return UnitInferenceResult(
            suggested_scale_factor=1.0,
            suggested_unit_label="meters",
            confidence=0.85,
            reasoning=(
                f"Detected small-to-medium geometry (avg horizontal: {avg_horizontal:.2f} units). "
                f"Typical of meter-scaled models (room interiors). "
                f"Scale factor: 1.0× (already in meters). "
                f"Confidence: 85% (high)."
            )
        )
    
    elif avg_horizontal < 15.0:
        # Medium model: could be meters, feet, or decimeters
        # Use aspect ratio to disambiguate
        if max_horizontal / avg_horizontal > 4:
            # Very unbalanced: possibly feet or inches
            return UnitInferenceResult(
                suggested_scale_factor=0.3048,
                suggested_unit_label="feet",
                confidence=0.70,
                reasoning=(
                    f"Detected medium, unbalanced geometry (avg: {avg_horizontal:.2f}, max: {max_horizontal:.2f}). "
                    f"Unbalanced dimensions suggest feet-scaled BIM export. "
                    f"Scale factor: 0.3048× (1 ft = 0.3048 m). "
                    f"Confidence: 70% (verify aspect ratio)."
                )
            )
        else:
            # Balanced medium model: likely meters
            return UnitInferenceResult(
                suggested_scale_factor=1.0,
                suggested_unit_label="meters",
                confidence=0.80,
                reasoning=(
                    f"Detected medium, balanced geometry (avg horizontal: {avg_horizontal:.2f} units). "
                    f"Likely large floor or small building in meters. "
                    f"Scale factor: 1.0× (already in meters). "
                    f"Confidence: 80%."
                )
            )
    
    elif avg_horizontal < 100.0:
        # Large model: could be feet, large meters, or decimeters
        if max_horizontal / avg_horizontal > 3:
            # Unbalanced: possibly feet from large CAD drawing
            return UnitInferenceResult(
                suggested_scale_factor=0.3048,
                suggested_unit_label="feet",
                confidence=0.65,
                reasoning=(
                    f"Detected large, unbalanced geometry (avg: {avg_horizontal:.2f}). "
                    f"Likely feet-scaled CAD (building floor plan). "
                    f"Scale factor: 0.3048× (1 ft = 0.3048 m). "
                    f"Confidence: 65% (CHECK - BIM software unit setting)."
                )
            )
        else:
            # Balanced large: likely meters or feet
            return UnitInferenceResult(
                suggested_scale_factor=1.0,
                suggested_unit_label="meters",
                confidence=0.75,
                reasoning=(
                    f"Detected large, balanced geometry (avg horizontal: {avg_horizontal:.2f} units). "
                    f"Could be meters (building) or feet (scaled CAD). "
                    f"Guessing meters (more common in modern BIM). "
                    f"Scale factor: 1.0×. Confidence: 75% (VERIFY in CAD software)."
                )
            )
    
    elif avg_horizontal < 1000.0:
        # Very large: likely millimeters, inches, or feet from BIG drawing
        if avg_horizontal > 300:
            # Over 300: probably millimeters
            return UnitInferenceResult(
                suggested_scale_factor=0.001,
                suggested_unit_label="millimeters",
                confidence=0.75,
                reasoning=(
                    f"Detected very large geometry (avg horizontal: {avg_horizontal:.0f} units). "
                    f"Likely millimeter-scaled BIM (Revit, Rhino). "
                    f"Scale factor: 0.001× (1 mm = 0.001 m). "
                    f"Confidence: 75%."
                )
            )
        else:
            # 100-300: ambiguous, guess inches
            return UnitInferenceResult(
                suggested_scale_factor=0.0254,
                suggested_unit_label="inches",
                confidence=0.60,
                reasoning=(
                    f"Detected large geometry (avg horizontal: {avg_horizontal:.0f} units). "
                    f"Could be inches (SketchUp, some Revit) or decimeters. "
                    f"Guessing inches. Scale factor: 0.0254× (1 in = 0.0254 m). "
                    f"Confidence: 60% (MUST verify - critical decision)."
                )
            )
    
    else:
        # Extremely large: almost certainly millimeters
        return UnitInferenceResult(
            suggested_scale_factor=0.001,
            suggested_unit_label="millimeters",
            confidence=0.85,
            reasoning=(
                f"Detected extremely large geometry (avg horizontal: {avg_horizontal:.0f} units). "
                f"Almost certainly millimeter-scaled from BIM software. "
                f"Scale factor: 0.001× (1 mm = 0.001 m). "
                f"Confidence: 85% (high)."
            )
        )


# ============================================================================
# ORIGIN REBASING HELPERS
# ============================================================================

def rebase_to_min_corner(vertices: List[Tuple[float, float, float]]) -> Tuple[List[Tuple[float, float, float]], Tuple[float, float, float]]:
    """
    Compute rebased vertices (and offset vector) to move min corner to (0, 0, 0).
    
    IMPORTANT: This function does NOT modify the input vertices list.
    It returns a NEW list with rebased coordinates as METADATA for inspection.
    The rebasing is NOT applied to the original mesh.
    
    FUTURE STEP (requires user confirmation in UI):
    Only apply this rebasing after user explicitly confirms in the UI.
    
    Args:
        vertices: List of (x, y, z) tuples (not modified)
    
    Returns:
        (rebased_vertices_copy, offset_vector)
        - rebased_vertices_copy: New list with vertices shifted so min corner is at (0,0,0)
        - offset_vector: The (dx, dy, dz) offset that was subtracted
    """
    if not vertices:
        return [], (0.0, 0.0, 0.0)
    
    # Find min corner
    min_x = min(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    min_z = min(v[2] for v in vertices)
    
    offset_vector = (min_x, min_y, min_z)
    
    # Create rebased copy (not modifying original)
    rebased = [
        (v[0] - min_x, v[1] - min_y, v[2] - min_z)
        for v in vertices
    ]
    
    return rebased, offset_vector


# ============================================================================
# STREAM-BASED OBJ PARSER
# ============================================================================

def _parse_obj_file_streamed(filepath: str) -> Tuple[int, int, List[Tuple[float, float, float]]]:
    """
    Stream-read an OBJ file to extract vertices without loading entire file in memory.
    
    Args:
        filepath: Path to OBJ file
    
    Returns:
        (vertex_count, face_count, vertices_list)
    
    Robustly handles:
    - Comments (# lines)
    - Vertex lines (v x y z)
    - Vertex data with or without w component
    - Non-vertex lines (vn, vt, f, g, o, usemtl, mtllib, etc.)
    - Malformed lines (skipped with warning)
    """
    vertices = []
    vertex_count = 0
    face_count = 0
    line_num = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                token = parts[0]
                
                # Parse vertex line
                if token == 'v':
                    try:
                        # Extract x, y, z (ignore optional w component)
                        if len(parts) >= 4:
                            x = float(parts[1])
                            y = float(parts[2])
                            z = float(parts[3])
                            vertices.append((x, y, z))
                            vertex_count += 1
                        else:
                            # Warn but continue
                            pass  # Not enough components, skip
                    except (ValueError, IndexError):
                        # Malformed vertex line, skip
                        pass
                
                # Count faces
                elif token == 'f':
                    face_count += 1
    
    except FileNotFoundError:
        raise IOError(f"OBJ file not found: {filepath}")
    except Exception as e:
        raise IOError(f"Error reading OBJ file: {str(e)}")
    
    return vertex_count, face_count, vertices


def _compute_bounding_box(vertices: List[Tuple[float, float, float]]) -> BoundingBox:
    """
    Compute axis-aligned bounding box from vertices.
    
    Args:
        vertices: List of (x, y, z) tuples
    
    Returns:
        BoundingBox object
    """
    if not vertices:
        return BoundingBox(0, 0, 0, 0, 0, 0)
    
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    
    return BoundingBox(
        min_x=min(xs),
        max_x=max(xs),
        min_y=min(ys),
        max_y=max(ys),
        min_z=min(zs),
        max_z=max(zs),
    )


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_obj_file(filepath: str) -> ObjImportAnalysisResult:
    """
    Analyze an OBJ file and return metadata about unit scale and origin.
    
    IMPORTANT: This is a READ-ONLY analysis. No geometry is modified.
    Results are suggestions only (metadata).
    
    Args:
        filepath: Path to OBJ file
    
    Returns:
        ObjImportAnalysisResult with complete analysis metadata
    
    Raises:
        IOError: If file cannot be read or parsed
    """
    filepath = str(filepath)
    
    if not os.path.exists(filepath):
        raise IOError(f"File not found: {filepath}")
    
    # Parse OBJ file (stream-based, memory-efficient)
    vertex_count, face_count, vertices = _parse_obj_file_streamed(filepath)
    
    if vertex_count == 0:
        raise IOError(f"No vertices found in OBJ file: {filepath}")
    
    # Compute bounding box
    bbox = _compute_bounding_box(vertices)
    
    # Infer unit scale
    unit_inference = infer_unit_scale(bbox.size_x, bbox.size_y, bbox.size_z)
    
    # Compute suggested origin offset
    _, origin_offset = rebase_to_min_corner(vertices)
    
    # Build notes
    notes = ""
    if face_count == 0:
        notes += "Warning: No faces found (vertices only). "
    if unit_inference.confidence < 0.7:
        notes += "Warning: Low confidence in unit inference. Verify manually in original CAD software. "
    
    return ObjImportAnalysisResult(
        filepath=filepath,
        vertex_count=vertex_count,
        face_count=face_count,
        bounding_box=bbox,
        unit_inference=unit_inference,
        suggested_origin_offset=origin_offset,
        notes=notes.strip(),
    )


# ============================================================================
# REPORTING
# ============================================================================

def report_text(result: ObjImportAnalysisResult) -> str:
    """
    Generate a human-readable report of analysis findings.
    
    This is the output that would be shown to a user in the UI.
    All values are SUGGESTIONS and metadata only - nothing is applied.
    
    Args:
        result: ObjImportAnalysisResult from analyze_obj_file()
    
    Returns:
        Formatted string report
    """
    lines = []
    
    lines.append("=" * 90)
    lines.append("OBJ IMPORT ANALYSIS REPORT".center(90))
    lines.append("=" * 90)
    lines.append("")
    
    # File info
    lines.append("FILE INFORMATION")
    lines.append("-" * 90)
    lines.append(f"  Path:                     {result.filepath}")
    lines.append(f"  Vertices:                 {result.vertex_count:,}")
    lines.append(f"  Faces:                    {result.face_count:,}")
    lines.append("")
    
    # Raw geometry statistics (unitless)
    lines.append("RAW GEOMETRY (Unitless Coordinates)")
    lines.append("-" * 90)
    lines.append(f"  Bounding Box Min Corner:  ({result.bounding_box.min_x:.6f}, {result.bounding_box.min_y:.6f}, {result.bounding_box.min_z:.6f})")
    lines.append(f"  Bounding Box Max Corner:  ({result.bounding_box.max_x:.6f}, {result.bounding_box.max_y:.6f}, {result.bounding_box.max_z:.6f})")
    lines.append(f"  Size (X × Y × Z):         {result.bounding_box.size_x:.6f} × {result.bounding_box.size_y:.6f} × {result.bounding_box.size_z:.6f}")
    lines.append("")
    
    # Unit inference (METADATA ONLY - NOT APPLIED)
    lines.append("UNIT SCALE INFERENCE (SUGGESTION ONLY - NOT APPLIED)")
    lines.append("-" * 90)
    lines.append(f"  Suggested Unit:           {result.unit_inference.suggested_unit_label}")
    lines.append(f"  Scale Factor:             {result.unit_inference.suggested_scale_factor}×")
    lines.append(f"    (Multiply all vertices by this to convert to meters)")
    lines.append(f"  Confidence:               {result.unit_inference.confidence * 100:.0f}%")
    lines.append("")
    lines.append("  Reasoning:")
    for reasoning_line in result.unit_inference.reasoning.split('\n'):
        lines.append(f"    {reasoning_line}")
    lines.append("")
    
    # FUTURE STEP notice for scale application
    lines.append("  FUTURE STEP (requires explicit user confirmation in UI):")
    lines.append("    Apply this scale factor to vertices to normalize to meters.")
    lines.append("    This will NOT happen automatically.")
    lines.append("")
    
    # Origin rebasing (METADATA ONLY - NOT APPLIED)
    lines.append("ORIGIN REBASING SUGGESTION (METADATA ONLY - NOT APPLIED)")
    lines.append("-" * 90)
    lines.append(f"  Current Min Corner:       ({result.bounding_box.min_x:.6f}, {result.bounding_box.min_y:.6f}, {result.bounding_box.min_z:.6f})")
    lines.append(f"  Offset to Apply:          ({result.suggested_origin_offset[0]:.6f}, {result.suggested_origin_offset[1]:.6f}, {result.suggested_origin_offset[2]:.6f})")
    lines.append(f"    (Subtract this offset from all vertices to move min corner to (0,0,0))")
    lines.append("")
    
    # FUTURE STEP notice for origin rebasing
    lines.append("  FUTURE STEP (requires explicit user confirmation in UI):")
    lines.append("    Apply this offset to rebase the geometry origin.")
    lines.append("    This will NOT happen automatically.")
    lines.append("")
    
    # Notes/warnings
    if result.notes:
        lines.append("WARNINGS / NOTES")
        lines.append("-" * 90)
        for note in result.notes.split('\n'):
            lines.append(f"  ⚠ {note}")
        lines.append("")
    
    # Summary
    lines.append("=" * 90)
    lines.append("SUMMARY")
    lines.append("=" * 90)
    lines.append(f"  This analysis is READ-ONLY metadata. No geometry has been modified.")
    lines.append(f"  All suggested transformations (scale, origin) require user confirmation")
    lines.append(f"  in the main UI before being applied.")
    lines.append("")
    lines.append(f"  Next steps:")
    lines.append(f"    1. Review the analysis above")
    lines.append(f"    2. Verify the suggested unit by checking your CAD software settings")
    lines.append(f"    3. In the main UI, confirm or override the scale factor")
    lines.append(f"    4. Confirm whether to apply origin rebasing")
    lines.append(f"    5. Only THEN will geometry be modified")
    lines.append("=" * 90)
    
    return "\n".join(lines)


# ============================================================================
# COMMAND-LINE TEST INTERFACE
# ============================================================================

def main():
    """
    Command-line test interface for analyzing OBJ files.
    
    Usage:
        python obj_import_analyzer.py path/to/file.obj
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python obj_import_analyzer.py <path/to/file.obj>")
        print("")
        print("Example:")
        print("  python obj_import_analyzer.py model.obj")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    try:
        print(f"Analyzing: {filepath}")
        print("")
        
        result = analyze_obj_file(filepath)
        print(report_text(result))
        
        print("\n" + "=" * 90)
        print("METADATA DICTIONARY (for programmatic access):")
        print("=" * 90)
        import json
        metadata = result.to_metadata_dict()
        print(json.dumps(metadata, indent=2))
        
    except IOError as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()