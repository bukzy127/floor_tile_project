def generate_qr_png(tile_id, tile_data, path, size=200):
    """Generate a QR code with a simple, readable URL format.

    Args:
        tile_id: Tile identifier (e.g., "T-0-10")
        tile_data: Dict with tile properties
        path: Output file path
        size: Image size in pixels
    """
    # Create a simple, readable text URL
    # Format: ID|Material|Thickness|Density|Weight|RValue
    qr_text = f"{tile_id}|{tile_data.get('material', 'N/A')}|{tile_data.get('thickness', 'N/A')}|{tile_data.get('density', 'N/A')}|{tile_data.get('weight', 'N/A')}|{tile_data.get('thermal_r', 'N/A')}"

    try:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=2,
        )
        qr.add_data(qr_text)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img = img.resize((size, size), Image.NEAREST)
        img.save(path)
    except Exception as e:
        print(f"Warning: QR code generation failed for {tile_id}: {e}")
import sys
import math
import json
try:
    import ezdxf
except ImportError:
    ezdxf = None
    print("ERROR: ezdxf not installed in this environment")
    print("Run:  .venv\\Scripts\\activate && pip install ezdxf")
try:
    import trimesh
except ImportError:
    trimesh = None
    print("Warning: trimesh not installed. 3D model import will be unavailable.")
try:
    import trimesh
except ImportError:
    trimesh = None
    print("Warning: trimesh not installed. 3D model import will be unavailable.")
import numpy as np
from pathlib import Path
from datetime import datetime
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QDialog,
    QLabel, QFormLayout, QPushButton, QDoubleSpinBox, QGroupBox,
    QScrollArea, QSplitter, QMessageBox
)
from PyQt6.QtCore import QPointF
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat, QVector3D, QMatrix4x4, QQuaternion, QVector2D
from OpenGL.GL import *
from OpenGL.GLU import *
import os, faulthandler
import qrcode
from PIL import Image
import io
faulthandler.enable()

# Import custom modules for 3D model loading and surface selection
try:
    from model_loader import load_model, normalize_mesh_units, ModelLoadError
    from surface_selector import SurfaceSelector, SelectedSurface
    MODEL_IMPORT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Model import modules not available: {e}")
    MODEL_IMPORT_AVAILABLE = False
# Workaround for macOS Qt5 + QOpenGLWidget crashes
os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
os.environ.setdefault("QT_OPENGL", "desktop")

EPSILON = 1e-6                   # Small value for float comparisons
PICK_EPSILON = 1e-5              # Tolerance used during mouse picking tests
# -----------------------------------------------------------------------------
# Helper: simple 2D polygon area
def polygon_area_2d(poly):
    if not poly or len(poly) < 3:
        return 0.0
    x = [p[0] for p in poly]
    y = [p[1] for p in poly]
    return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(poly)-1)))

def polygon_centroid_2d(poly):
    """Return the centroid of a simple 2D polygon."""
    if not poly or len(poly) < 3:
        return None
    signed_area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        cross = x0 * y1 - x1 * y0
        signed_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    signed_area *= 0.5
    if abs(signed_area) < EPSILON:
        avg_x = sum(p[0] for p in poly) / len(poly)
        avg_y = sum(p[1] for p in poly) / len(poly)
        return (avg_x, avg_y)
    cx /= (6.0 * signed_area)
    cy /= (6.0 * signed_area)
    return (cx, cy)

def point_on_segment_2d(pt, a, b, tolerance=0.0):
    """Return True if pt lies within tolerance of the line segment AB."""
    ax, ay = a
    bx, by = b
    px, py = pt
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < EPSILON:
        return math.hypot(apx, apy) <= tolerance
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(px - closest_x, py - closest_y) <= tolerance

def point_in_polygon(pt, poly, tolerance=0.0):
    if not poly or len(poly) < 3:
        return False
    x, y = pt
    inside = False
    n = len(poly)
    # Standard ray-casting algorithm, robust against horizontal edges
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if tolerance > 0.0 and point_on_segment_2d((x, y), (xi, yi), (xj, yj), tolerance):
            return True
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside

# -----------------------------------------------------------------------------
# Sutherland-Hodgman polygon clipping (clip subject polygon by convex clip polygon)
def sutherland_hodgman_clip(subject_poly, clip_poly):
    def inside(p, edge_start, edge_end):
        # return True if p is on the left side of edge (edge_start->edge_end)
        return ((edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) - (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0])) >= -EPSILON
    def compute_intersection(s, e, cp1, cp2):
        # intersection of segment s->e with clip edge cp1->cp2
        x1,y1 = s; x2,y2 = e
        x3,y3 = cp1; x4,y4 = cp2
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < EPSILON:
            return e
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
        return (px, py)
    output_list = subject_poly
    cp_count = len(clip_poly)
    for i in range(cp_count):
        input_list = output_list
        output_list = []
        if not input_list:
            break
        cp1 = clip_poly[i]
        cp2 = clip_poly[(i+1) % cp_count]
        s = input_list[-1]
        for e in input_list:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    inter = compute_intersection(s, e, cp1, cp2)
                    output_list.append(inter)
                output_list.append(e)
            elif inside(s, cp1, cp2):
                inter = compute_intersection(s, e, cp1, cp2)
                output_list.append(inter)
            s = e
    return output_list

# -----------------------------------------------------------------------------
class Tile3D:
    def __init__(self, origin_xy, xtile, ytile, thickness,
                 is_cut=False, cut_polygon_xy=None, qr_data=None):
        self.origin_xy = origin_xy
        self.xtile = xtile
        self.ytile = ytile
        self.thickness = thickness
        self.is_cut = is_cut
        self.cut_polygon_xy = cut_polygon_xy
        self.qr_data = qr_data
        self.qr_texture_id = None
        self.qr_size = None
        self.pedestal_list = []
        self.corners_bottom_xyz = []
        self.corners_top_xyz = []
      # Cached picking helpers (set by prepare_pick_geometry)
        self.pick_origin = None
        self.pick_normal = None
        self.pick_u_axis = None
        self.pick_v_axis = None
        self.pick_polygon2d = []
        self.pick_bounds = None

    def get_actual_xy_footprint(self):
        if self.cut_polygon_xy and len(self.cut_polygon_xy) >=3:
            return self.cut_polygon_xy
        return [] # Return empty list if no valid footprint

    def compute_3d_corners(self, tile_bottom_z_level):
        footprint_xy = self.get_actual_xy_footprint()
        if not footprint_xy: return

        self.corners_bottom_xyz = []
        self.corners_top_xyz = []
        for (x, y) in footprint_xy:
            z_bottom = tile_bottom_z_level
            z_top = z_bottom + self.thickness
            self.corners_bottom_xyz.append(QVector3D(x, y, z_bottom))
            self.corners_top_xyz.append(QVector3D(x, y, z_top))
            self.prepare_pick_geometry()

    def prepare_pick_geometry(self):
        """Pre-compute an orthonormal basis used during mouse picking."""
        self.pick_origin = None
        self.pick_normal = None
        self.pick_u_axis = None
        self.pick_v_axis = None
        self.pick_polygon2d = []
        self.pick_bounds = None
        if not self.corners_top_xyz or len(self.corners_top_xyz) < 3:
            return
        # Use the centroid of the polygon as the origin for better numerical stability.
        centroid = QVector3D(0.0, 0.0, 0.0)
        for p in self.corners_top_xyz:
            centroid += p
        centroid /= float(len(self.corners_top_xyz))
        # Estimate plane normal from the polygon fan.
        plane_normal = QVector3D(0.0, 0.0, 0.0)
        for i in range(len(self.corners_top_xyz)):
            p_curr = self.corners_top_xyz[i] - centroid
            p_next = self.corners_top_xyz[(i + 1) % len(self.corners_top_xyz)] - centroid
            plane_normal += QVector3D.crossProduct(p_curr, p_next)
        if plane_normal.lengthSquared() < EPSILON:
            return
        plane_normal.normalize()
        # Build a stable orthonormal basis on the plane using the longest edge.
        longest_edge = None
        longest_len_sq = -1.0
        for i in range(len(self.corners_top_xyz)):
            a = self.corners_top_xyz[i]
            b = self.corners_top_xyz[(i + 1) % len(self.corners_top_xyz)]
            edge = b - a
            length_sq = edge.lengthSquared()
            if length_sq > longest_len_sq and length_sq > EPSILON:
                longest_len_sq = length_sq
                longest_edge = edge
        if longest_edge is None:
            return
        u_axis = longest_edge.normalized()
        v_axis = QVector3D.crossProduct(plane_normal, u_axis)
        if v_axis.lengthSquared() < EPSILON:
            return
        v_axis.normalize()
        # Cache for picking.
        self.pick_origin = centroid
        self.pick_normal = plane_normal
        self.pick_u_axis = u_axis
        self.pick_v_axis = v_axis
        # Pre-project the polygon into the orthonormal basis.
        poly2d = []
        min_u = min_v = float('inf')
        max_u = max_v = float('-inf')
        for p in self.corners_top_xyz:
            rel = p - centroid
            u = QVector3D.dotProduct(rel, u_axis)
            v = QVector3D.dotProduct(rel, v_axis)
            poly2d.append((u, v))
            min_u = min(min_u, u)
            max_u = max(max_u, u)
            min_v = min(min_v, v)
            max_v = max(max_v, v)
        self.pick_polygon2d = poly2d
        self.pick_bounds = (min_u, max_u, min_v, max_v)
# -----------------------------------------------------------------------------
class InfoDialog(QDialog):
    def __init__(self, tile: Tile3D, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clicked Tile Information")
        layout = QFormLayout(self)
        layout.addRow("QR Data (ID):", QLabel(str(tile.qr_data or "N/A")))

        # Generate an inline QR image for the dialog (PIL -> QPixmap via bytes)
        try:
            tile_id = str(tile.qr_data or "TILE")
            # Get tile data from parent window if available
            tile_data = {
                'material': 'Unknown',
                'density': 0,
                'weight': 0,
                'thermal_r': 0,
                'thickness': f"{tile.thickness*1000:.1f}mm" if tile.thickness else "N/A"
            }
            if parent and hasattr(parent, 'density_in'):
                tile_data['material'] = parent.material_cb.currentText() if hasattr(parent, 'material_cb') else 'Tile'
                tile_data['density'] = int(parent.density_in.value())
                tile_data['weight'] = parent.weight_in.value()
                tile_data['thermal_r'] = parent.thermal_r_in.value()

            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=6,
                border=1,
            )
            # Create simple readable text format: ID|Material|Thickness|Density|Weight|RValue
            qr_text = f"{tile_id}|{tile_data.get('material', 'N/A')}|{tile_data.get('thickness', 'N/A')}|{tile_data.get('density', 'N/A')}|{tile_data.get('weight', 'N/A')}|{tile_data.get('thermal_r', 'N/A')}"

            qr.add_data(qr_text)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white")
            if hasattr(qr_img, 'get_image'):
                qr_img = qr_img.get_image()
            qr_img = qr_img.convert('RGBA')
            # Save to bytes
            bio = io.BytesIO()
            qr_img.save(bio, format='PNG')
            bio.seek(0)
            from PyQt6.QtGui import QPixmap
            pix = QPixmap()
            pix.loadFromData(bio.getvalue(), 'PNG')
            # Scale pixmap to reasonable dialog size
            pix = pix.scaled(128, 128, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            img_label = QLabel()
            img_label.setPixmap(pix)
            layout.addRow("QR Code:", img_label)
        except Exception:
            # If QR generation fails, skip image silently
            pass

        footprint = tile.get_actual_xy_footprint()
        origin_display, actual_dims_display = "N/A", "N/A"

        if footprint:
            min_x = min(p[0] for p in footprint); max_x = max(p[0] for p in footprint)
            min_y = min(p[1] for p in footprint); max_y = max(p[1] for p in footprint)
            origin_display = f"{tile.origin_xy[0]:.3f}, {tile.origin_xy[1]:.3f}"
            actual_dims_display = f"W:{(max_x - min_x):.3f} × L:{(max_y - min_y):.3f}"

        layout.addRow("Logical XY Origin:", QLabel(origin_display))
        layout.addRow("Original Full Tile Size (W×L):", QLabel(f"{tile.xtile:.3f} × {tile.ytile:.3f}"))
        layout.addRow("Actual Footprint BBox (W×L):", QLabel(actual_dims_display))
        layout.addRow("Is Cut Tile:", QLabel(str(tile.is_cut)))
        layout.addRow("Thickness:", QLabel(f"{tile.thickness:.3f}"))

        if tile.corners_bottom_xyz and tile.corners_top_xyz and tile.corners_bottom_xyz[0] is not None :
            bottom_z = tile.corners_bottom_xyz[0].z()
            top_z = tile.corners_top_xyz[0].z()
            layout.addRow("Bottom Z (top of pedestal):", QLabel(f"{bottom_z:.3f}"))
            layout.addRow("Top Z (tile surface):", QLabel(f"{top_z:.3f}"))

        close_button = QPushButton("Close"); close_button.clicked.connect(self.accept)
        layout.addRow(close_button)
        self.setMinimumWidth(350)

# -----------------------------------------------------------------------------
class RoomInputHandler:
    """Handles room input modes: manual width/length, polygon points, file import."""
    def __init__(self):
        self.mode = 'manual'  # 'manual'|'polygon'|'file'
        self.polygon = []
        self.file_path = None

    def set_manual(self):
        self.mode = 'manual'; self.polygon = []; self.file_path = None

    def set_polygon(self, poly_points):
        # Expect list of (x,y) tuples
        self.mode = 'polygon'; self.polygon = poly_points; self.file_path = None

    def load_from_file(self, path):
        # Accept simple CSV or whitespace separated lines with x,y (optionally z)
        pts = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln: continue
                    parts = [p for p in ln.replace(',', ' ').split() if p]
                    if len(parts) < 2: continue
                    try:
                        x = float(parts[0]); y = float(parts[1])
                        pts.append((x, y))
                    except ValueError:
                        continue
            if len(pts) < 3:
                raise ValueError('Polygon must contain at least 3 points')
            # ensure closed
            if pts[0] != pts[-1]: pts.append(pts[0])
            self.mode = 'polygon'; self.polygon = pts; self.file_path = path
            return True, None
        except Exception as e:
            return False, str(e)

    def get_polygon(self):
        return self.polygon

# -----------------------------------------------------------------------------
class ElevationModel:
    """Supports flat, planar slope, and irregular map (IDW interpolation)"""
    def __init__(self):
        self.mode = 'flat'  # 'flat'|'planar'|'irregular'
        self.flat_z = 0.0
        self.base_z = 0.0
        self.slope_x = 0.0
        self.slope_y = 0.0
        self.points = []  # list of (x,y,z) for irregular

    def z_at(self, x, y):
        if self.mode == 'flat':
            return self.flat_z
        elif self.mode == 'planar':
            return self.base_z + self.slope_x * x + self.slope_y * y
        else:
            # irregular: use inverse distance weighting with nearest 4
            if not self.points:
                return 0.0
            dists = []
            for px,py,pz in self.points:
                dx = x-px; dy = y-py; d = math.hypot(dx,dy)
                if d < EPSILON:
                    return pz
                dists.append((d,pz))
            dists.sort(key=lambda t: t[0])
            k = min(8, len(dists))
            num = 0.0; den = 0.0
            for i in range(k):
                w = 1.0 / (dists[i][0]**2 + 1e-12)
                num += w * dists[i][1]; den += w
            return num/den if den > 0 else 0.0

    def load_points_from_file(self, path):
        pts = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln: continue
                    parts = [p for p in ln.replace(',', ' ').split() if p]
                    if len(parts) < 3: continue
                    try:
                        x = float(parts[0]); y = float(parts[1]); z = float(parts[2])
                        pts.append((x,y,z))
                    except ValueError:
                        continue
            if not pts:
                raise ValueError('No valid elevation points found')
            self.points = pts; self.mode = 'irregular'
            return True, None
        except Exception as e:
            return False, str(e)

# -----------------------------------------------------------------------------
class GLWidget(QOpenGLWidget):
    tileClicked = QtCore.pyqtSignal(object)
    pedestalClicked = QtCore.pyqtSignal(object)  # Signal for pedestal clicks

    def __init__(self, parent=None):
        super().__init__(parent)
       
        self.tiles = []
        self.pedestals = []
        self.original_floor_mesh = []
        self.setMouseTracking(True)
        self.selected_tile_index = -1
        self.selected_pedestal_index = -1  # Track selected pedestal

        self.camera_azimuth = 45.0
        self.camera_elevation = 30.0
        self.camera_distance = 15.0
        self.camera_target = QVector3D(0.0, 0.0, 0.0)
        #self.last_mouse_pos = QtCore.QPoint()
        self.last_mouse_pos = QtCore.QPointF()
        # Start with no default room so the viewport is empty until the user computes a layout.
        self.room_center_for_rotation = QVector3D(0.0, 0.0, 0.0)

        # Initialize room dimensions to zero to avoid drawing a small default square in the view.
        self.room_dims = {'width': 0.0, 'length': 0.0, 'target_top_z': 0.0}
        self.original_slope_func = lambda x, y: 0.0
        self.unit_cylinder_dl = -1
        # Cached GL transforms captured during paint to avoid GL calls in event handlers
        self._cached_modelview = None
        self._cached_projection = None
        self._cached_viewport = None

        # New: Room and elevation models
        self.room_input = RoomInputHandler()
        self.elevation_model = ElevationModel()
        self.room_polygon_xy = []
        # 3D Model Import support
        self.imported_mesh = None
        self.surface_selection_mode = False
        self.selected_surface = None
        self.selected_surfaces = set()  # Multi-selection: set of face indices (highlighted in red)
        self.hovered_surface = None
        self.surface_selector = None  # Will be initialized when mesh is loaded

        # Floor detection for uneven-floor pedestal handling
        self.floor_triangles = []
        self._floor_vertices_flat = []
        self.ceiling_z = None  # Detected ceiling height from imported model
        self.has_imported_model = False  # Flag to prevent default floor rendering in import mode

        # Visualization toggles
        self.show_wireframe = False
        self.show_elevation_map = False
        self.show_tiles = True
        self.show_qr_codes = True 

        self.qr_pct = 0.10  # Default QR code size percent

        # Material texture support
        self.material_texture_id = None
        self.material_texture_path = None
        self.material_texture_size = (0, 0)

        # 3D Model Import support
        self.imported_mesh = None
        self.surface_selection_mode = False
        self.selected_surface = None
        self.selected_surfaces = set()  # Multi-selection: set of face indices (highlighted in red)
        self.hovered_surface = None

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [self.camera_distance, self.camera_distance, self.camera_distance, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        glEnable(GL_NORMALIZE)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glClearColor(0.85, 0.85, 0.95, 1.0)
        self.unit_cylinder_dl = -1

    def load_material_texture(self, image_path):
        """Load a seamless material texture from PNG or JPG file."""
        try:
            # Open image with PIL
            img = Image.open(image_path)
            img = img.convert('RGB')  # Ensure RGB format

            # Store original size for seamless tiling calculations
            original_width, original_height = img.size

            # Resize to power of 2 for better OpenGL compatibility (max 2048x2048)
            max_size = 2048
            if original_width > max_size or original_height > max_size:
                scale_factor = min(max_size / original_width, max_size / original_height)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                img = img.resize((new_width, new_height), Image.LANCZOS)

            # Flip image for OpenGL coordinate system
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = img.tobytes()
            img_width, img_height = img.size

            # Ensure OpenGL context is current
            self.makeCurrent()

            # Delete old texture if exists
            if self.material_texture_id is not None:
                try:
                    glDeleteTextures([self.material_texture_id])
                except:
                    pass

            # Generate new texture
            texture_id = glGenTextures(1)
            if isinstance(texture_id, (list, tuple)):
                texture_id = texture_id[0]
            texture_id = int(texture_id)

            glBindTexture(GL_TEXTURE_2D, texture_id)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

            # Use GL_REPEAT for seamless tiling
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

            # Use mipmaps for better quality at different scales
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_width, img_height,
                        0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

            # Generate mipmaps for better performance
            try:
                from OpenGL.GL import glGenerateMipmap
                glGenerateMipmap(GL_TEXTURE_2D)
            except:
                # Fallback for older OpenGL
                pass

            # Store texture info
            self.material_texture_id = texture_id
            self.material_texture_path = image_path
            self.material_texture_size = (img_width, img_height)

            self.doneCurrent()

            return True, None

        except Exception as e:
            return False, str(e)

    def _update_camera_target(self, room_width, room_length, effective_tile_top_z):
        self.room_center_for_rotation = QVector3D(room_width / 2.0, room_length / 2.0, effective_tile_top_z / 2.0)
        if self.camera_distance <= 1.0 or self.camera_distance < max(room_width, room_length, abs(effective_tile_top_z)) +1.0 :
             self.camera_distance = max(room_width, room_length, abs(effective_tile_top_z), 5.0) * 1.75 + 1.0

    def resizeGL(self, w, h):
        if w <= 0 or h <= 0:
            return
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / max(h, 1), 0.1, max(50.0, self.camera_distance * 3.0))
        glMatrixMode(GL_MODELVIEW)

    def extract_floor_from_mesh(self, mesh):
        """Extract floor triangles from imported 3D mesh for uneven-floor pedestal support."""
        import numpy as np
        self.floor_triangles = []
        self._floor_vertices_flat = []
        
        if mesh is None or not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            return
        
        vertices = np.array(mesh.vertices, dtype=float)
        faces = np.array(mesh.faces, dtype=int)
        
        if len(vertices) == 0 or len(faces) == 0:
            return
        
        # Compute triangle data: centroids, normals, areas
        triangles = []
        for face in faces:
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
                'vertices': [v0, v1, v2],
                'centroid': centroid,
                'normal': normal,
                'area': area,
                'face': face
            })
        
        if not triangles:
            return
        
        # Filter triangles: facing +Z, non-trivial area, in lower Z band
        z_threshold = 0.85
        all_z = [t['centroid'][2] for t in triangles]
        min_z, max_z = min(all_z), max(all_z)
        z_range = max_z - min_z
        z_band_threshold = min_z + 0.35 * z_range if z_range > 1e-6 else min_z + 1.0
        
        floor_candidates = [
            t for t in triangles
            if t['normal'][2] >= z_threshold and t['area'] > 1e-6 and t['centroid'][2] <= z_band_threshold
        ]
        
        if not floor_candidates:
            return
        
        # Build connectivity via shared vertices
        from collections import defaultdict
        vertex_to_triangles = defaultdict(list)
        for i, t in enumerate(floor_candidates):
            for v_idx in t['face']:
                vertex_to_triangles[v_idx].append(i)
        
        # Find largest connected component using BFS
        visited = set()
        components = []
        for start_idx in range(len(floor_candidates)):
            if start_idx in visited:
                continue
            component = []
            queue = [start_idx]
            visited.add(start_idx)
            while queue:
                current = queue.pop(0)
                component.append(current)
                for v_idx in floor_candidates[current]['face']:
                    for neighbor_idx in vertex_to_triangles[v_idx]:
                        if neighbor_idx not in visited:
                            visited.add(neighbor_idx)
                            queue.append(neighbor_idx)
            components.append(component)
        
        if not components:
            return
        
        largest_component = max(components, key=len)
        self.floor_triangles = [floor_candidates[i] for i in largest_component]
        
        # Build flat list of floor vertices for fallback
        unique_verts = set()
        for t in self.floor_triangles:
            for v in t['vertices']:
                unique_verts.add(tuple(v))
        self._floor_vertices_flat = [np.array(v) for v in unique_verts]
        
        print(f"[Floor Detection] Extracted {len(self.floor_triangles)} floor triangles from mesh")

    def sample_floor_z(self, x, y):
        """Sample Z coordinate from detected floor triangles at (x, y) using barycentric interpolation."""
        import numpy as np
        
        # If no floor detected, fall back to original slope function
        if not self.floor_triangles:
            return self.original_slope_func(x, y)
        
        # Try barycentric containment on each triangle
        for tri in self.floor_triangles:
            v0, v1, v2 = tri['vertices']
            # 2D barycentric coords
            x0, y0, z0 = v0[0], v0[1], v0[2]
            x1, y1, z1 = v1[0], v1[1], v1[2]
            x2, y2, z2 = v2[0], v2[1], v2[2]
            
            denom = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
            if abs(denom) < 1e-9:
                continue
            
            w0 = ((y1 - y2)*(x - x2) + (x2 - x1)*(y - y2)) / denom
            w1 = ((y2 - y0)*(x - x2) + (x0 - x2)*(y - y2)) / denom
            w2 = 1.0 - w0 - w1
            
            # Check if point is inside triangle
            if w0 >= -1e-6 and w1 >= -1e-6 and w2 >= -1e-6:
                # Interpolate Z
                z = w0 * z0 + w1 * z1 + w2 * z2
                return z
        
        # Fallback: nearest floor vertex
        if self._floor_vertices_flat:
            min_dist_sq = float('inf')
            nearest_z = 0.0
            for v in self._floor_vertices_flat:
                dist_sq = (v[0] - x)**2 + (v[1] - y)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    nearest_z = v[2]
            return nearest_z
        
        # Final fallback
        return self.original_slope_func(x, y)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        rad_azimuth = math.radians(self.camera_azimuth)
        rad_elevation = math.radians(self.camera_elevation)

        eye_x = self.room_center_for_rotation.x() + self.camera_distance * math.cos(rad_elevation) * math.sin(rad_azimuth)
        eye_y = self.room_center_for_rotation.y() + self.camera_distance * math.cos(rad_elevation) * math.cos(rad_azimuth)
        eye_z = self.room_center_for_rotation.z() + self.camera_distance * math.sin(rad_elevation)
        eye_z = max(eye_z, self.room_center_for_rotation.z() * 0.1 + 0.01)

        gluLookAt(eye_x, eye_y, eye_z,
                  self.room_center_for_rotation.x(), self.room_center_for_rotation.y(), self.room_center_for_rotation.z(),
                  0, 0, 1)
        # Draw imported 3D model if available
        if self.imported_mesh is not None:
            self.draw_imported_mesh()


        # Cache current transforms while the context is current
        try:
            self._cached_modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            self._cached_projection = glGetDoublev(GL_PROJECTION_MATRIX)
            self._cached_viewport = glGetIntegerv(GL_VIEWPORT)
        except Exception:
            self._cached_modelview = self._cached_projection = self._cached_viewport = None

        if self.show_elevation_map:
            # Draw elevation heatmap under tiles
            self.draw_elevation_heatmap()
        else:
            self.draw_original_floor()

        # Draw room polygon boundary if available
        self.draw_room_boundary()

        for i, ped in enumerate(self.pedestals):
            self.draw_pedestal(ped, is_selected=(i == self.selected_pedestal_index))
        # Draw imported 3D model if available
        if self.imported_mesh is not None:
            self.draw_imported_mesh()

        if self.show_tiles:
            for i, t in enumerate(self.tiles):
                self.draw_tile(t, is_selected=(i == self.selected_tile_index))
                if self.show_qr_codes:
                    self.draw_tile_qr(t)
        self.draw_axes()

    def nudge_pedestal_inside_tile(self, ped_pos, ped_radius, footprint):
        """
        Ensure pedestal (circle center + radius) is fully inside the polygon footprint.
        If outside, move inward along direction toward polygon centroid by radius.
        """
        px, py = ped_pos

        # quick success path: if already inside by radius margin, keep
        if point_in_polygon((px, py), footprint, tolerance=ped_radius + 1e-6):
            min_dist = float('inf')
            for i in range(len(footprint)):
                a = footprint[i]
                b = footprint[(i + 1) % len(footprint)]
                ax, ay = a; bx, by = b
                seg_len = math.hypot(bx - ax, by - ay)
                if seg_len < EPSILON:
                    continue
                t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / (seg_len * seg_len)))
                cx = ax + t * (bx - ax); cy = ay + t * (by - ay)
                d = math.hypot(px - cx, py - cy)
                min_dist = min(min_dist, d)

            if min_dist >= ped_radius - 1e-6:
                return (px, py)

        # move toward centroid until inside
        cx, cy = polygon_centroid_2d(footprint)
        if cx is None:
            cx = sum(p[0] for p in footprint) / len(footprint)
            cy = sum(p[1] for p in footprint) / len(footprint)

        dir_x = px - cx
        dir_y = py - cy
        length = math.hypot(dir_x, dir_y)
        if length < EPSILON:
            return (px, py)

        dir_x /= length
        dir_y /= length

        attempt_pos = (px - dir_x * ped_radius,
                       py - dir_y * ped_radius)

        for k in range(10):
            if point_in_polygon(attempt_pos, footprint, tolerance=ped_radius):
                return attempt_pos
            attempt_pos = (attempt_pos[0] - dir_x * (ped_radius * 0.4),
                           attempt_pos[1] - dir_y * (ped_radius * 0.4))

        return (cx, cy)

    def compute_and_build_layout(self, room_params_input, tile_params, slope_params):
        """Main layout algorithm updated to support polygonal rooms and irregular elevation models.
        room_params_input should include keys:
          - mode: 'manual'|'polygon'|'file'
          - width, length, target_top_z (for manual)
          - polygon: list of (x,y) for polygon if provided
        slope_params should be considered only for planar mode; elevation model used otherwise.
        """
        self.tiles.clear(); self.pedestals.clear(); self.original_floor_mesh.clear()
        self.selected_tile_index = -1

        # Unpack params and configure elevation model
        room_mode = room_params_input.get('mode', 'manual')
        room_polygon = room_params_input.get('polygon', [])
        rw_param = room_params_input.get('width', 0.0)
        rl_param = room_params_input.get('length', 0.0)
        
        print(f"[DEBUG] Room setup: mode={room_mode}, width={rw_param:.3f}, length={rl_param:.3f}, polygon_points={len(room_polygon)}")

        # configure elevation model
        elev_mode = room_params_input.get('elevation_mode', 'flat')
        self.elevation_model.mode = elev_mode
        if elev_mode == 'flat':
            self.elevation_model.flat_z = room_params_input.get('flat_z', 0.0)
            self.original_slope_func = lambda x,y: self.elevation_model.z_at(x,y)
        elif elev_mode == 'planar':
            self.elevation_model.base_z = slope_params.get('base_z', 0.0)
            self.elevation_model.slope_x = slope_params.get('slope_x', 0.0)
            self.elevation_model.slope_y = slope_params.get('slope_y', 0.0)
            self.original_slope_func = lambda x,y: self.elevation_model.z_at(x,y)
        else:
            # irregular
            # assume elevation_model.points already loaded externally
            self.original_slope_func = lambda x,y: self.elevation_model.z_at(x,y)

        # Determine effective room polygon and bounds
        origin_x = 0.0; origin_y = 0.0
        
        # Check if we're in file/import mode without selected surfaces
        if room_mode == 'file' and not room_polygon:
            # File mode but no surfaces selected - don't generate default room
            print("[DEBUG] File/import mode active but no surfaces selected - skipping tile generation")
            self.tiles.clear()
            self.pedestals.clear()
            self.room_polygon_xy = []
            self.room_dims = {'width': 0.0, 'length': 0.0, 'target_top_z': 0.0}
            self.update()
            return
        
        if room_mode == 'manual' or not room_polygon:
            room_poly = [(0.0,0.0),(rw_param,0.0),(rw_param,rl_param),(0.0,rl_param)]
            area_room = rw_param * rl_param
            origin_x, origin_y = 0.0, 0.0
            self.room_polygon_xy = room_poly
        else:
            # keep polygon in world coordinates (do NOT shift)
            room_poly = list(room_polygon)
            if room_poly[0] != room_poly[-1]: room_poly.append(room_poly[0])
            area_room = polygon_area_2d(room_poly)
            if area_room <= EPSILON:
                QMessageBox.critical(self, "Invalid Polygon", "Provided polygon has zero area or is invalid.")
                return
            # derive bounding box
            xs = [p[0] for p in room_poly]; ys = [p[1] for p in room_poly]
            min_x, max_x = min(xs), max(xs); min_y, max_y = min(ys), max(ys)
            rw_param = max_x - min_x; rl_param = max_y - min_y
            # set origin to bounding box min so grid aligns with polygon
            origin_x, origin_y = min_x, min_y
            # store polygon in widget for drawing (world coords)
            self.room_polygon_xy = room_poly

        # Determine raised floor Z levels
        # When ceiling_z is available (from imported model), interpret room_height as HEADROOM (clearance from tile top to ceiling)
        # Otherwise, treat it as absolute tile top Z coordinate for backward compatibility
        headroom_or_height = room_params_input.get('room_height', room_params_input.get('target_top_z', 0.4))
        
        if self.ceiling_z is not None:
            # Headroom mode: tile_top_z = ceiling_z - headroom
            target_top_z = self.ceiling_z - headroom_or_height
            print(f"[DEBUG] Headroom mode: ceiling_z={self.ceiling_z:.3f}, headroom={headroom_or_height:.3f}, tile_top_z={target_top_z:.3f}")
        else:
            # Absolute mode (backward compatibility): tile_top_z = user input
            target_top_z = headroom_or_height
            print(f"[DEBUG] Absolute mode: tile_top_z={target_top_z:.3f} (no ceiling detected)")
        
        actual_tile_top_z = target_top_z
        actual_tile_bottom_z = target_top_z - tile_params['thickness']
        
        print(f"[DEBUG] Tile heights: target_top_z={target_top_z:.3f}, tile_bottom_z={actual_tile_bottom_z:.3f}, thickness={tile_params['thickness']:.3f}")

        self.room_dims = {'width': rw_param, 'length': rl_param, 'target_top_z': actual_tile_top_z}
        # update camera target using world center
        self._update_camera_target(origin_x + rw_param, origin_y + rl_param, actual_tile_top_z)

        # Generate tiles across bounding box and clip them to polygon (if polygonal room)
        initial_tiles = []
        num_x = math.ceil(rw_param / tile_params['width']) if tile_params['width'] > EPSILON else 0
        num_y = math.ceil(rl_param / tile_params['length']) if tile_params['length'] > EPSILON else 0

        for i in range(num_x):
            for j in range(num_y):
                x_orig = origin_x + i * tile_params['width']; y_orig = origin_y + j * tile_params['length']
                rect = [(x_orig, y_orig), (x_orig + tile_params['width'], y_orig), (x_orig + tile_params['width'], y_orig + tile_params['length']), (x_orig, y_orig + tile_params['length'])]

                if room_mode == 'manual' or not room_polygon:
                    # simple rectangular fitting within bounds (origin_x/origin_y are zero in manual mode)
                    actual_w = min(tile_params['width'], origin_x + rw_param - x_orig)
                    actual_l = min(tile_params['length'], origin_y + rl_param - y_orig)
                    if actual_w <= EPSILON or actual_l <= EPSILON:
                        continue
                    is_cut = (abs(actual_w - tile_params['width']) > EPSILON or abs(actual_l - tile_params['length']) > EPSILON)
                    footprint_xy = [(x_orig, y_orig), (x_orig + actual_w, y_orig), (x_orig + actual_w, y_orig + actual_l), (x_orig, y_orig + actual_l)]
                    tile = Tile3D(origin_xy=(x_orig, y_orig), xtile=tile_params['width'], ytile=tile_params['length'], thickness=tile_params['thickness'], is_cut=is_cut, cut_polygon_xy=footprint_xy, qr_data=f"T-{i}-{j}")
                    tile.compute_3d_corners(actual_tile_bottom_z)
                    initial_tiles.append(tile)
                else:
                    # polygonal room: compute intersection poly between room_poly (world coords) and rectangle
                    inter = sutherland_hodgman_clip(room_poly, rect)
                    if not inter or polygon_area_2d(inter) < (tile_params['width']*tile_params['length'])*0.01:
                        continue
                    # inter is in world coords and represents the clipped polygon of this tile
                    is_cut = True if abs(polygon_area_2d(inter) - (tile_params['width']*tile_params['length'])) > EPSILON else False
                    tile = Tile3D(origin_xy=(x_orig, y_orig), xtile=tile_params['width'], ytile=tile_params['length'], thickness=tile_params['thickness'], is_cut=is_cut, cut_polygon_xy=inter, qr_data=f"T-{i}-{j}")
                    tile.compute_3d_corners(actual_tile_bottom_z)
                    initial_tiles.append(tile)

      # Characterize tiles so we can reduce pedestals under very small edge pieces
        full_tile_area = tile_params['width'] * tile_params['length']
        small_piece_threshold = full_tile_area * 0.25 if full_tile_area > 0 else 0.0
        edge_epsilon = 1e-4
        tile_characteristics = []
        for tile in initial_tiles:
            footprint = tile.get_actual_xy_footprint()
            area = abs(polygon_area_2d(footprint))
            touches_bbox = any(
                abs(px - origin_x) < edge_epsilon
                or abs(px - (origin_x + rw_param)) < edge_epsilon
                or abs(py - origin_y) < edge_epsilon
                or abs(py - (origin_y + rl_param)) < edge_epsilon
                for px, py in footprint
            ) if footprint else False
            is_small_edge_piece = bool(tile.is_cut and touches_bbox and area < small_piece_threshold)
            tile_characteristics.append({'is_small_edge_piece': is_small_edge_piece, 'area': area})
            tile.is_small_edge_piece = is_small_edge_piece

        # Generate and adjust pedestals (existing algorithm works with arbitrary corner coords)
        pedestal_cap_radius = 0.035
        min_pedestal_height = 0.01

        # 1. Generate logical pedestals and classify them
        logical_pedestals = {}
        # Collect which tiles reference each corner and keep the original corner position.
        # This allows us to later nudge edge/corner pedestals inward based on actual
        # adjacent tile geometry (so pedestals never end up outside cut tile footprints).
        for t_idx, tile in enumerate(initial_tiles):
            for px, py in tile.get_actual_xy_footprint():
                key = (round(px, 4), round(py, 4))
                if key not in logical_pedestals:
                    # classify against global room bbox as before (keeps legacy behavior)
                    is_on_x_edge = abs(px - origin_x) < EPSILON or abs(px - (origin_x + rw_param)) < EPSILON
                    is_on_y_edge = abs(py - origin_y) < EPSILON or abs(py - (origin_y + rl_param)) < EPSILON
                    ped_type = 3
                    if is_on_x_edge and is_on_y_edge:
                        ped_type = 1
                    elif is_on_x_edge or is_on_y_edge:
                        ped_type = 2
                    logical_pedestals[key] = {'type': ped_type, 'pos': (px, py), 'tiles': [t_idx]}
                else:
                    logical_pedestals[key]['tiles'].append(t_idx)

        # 2. Add Type 3 pedestals first
        final_pedestals_map = {}
        rejected_count = 0
        for key, data in logical_pedestals.items():
            if data['type'] == 3:
                px, py = data['pos']
                base_z = self.sample_floor_z(px, py)
                height = actual_tile_bottom_z - base_z
                if height >= min_pedestal_height:
                    final_pedestals_map[key] = {'pos_xy': (px, py), 'base_z': base_z, 'height': height, 'radius': pedestal_cap_radius}
                else:
                    rejected_count += 1
        
        print(f"[DEBUG] Type 3 pedestals: {len([d for d in logical_pedestals.values() if d['type']==3])} attempted, {len([p for p in final_pedestals_map.values()])} added, {rejected_count} rejected (height < {min_pedestal_height})")
        if rejected_count > 0:
            # Sample one rejected pedestal to show why
            for key, data in logical_pedestals.items():
                if data['type'] == 3:
                    px, py = data['pos']
                    base_z = self.sample_floor_z(px, py)
                    height = actual_tile_bottom_z - base_z
                    if height < min_pedestal_height:
                        print(f"[DEBUG] Sample rejected pedestal: pos=({px:.3f},{py:.3f}), base_z={base_z:.3f}, tile_bottom={actual_tile_bottom_z:.3f}, height={height:.3f}")
                        break

        # 3. Add adjusted Type 1 & 2 pedestals, checking for collision
        type3_positions = [p['pos_xy'] for p in final_pedestals_map.values()]
        min_dist_sq = (2 * pedestal_cap_radius)**2

        for key, data in logical_pedestals.items():
            if data['type'] == 1 or data['type'] == 2:
                tile_indexes = data.get('tiles', [])
                if tile_indexes and all(tile_characteristics[t_idx]['is_small_edge_piece'] for t_idx in tile_indexes):
                    continue
                px, py = data['pos']
                R = pedestal_cap_radius
                adj_px, adj_py = px, py

                # New strategy: nudge the pedestal INWARD toward the centroid(s) of the
                # adjacent tile footprint(s). This ensures pedestals for cut tiles are
                # moved into the tile geometry (not outside it), avoiding visible
                # protrusion beyond cut edges.
                vecx = vecy = 0.0
                cnt = 0
                for t_idx in data.get('tiles', []):
                    tpoly = initial_tiles[t_idx].get_actual_xy_footprint()
                    if not tpoly:
                        continue
                    cx = sum(p[0] for p in tpoly) / len(tpoly)
                    cy = sum(p[1] for p in tpoly) / len(tpoly)
                    dx = cx - px; dy = cy - py
                    dlen = math.hypot(dx, dy)
                    if dlen > EPSILON:
                        vecx += dx / dlen; vecy += dy / dlen; cnt += 1
                if cnt > 0:
                    vx = vecx / cnt; vy = vecy / cnt
                    vlen = math.hypot(vx, vy)
                    if vlen > EPSILON:
                        vx /= vlen; vy /= vlen
                        adj_px = px + vx * R
                        adj_py = py + vy * R

                # Ensure the adjusted point lies inside all adjacent tile polygons.
                # If it doesn't, iteratively nudge it toward the polygon centroid
                # until it falls inside (binary search along the segment).
                for t_idx in data.get('tiles', []):
                    tpoly = initial_tiles[t_idx].get_actual_xy_footprint()
                    if not tpoly:
                        continue
                    if point_in_polygon((adj_px, adj_py), tpoly):
                        continue
                    cx = sum(p[0] for p in tpoly) / len(tpoly)
                    cy = sum(p[1] for p in tpoly) / len(tpoly)
                    low = 0.0; high = 1.0
                    good_x, good_y = adj_px, adj_py
                    for _ in range(18):
                        mid = (low + high) / 2.0
                        mx = adj_px + (cx - adj_px) * mid
                        my = adj_py + (cy - adj_py) * mid
                        if point_in_polygon((mx, my), tpoly):
                            good_x, good_y = mx, my
                            high = mid
                        else:
                            low = mid
                    adj_px, adj_py = good_x, good_y

                # Check for collision with Type 3 pedestals
                is_colliding = any(((adj_px - p3[0])**2 + (adj_py - p3[1])**2) < min_dist_sq for p3 in type3_positions)

                if not is_colliding:
                    base_z = self.sample_floor_z(adj_px, adj_py)
                    height = actual_tile_bottom_z - base_z
                    if height >= min_pedestal_height:
                        final_pedestals_map[key] = {'pos_xy': (adj_px, adj_py), 'base_z': base_z, 'height': height, 'radius': pedestal_cap_radius}

        # --- Nudge pedestals inside tile footprints ---
        adjusted_pedestals = []
        for ped in final_pedestals_map.values():
            px, py = ped['pos_xy']
            radius = ped.get('radius', 0.0)

            best_tile = None
            best_dist = float('inf')

            for tile in initial_tiles:
                fp = tile.get_actual_xy_footprint()
                if not fp:
                    continue

                if point_in_polygon((px, py), fp, tolerance=radius):
                    best_tile = tile
                    break

                centroid = polygon_centroid_2d(fp)
                if centroid:
                    cx, cy = centroid
                    d = math.hypot(px - cx, py - cy)
                    if d < best_dist:
                        best_dist = d
                        best_tile = tile

            if best_tile:
                fp = best_tile.get_actual_xy_footprint()
                new_pos = self.nudge_pedestal_inside_tile((px, py), radius, fp)
                ped['pos_xy'] = new_pos

            adjusted_pedestals.append(ped)
        self.pedestals = adjusted_pedestals

        # ============================================================
        # PEDESTAL OPTIMIZATION (Option A + Option D)
        # ============================================================

        # A) Adaptive minimum spacing based on pedestal radius
        base_spacing = 0.10        # default spacing
        large_tile_spacing = 0.15  # for large tiles
        small_tile_spacing = 0.07  # for micro pedestals

        def pedestal_min_spacing(pedestal):
            radius = pedestal.get("radius", 0.035)
            if radius > 0.033:
                return large_tile_spacing     # full tile corner
            elif radius < 0.020:
                return small_tile_spacing     # tiny cut-tile pedestal
            return base_spacing

        def _tile_centroid_for_idx(idx):
            footprint = initial_tiles[idx].get_actual_xy_footprint()
            if not footprint:
                return None
            centroid = polygon_centroid_2d(footprint)
            if centroid:
                return centroid
            avg_x = sum(p[0] for p in footprint) / len(footprint)
            avg_y = sum(p[1] for p in footprint) / len(footprint)
            return (avg_x, avg_y)

        def _point_in_any_tile(px, py, tile_indices):
            for t_idx in tile_indices:
                footprint = initial_tiles[t_idx].get_actual_xy_footprint()
                if footprint and point_in_polygon((px, py), footprint, tolerance=1e-6):
                    return True
            return False

        # B) Merge pedestals into centroid clusters
        merged = []

        for ped in self.pedestals:
            px, py = ped["pos_xy"]
            placed = False

            for cluster in merged:
                cx, cy = cluster["pos_xy"]
                spacing = pedestal_min_spacing(cluster)

                if (px - cx)**2 + (py - cy)**2 < spacing**2:
                    # merge pedestals into centroid
                    new_x = (cx + px) / 2.0
                    new_y = (cy + py) / 2.0
                    cluster["pos_xy"] = (new_x, new_y)

                    # average height + base_z for consistency
                    cluster["base_z"] = (cluster["base_z"] + ped["base_z"]) / 2.0
                    cluster["height"] = (cluster["height"] + ped["height"]) / 2.0
                    cluster["tiles"] = list(dict.fromkeys(cluster.get("tiles", []) + ped.get("tiles", [])))

                    placed = True
                    break

            if not placed:
                merged.append(ped)

        self.pedestals = merged

        # Ensure every pedestal tied to edge tiles remains inside at least one of its footprints.
        for ped in self.pedestals:
            tile_indices = ped.get("tiles", [])
            if not tile_indices:
                continue
            px, py = ped["pos_xy"]
            if _point_in_any_tile(px, py, tile_indices):
                continue
            dir_x = dir_y = 0.0
            contributors = 0
            for t_idx in tile_indices:
                centroid = _tile_centroid_for_idx(t_idx)
                if not centroid:
                    continue
                dx = centroid[0] - px
                dy = centroid[1] - py
                length = math.hypot(dx, dy)
                if length > EPSILON:
                    dir_x += dx / length
                    dir_y += dy / length
                    contributors += 1
            if contributors == 0:
                continue
            norm = math.hypot(dir_x, dir_y)
            if norm < EPSILON:
                continue
            dir_x /= norm
            dir_y /= norm
            step = pedestal_min_spacing(ped) * 0.5
            adjusted = False
            for _ in range(12):
                test_x = px + dir_x * step
                test_y = py + dir_y * step
                if _point_in_any_tile(test_x, test_y, tile_indices):
                    ped["pos_xy"] = (test_x, test_y)
                    adjusted = True
                    break
                step *= 0.5
            if not adjusted:
                for t_idx in tile_indices:
                    centroid = _tile_centroid_for_idx(t_idx)
                    if centroid:
                        ped["pos_xy"] = centroid
                        break
# Build pedestal associations for each tile footprint
        for tile in initial_tiles:
            footprint = tile.get_actual_xy_footprint()
            if not footprint:
                tile.pedestal_list = []
                continue

            ped_entries = []
            for ped in self.pedestals:
                px, py = ped["pos_xy"]
                tolerance = max(ped.get("radius", 0.0), 1e-6)
                if point_in_polygon((px, py), footprint, tolerance=tolerance):
                    ped_entries.append((px * 1000.0, py * 1000.0, ped["height"] * 1000.0))

            ped_entries.sort(key=lambda item: (item[1], item[0], item[2]))
            tile.pedestal_list = ped_entries

        # ============================================================

        # 4. Filter tiles to keep only those that are fully supported: all corner points must have pedestals nearby
        supported_corners_keys = final_pedestals_map.keys()
        self.tiles = []
         
        for idx, tile in enumerate(initial_tiles):
            if tile_characteristics[idx]['is_small_edge_piece']:
                self.tiles.append(tile)
                continue
            is_supported = all((round(cx, 4), round(cy, 4)) in supported_corners_keys for cx, cy in tile.get_actual_xy_footprint())
            if is_supported:
                self.tiles.append(tile)
        
        print(f"[DEBUG] Tiles: {len(initial_tiles)} initial, {len(self.tiles)} supported, {len(self.pedestals)} pedestals")
        if len(self.tiles) == 0 and len(initial_tiles) > 0:
            # Check why tiles were rejected
            sample_tile = initial_tiles[0]
            corners = sample_tile.get_actual_xy_footprint()
            print(f"[DEBUG] Sample tile corners: {[(round(cx,4), round(cy,4)) for cx,cy in corners]}")
            print(f"[DEBUG] Available pedestal corners: {list(supported_corners_keys)[:10]}...")

        # Generate floor mesh for drawing (same as before but use elevation model z)
        # Only generate a floor mesh if the room has non-zero dimensions AND not in import mode.
        floor_segments = 40
        self.original_floor_mesh = []
        if not self.has_imported_model and rw_param > EPSILON and rl_param > EPSILON and floor_segments > 0:
            w_step = rw_param / floor_segments
            l_step = rl_param / floor_segments
            for i in range(floor_segments):
                for j in range(floor_segments):
                    x0 = origin_x + i * w_step
                    y0 = origin_y + j * l_step
                    x1 = origin_x + (i + 1) * w_step
                    y1 = origin_y + (j + 1) * l_step
                    p1 = QVector3D(x0, y0, self.original_slope_func(x0, y0))
                    p2 = QVector3D(x1, y0, self.original_slope_func(x1, y0))
                    p3 = QVector3D(x1, y1, self.original_slope_func(x1, y1))
                    p4 = QVector3D(x0, y1, self.original_slope_func(x0, y1))
                    self.original_floor_mesh.append([p1, p2, p3, p4])
        else:
            # No room defined - leave floor mesh empty so no default square appears.
            self.original_floor_mesh = []

        # Debug: report floor mesh size to confirm default square removal
        try:
            print(f"[DEBUG] compute_and_build_layout: original_floor_mesh quads = {len(self.original_floor_mesh)}")
            sys.stdout.flush()
        except Exception:
            pass

        self.update()

    def draw_tile(self, tile: Tile3D, is_selected=False):
        if not tile.corners_top_xyz: return

        # Enable texture if material texture is loaded
        texture_enabled = False
        if self.material_texture_id is not None:
            texture_enabled = True
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.material_texture_id)
            glColor3f(1.0, 1.0, 1.0)  # White to show texture colors
        else:
            if is_selected: glColor3f(1.0, 0.7, 0.0)
            elif tile.is_cut: glColor3f(0.65, 0.75, 0.9)
            else: glColor3f(0.9, 0.9, 0.8)

        # Get tile footprint for texture coordinate calculation
        footprint = tile.get_actual_xy_footprint()
        if footprint:
            xs = [p[0] for p in footprint]
            ys = [p[1] for p in footprint]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            tile_width = max_x - min_x
            tile_height = max_y - min_y
        else:
            tile_width = tile.xtile
            tile_height = tile.ytile
            min_x = tile.origin_xy[0]
            min_y = tile.origin_xy[1]

        # Use physical dimensions for seamless tiling (texture repeats based on actual size)
        # This ensures texture scale is consistent across all tiles
        texture_scale = 1.0  # 1 texture repeat per meter

        # Top face with texture coordinates
        glBegin(GL_POLYGON)
        glNormal3f(0, 0, 1)
        for v in tile.corners_top_xyz:
            if texture_enabled:
                # Calculate texture coordinates based on world position for seamless tiling
                tex_u = (v.x() - min_x) * texture_scale
                tex_v = (v.y() - min_y) * texture_scale
                glTexCoord2f(tex_u, tex_v)
            glVertex3f(v.x(), v.y(), v.z())
        glEnd()

        # Disable texture for bottom and sides (or apply differently if desired)
        if texture_enabled:
            glDisable(GL_TEXTURE_2D)
            if is_selected: glColor3f(1.0, 0.7, 0.0)
            elif tile.is_cut: glColor3f(0.65, 0.75, 0.9)
            else: glColor3f(0.9, 0.9, 0.8)

        # Bottom
        glBegin(GL_POLYGON); glNormal3f(0,0,-1)
        for v in reversed(tile.corners_bottom_xyz): glVertex3f(v.x(), v.y(), v.z())
        glEnd()

        # Sides
        num_corners = len(tile.corners_top_xyz)
        for i in range(num_corners):
            p1t,p2t = tile.corners_top_xyz[i], tile.corners_top_xyz[(i + 1) % num_corners]
            p1b,p2b = tile.corners_bottom_xyz[i], tile.corners_bottom_xyz[(i + 1) % num_corners]
            side_normal = QVector3D.crossProduct(p2b - p1b, p1t - p1b).normalized()
            glNormal3f(side_normal.x(), side_normal.y(), side_normal.z())
            glBegin(GL_QUADS)
            glVertex3f(p1t.x(),p1t.y(),p1t.z()); glVertex3f(p2t.x(),p2t.y(),p2t.z())
            glVertex3f(p2b.x(),p2b.y(),p2b.z()); glVertex3f(p1b.x(),p1b.y(),p1b.z())
            glEnd()

        glColor3f(0.2,0.2,0.2); glLineWidth(1.5 if is_selected else 1.0)
        if self.show_wireframe:
            glBegin(GL_LINE_LOOP)
            for v in tile.corners_top_xyz: glVertex3f(v.x(), v.y(), v.z() + 0.0005)
            glEnd()
        else:
            glBegin(GL_LINE_LOOP)
            for v in tile.corners_top_xyz: glVertex3f(v.x(), v.y(), v.z() + 0.0005)
            glEnd()
        glLineWidth(1.0)
        # Attempt to draw a QR overlay for the tile (created lazily)
        if self.show_qr_codes:
            try:
                self.draw_tile_qr(tile)
            except Exception:
                # Don't let QR generation break rendering
                pass

    def _ensure_tile_qr_texture(self, tile):
        """Create and cache an OpenGL texture for the tile's QR code."""
        if tile is None or getattr(tile, "qr_texture_id", None):
            return

        tile_id = str(tile.qr_data or "")
        if not tile_id:
            idx = self.tiles.index(tile) + 1 if tile in self.tiles else (id(tile) & 0xFFFF)
            tile_id = f"TILE-{idx}"

        # Create tile data dict
        tile_data = {
            'material': 'Tile',
            'density': 1900,
            'weight': 15,
            'thermal_r': 0.05,
            'thickness': f"{tile.thickness*1000:.1f}mm" if tile.thickness else "N/A"
        }

        # Create simple readable text: ID|Material|Thickness|Density|Weight|RValue
        qr_text = f"{tile_id}|{tile_data.get('material', 'N/A')}|{tile_data.get('thickness', 'N/A')}|{tile_data.get('density', 'N/A')}|{tile_data.get('weight', 'N/A')}|{tile_data.get('thermal_r', 'N/A')}"

        try:
            qr_code = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=6,
                border=1,
            )
            qr_code.add_data(qr_text)
            qr_code.make(fit=True)
        except Exception as e:
            print(f"QR generation failed: {e}")
            return

        qr_img = qr_code.make_image(fill_color="black", back_color="white")
        if hasattr(qr_img, "get_image"):
            qr_img = qr_img.get_image()
        if not isinstance(qr_img, Image.Image):
            raise RuntimeError("QR generation did not return a PIL image.")

        qr_img = qr_img.convert("RGBA")
        side = max(qr_img.width, qr_img.height)
        if qr_img.width != qr_img.height:
            square = Image.new("RGBA", (side, side), (255, 255, 255, 0))
            offset = ((side - qr_img.width) // 2, (side - qr_img.height) // 2)
            square.paste(qr_img, offset)
            qr_img = square
        max_tex_size = 256
        if side > max_tex_size:
            qr_img = qr_img.resize((max_tex_size, max_tex_size), Image.NEAREST)
            side = max_tex_size
        else:
            side = qr_img.width

        qr_img = qr_img.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = qr_img.tobytes()

        texture_id = glGenTextures(1)
        if isinstance(texture_id, (list, tuple)):
            texture_id = texture_id[0]
        texture_id = int(texture_id)
        if texture_id <= 0:
            raise RuntimeError("Failed to allocate QR texture.")

        glBindTexture(GL_TEXTURE_2D, texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, side, side, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

        footprint = tile.get_actual_xy_footprint()
        qr_extent = None
        if footprint:
            xs = [p[0] for p in footprint]
            ys = [p[1] for p in footprint]
            width = max(xs) - min(xs)
            length = max(ys) - min(ys)
            qr_pct = getattr(self, "qr_pct", 0.10)
            qr_extent = min(width, length) * qr_pct
        if qr_extent is None or qr_extent <= EPSILON:
            qr_extent = max(EPSILON, min(tile.xtile, tile.ytile) * getattr(self, "qr_pct", 0.10))

        tile.qr_texture_id = texture_id
        tile.qr_size = qr_extent

    def draw_tile_qr(self, tile):
        """Draw the tile's QR texture and a thin white bounding frame."""
        if tile is None or not tile.corners_top_xyz:
            return

        # Ensure the texture exists (lazy creation inside a current GL context)
        if not getattr(tile, "qr_texture_id", None):
            try:
                self._ensure_tile_qr_texture(tile)
            except Exception:
                return

        texture_id = getattr(tile, "qr_texture_id", None)
        if not texture_id:
            return

        qr_size = getattr(tile, "qr_size", min(tile.xtile, tile.ytile) * 0.5)
        if qr_size <= EPSILON:
            return

        # Default center: centroid of top corners
        num_pts = len(tile.corners_top_xyz)
        cz = sum(v.z() for v in tile.corners_top_xyz) / num_pts

        # Position QR at top-right corner of the tile footprint (with small margin)
        footprint = tile.get_actual_xy_footprint()
        if footprint:
            xs = [p[0] for p in footprint]
            ys = [p[1] for p in footprint]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            margin = min(tile.xtile, tile.ytile) * 0.02
            # lower-left corner of QR image
            insert_x = max_x - margin - qr_size
            insert_y = max_y - margin - qr_size
            cx = insert_x + (qr_size / 2.0)
            cy = insert_y + (qr_size / 2.0)
        else:
            cx = sum(v.x() for v in tile.corners_top_xyz) / num_pts
            cy = sum(v.y() for v in tile.corners_top_xyz) / num_pts

        half = qr_size / 2.0
        was_texture_enabled = bool(glIsEnabled(GL_TEXTURE_2D))
        if not was_texture_enabled:
            glEnable(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, int(texture_id))
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3f(cx - half, cy - half, cz + 0.002)
        glTexCoord2f(1.0, 0.0); glVertex3f(cx + half, cy - half, cz + 0.002)
        glTexCoord2f(1.0, 1.0); glVertex3f(cx + half, cy + half, cz + 0.002)
        glTexCoord2f(0.0, 1.0); glVertex3f(cx - half, cy + half, cz + 0.002)
        glEnd()

        border = qr_size * 1.05
        glDisable(GL_TEXTURE_2D)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(cx - border / 2.0, cy - border / 2.0, cz + 0.003)
        glVertex3f(cx + border / 2.0, cy - border / 2.0, cz + 0.003)
        glVertex3f(cx + border / 2.0, cy + border / 2.0, cz + 0.003)
        glVertex3f(cx - border / 2.0, cy + border / 2.0, cz + 0.003)
        glEnd()

        if was_texture_enabled:
            glEnable(GL_TEXTURE_2D)

    def create_unit_cylinder_dl(self, segments=12):
        dl = glGenLists(1)
        glNewList(dl, GL_COMPILE)
        radius, height = 1.0, 1.0

        # Top cap
        glBegin(GL_TRIANGLE_FAN); glNormal3f(0,0,1)
        glVertex3f(0,0,height)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex3f(radius * math.cos(angle), radius * math.sin(angle), height)
        glEnd()

        # Bottom cap
        glBegin(GL_TRIANGLE_FAN); glNormal3f(0,0,-1)
        glVertex3f(0,0,0)
        for i in range(segments, -1, -1):
            angle = 2 * math.pi * i / segments
            glVertex3f(radius * math.cos(angle), radius * math.sin(angle), 0)
        glEnd()

        # Sides
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            nx, ny = math.cos(angle), math.sin(angle)
            glNormal3f(nx,ny,0)
            glVertex3f(radius * nx, radius * ny, height)
            glVertex3f(radius * nx, radius * ny, 0)
        glEnd()
        glEndList()
        return dl

    def draw_pedestal(self, pedestal, is_selected=False):
        if self.unit_cylinder_dl == -1: self.unit_cylinder_dl = self.create_unit_cylinder_dl()

        glPushMatrix()
        glTranslatef(pedestal['pos_xy'][0], pedestal['pos_xy'][1], pedestal['base_z'])

        total_h = pedestal.get('height', 0.0)
        base_h = pedestal.get('min_height', 0.0)
        adjustable_h = pedestal.get('adjustable_height', max(total_h - base_h, 0.0))
        cap_r = pedestal['radius']

        current_z = 0.0

        # Fixed base segment
        if base_h > EPSILON:
            base_main = max(base_h * 0.7, 0.003)
            base_cap = max(base_h - base_main, 0.0)
            glPushMatrix(); glTranslatef(0, 0, current_z); glScalef(cap_r, cap_r, base_main)
            if is_selected:
                glColor3f(1.0, 0.7, 0.0)
            else:
                glColor3f(0.20, 0.20, 0.23)
            glCallList(self.unit_cylinder_dl); glPopMatrix()
            current_z += base_main

            if base_cap > EPSILON:
                glPushMatrix(); glTranslatef(0, 0, current_z); glScalef(cap_r * 0.9, cap_r * 0.9, base_cap)
                if is_selected:
                    glColor3f(1.0, 0.78, 0.1)
                else:
                    glColor3f(0.26, 0.26, 0.30)
                glCallList(self.unit_cylinder_dl); glPopMatrix()
                current_z += base_cap

        # Adjustable segment
        if adjustable_h > EPSILON:
            adj_main = max(adjustable_h * 0.8, 0.003)
            adj_cap = max(adjustable_h - adj_main, 0.0)
            glPushMatrix(); glTranslatef(0, 0, current_z); glScalef(cap_r * 0.75, cap_r * 0.75, adj_main)
            if is_selected:
                glColor3f(1.0, 0.82, 0.2)
            else:
                glColor3f(0.15, 0.55, 0.65)
            glCallList(self.unit_cylinder_dl); glPopMatrix()
            current_z += adj_main

            if adj_cap > EPSILON:
                glPushMatrix(); glTranslatef(0, 0, current_z); glScalef(cap_r * 0.85, cap_r * 0.85, adj_cap)
                if is_selected:
                    glColor3f(1.0, 0.88, 0.3)
                else:
                    glColor3f(0.22, 0.65, 0.72)
                glCallList(self.unit_cylinder_dl); glPopMatrix()
                current_z += adj_cap

        # Fill any residual height to hit the target total
        remaining = max(total_h - current_z, 0.0)
        if remaining > EPSILON:
            glPushMatrix(); glTranslatef(0, 0, current_z); glScalef(cap_r * 0.9, cap_r * 0.9, remaining)
            if is_selected:
                glColor3f(1.0, 0.85, 0.25)
            else:
                glColor3f(0.30, 0.30, 0.34)
            glCallList(self.unit_cylinder_dl); glPopMatrix()

        glPopMatrix()

    def draw_original_floor(self):
        # Don't draw default floor when in import mode
        if self.has_imported_model:
            return
        glColor3f(0.6, 0.6, 0.55)
        for quad in self.original_floor_mesh:
            glBegin(GL_QUADS)
            n = QVector3D.crossProduct(quad[1]-quad[0], quad[3]-quad[0]).normalized()
            if n.z() < 0: n = -n
            glNormal3f(n.x(), n.y(), n.z())
            for vertex in quad: glVertex3f(vertex.x(), vertex.y(), vertex.z())
            glEnd()

    def draw_elevation_heatmap(self):
        # draw colored quads based on elevation
        # Don't draw default floor when in import mode
        if self.has_imported_model:
            return
        if not self.original_floor_mesh:
            self.draw_original_floor(); return
        # compute min/max z
        zs = [v.z() for quad in self.original_floor_mesh for v in quad]
        if not zs: return
        zmin, zmax = min(zs), max(zs)
        rng = max(zmax - zmin, 1e-6)

        # Draw heatmap quads
        for quad in self.original_floor_mesh:
            avg_z = sum(v.z() for v in quad) / 4.0
            t = (avg_z - zmin) / rng
            # Color gradient: blue (low) -> green -> red (high)
            if t < 0.5:
                r, g, b = 0, t * 2, 1 - t * 2
            else:
                r, g, b = (t - 0.5) * 2, 1 - (t - 0.5) * 2, 0
            glColor3f(r, g, b)
            glBegin(GL_QUADS)
            for v in quad:
                glVertex3f(v.x(), v.y(), v.z())
            glEnd()

    def draw_room_boundary(self):
        if not self.room_polygon_xy or len(self.room_polygon_xy) < 3: return
        glDisable(GL_LIGHTING); glLineWidth(2.0)
        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_LINE_LOOP)
        for x,y in self.room_polygon_xy:
            z = self.original_slope_func(x,y)
            glVertex3f(x, y, z + 0.002)
        glEnd()
        glEnable(GL_LIGHTING); glLineWidth(1.0)

    def draw_imported_mesh(self):
        """Draw the imported 3D model mesh with surface selection highlighting - OPTIMIZED."""
        if self.imported_mesh is None:
            return

        mesh = self.imported_mesh

        # For large meshes, use simplified rendering
        use_simplified = len(mesh.faces) > 10000

        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)

        if use_simplified:
            # Fast rendering: draw entire mesh as one color with wireframe
            glColor3f(0.7, 0.7, 0.8)  # Default gray-blue
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  # Wireframe mode
            glLineWidth(1.0)

        # Draw the mesh efficiently
        glBegin(GL_TRIANGLES)

        if use_simplified:
            # Simplified: only draw geometry, no per-face color checks
            for face_idx, face in enumerate(mesh.faces):
                if hasattr(mesh, 'face_normals'):
                    normal = mesh.face_normals[face_idx]
                    glNormal3f(normal[0], normal[1], normal[2])

                for vertex_idx in face:
                    vertex = mesh.vertices[vertex_idx]
                    glVertex3f(vertex[0], vertex[1], vertex[2])
        else:
            # Full rendering with selection highlighting
            for face_idx, face in enumerate(mesh.faces):
                # Check if this face is in multi-selection set (red), single-selected (green), or hovered (yellow)
                is_multi_selected = face_idx in self.selected_surfaces
                is_selected = (self.selected_surface is not None and
                              self.selected_surface == face_idx)
                is_hovered = (self.hovered_surface is not None and
                             self.hovered_surface == face_idx and
                             self.surface_selection_mode)

                # Set color based on state (priority: multi-selected > single-selected > hovered > default)
                if is_multi_selected:
                    glColor3f(0.9, 0.2, 0.2)  # Red for multi-selected surfaces
                elif is_selected:
                    glColor3f(0.2, 0.8, 0.2)  # Green for single-selected
                elif is_hovered:
                    glColor3f(1.0, 1.0, 0.0)  # Yellow for hovered
                else:
                    glColor3f(0.7, 0.7, 0.8)  # Default gray-blue

                # Calculate normal for lighting
                if hasattr(mesh, 'face_normals'):
                    normal = mesh.face_normals[face_idx]
                    glNormal3f(normal[0], normal[1], normal[2])

                # Draw the triangle face
                for vertex_idx in face:
                    vertex = mesh.vertices[vertex_idx]
                    glVertex3f(vertex[0], vertex[1], vertex[2])

        glEnd()

        # Reset polygon mode if simplified
        if use_simplified:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Draw wireframe for all multi-selected surfaces (red outline)
        if self.selected_surfaces:
            glDisable(GL_LIGHTING)
            glLineWidth(3.0)
            glColor3f(1.0, 0.3, 0.3)  # Red wireframe for multi-selected
            for surf_idx in self.selected_surfaces:
                if surf_idx < len(mesh.faces):
                    face = mesh.faces[surf_idx]
                    glBegin(GL_LINE_LOOP)
                    for vertex_idx in face:
                        vertex = mesh.vertices[vertex_idx]
                        glVertex3f(vertex[0], vertex[1], vertex[2])
                    glEnd()
            glEnable(GL_LIGHTING)
            glLineWidth(1.0)

        # Draw wireframe for single-selected surface (always visible)
        if self.selected_surface is not None and self.selected_surface < len(mesh.faces):
            glDisable(GL_LIGHTING)
            glLineWidth(4.0)
            glColor3f(1.0, 0.8, 0.0)  # Orange wireframe for visibility

            face = mesh.faces[self.selected_surface]
            glBegin(GL_LINE_LOOP)
            for vertex_idx in face:
                vertex = mesh.vertices[vertex_idx]
                glVertex3f(vertex[0], vertex[1], vertex[2])
            glEnd()

            glEnable(GL_LIGHTING)
            glLineWidth(1.0)

    def draw_axes(self):
        glLineWidth(2.5); glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        ax_len = max(0.5, self.room_dims.get('width', 1.0) * 0.15, self.room_dims.get('length', 1.0) * 0.15)

        if self.room_polygon_xy:
            xs = [p[0] for p in self.room_polygon_xy]
            ys = [p[1] for p in self.room_polygon_xy]
            origin_x = min(xs)
            origin_y = min(ys)
        else:
            origin_x = origin_y = 0.0

        origin_z = self.original_slope_func(origin_x, origin_y)

        glColor3f(1,0,0); glVertex3f(origin_x, origin_y, origin_z); glVertex3f(origin_x + ax_len, origin_y, origin_z)
        glColor3f(0,1,0); glVertex3f(origin_x, origin_y, origin_z); glVertex3f(origin_x, origin_y + ax_len, origin_z)
        glColor3f(0,0,1); glVertex3f(origin_x, origin_y, origin_z); glVertex3f(origin_x, origin_y, origin_z + ax_len)

        glEnd()
        glEnable(GL_LIGHTING); glLineWidth(1.0)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        #self.last_mouse_pos = event.pos()
        self.last_mouse_pos = event.position()
        pixel_ratio = self.devicePixelRatioF()
        screen_x = event.position().x() * pixel_ratio
        screen_y = event.position().y() * pixel_ratio

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Check if in surface selection mode for 3D model
            if self.surface_selection_mode and self.imported_mesh is not None:
                surface_idx = self.pick_surface(screen_x, screen_y)
                if surface_idx is not None:
                    # Check for Ctrl modifier for multi-selection
                    ctrl_held = event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier

                    # Determine the full coplanar group for the picked face (if selector available)
                    if hasattr(self, 'surface_selector') and self.surface_selector is not None:
                        try:
                            group_faces = set(self.surface_selector.get_group_faces(surface_idx))
                        except Exception:
                            group_faces = {surface_idx}
                    else:
                        group_faces = {surface_idx}

                    if ctrl_held:
                        # Multi-select toggle: add/remove the entire group
                        if group_faces.issubset(self.selected_surfaces):
                            self.selected_surfaces.difference_update(group_faces)
                        else:
                            self.selected_surfaces.update(group_faces)
                        # Keep selection mode active for more picks
                        if hasattr(self.parent(), 'update_selection_info'):
                            self.parent().update_selection_info()
                    else:
                        # Single-select: replace selection with the group and exit selection mode
                        self.selected_surfaces.clear()
                        self.selected_surfaces.update(group_faces)
                        self.selected_surface = surface_idx
                        self.surface_selection_mode = False

                        # Extract surface dimensions and update parent window (pass the picked face)
                        if hasattr(self.parent(), 'on_surface_selected'):
                            self.parent().on_surface_selected(surface_idx)

                    self.update()
                return

            # Try to pick pedestal first (pedestals are smaller, give priority)
            ped_obj, ped_idx = self.pick_pedestal(screen_x, screen_y)
            if ped_obj:
                prev_ped_idx = self.selected_pedestal_index
                self.selected_pedestal_index = ped_idx
                self.selected_tile_index = -1  # Deselect tile
                self.pedestalClicked.emit(ped_obj)
                if prev_ped_idx != self.selected_pedestal_index:
                    self.update()
            else:
                # No pedestal hit, try tile picking
                prev_idx = self.selected_tile_index
                prev_ped_idx = self.selected_pedestal_index
                self.selected_tile_index = -1
                self.selected_pedestal_index = -1
                tile_obj, tile_idx = self.pick_tile_accurate(screen_x, screen_y)
                if tile_obj:
                    self.selected_tile_index = tile_idx
                    self.tileClicked.emit(tile_obj)
                if prev_idx != self.selected_tile_index or prev_ped_idx != -1:
                    self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        #dx = event.x() - self.last_mouse_pos.x(); dy = event.y() - self.last_mouse_pos.y()
        dx = event.position().x() - self.last_mouse_pos.x()
        dy = event.position().y() - self.last_mouse_pos.y()
        buttons = event.buttons()
        #if buttons & QtCore.Qt.LeftButton:
        if buttons & QtCore.Qt.MouseButton.LeftButton:
            self.camera_azimuth += dx * 0.35
            self.camera_elevation += dy * 0.35
            self.camera_elevation = max(-89.9, min(89.9, self.camera_elevation))
            self.update()
        #elif buttons & QtCore.Qt.MouseButton.MiddleButton:
        elif buttons & QtCore.Qt.MouseButton.MiddleButton:
            pan_speed = 0.0015 * self.camera_distance
            right_rad = math.radians(self.camera_azimuth + 90.0)
            self.room_center_for_rotation.setX(self.room_center_for_rotation.x() - dx * math.cos(right_rad) * pan_speed)
            self.room_center_for_rotation.setY(self.room_center_for_rotation.y() - dx * math.sin(right_rad) * pan_speed)
            self.room_center_for_rotation.setZ(self.room_center_for_rotation.z() + dy * pan_speed)
            self.update()
        self.last_mouse_pos = event.position()
        #self.last_mouse_pos = event.pos()

    def wheelEvent(self, event: QtGui.QWheelEvent):
        delta = event.angleDelta().y() / 120
        zoom_inc = 0.08
        self.camera_distance *= (1.0 - zoom_inc) if delta > 0 else (1.0 + zoom_inc)
        self.camera_distance = max(0.05, self.camera_distance)
        # Ensure OpenGL context is current before calling resizeGL to update projection
        try:
            self.makeCurrent()
            self.resizeGL(self.width(), self.height())
        finally:
            try:
                self.doneCurrent()
            except Exception:
                pass
        self.update()

    def pick_tile_accurate(self, x_screen, y_screen):
        # Use cached matrices captured during paintGL to avoid GL calls here
        if self._cached_modelview is None or self._cached_projection is None or self._cached_viewport is None:
            return None, -1
        mvm = self._cached_modelview
        pjm = self._cached_projection
        vp = self._cached_viewport
        # QOpenGLWidget renders into a framebuffer whose resolution can be
        # larger than the logical widget size on high-DPI displays.  The
        # incoming mouse position is in device-independent coordinates, so it
        # needs to be converted to the framebuffer's pixel space before we
        # hand it to gluUnProject (which expects raw viewport pixels).
        # Guard against re-scaling positions that are already provided in
        # framebuffer coordinates by clamping to the viewport bounds.
        x_screen = max(vp[0], min(vp[0] + vp[2], x_screen))
        y_screen = max(vp[1], min(vp[1] + vp[3], y_screen))
        nx, ny, nz = gluUnProject(x_screen, vp[3] - y_screen, 0.0, mvm, pjm, vp)
        fx, fy, fz = gluUnProject(x_screen, vp[3] - y_screen, 1.0, mvm, pjm, vp)
        ro = QVector3D(nx, ny, nz)
        rd = QVector3D(fx - nx, fy - ny, fz - nz).normalized()
        closest_tile = None
        closest_index = -1
        min_t = float('inf')
        for idx, tile in enumerate(self.tiles):
            if tile.pick_origin is None or tile.pick_normal is None:
                continue
            denom = QVector3D.dotProduct(tile.pick_normal, rd)
            if abs(denom) < EPSILON:
                continue
            t_int = QVector3D.dotProduct(tile.pick_origin - ro, tile.pick_normal) / denom
            if t_int < -PICK_EPSILON or t_int >= min_t:
                continue
            ipt3d = ro + rd * t_int
            rel = ipt3d - tile.pick_origin
            u = QVector3D.dotProduct(rel, tile.pick_u_axis)
            v = QVector3D.dotProduct(rel, tile.pick_v_axis)
            if tile.pick_bounds is not None:
                min_u, max_u, min_v, max_v = tile.pick_bounds
                if (u < min_u - PICK_EPSILON or u > max_u + PICK_EPSILON or
                        v < min_v - PICK_EPSILON or v > max_v + PICK_EPSILON):
                    continue
            if point_in_polygon((u, v), tile.pick_polygon2d, tolerance=PICK_EPSILON):
                min_t = t_int
                closest_tile = tile
                closest_index = idx
        return closest_tile, closest_index

    def pick_surface(self, x_screen, y_screen):
        """Pick a surface from the imported 3D mesh using ray-triangle intersection.

        Returns: face_index or None if no hit.
        """
        if self.imported_mesh is None:
            return None

        if self._cached_modelview is None or self._cached_projection is None or self._cached_viewport is None:
            return None

        mvm = self._cached_modelview
        pjm = self._cached_projection
        vp = self._cached_viewport

        # Clamp to viewport bounds
        x_screen = max(vp[0], min(vp[0] + vp[2], x_screen))
        y_screen = max(vp[1], min(vp[1] + vp[3], y_screen))

        # Unproject to get ray
        nx, ny, nz = gluUnProject(x_screen, vp[3] - y_screen, 0.0, mvm, pjm, vp)
        fx, fy, fz = gluUnProject(x_screen, vp[3] - y_screen, 1.0, mvm, pjm, vp)
        ray_origin = np.array([nx, ny, nz])
        ray_dir = np.array([fx - nx, fy - ny, fz - nz])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        # Use SurfaceSelector if available for more robust picking
        if hasattr(self, 'surface_selector') and self.surface_selector is not None:
            try:
                face_idx = self.surface_selector.raycast(ray_origin, ray_dir)
                return face_idx
            except Exception as e:
                print(f"SurfaceSelector raycast failed: {e}, falling back to basic method")

        closest_face = None
        min_t = float('inf')

        # Test each triangle face
        mesh = self.imported_mesh
        for face_idx, face in enumerate(mesh.faces):
            v0 = mesh.vertices[face[0]]
            v1 = mesh.vertices[face[1]]
            v2 = mesh.vertices[face[2]]

            # Möller–Trumbore ray-triangle intersection algorithm
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = np.cross(ray_dir, edge2)
            a = np.dot(edge1, h)

            if abs(a) < EPSILON:
                continue  # Ray parallel to triangle

            f = 1.0 / a
            s = ray_origin - v0
            u = f * np.dot(s, h)

            if u < 0.0 or u > 1.0:
                continue

            q = np.cross(s, edge1)
            v = f * np.dot(ray_dir, q)

            if v < 0.0 or u + v > 1.0:
                continue

            # Compute t to find intersection point
            t = f * np.dot(edge2, q)

            if t > PICK_EPSILON and t < min_t:
                min_t = t
                closest_face = face_idx

        return closest_face

    def pick_pedestal(self, x_screen, y_screen):
        """Pick a pedestal using ray-cylinder intersection.

        Returns: (pedestal_dict, pedestal_index) or (None, -1) if no hit.
        """
        if self._cached_modelview is None or self._cached_projection is None or self._cached_viewport is None:
            return None, -1

        mvm = self._cached_modelview
        pjm = self._cached_projection
        vp = self._cached_viewport

        # Clamp to viewport bounds
        x_screen = max(vp[0], min(vp[0] + vp[2], x_screen))
        y_screen = max(vp[1], min(vp[1] + vp[3], y_screen))

        # Unproject to get ray
        nx, ny, nz = gluUnProject(x_screen, vp[3] - y_screen, 0.0, mvm, pjm, vp)
        fx, fy, fz = gluUnProject(x_screen, vp[3] - y_screen, 1.0, mvm, pjm, vp)
        ray_origin = QVector3D(nx, ny, nz)
        ray_dir = QVector3D(fx - nx, fy - ny, fz - nz).normalized()

        closest_pedestal = None
        closest_index = -1
        min_t = float('inf')

        # Test each pedestal as a cylinder
        for idx, ped in enumerate(self.pedestals):
            px, py = ped['pos_xy']
            base_z = ped['base_z']
            height = ped['height']
            radius = ped['radius']

            # Cylinder axis is vertical (along Z)
            cyl_base = QVector3D(px, py, base_z)
            cyl_top = QVector3D(px, py, base_z + height)
            cyl_axis = QVector3D(0, 0, 1)  # Vertical

            # Ray-cylinder intersection (infinite cylinder first)
            # Using parametric form: P = ray_origin + t * ray_dir
            # Cylinder: (P - cyl_base - ((P - cyl_base) · cyl_axis) * cyl_axis)² = radius²

            # For vertical cylinder, simplify to 2D circle test in XY plane
            ro_xy = QVector3D(ray_origin.x() - px, ray_origin.y() - py, 0)
            rd_xy = QVector3D(ray_dir.x(), ray_dir.y(), 0)

            # Quadratic equation: |ro_xy + t * rd_xy|² = radius²
            a = rd_xy.lengthSquared()
            b = 2.0 * QVector3D.dotProduct(ro_xy, rd_xy)
            c = ro_xy.lengthSquared() - radius * radius

            discriminant = b * b - 4.0 * a * c

            if discriminant < 0 or abs(a) < EPSILON:
                continue  # No intersection with infinite cylinder

            # Find intersection points
            sqrt_disc = math.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            # Check both intersection points
            for t in [t1, t2]:
                if t < PICK_EPSILON or t >= min_t:
                    continue

                # Check if intersection point is within cylinder height
                hit_point = ray_origin + ray_dir * t
                if hit_point.z() >= base_z - PICK_EPSILON and hit_point.z() <= base_z + height + PICK_EPSILON:
                    min_t = t
                    closest_pedestal = ped
                    closest_index = idx
                    break

        return closest_pedestal, closest_index

    def get_pedestal_height_mm(self, pedestal):
        """Calculate pedestal height in millimeters.

        Args:
            pedestal: Pedestal dictionary with 'height' key in meters

        Returns:
            float: Height in millimeters
        """
        if pedestal is None:
            return 0.0
        return pedestal.get('height', 0.0) * 1000.0  # Convert meters to mm

    # --- START: New/Modified methods for Exporting ---

    def generate_layout_report_string(self):
        """Generates a formatted, human-readable string of the layout."""
        report_lines = []

        report_lines.append("=" * 84)
        report_lines.append("RAISED FLOOR LAYOUT REPORT".center(84))
        report_lines.append("=" * 84)
        # Use Python datetime formatting to avoid PyQt6 enum compatibility issues
        report_lines.append(f"Generated on: {datetime.now().strftime('%c')}")
        report_lines.append("\n")

        # Tiles Section
        report_lines.append("-" * 84)
        report_lines.append("TILES LIST".center(84))
        report_lines.append("-" * 84)
        header = f"{'ID':<5} | {'Center X (mm)':<17} | {'Center Y (mm)':<17} | {'Width (mm)':<14} | {'Length (mm)':<14} | {'Is Cut':<8}"
        report_lines.append(header)
        report_lines.append("-" * len(header))

        if not self.tiles:
            report_lines.append("No tiles in the current layout.")
        else:
            sorted_tiles = sorted(self.tiles, key=lambda t: (t.origin_xy[1], t.origin_xy[0]))
            for i, tile in enumerate(sorted_tiles, 1):
                footprint = tile.get_actual_xy_footprint()
                if not footprint: continue
                min_x = min(p[0] for p in footprint); max_x = max(p[0] for p in footprint)
                min_y = min(p[1] for p in footprint); max_y = max(p[1] for p in footprint)
                center_x = (min_x + max_x) / 2.0; center_y = (min_y + max_y) / 2.0
                width = max_x - min_x; length = max_y - min_y
                line = (f"{i:<5} | {center_x*1000:<17.2f} | {center_y*1000:<17.2f} | "
                        f"{width*1000:<14.2f} | {length*1000:<14.2f} | {'Yes' if tile.is_cut else 'No':<8}")
                report_lines.append(line)

        report_lines.append("\n\n")

        # Supports Section
        report_lines.append("-" * 84)
        report_lines.append("SUPPORTS (PEDESTALS) LIST".center(84))
        report_lines.append("-" * 84)
        header = f"{'ID':<5} | {'Position X (mm)':<17} | {'Position Y (mm)':<17} | {'Base Z (mm)':<15} | {'Height (mm)':<12}"
        report_lines.append(header)
        report_lines.append("-" * len(header))

        if not self.pedestals:
            report_lines.append("No supports in the current layout.")
        else:
            sorted_pedestals = sorted(self.pedestals, key=lambda p: (p['pos_xy'][1], p['pos_xy'][0]))
            for i, ped in enumerate(sorted_pedestals, 1):
                line = (f"{i:<5} | {ped['pos_xy'][0]*1000:<17.2f} | {ped['pos_xy'][1]*1000:<17.2f} | "
                        f"{ped['base_z']*1000:<15.2f} | {ped['height']*1000:<12.2f}")
                report_lines.append(line)

        report_lines.append("\n\n")
        report_lines.append("=" * 28 + " END OF REPORT " + "=" * 29)
        return "\n".join(report_lines)

    def _generate_cylinder_mesh_data(self, origin, radius, height, segments):
        """Generates vertices, normals, and faces for a single cylinder."""
        vertices = []; normals = []; faces = []

        normals.append(QVector3D(0, 0, 1))
        normals.append(QVector3D(0, 0, -1))
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            normals.append(QVector3D(math.cos(angle), math.sin(angle), 0))

        vertices.append(origin + QVector3D(0, 0, height))
        vertices.append(origin + QVector3D(0, 0, 0))
        for i in range(segments):
            angle = 2 * math.pi * i / segments
           
            vertices.append(origin + QVector3D(radius * math.cos(angle), radius * math.sin(angle), height))
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            vertices.append(origin + QVector3D(radius * math.cos(angle), radius * math.sin(angle), 0))

        top_n, bot_n = 1, 2
        for i in range(segments):
            v1_top = 3 + i; v2_top = 3 + ((i + 1) % segments)
            faces.append([(1, top_n), (v2_top, top_n), (v1_top, top_n)])
            v1_bot = 3 + segments + i; v2_bot = 3 + segments + ((i + 1) % segments)
            faces.append([(2, bot_n), (v1_bot, bot_n), (v2_bot, bot_n)])
            n1_side = 3 + i; n2_side = 3 + ((i + 1) % segments)
            faces.append([(v1_bot, n1_side), (v2_bot, n2_side), (v2_top, n2_side), (v1_top, n1_side)])
        return vertices, normals, faces

    def _generate_pedestal_mesh(self, pedestal_data):
        """Generates combined mesh data for a multi-part pedestal."""
        segments = 12
        total_h, cap_r = pedestal_data['height'], pedestal_data['radius']
        pos_xy, base_z = pedestal_data['pos_xy'], pedestal_data['base_z']

        BASE_H_F, STEM_H_F, CAP_H_F = 0.10, 0.80, 0.10
        BASE_R_S, STEM_R_S = 1.0, 0.7
        base_h, base_r = total_h * BASE_H_F, cap_r * BASE_R_S
        stem_h, stem_r = total_h * STEM_H_F, cap_r * STEM_R_S
        cap_h = total_h * CAP_H_F

        params = [
            (QVector3D(pos_xy[0], pos_xy[1], base_z), base_r, base_h),
            (QVector3D(pos_xy[0], pos_xy[1], base_z + base_h), stem_r, stem_h),
            (QVector3D(pos_xy[0], pos_xy[1], base_z + base_h + stem_h), cap_r, cap_h),
        ]
        all_v, all_n, all_f = [], [], []
        v_offset, n_offset = 0, 0
        for origin, radius, height in params:
            verts, norms, faces = self._generate_cylinder_mesh_data(origin, radius, height, segments)
            all_v.extend(verts); all_n.extend(norms)
            for face in faces:
                all_f.append([(v + v_offset, n + n_offset) for v, n in face])
            v_offset += len(verts); n_offset += len(norms)
        return all_v, all_n, all_f

    def export_scene_to_obj_file(self, file_path):
        """Writes the entire scene geometry to a single .obj file."""
        v_offset = 1; n_offset = 1
        try:
            with open(file_path, 'w') as f:
                f.write("# 3D Tile Layout Export\n\n")

                f.write("# --- TILES ---\n")
                for i, tile in enumerate(self.tiles):
                    if not tile.corners_top_xyz: continue
                    f.write(f"g tile_{tile.qr_data or i}\n")
                    verts = tile.corners_bottom_xyz + tile.corners_top_xyz
                    for v in verts: f.write(f"v {v.x():.6f} {v.y():.6f} {v.z():.6f}\n")
                    num_c = len(tile.corners_top_xyz)
                    norms = [QVector3D(0, 0, 1), QVector3D(0, 0, -1)]
                    for j in range(num_c):
                        p1b, p1t = tile.corners_bottom_xyz[j], tile.corners_top_xyz[j]
                        p2b = tile.corners_bottom_xyz[(j + 1) % num_c]
                        norms.append(QVector3D.crossProduct(p2b-p1b, p1t-p1b).normalized())
                    for n in norms: f.write(f"vn {n.x():.6f} {n.y():.6f} {n.z():.6f}\n")
                    f_top = ' '.join([f"{v_offset + num_c + j}//{n_offset}" for j in range(num_c)])
                    f_bot = ' '.join([f"{v_offset + j}//{n_offset+1}" for j in reversed(range(num_c))])
                    f.write(f"f {f_top}\n"); f.write(f"f {f_bot}\n")
                    for j in range(num_c):
                        v1,v2=v_offset+j, v_offset+((j+1)%num_c)
                        v3,v4=v_offset+num_c+((j+1)%num_c), v_offset+num_c+j
                        n = n_offset+2+j
                        f.write(f"f {v1}//{n} {v2}//{n} {v3}//{n} {v4}//{n}\n")
                    v_offset += len(verts); n_offset += len(norms)

                f.write("\n# --- PEDESTALS ---\n")
                for i, ped in enumerate(self.pedestals):
                    verts, norms, faces = self._generate_pedestal_mesh(ped)
                    f.write(f"g pedestal_{i}\n")
                    for v in verts: f.write(f"v {v.x():.6f} {v.y():.6f} {v.z():.6f}\n")
                    for n in norms: f.write(f"vn {n.x():.6f} {n.y():.6f} {n.z():.6f}\n")
                    for face in faces:
                        f_str = ' '.join([f"{(v-1)+v_offset}//{(n-1)+n_offset}" for v, n in face])
                        f.write(f"f {f_str}\n")
                    v_offset += len(verts); n_offset += len(norms)

                f.write("\n# --- SUBFLOOR ---\n\n")
                f.write("g subfloor\n")
                for quad in self.original_floor_mesh:
                    for v in quad: f.write(f"v {v.x():.6f} {v.y():.6f} {v.z():.6f}\n")
                    n = QVector3D.crossProduct(quad[1]-quad[0], quad[3]-quad[0]).normalized()
                    if n.z() < 0: n = -n
                    f.write(f"vn {n.x():.6f} {n.y():.6f} {n.z():.6f}\n")
                    f.write(f"f {v_offset}//{n_offset} {v_offset+1}//{n_offset} {v_offset+2}//{n_offset} {v_offset+3}//{n_offset}\n")
                    v_offset += 4; n_offset += 1
        except IOError as e:
            print(f"Error exporting OBJ file: {e}")

    def export_scene_to_dxf_file(self, file_path, qr_pct=0.10):
        import ezdxf
        try:
            doc = ezdxf.new(dxfversion="R2010")
            msp = doc.modelspace()

            # Create separate layers for tiles and pedestals with distinct colors
            if 'TILES' not in doc.layers:
                doc.layers.new('TILES', dxfattribs={'color': 160})  # Gray for tiles
            if 'PEDESTALS' not in doc.layers:
                doc.layers.new('PEDESTALS', dxfattribs={'color': 3})  # Green for pedestals

            # -----------------------------------------
            # 1. EXPORT TILES AS CLEAN RECTANGLES
            # -----------------------------------------
            for tile_idx, tile in enumerate(self.tiles):
                top = tile.corners_top_xyz
                bottom = tile.corners_bottom_xyz
                if not top or not bottom or len(top) < 3:
                    continue

                # Add 3D faces without triangulation for clean appearance
                # Top face as single polygon (no triangulation)
                top_points = [(p.x(), p.y(), p.z()) for p in top]
                msp.add_3dface(
                    top_points,
                    dxfattribs={'layer': 'TILES', 'color': 160}
                )
                
                # Bottom face as single polygon (reversed)
                bottom_points = [(p.x(), p.y(), p.z()) for p in reversed(bottom)]
                msp.add_3dface(
                    bottom_points,
                    dxfattribs={'layer': 'TILES', 'color': 160}
                )
                
                # Side faces
                for i in range(len(top)):
                    p1t = top[i]
                    p2t = top[(i + 1) % len(top)]
                    p1b = bottom[i]
                    p2b = bottom[(i + 1) % len(bottom)]
                    
                    msp.add_3dface(
                        [(p1b.x(), p1b.y(), p1b.z()),
                         (p2b.x(), p2b.y(), p2b.z()),
                         (p2t.x(), p2t.y(), p2t.z()),
                         (p1t.x(), p1t.y(), p1t.z())],
                        dxfattribs={'layer': 'TILES', 'color': 160}
                    )

            # -----------------------------------------
            # 2. EXPORT PEDESTALS AS 3D CYLINDERS
            # -----------------------------------------
            for ped_idx, ped in enumerate(self.pedestals):
                x, y = ped["pos_xy"]
                z = ped["base_z"]
                r = ped["radius"]
                h = ped["height"]
                segments = 16

                for i in range(segments):
                    a1 = 2 * math.pi * i / segments
                    a2 = 2 * math.pi * (i + 1) / segments

                    p1b = (x + r * math.cos(a1), y + r * math.sin(a1), z)
                    p2b = (x + r * math.cos(a2), y + r * math.sin(a2), z)
                    p1t = (p1b[0], p1b[1], z + h)
                    p2t = (p2b[0], p2b[1], z + h)

                    msp.add_3dface([
                        p1b, p2b, p2t, p1t
                    ], dxfattribs={'layer': 'PEDESTALS', 'color': 3})

            # -----------------------------------------
            # 3. EXPORT REAL QR CODE IMAGES
            # -----------------------------------------
            import os

            # Ensure layers exist
            if 'QRCODES' not in doc.layers:
                doc.layers.new('QRCODES', dxfattribs={'color': 253})
            if 'ELEVATION_MAP' not in doc.layers:
                doc.layers.new('ELEVATION_MAP', dxfattribs={'color': 7})

            # Output folder next to DXF file
            img_dir = os.path.join(os.path.dirname(file_path), "qr_images")
            os.makedirs(img_dir, exist_ok=True)

            # -----------------------------------------
            # 3. EXPORT ELEVATION MAP (if available)
            # -----------------------------------------
            if self.original_floor_mesh:
                for quad in self.original_floor_mesh:
                    if len(quad) >= 4:
                        quad_points = [(v.x(), v.y(), v.z()) for v in quad]
                        msp.add_3dface(
                            quad_points,
                            dxfattribs={'layer': 'ELEVATION_MAP', 'color': 7}
                        )

            # -----------------------------------------
            # 4. EXPORT QR CODES
            # -----------------------------------------
            for tile in self.tiles:
                fp = tile.get_actual_xy_footprint()
                if not fp:
                    continue

                try:
                    # Tile bbox and top Z
                    xs = [p[0] for p in fp]
                    ys = [p[1] for p in fp]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    tile_w = max_x - min_x
                    tile_h = max_y - min_y
                    tile_top_z = tile.corners_top_xyz[0].z()

                    # QR sizing and placement - match draw_tile_qr logic
                    qr_pct_size = getattr(self, "qr_pct", qr_pct if qr_pct is not None else 0.10)
                    qr_size = min(tile_w, tile_h) * qr_pct_size
                    margin = min(tile_w, tile_h) * 0.02
                    
                    # Position at top-right corner with margin (lower-left corner of QR image)
                    insert_x = max_x - margin - qr_size
                    insert_y = max_y - margin - qr_size
                    cz = tile_top_z + 0.005
                    insert = (insert_x, insert_y, cz)

                    label = str(tile.qr_data)
                    img_path = os.path.join(img_dir, f"{label}.png")

                    # Create tile data with material properties for QR code
                    tile_data = {
                        'material': 'Tile',
                        'density': 1900,
                        'weight': 15,
                        'thermal_r': 0.05,
                        'thickness': f"{tile.thickness*1000:.1f}mm" if tile.thickness else "N/A"
                    }

                    # Generate QR PNG to disk and open for pixel size
                    generate_qr_png(label, tile_data, img_path, size=250)
                    img = Image.open(img_path)
                    abs_img_path = os.path.abspath(img_path)

                    # Register image definition (absolute path required by AutoCAD)
                    img_def = doc.add_image_def(
                        filename=abs_img_path,
                        size_in_pixel=img.size,
                    )
                    print(f"[DXF] IMAGEDEF for {label}: handle={img_def.dxf.handle}, path={abs_img_path}")

                    # Add IMAGE entity to modelspace using official ezdxf signature
                    img_entity = msp.add_image(
                        image_def=img_def,
                        insert=insert,
                        size_in_units=(qr_size, qr_size),
                        dxfattribs={"layer": "QRCODES"},
                    )
                    assert img_entity is not None
                    print(f"[DXF] IMAGE for {label}: handle={img_entity.dxf.handle}, insert={insert}, size={qr_size}")

                except Exception as e:
                    print(f"[DXF] ERROR exporting QR for tile {label}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # -----------------------------------------
            # Save output (robustly, handling permission errors)
            # -----------------------------------------
            try:
                out_path = Path(file_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # If destination exists, attempt to remove it first (may fail if file locked)
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except PermissionError:
                        print(f"Permission denied removing existing file: {out_path}")
                        # We'll save to a temporary file and attempt to replace

                # Save to a temporary file first
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                tmp_path = out_path.with_name(out_path.stem + f".tmp_{timestamp}.dxf")
                try:
                    doc.saveas(str(tmp_path))
                except PermissionError as e:
                    print("Permission denied when writing temporary DXF:", e)
                    raise

                # Move temp file to final destination (atomic on most platforms)
                try:
                    os.replace(str(tmp_path), str(out_path))
                    print("DXF successfully written to:", out_path)
                except PermissionError:
                    # Destination still not writable; save to alternative timestamped file
                    alt_path = out_path.with_name(out_path.stem + f".{timestamp}.dxf")
                    try:
                        os.replace(str(tmp_path), str(alt_path))
                        print(f"Destination locked; saved DXF to alternative path: {alt_path}")
                    except Exception as e:
                        print("Failed to move DXF to final or alternative location:", e)
                        raise

            except PermissionError as e:
                print("DXF Export Permission Error:", e)
                print(f"Check file is not open in another program and you have write permissions to: {file_path}")
            except Exception as e:
                print("DXF Export Error:", e)

        except Exception as e:
            # Outer try-catch for the DXF exporter
            print("DXF Export Error (fatal):", e)

    # --- END: Exporting Methods ---

# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def on_visualization_toggle(self):
        """Instantly update visualization toggles without recomputing layout."""
        self.gl_widget.show_wireframe = self.wireframe_cb.isChecked()
        self.gl_widget.show_elevation_map = self.elevmap_cb.isChecked()
        self.gl_widget.show_tiles = self.showtiles_cb.isChecked()
        self.gl_widget.update()

    def toggle_qr_visibility(self, state):
        self.gl_widget.show_qr_codes = bool(state)
        self.gl_widget.update()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Tile Layout Application (v3)"); self.setGeometry(50,50,1400,900)

        self.central_widget = QWidget(); self.setCentralWidget(self.central_widget)
        self.splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.scroll_area = QScrollArea(); self.scroll_area.setWidgetResizable(True)
        self.control_panel_widget = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel_widget)

        # Room input mode group
        room_mode_g = QGroupBox("Room Input")
        room_mode_f = QFormLayout()
        self.room_mode_cb = QtWidgets.QComboBox()
        self.room_mode_cb.addItems(["Manual (W×L)", "Polygon Points", "Import from File", "Import 3D Model"])
        self.room_mode_cb.currentTextChanged.connect(self.on_room_mode_changed)
        room_mode_f.addRow("Input Mode:", self.room_mode_cb)
        self.import_room_btn = QPushButton("Import Room File")
        self.import_room_btn.clicked.connect(self.import_room_file)
        self.edit_polygon_btn = QPushButton("Edit Polygon Points")
        self.edit_polygon_btn.clicked.connect(self.edit_polygon_points)
        room_mode_f.addRow(self.import_room_btn, self.edit_polygon_btn)

        # 3D Model Import controls
        self.import_3d_model_btn = QPushButton("Browse 3D Model (.obj/.dwg/.skp)")
        self.import_3d_model_btn.clicked.connect(self.import_3d_model)
        self.import_3d_model_btn.hide()
        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setStyleSheet("color: gray; font-size: 9pt;")
        self.model_status_label.hide()
        self.select_surface_btn = QPushButton("Select Surface for Tiling")
        self.select_surface_btn.clicked.connect(self.activate_surface_selection)
        self.select_surface_btn.hide()
        self.clear_selection_btn = QPushButton("Clear Selection")
        self.clear_selection_btn.clicked.connect(self.clear_surface_selection)
        self.clear_selection_btn.hide()
        self.surface_info_label = QLabel("No surface selected")
        self.surface_info_label.setStyleSheet("color: gray; font-size: 9pt;")
        self.surface_info_label.hide()

        room_mode_f.addRow(self.import_3d_model_btn)
        room_mode_f.addRow("Model:", self.model_status_label)
        room_mode_f.addRow(self.select_surface_btn, self.clear_selection_btn)
        room_mode_f.addRow("Surface:", self.surface_info_label)

        room_mode_g.setLayout(room_mode_f); self.control_layout.addWidget(room_mode_g)

        room_g = QGroupBox("Room Dimensions (meters)")
        room_f = QFormLayout()
        self.rw_in=QDoubleSpinBox(minimum=0.0,maximum=100,value=5.0,decimals=2,singleStep=0.1)
        self.rl_in=QDoubleSpinBox(minimum=0.0,maximum=100,value=4.0,decimals=2,singleStep=0.1)
        self.r_ttz_in=QDoubleSpinBox(minimum=-10,maximum=10,value=0.15,decimals=2,singleStep=0.05)
        room_f.addRow("Width:",self.rw_in); room_f.addRow("Length:",self.rl_in)
        room_f.addRow("Target Tile Top Z:", self.r_ttz_in)
        room_g.setLayout(room_f); self.control_layout.addWidget(room_g)

        tile_g = QGroupBox("Tile Dimensions (meters)")
        tile_f = QFormLayout()
        self.tw_in=QDoubleSpinBox(minimum=0.01,maximum=5,value=0.6,decimals=2,singleStep=0.01)
        self.tl_in=QDoubleSpinBox(minimum=0.01,maximum=5,value=0.6,decimals=2,singleStep=0.01)
        self.tt_in=QDoubleSpinBox(minimum=0.001,maximum=0.5,value=0.02,decimals=3,singleStep=0.001)
        tile_f.addRow("Width:",self.tw_in); tile_f.addRow("Length:",self.tl_in); tile_f.addRow("Thickness:",self.tt_in)
        tile_g.setLayout(tile_f); self.control_layout.addWidget(tile_g)

        # Material Properties Group
        material_g = QGroupBox("Material Properties")
        material_f = QFormLayout()

        # Material selector dropdown with default value set to "Tile"
        self.material_cb = QtWidgets.QComboBox()
        self.material_cb.addItems(["Stone", "Wood", "Tile", "Concrete", "Glass", "Gypsum board"])
        self.material_cb.setCurrentIndex(2)  # Set "Tile" as default
        self.material_cb.currentTextChanged.connect(self.on_material_changed)
        material_f.addRow("Material:", self.material_cb)

        # Import Material Texture Button
        self.import_material_btn = QPushButton("Import Material Texture")
        self.import_material_btn.clicked.connect(self.import_material_texture)
        self.material_texture_label = QLabel("No texture loaded")
        self.material_texture_label.setWordWrap(True)
        self.material_texture_label.setStyleSheet("color: gray; font-size: 9pt;")
        material_f.addRow(self.import_material_btn)
        material_f.addRow("Texture:", self.material_texture_label)

        # Material property input fields (all editable)
        self.density_in = QDoubleSpinBox(minimum=0, maximum=10000, value=1900, decimals=1, singleStep=10)
        self.weight_in = QDoubleSpinBox(minimum=0, maximum=500, value=15, decimals=2, singleStep=0.5)
        self.thermal_r_in = QDoubleSpinBox(minimum=0, maximum=10, value=0.05, decimals=3, singleStep=0.01)

        material_f.addRow("Density (kg/m³):", self.density_in)
        material_f.addRow("Weight (kg/m²):", self.weight_in)
        material_f.addRow("Thermal R-value (ft²·°F·h/Btu):", self.thermal_r_in)

        # Store default values for each material (density, weight, thermal_r)
        self.material_defaults = {
            "Stone": {"density": 2600, "weight": 27, "thermal_r": 0.05},
            "Wood": {"density": 650, "weight": 10, "thermal_r": 0.8},
            "Tile": {"density": 1900, "weight": 15, "thermal_r": 0.05},
            "Concrete": {"density": 2200, "weight": 110, "thermal_r": 0.4},
            "Glass": {"density": 2500, "weight": 47, "thermal_r": 0.05},
            "Gypsum board": {"density": 600, "weight": 6, "thermal_r": 0.3}
        }

        # Track which fields have been manually edited
        self.material_edited = False
        self.density_edited = False
        self.weight_edited = False
        self.thermal_r_edited = False

        # Connect to valueChanged signals to track manual edits
        self.material_cb.currentTextChanged.connect(lambda: setattr(self, 'material_edited', True))
        self.density_in.valueChanged.connect(lambda: setattr(self, 'density_edited', True))
        self.weight_in.valueChanged.connect(lambda: setattr(self, 'weight_edited', True))
        self.thermal_r_in.valueChanged.connect(lambda: setattr(self, 'thermal_r_edited', True))

        material_g.setLayout(material_f)
        self.control_layout.addWidget(material_g)

        # Slope/Elevation controls
        slope_g = QGroupBox("Original Subfloor Slope / Elevation")
        slope_f = QFormLayout()
        self.sbz_in = QDoubleSpinBox(minimum=-10, maximum=10, value=-0.05, decimals=3, singleStep=0.01)
        self.sx_in = QDoubleSpinBox(minimum=-0.5, maximum=0.5, value=0.02, decimals=3, singleStep=0.001)
        self.sy_in = QDoubleSpinBox(minimum=-0.5, maximum=0.5, value=0.01, decimals=3, singleStep=0.001)
        slope_f.addRow("Base Z (at origin):", self.sbz_in)
        slope_f.addRow("Slope X (Z-change/m):", self.sx_in)
        slope_f.addRow("Slope Y (Z-change/m):", self.sy_in)
        self.elev_mode_cb = QtWidgets.QComboBox()
        self.elev_mode_cb.addItems(["Flat", "Planar", "Irregular Map"])
        self.import_elev_btn = QPushButton("Import Elevation Map")
        self.import_elev_btn.clicked.connect(self.import_elevation_file)
        slope_f.addRow("Elevation Mode:", self.elev_mode_cb)
        slope_f.addRow(self.import_elev_btn)
        slope_g.setLayout(slope_f); self.control_layout.addWidget(slope_g)

        # Raised floor controls
        raised_g = QGroupBox("Raised Floor Height")
        raised_f = QFormLayout()
        self.room_height_in = QDoubleSpinBox(minimum=0.0, maximum=10.0, value=0.40, decimals=3, singleStep=0.05)
        self.room_height_in.setToolTip("Desired clearance from tile surface to ceiling (headroom in meters). When importing a 3D model, ceiling height is detected automatically.")
        self.pedestal_min_height_in = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.10, decimals=3, singleStep=0.01)
        self.pedestal_min_height_in.setToolTip("Minimum fixed pedestal base height (meters)")
        
        # Create horizontal layouts with unit labels
        room_height_layout = QHBoxLayout()
        room_height_layout.addWidget(self.room_height_in)
        room_height_layout.addWidget(QLabel("m"))
        room_height_layout.addStretch()
        
        pedestal_height_layout = QHBoxLayout()
        pedestal_height_layout.addWidget(self.pedestal_min_height_in)
        pedestal_height_layout.addWidget(QLabel("m"))
        pedestal_height_layout.addStretch()
        
        raised_f.addRow("Desired headroom (tile to ceiling):", room_height_layout)
        raised_f.addRow("Pedestal min height:", pedestal_height_layout)
        raised_g.setLayout(raised_f)
        self.control_layout.addWidget(raised_g)

        # QR Code Size Slider
        qrsize_g = QGroupBox("QR Code Size")
        qrsize_f = QFormLayout()
        self.qrsize_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.qrsize_slider.setMinimum(0)
        self.qrsize_slider.setMaximum(50)
        self.qrsize_slider.setValue(10)
        self.qrsize_slider.setTickInterval(1)
        self.qrsize_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.qrsize_label = QLabel("QR Size: 10% of tile")
        self.qrsize_slider.valueChanged.connect(lambda v: self.qrsize_label.setText(f"QR Size: {v}% of tile"))
        self.qrsize_slider.valueChanged.connect(self.on_qrsize_changed)
        qrsize_f.addRow(self.qrsize_label, self.qrsize_slider)
        qrsize_g.setLayout(qrsize_f)
        self.control_layout.addWidget(qrsize_g)

        # Visualization toggles
        vis_g = QGroupBox("Visualization")
        vis_f = QFormLayout()
        self.wireframe_cb = QtWidgets.QCheckBox("Wireframe Tiles")
        self.elevmap_cb = QtWidgets.QCheckBox("Show Elevation Map")
        self.showtiles_cb = QtWidgets.QCheckBox("Show Tiles")
        self.showtiles_cb.setChecked(True)
        self.showqr_cb = QtWidgets.QCheckBox("Show QR Codes")
        self.showqr_cb.setChecked(True)
        self.wireframe_cb.stateChanged.connect(self.on_visualization_toggle)
        self.elevmap_cb.stateChanged.connect(self.on_visualization_toggle)
        self.showtiles_cb.stateChanged.connect(self.on_visualization_toggle)
        self.showqr_cb.stateChanged.connect(self.toggle_qr_visibility)
        vis_f.addRow(self.wireframe_cb)
        vis_f.addRow(self.elevmap_cb)
        vis_f.addRow(self.showtiles_cb)
        vis_f.addRow(self.showqr_cb)
        vis_g.setLayout(vis_f)
        self.control_layout.addWidget(vis_g)

        # Compute button
        self.comp_btn = QPushButton("Compute and Visualize Layout")
        self.comp_btn.clicked.connect(self.update_visualization)
        self.control_layout.addWidget(self.comp_btn)

        # Export buttons
        self.export_obj_btn = QPushButton("Export Scene to .OBJ File")
        self.export_obj_btn.clicked.connect(self.export_scene_to_obj)
        self.control_layout.addWidget(self.export_obj_btn)

        self.export_report_btn = QPushButton("Export Layout Report (.txt)")
        self.export_report_btn.clicked.connect(self.export_layout_report)
        self.control_layout.addWidget(self.export_report_btn)

        self.export_dxf_btn = QPushButton("Export to AutoCAD (.DXF)")
        self.export_dxf_btn.clicked.connect(self.export_scene_to_dxf)
        self.control_layout.addWidget(self.export_dxf_btn)

        self.control_layout.addStretch()
        self.scroll_area.setWidget(self.control_panel_widget)

        # Create GL widget and container
        self.splitter.addWidget(self.scroll_area)
        self.gl_widget = GLWidget(self)
        self.gl_widget.tileClicked.connect(self.show_tile_info_dialog)
        self.gl_widget.pedestalClicked.connect(self.show_pedestal_height)

        gl_container = QWidget()
        gl_container_layout = QVBoxLayout(gl_container)
        gl_container_layout.setContentsMargins(0, 0, 0, 0)
        gl_container_layout.addWidget(self.gl_widget)

        # Pedestal height display panel
        self.pedestal_height_panel = QLabel()
        self.pedestal_height_panel.setStyleSheet("""
            QLabel {
                background-color: rgba(50, 50, 50, 220);
                color: white;
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 14pt;
                font-weight: bold;
                border: 2px solid #FFA500;
            }
        """)
        self.pedestal_height_panel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pedestal_height_panel.hide()
        self.pedestal_height_panel.setParent(self.gl_widget)

        self.splitter.addWidget(gl_container)
        self.splitter.setSizes([380, self.width() - 380])
        self.splitter.setStretchFactor(1, 1)

        main_hbox = QHBoxLayout(self.central_widget)
        main_hbox.addWidget(self.splitter)

        # Defer first visualization
        QtCore.QTimer.singleShot(0, self.update_visualization)

    def on_room_mode_changed(self, mode_text):
        """Show/hide controls based on selected room input mode."""
        # Hide all mode-specific controls first
        self.import_room_btn.hide()
        self.edit_polygon_btn.hide()
        self.import_3d_model_btn.hide()
        self.model_status_label.hide()
        self.select_surface_btn.hide()
        self.clear_selection_btn.hide()
        self.surface_info_label.hide()

        # Show relevant controls based on mode
        if "Manual" in mode_text:
            pass  # Manual mode uses default W×L inputs (always visible)
            # Reset import mode flag when switching to manual
            self.gl_widget.has_imported_model = False
        elif "Polygon" in mode_text:
            self.edit_polygon_btn.show()
            # Reset import mode flag when switching to polygon mode
            self.gl_widget.has_imported_model = False
        elif "Import from File" in mode_text:
            self.import_room_btn.show()
            # Reset import mode flag when switching to file import mode
            self.gl_widget.has_imported_model = False
        elif "Import 3D Model" in mode_text:
            self.import_3d_model_btn.show()
            self.model_status_label.show()
            self.select_surface_btn.show()
            self.clear_selection_btn.show()
            self.surface_info_label.show()

    def export_scene_to_obj(self):
        """Save scene to an OBJ file using the GL widget exporter."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Scene to OBJ", "", "OBJ Files (*.obj);;All Files (*)")
        if not path:
            return
        try:
            self.gl_widget.export_scene_to_obj_file(path)
            QMessageBox.information(self, "Export Complete", f"OBJ exported to: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export OBJ:\n{e}")

    def export_layout_report(self):
        """Write the current generated layout report to a text file."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Layout Report", "", "Text Files (*.txt);;All Files (*)")
        if not path:
            return
        try:
            report = self.gl_widget.generate_layout_report_string()
            with open(path, 'w', encoding='utf-8') as f:
                f.write(report)
            QMessageBox.information(self, "Report Saved", f"Layout report saved to: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Failed to save report:\n{e}")

    def export_scene_to_dxf(self):
        """Save scene to a DXF file using the GL widget exporter."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Scene to DXF", "", "DXF Files (*.dxf);;All Files (*)")
        if not path:
            return
        try:
            self.gl_widget.export_scene_to_dxf_file(path, qr_pct=getattr(self.gl_widget, 'qr_pct', 0.10))
            QMessageBox.information(self, "Export Complete", f"DXF exported to: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export DXF:\n{e}")

    def import_3d_model(self):
        """Import a 3D model file (.obj, .dwg, .dxf, .skp) using unified model_loader."""
        if not MODEL_IMPORT_AVAILABLE:
            QMessageBox.critical(
                self,
                "Model Import Unavailable",
                "3D model import modules are not available.\n\n"
                "Required libraries:\n"
                "• trimesh - for mesh loading\n"
                "• ezdxf - for DWG/DXF files\n\n"
                "Install with:\n"
                "pip install trimesh ezdxf\n\n"
                "For SketchUp support, also install:\n"
                "pip install trimesh[easy]"
            )
            return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import 3D Model",
            "",
            "3D Model Files (*.obj *.dwg *.dxf *.skp);;OBJ Files (*.obj);;DWG/DXF Files (*.dwg *.dxf);;SketchUp Files (*.skp);;All Files (*)"
        )
        if not path:
            return

        try:
            # Use unified model loader from model_loader.py
            mesh = load_model(path)

            # Normalize units (convert mm to meters if needed)
            mesh = normalize_mesh_units(mesh, target_unit='meters')


            # Convert UnifiedMesh to trimesh for compatibility with SurfaceSelector
            if hasattr(mesh, 'to_trimesh'):
                mesh = mesh.to_trimesh()

            # Store the mesh in GL widget
            self.gl_widget.imported_mesh = mesh
            
            # Extract floor geometry for uneven-floor pedestal support
            self.gl_widget.extract_floor_from_mesh(mesh)
            
            # Detect ceiling height (max Z of all vertices)
            if len(mesh.vertices) > 0:
                self.gl_widget.ceiling_z = float(np.max(mesh.vertices[:, 2]))
                print(f"🔍 DEBUG: Detected ceiling_z = {self.gl_widget.ceiling_z:.3f}m from imported model")
            else:
                self.gl_widget.ceiling_z = None
            
            # Initialize SurfaceSelector immediately so groups are available for pick/selection
            try:
                self.gl_widget.surface_selector = SurfaceSelector(mesh)
            except Exception as e:
                # Non-fatal: leave selector None and fallback to per-face selection
                print(f"Warning: SurfaceSelector initialization failed: {e}")
            self.gl_widget.surface_selection_mode = False
            self.gl_widget.selected_surface = None

            # Set import mode flag to prevent default floor rendering
            self.gl_widget.has_imported_model = True
            
            # Clear existing tiles, pedestals, and default floor mesh so only imported model is visible
            self.gl_widget.tiles.clear()
            self.gl_widget.pedestals.clear()
            self.gl_widget.original_floor_mesh.clear()

            # Auto-position camera to frame the imported model
            self._frame_imported_mesh(mesh)

            # Update UI
            import os
            filename = os.path.basename(path)
            self.model_status_label.setText(f"✓ {filename}")
            self.model_status_label.setStyleSheet("color: green; font-size: 9pt;")
            self.surface_info_label.setText("No surface selected")
            self.surface_info_label.setStyleSheet("color: gray; font-size: 9pt;")

            # Hide tiles during 3D model viewing
            self.showtiles_cb.setChecked(False)

            # Update the 3D view to show the texture
            self.gl_widget.update()

            # Inform user about precomputed surface groups (if available)
            try:
                if hasattr(self.gl_widget, 'surface_selector') and self.gl_widget.surface_selector is not None:
                    group_count = self.gl_widget.surface_selector.get_group_count()
                    face_count = len(mesh.faces)
                    perf_mode = "Wireframe (Fast)" if face_count > 10000 else "Full Detail"
                    perf_tip = "\n⚡ Large model detected - using optimized rendering mode." if face_count > 10000 else ""
                    QMessageBox.information(
                    self,
                    "Model Loaded Successfully",
                    f"✅ 3D model loaded successfully!\n\n"
                    f"📁 File: {filename}\n"
                    f"📊 Faces: {face_count:,}\n"
                    f"📍 Vertices: {len(mesh.vertices):,}\n"
                    f"🔗 Surface groups detected: {group_count}\n"
                    f"🎨 Render Mode: {perf_mode}{perf_tip}\n\n"
                    f"📷 Camera automatically positioned to frame the model.\n\n"
                    f"Next steps:\n"
                    f"1. Rotate/zoom to view the model (drag mouse)\n"
                    f"2. Click 'Select Surface for Tiling'\n"
                    f"3. Click on any surface in the 3D view (Ctrl+click for multi-select)\n"
                    f"4. Click 'Compute and Visualize Layout' to generate tiles."
                )
                else:
                    QMessageBox.information(
                        self,
                        "Model Loaded Successfully",
                        f"✅ 3D model loaded successfully!\n\nFile: {filename}\nFaces: {len(mesh.faces):,}\nVertices: {len(mesh.vertices):,}\n\n"
                        f"Surface grouping not available; selection will be per-triangle."
                    )
            except Exception:
                pass

        except ModelLoadError as e:
            # Friendly error from model_loader.py with specific guidance
            QMessageBox.critical(
                self,
                "Could not load 3D Model",
                str(e)
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Unexpected Error",
                f"An unexpected error occurred:\n{str(e)}\n\n"
                f"Please try exporting your model as OBJ format for best compatibility."
            )

    # NOTE: _load_dwg_as_mesh is now replaced by model_loader.load_dwg_model()
    # Keeping this as reference only - not used anymore
    def _load_dwg_as_mesh_DEPRECATED(self, path):
        """Convert DWG/DXF 3D faces to a trimesh object with enhanced support for multiple entity types."""
        try:
            doc = ezdxf.readfile(path)
        except Exception as e:
            raise ValueError(f"Cannot read DWG/DXF file: {str(e)}")

        msp = doc.modelspace()

        vertices = []
        faces = []
        vertex_map = {}

        def get_vertex_index(v):
            """Add vertex to list if not exists, return index."""
            key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vertices.append([v[0], v[1], v[2]])
            return vertex_map[key]

        entity_count = {'3DFACE': 0, 'POLYLINE': 0, 'MESH': 0, 'POLYFACE': 0, 'OTHER': 0}

        # Extract 3DFACE entities (most common in CAD exports)
        for entity in msp.query('3DFACE'):
            entity_count['3DFACE'] += 1
            try:
                pts = [entity.dxf.vtx0, entity.dxf.vtx1, entity.dxf.vtx2]
                # Check if it's a quad (4th vertex different from 3rd)
                if hasattr(entity.dxf, 'vtx3'):
                    vtx3 = entity.dxf.vtx3
                    # If quad, split into two triangles
                    if vtx3 != entity.dxf.vtx2:
                        # Triangle 1: v0, v1, v2
                        indices = [get_vertex_index((p[0], p[1], p[2])) for p in pts]
                        faces.append(indices)
                        # Triangle 2: v0, v2, v3
                        indices2 = [get_vertex_index((pts[0][0], pts[0][1], pts[0][2])),
                                   get_vertex_index((pts[2][0], pts[2][1], pts[2][2])),
                                   get_vertex_index((vtx3[0], vtx3[1], vtx3[2]))]
                        faces.append(indices2)
                    else:
                        # Just a triangle
                        indices = [get_vertex_index((p[0], p[1], p[2])) for p in pts]
                        faces.append(indices)
                else:
                    # Triangle only
                    indices = [get_vertex_index((p[0], p[1], p[2])) for p in pts]
                    faces.append(indices)
            except Exception:
                continue

        # Extract MESH entities (AutoCAD mesh objects)
        for entity in msp.query('MESH'):
            entity_count['MESH'] += 1
            try:
                if hasattr(entity, 'vertices') and hasattr(entity, 'faces'):
                    vert_start_idx = len(vertices)
                    for v in entity.vertices:
                        vertices.append([v[0], v[1], v[2]])
                    for face in entity.faces:
                        if len(face) >= 3:
                            indices = [vert_start_idx + face[i] for i in range(3)]
                            faces.append(indices)
            except Exception:
                continue

        # Extract POLYFACE entities
        for entity in msp.query('POLYLINE'):
            if entity.is_poly_face_mesh:
                entity_count['POLYFACE'] += 1
                try:
                    mesh_vertices = list(entity.points())
                    vert_start_idx = len(vertices)
                    for v in mesh_vertices:
                        vertices.append([v[0], v[1], v[2]])
                    # Extract faces from polyface
                    if hasattr(entity, 'faces'):
                        for face in entity.faces():
                            face_indices = [i for i in face if i > 0]
                            if len(face_indices) >= 3:
                                indices = [vert_start_idx + face_indices[i] - 1 for i in range(3)]
                                faces.append(indices)
                except Exception:
                    continue
            else:
                entity_count['POLYLINE'] += 1

        # Build error message if no faces found
        if not faces:
            error_msg = "No 3D faces found in DWG/DXF file.\n\n"
            error_msg += "Entities found:\n"
            for entity_type, count in entity_count.items():
                if count > 0:
                    error_msg += f"  - {entity_type}: {count}\n"

            error_msg += "\n💡 Tips:\n"
            error_msg += "• Make sure the file contains 3D geometry (not just 2D drawings)\n"
            error_msg += "• In AutoCAD, use '3DFACE' or 'MESH' objects\n"
            error_msg += "• Try exporting as OBJ format for better compatibility\n"
            error_msg += "• In AutoCAD: File → Export → Select 'OBJ (*.obj)'"

            raise ValueError(error_msg)

        print(f"Loaded DWG/DXF: {len(vertices)} vertices, {len(faces)} faces")
        return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

    def _frame_imported_mesh(self, mesh):
        """Automatically position the camera to frame the imported 3D model."""
        # Calculate bounding box of the mesh
        vertices = np.array(mesh.vertices)
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        center = (min_coords + max_coords) / 2.0

        # Calculate the size of the bounding box
        size = max_coords - min_coords
        max_dimension = max(size)

        # Set camera to look at the center of the model
        self.gl_widget.room_center_for_rotation = QVector3D(
            float(center[0]),
            float(center[1]),
            float(center[2])
        )

        # Set camera distance based on model size (add some margin)
        self.gl_widget.camera_distance = max_dimension * 2.0

        # Reset camera angles to get a good initial view
        self.gl_widget.camera_azimuth = 45.0
        self.gl_widget.camera_elevation = 30.0

        # Update the GL widget
        self.gl_widget.makeCurrent()
        self.gl_widget.resizeGL(self.gl_widget.width(), self.gl_widget.height())
        self.gl_widget.doneCurrent()

        print(f"Camera positioned: center={center}, distance={self.gl_widget.camera_distance:.2f}")

    def activate_surface_selection(self):
        """Activate surface selection mode in the 3D viewer."""
        if not hasattr(self.gl_widget, 'imported_mesh') or self.gl_widget.imported_mesh is None:
            QMessageBox.warning(
                self,
                "No Model Loaded",
                "Please import a 3D model first before selecting a surface."
            )
            return

        # Toggle selection mode
        self.gl_widget.surface_selection_mode = not self.gl_widget.surface_selection_mode

        if self.gl_widget.surface_selection_mode:
            self.select_surface_btn.setText("Cancel Surface Selection")
            self.select_surface_btn.setStyleSheet("background-color: #FFA500; font-weight: bold;")
            QMessageBox.information(
                self,
                "Surface Selection Active",
                "Click on any surface in the 3D view to select it for tiling.\n\n"
                "The selected surface will turn green.\n"
                "Click 'Cancel Surface Selection' to exit this mode."
            )
        else:
            self.select_surface_btn.setText("Select Surface for Tiling")
            self.select_surface_btn.setStyleSheet("")

        self.gl_widget.update()

    def clear_surface_selection(self):
        """Clear all selected surfaces from the 3D model."""
        self.gl_widget.selected_surfaces.clear()
        self.gl_widget.selected_surface = None
        self.update_selection_info()
        self.gl_widget.update()

    def update_selection_info(self):
        """Update the surface info label to show the count of selected surfaces."""
        # Prefer to show number of logical surface groups selected when a selector exists
        sel = self.gl_widget.selected_surfaces
        if not sel:
            self.surface_info_label.setText("No surface selected")
            self.surface_info_label.setStyleSheet("color: gray; font-size: 9pt;")
            return

        if hasattr(self.gl_widget, 'surface_selector') and self.gl_widget.surface_selector is not None:
            # Map selected faces -> group ids
            gids = set()
            for f in sel:
                gid = self.gl_widget.surface_selector.get_group_id_for_face(f)
                if gid is not None:
                    gids.add(gid)
            group_count = len(gids) if gids else len(sel)
            if group_count == 1:
                self.surface_info_label.setText(f"✓ 1 surface selected (red)")
            else:
                self.surface_info_label.setText(f"✓ {group_count} surfaces selected (red)")
            self.surface_info_label.setStyleSheet("color: #CC3333; font-size: 9pt; font-weight: bold;")
        else:
            # Fallback: show number of faces selected
            count = len(sel)
            if count == 1:
                self.surface_info_label.setText(f"✓ 1 surface selected (red)")
            else:
                self.surface_info_label.setText(f"✓ {count} surfaces selected (red)")
            self.surface_info_label.setStyleSheet("color: #CC3333; font-size: 9pt; font-weight: bold;")

    def on_surface_selected(self, surface_idx):
        """Callback when a surface is selected from the 3D model.
        Extract dimensions using SurfaceSelector and update UI controls."""
        if not hasattr(self.gl_widget, 'imported_mesh') or self.gl_widget.imported_mesh is None:
            return

        mesh = self.gl_widget.imported_mesh

        # Use SurfaceSelector for enhanced selection
        if self.gl_widget.surface_selector is None:
            if MODEL_IMPORT_AVAILABLE:
                self.gl_widget.surface_selector = SurfaceSelector(mesh)
            else:
                # Fallback to simple bounding box
                face = mesh.faces[surface_idx]
                vertices = np.array([mesh.vertices[idx] for idx in face])
                min_coords = vertices.min(axis=0)
                max_coords = vertices.max(axis=0)
                dimensions = max_coords - min_coords
                width = float(max(dimensions[0], dimensions[1]))
                length = float(min(dimensions[0], dimensions[1]))
                self.rw_in.setValue(width)
                self.rl_in.setValue(length)
                self.surface_info_label.setText(f"✓ Selected: {width:.2f}m × {length:.2f}m")
                self.surface_info_label.setStyleSheet("color: green; font-size: 9pt;")
                return

        selector = self.gl_widget.surface_selector

        # Select the face and expand to coplanar region
        selector.select_face(surface_idx)
        coplanar_faces = selector.select_coplanar_region(surface_idx, angle_threshold=5.0)

        # Validate the surface
        validation = selector.validate_surface(coplanar_faces)

        if not validation.is_valid:
            QMessageBox.warning(
                self,
                "Surface Planarity Warning",
                f"{validation.message}\n\n"
                f"The surface may not be ideal for tiling.\n"
                f"Consider selecting a flatter surface or using manual input."
            )

        # Get projected dimensions (more accurate for tilted surfaces)
        projected_dims = selector.get_surface_bounds_projected(coplanar_faces)

        if projected_dims:
            width, length = projected_dims
        else:
            # Fallback to simple bounding box
            face = mesh.faces[surface_idx]
            vertices = np.array([mesh.vertices[idx] for idx in face])
            min_coords = vertices.min(axis=0)
            max_coords = vertices.max(axis=0)
            dimensions = max_coords - min_coords
            width = float(max(dimensions[0], dimensions[1]))
            length = float(min(dimensions[0], dimensions[1]))

        # Update room dimensions in UI
        self.rw_in.setValue(float(width))
        self.rl_in.setValue(float(length))

        # Update UI labels with coplanar face count
        face_info = f"{len(coplanar_faces)} faces" if len(coplanar_faces) > 1 else "1 face"
        self.surface_info_label.setText(f"✓ Selected: {width:.2f}m × {length:.2f}m ({face_info})")
        self.surface_info_label.setStyleSheet("color: green; font-size: 9pt; font-weight: bold;")

        self.select_surface_btn.setText("Select Surface for Tiling")
        self.select_surface_btn.setStyleSheet("")

        # Store surface info for layout generation
        self._selected_surface_info = selector.get_surface_info(coplanar_faces)

        QMessageBox.information(
            self,
            "Surface Selected",
            f"Surface dimensions extracted:\n\n"
            f"Width: {width:.2f} meters\n"
            f"Length: {length:.2f} meters\n"
            f"Faces: {len(coplanar_faces)}\n"
            f"Planar: {'Yes ✓' if validation.is_planar else 'No ⚠'}\n\n"
            f"The room dimensions have been updated.\n"
            f"Click 'Compute and Visualize Layout' to generate tiles."
        )

    def on_material_changed(self, material_name):
        """Update material properties when material is selected from dropdown.
        Only fills in values if they haven't been manually edited by the user."""
        if material_name not in self.material_defaults:
            return

        defaults = self.material_defaults[material_name]

        # Block signals temporarily to avoid triggering edited flags
        self.density_in.blockSignals(True)
        self.weight_in.blockSignals(True)
        self.thermal_r_in.blockSignals(True)

        # Only update if the field hasn't been manually edited
        if not self.density_edited:
            self.density_in.setValue(defaults["density"])
        if not self.weight_edited:
            self.weight_in.setValue(defaults["weight"])
        if not self.thermal_r_edited:
            self.thermal_r_in.setValue(defaults["thermal_r"])

        # Re-enable signals
        self.density_in.blockSignals(False)
        self.weight_in.blockSignals(False)
        self.thermal_r_in.blockSignals(False)

    def on_qrsize_changed(self, value):
        # Update GL widget QR percent and clear cached QR textures so they are recreated
        pct = max(0.0, min(1.0, value / 100.0))
        self.gl_widget.qr_pct = pct
        for t in self.gl_widget.tiles:
            try:
                if getattr(t, 'qr_texture_id', None):
                    # mark for recreation; do not attempt to delete GL texture here (context may be different)
                    t.qr_texture_id = None
                t.qr_size = None
            except Exception:
                pass
        self.gl_widget.update()

    def import_room_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Room Polygon", "", "Text/CSV Files (*.txt *.csv);;All Files (*)")
        if not path: return
        ok, err = self.gl_widget.room_input.load_from_file(path)
        if not ok:
            QMessageBox.critical(self, "Import Error", f"Could not parse room file:\n{err}")
            return
        QMessageBox.information(self, "Import Successful", f"Loaded polygon with {len(self.gl_widget.room_input.polygon)} points")

    def import_elevation_file(self):
        """Import elevation map from a text file with x,y,z coordinates."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Elevation Map (x,y,z per line)", "", "Text/CSV Files (*.txt *.csv);;All Files (*)")
        if not path:
            return
        ok, err = self.gl_widget.elevation_model.load_points_from_file(path)
        if not ok:
            QMessageBox.critical(self, "Import Error", f"Could not parse elevation file:\n{err}")
            return
        QMessageBox.information(self, "Import Successful", f"Loaded elevation map with {len(self.gl_widget.elevation_model.points)} points")

    def import_material_texture(self):
        """Import a seamless material texture (PNG or JPG) and apply it to all tiles."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Material Texture",
            "",
            "Image Files (*.png *.jpg *.jpeg);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        if not path:
            return

        # Load the texture into OpenGL
        success, error = self.gl_widget.load_material_texture(path)

        if not success:
            QMessageBox.critical(self, "Import Error", f"Could not load texture:\n{error}")
            return

        # Update UI to show loaded texture
        import os
        filename = os.path.basename(path)
        self.material_texture_label.setText(f"✓ {filename}")
        self.material_texture_label.setStyleSheet("color: green; font-size: 9pt;")

        # Update the 3D view to show the texture
        self.gl_widget.update()

        QMessageBox.information(
            self,
            "Texture Loaded",
            f"Material texture loaded successfully!\n\n"
            f"The texture will be applied to all tiles.\n"
            f"Click 'Compute and Visualize Layout' to see the textured tiles."
        )

    def edit_polygon_points(self):
        # small dialog to paste points (x y per line)
        d = QDialog(self)
        d.setWindowTitle("Edit Polygon Points (x y per line)")
        v = QVBoxLayout(d)
        te = QtWidgets.QPlainTextEdit()
        # populate existing
        if self.gl_widget.room_input.polygon:
            txt = "\n".join(f"{x} {y}" for x,y in self.gl_widget.room_input.polygon)
            te.setPlainText(txt)
        v.addWidget(te)
        hb = QHBoxLayout(); okb = QPushButton("OK"); cb = QPushButton("Cancel")
        hb.addWidget(okb); hb.addWidget(cb)
        v.addLayout(hb)
        def do_ok():
            txt = te.toPlainText().strip()
            pts = []
            for ln in txt.splitlines():
                ln = ln.strip()
                if not ln: continue
                parts = [p for p in ln.replace(',', ' ').split() if p]
                if len(parts) < 2: continue
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except Exception:
                    continue
            if len(pts) < 3:
                QMessageBox.critical(self, "Invalid Polygon", "Please enter at least 3 points.")
                return
            self.gl_widget.room_input.set_polygon(pts)
            d.accept()
        okb.clicked.connect(do_ok); cb.clicked.connect(d.reject)
        d.exec()


    def get_parameters(self):
        # derive room mode
        rm_text = self.room_mode_cb.currentText()
        if 'Manual' in rm_text:
            mode = 'manual'
        elif 'Polygon' in rm_text:
            mode = 'polygon'
        else:
            mode = 'file'

        polygon = []
        if self.gl_widget.room_input.polygon:
            polygon = list(self.gl_widget.room_input.polygon)

        elev_text = self.elev_mode_cb.currentText()
        if 'Planar' in elev_text:
            elev_mode = 'planar'
        elif 'Flat' in elev_text:
            elev_mode = 'flat'
        else:
            elev_mode = 'irregular'

        params = {
            'width':    self.rw_in.value(),'length':   self.rl_in.value(),'target_top_z': self.r_ttz_in.value(),
            'mode': mode,
            'polygon': polygon,
            'elevation_mode': elev_mode,
            'flat_z': self.sbz_in.value() if elev_mode=='flat' else 0.0,
            'tile': {'width':self.tw_in.value(),'length':self.tl_in.value(),'thickness': self.tt_in.value()},
            'slope': {'base_z':self.sbz_in.value(),'slope_x':self.sx_in.value(),'slope_y': self.sy_in.value()},
            'room_height': self.room_height_in.value(),
            'pedestal_min_height': self.pedestal_min_height_in.value()
        }
        # visualization toggles
        self.gl_widget.show_wireframe = self.wireframe_cb.isChecked()
        self.gl_widget.show_elevation_map = self.elevmap_cb.isChecked()
        self.gl_widget.show_tiles = self.showtiles_cb.isChecked()
        self.gl_widget.show_qr_codes = self.showqr_cb.isChecked()
        return params

    def update_visualization(self):
        """
        Main callback for 'Compute and Visualize Layout' button.
        If surfaces are selected from a 3D model, generates tiles/pedestals
        ONLY on those surfaces. Otherwise uses standard room-based layout.
        """
        params = self.get_parameters()
        
        # If surfaces are selected from imported 3D model, extract floor polygon
        if (self.gl_widget.imported_mesh is not None and 
            self.gl_widget.selected_surfaces):
            
            import numpy as np
            mesh = self.gl_widget.imported_mesh
            selected_faces = list(self.gl_widget.selected_surfaces)
            
            # Extract all vertices from selected faces
            all_vertices = []
            for face_idx in selected_faces:
                if face_idx < len(mesh.faces):
                    face = mesh.faces[face_idx]
                    for vertex_idx in face:
                        if vertex_idx < len(mesh.vertices):
                            v = mesh.vertices[vertex_idx]
                            all_vertices.append([v[0], v[1], v[2]])
            
            if all_vertices:
                vertices_array = np.array(all_vertices)
                
                # Project to XY plane and find boundary
                xy_points = vertices_array[:, :2]
                
                # Compute convex hull to get boundary polygon
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(xy_points)
                    boundary_polygon = [(xy_points[i, 0], xy_points[i, 1]) for i in hull.vertices]
                    
                    # Update params to use polygon mode
                    params['mode'] = 'polygon'
                    params['polygon'] = boundary_polygon
                    
                    print(f"[DEBUG] Extracted floor polygon from {len(selected_faces)} selected faces, {len(boundary_polygon)} boundary points")
                    
                except Exception as e:
                    print(f"[WARNING] Could not compute floor polygon: {e}")
        
        self.gl_widget.compute_and_build_layout(params, params['tile'], params['slope'])

        # Refresh the 3D view
        self.gl_widget.update()

    def show_tile_info_dialog(self, tile: Tile3D):
        InfoDialog(tile, self).exec()

    def show_pedestal_height(self, pedestal):
        """Display the height of a clicked pedestal in a floating panel.

        Args:
            pedestal: Dictionary containing pedestal data including height
        """
        if pedestal is None:
            self.pedestal_height_panel.hide()
            return

        # Get height in millimeters
        height_mm = self.gl_widget.get_pedestal_height_mm(pedestal)
        height_cm = height_mm / 10.0
        min_height_mm = pedestal.get('min_height', 0.0) * 1000.0
        adjustable_mm = max(height_mm - min_height_mm, 0.0)

        # Get pedestal position for additional info
        px, py = pedestal['pos_xy']
        base_z_mm = pedestal['base_z'] * 1000.0

        # Format the display text
        display_text = (
            f"Pedestal Height: {height_mm:.1f} mm ({height_cm:.1f} cm)\n"
            f"Base segment: {min_height_mm:.1f} mm\n"
            f"Adjustable segment: {adjustable_mm:.1f} mm"
        )

        # Update label
        self.pedestal_height_panel.setText(display_text)
        self.pedestal_height_panel.adjustSize()

        # Position the panel at bottom-right of GL widget
        gl_width = self.gl_widget.width()
        gl_height = self.gl_widget.height()
        panel_width = self.pedestal_height_panel.width()
        panel_height = self.pedestal_height_panel.height()

        # Place at bottom-right with margin
        margin = 20
        x_pos = gl_width - panel_width - margin
        y_pos = gl_height - panel_height - margin

        self.pedestal_height_panel.move(x_pos, y_pos)
        self.pedestal_height_panel.show()
        self.pedestal_height_panel.raise_()  # Bring to front


def _run_app():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    try:
        sys.exit(app.exec())
    except SystemExit:
        pass


if __name__ == '__main__':
    # When launched directly, start the GUI so compute_and_build_layout runs once and
    # we can observe the debug output confirming no default floor square is created.
    _run_app()
