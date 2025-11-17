import sys
import math
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
faulthandler.enable()
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
        self.mode = 'planar'  # 'flat'|'planar'|'irregular'
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tiles = []
        self.pedestals = []
        self.original_floor_mesh = []
        self.setMouseTracking(True)
        self.selected_tile_index = -1

        self.camera_azimuth = 45.0
        self.camera_elevation = 30.0
        self.camera_distance = 15.0
        self.camera_target = QVector3D(0.05, 0.05, 0.0)
        #self.last_mouse_pos = QtCore.QPoint()
        self.last_mouse_pos = QtCore.QPointF()
        self.room_center_for_rotation = QVector3D(0.05,0.05,0)

        self.room_dims = {'width': 0.1, 'length': 0.1, 'target_top_z': 0.0}
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

        # Visualization toggles
        self.show_wireframe = False
        self.show_elevation_map = False
        self.show_tiles = True

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

        for ped in self.pedestals: self.draw_pedestal(ped)
        if self.show_tiles:
            for i, t in enumerate(self.tiles): self.draw_tile(t, is_selected=(i == self.selected_tile_index))
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

        # configure elevation model
        elev_mode = room_params_input.get('elevation_mode', 'planar')
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
        min_required_clearance = 0.1
        # sample four corners of bounding box for max subfloor (use world coords)
        subfloor_corners_z = [self.original_slope_func(origin_x, origin_y), self.original_slope_func(origin_x + rw_param, origin_y), self.original_slope_func(origin_x, origin_y + rl_param), self.original_slope_func(origin_x + rw_param, origin_y + rl_param)]
        max_subfloor_z_in_room = max(subfloor_corners_z)
        actual_tile_bottom_z = max(room_params_input.get('target_top_z', 0.0) - tile_params['thickness'], max_subfloor_z_in_room + min_required_clearance)
        actual_tile_top_z = actual_tile_bottom_z + tile_params['thickness']

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
        for key, data in logical_pedestals.items():
            if data['type'] == 3:
                px, py = data['pos']
                base_z = self.original_slope_func(px, py)
                height = actual_tile_bottom_z - base_z
                if height >= min_pedestal_height:
                    final_pedestals_map[key] = {'pos_xy': (px, py), 'base_z': base_z, 'height': height, 'radius': pedestal_cap_radius}

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
                    base_z = self.original_slope_func(adj_px, adj_py)
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
                    combined_tiles = list(dict.fromkeys(cluster.get("tiles", []) + ped.get("tiles", [])))
                    if combined_tiles:
                        cluster["tiles"] = combined_tiles

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

        # Generate floor mesh for drawing (same as before but use elevation model z)
        floor_segments = 40
        w_step = rw_param / floor_segments if floor_segments>0 else rw_param
        l_step = rl_param / floor_segments if floor_segments>0 else rl_param
        for i in range(floor_segments):
            for j in range(floor_segments):
                x0, y0 = origin_x + i * w_step, origin_y + j * l_step
                x1, y1 = origin_x + (i + 1) * w_step, origin_y + (j + 1) * l_step
                p1 = QVector3D(x0, y0, self.original_slope_func(x0, y0))
                p2 = QVector3D(x1, y0, self.original_slope_func(x1, y0))
                p3 = QVector3D(x1, y1, self.original_slope_func(x1, y1))
                p4 = QVector3D(x0, y1, self.original_slope_func(x0, y1))
                self.original_floor_mesh.append([p1, p2, p3, p4])

        self.update()

    def draw_tile(self, tile: Tile3D, is_selected=False):
        if not tile.corners_top_xyz: return
        if is_selected: glColor3f(1.0, 0.7, 0.0)
        elif tile.is_cut: glColor3f(0.65, 0.75, 0.9)
        else: glColor3f(0.9, 0.9, 0.8)

        # Top
        glBegin(GL_POLYGON); glNormal3f(0,0,1)
        for v in tile.corners_top_xyz: glVertex3f(v.x(), v.y(), v.z())
        glEnd()

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

    def draw_pedestal(self, pedestal):
        if self.unit_cylinder_dl == -1: self.unit_cylinder_dl = self.create_unit_cylinder_dl()

        glPushMatrix()
        glTranslatef(pedestal['pos_xy'][0], pedestal['pos_xy'][1], pedestal['base_z'])

        total_h, cap_r = pedestal['height'], pedestal['radius']
        BASE_H_F, STEM_H_F, CAP_H_F = 0.10, 0.80, 0.10
        BASE_R_S, STEM_R_S = 1.0, 0.7

        current_z = 0.0
        # Draw Base
        base_h, base_r = total_h * BASE_H_F, cap_r * BASE_R_S
        glPushMatrix(); glTranslatef(0, 0, current_z); glScalef(base_r, base_r, base_h)
        glColor3f(0.22, 0.22, 0.25); glCallList(self.unit_cylinder_dl); glPopMatrix()
        current_z += base_h
        # Draw Stem
        stem_h, stem_r = total_h * STEM_H_F, cap_r * STEM_R_S
        glPushMatrix(); glTranslatef(0, 0, current_z); glScalef(stem_r, stem_r, stem_h)
        glColor3f(0.28, 0.28, 0.31); glCallList(self.unit_cylinder_dl); glPopMatrix()
        current_z += stem_h
        # Draw Cap
        cap_h = total_h * CAP_H_F
        glPushMatrix(); glTranslatef(0, 0, current_z); glScalef(cap_r, cap_r, cap_h)
        glColor3f(0.35, 0.35, 0.38); glCallList(self.unit_cylinder_dl); glPopMatrix()

        glPopMatrix()

    def draw_original_floor(self):
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
        if not self.original_floor_mesh:
            self.draw_original_floor(); return
        # compute min/max z
        zs = [v.z() for quad in self.original_floor_mesh for v in quad]
        if not zs: return
        zmin, zmax = min(zs), max(zs)
        rng = max(zmax - zmin, 1e-6)
        for quad in self.original_floor_mesh:
            avgz = sum(v.z() for v in quad)/4.0
            t = (avgz - zmin)/rng
            # color map: blue (low) -> red (high)
            r = t; g = 0.2 + 0.6*(1.0 - abs(0.5 - t)); b = 1.0 - t
            glColor3f(r, g, b)
            glBegin(GL_QUADS)
            n = QVector3D.crossProduct(quad[1]-quad[0], quad[3]-quad[0]).normalized()
            if n.z() < 0: n = -n
            glNormal3f(n.x(), n.y(), n.z())
            for vertex in quad: glVertex3f(vertex.x(), vertex.y(), vertex.z())
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

    def draw_axes(self):
        glLineWidth(2.5); glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        ax_len = max(0.5, self.room_dims.get('width', 1.0) * 0.15, self.room_dims.get('length', 1.0) * 0.15)
        glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(ax_len,0,0)
        glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,ax_len,0)
        glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,ax_len)
        glEnd()
        glEnable(GL_LIGHTING); glLineWidth(1.0)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        #self.last_mouse_pos = event.pos()
        self.last_mouse_pos = event.position()
        if event.buttons() == QtCore.Qt.MouseButton.LeftButton:
            prev_idx = self.selected_tile_index
            self.selected_tile_index = -1
            pixel_ratio = self.devicePixelRatioF()
            tile_obj, tile_idx = self.pick_tile_accurate(
                event.position().x() * pixel_ratio,
                event.position().y() * pixel_ratio,
            )
            if tile_obj:
                self.selected_tile_index = tile_idx
                self.tileClicked.emit(tile_obj)
            if prev_idx != self.selected_tile_index: self.update()

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

    # --- START: New/Modified methods for Exporting ---

    def generate_layout_report_string(self):
        """Generates a formatted, human-readable string of the layout."""
        report_lines = []

        report_lines.append("=" * 84)
        report_lines.append("RAISED FLOOR LAYOUT REPORT".center(84))
        report_lines.append("=" * 84)
        # FIX: PyQt6 removed Qt.DefaultLocaleLongDate → use Python datetime formatting         generated_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")         report_lines.append(f"Generated on: {generated_timestamp}")
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

    # --- END: Exporting Methods ---

# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
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
        self.room_mode_cb.addItems(["Manual (W×L)", "Polygon Points", "Import from File"])
        room_mode_f.addRow("Input Mode:", self.room_mode_cb)
        self.import_room_btn = QPushButton("Import Room File")
        self.import_room_btn.clicked.connect(self.import_room_file)
        self.edit_polygon_btn = QPushButton("Edit Polygon Points")
        self.edit_polygon_btn.clicked.connect(self.edit_polygon_points)
        room_mode_f.addRow(self.import_room_btn, self.edit_polygon_btn)
        room_mode_g.setLayout(room_mode_f); self.control_layout.addWidget(room_mode_g)

        room_g = QGroupBox("Room Dimensions (meters)")
        room_f = QFormLayout()
        self.rw_in=QDoubleSpinBox(minimum=0.1,maximum=100,value=5.0,decimals=2,singleStep=0.1)
        self.rl_in=QDoubleSpinBox(minimum=0.1,maximum=100,value=4.0,decimals=2,singleStep=0.1)
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

        slope_g = QGroupBox("Original Subfloor Slope / Elevation")
        slope_f = QFormLayout()
        self.sbz_in=QDoubleSpinBox(minimum=-10,maximum=10,value=-0.05,decimals=3,singleStep=0.01)
        self.sx_in=QDoubleSpinBox(minimum=-0.5,maximum=0.5,value=0.02,decimals=3,singleStep=0.001)
        self.sy_in=QDoubleSpinBox(minimum=-0.5,maximum=0.5,value=0.01,decimals=3,singleStep=0.001)
        slope_f.addRow("Base Z (at origin):",self.sbz_in)
        slope_f.addRow("Slope X (Z-change/m):",self.sx_in); slope_f.addRow("Slope Y (Z-change/m):",self.sy_in)
        # Elevation mode and import
        self.elev_mode_cb = QtWidgets.QComboBox(); self.elev_mode_cb.addItems(["Planar", "Flat", "Irregular Map"])
        self.import_elev_btn = QPushButton("Import Elevation Map")
        self.import_elev_btn.clicked.connect(self.import_elevation_file)
        slope_f.addRow("Elevation Mode:", self.elev_mode_cb)
        slope_f.addRow(self.import_elev_btn)
        slope_g.setLayout(slope_f); self.control_layout.addWidget(slope_g)

        # Visualization toggles
        vis_g = QGroupBox("Visualization")
        vis_f = QFormLayout()
        self.wireframe_cb = QtWidgets.QCheckBox("Wireframe Tiles")
        self.elevmap_cb = QtWidgets.QCheckBox("Show Elevation Map")
        self.showtiles_cb = QtWidgets.QCheckBox("Show Tiles")
        self.showtiles_cb.setChecked(True)
        vis_f.addRow(self.wireframe_cb); vis_f.addRow(self.elevmap_cb); vis_f.addRow(self.showtiles_cb)
        vis_g.setLayout(vis_f); self.control_layout.addWidget(vis_g)

        self.comp_btn = QPushButton("Compute and Visualize Layout"); self.comp_btn.clicked.connect(self.update_visualization)
        self.control_layout.addWidget(self.comp_btn)

        # --- START: New UI elements for Exporting ---
        self.export_obj_btn = QPushButton("Export Scene to .OBJ File")
        self.export_obj_btn.clicked.connect(self.export_scene_to_obj)
        self.control_layout.addWidget(self.export_obj_btn)

        self.export_report_btn = QPushButton("Export Layout Report (.txt)")
        self.export_report_btn.clicked.connect(self.export_layout_report)
        self.control_layout.addWidget(self.export_report_btn)
        # --- END: New UI elements for Exporting ---

        self.control_layout.addStretch()
        self.scroll_area.setWidget(self.control_panel_widget)

        self.splitter.addWidget(self.scroll_area)
        self.gl_widget = GLWidget(self); self.gl_widget.tileClicked.connect(self.show_tile_info_dialog)
        self.splitter.addWidget(self.gl_widget)
        self.splitter.setSizes([380, self.width() - 380])
        self.splitter.setStretchFactor(1, 1)

        main_hbox = QHBoxLayout(self.central_widget)
        main_hbox.addWidget(self.splitter)

        # Defer first visualization until after the window is shown to ensure a GL context
        QtCore.QTimer.singleShot(0, self.update_visualization)

    def import_room_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Room Polygon", "", "Text/CSV Files (*.txt *.csv);;All Files (*)")
        if not path: return
        ok, err = self.gl_widget.room_input.load_from_file(path)
        if not ok:
            QMessageBox.critical(self, "Import Error", f"Could not parse room file:\n{err}")
            return
        QMessageBox.information(self, "Import Successful", f"Loaded polygon with {len(self.gl_widget.room_input.polygon)} points")

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

    def import_elevation_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Elevation Map (x,y,z per line)", "", "Text/CSV Files (*.txt *.csv);;All Files (*)")
        if not path: return
        ok, err = self.gl_widget.elevation_model.load_points_from_file(path)
        if not ok:
            QMessageBox.critical(self, "Import Error", f"Could not parse elevation file:\n{err}")
            return
        QMessageBox.information(self, "Import Successful", f"Loaded elevation map with {len(self.gl_widget.elevation_model.points)} points")

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
            'slope': {'base_z':self.sbz_in.value(),'slope_x':self.sx_in.value(),'slope_y': self.sy_in.value()}
        }
        # visualization toggles
        self.gl_widget.show_wireframe = self.wireframe_cb.isChecked()
        self.gl_widget.show_elevation_map = self.elevmap_cb.isChecked()
        self.gl_widget.show_tiles = self.showtiles_cb.isChecked()
        return params

    def update_visualization(self):
        params = self.get_parameters()
        self.gl_widget.compute_and_build_layout(params, params['tile'], params['slope'])

    def show_tile_info_dialog(self, tile: Tile3D):
        InfoDialog(tile, self).exec()

    # --- START: New methods for Exporting ---
    def export_scene_to_obj(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Scene to OBJ", "3d_tile_layout.obj", "Wavefront OBJ (*.obj)")
        if path:
            self.gl_widget.export_scene_to_obj_file(path)
            QMessageBox.information(self, "Export Successful", f"Scene successfully exported to:\n{path}")

    def export_layout_report(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Layout Report",
            "layout_report.txt",
            "Text Files (*.txt)"
        )

        if not path:
            return

        resolved_path = Path(path)

        if resolved_path.suffix.lower() != ".txt":
            if resolved_path.is_dir():
                resolved_path = resolved_path / "layout_report.txt"
            else:
                resolved_path = resolved_path.with_suffix(".txt")

        path = str(resolved_path)

        try:
            report_string = self.gl_widget.generate_layout_report_string()
            with open(path, "w", encoding="utf-8") as f:
                f.write(report_string)

            # SAFE: no QMessageBox here (avoids GL crash)
            print(f"[OK] Layout report saved to: {path}")

        except Exception as e:
            print("[ERROR - Could not write layout report]", e)
            # Error box is safe
            QMessageBox.critical(self, "Export Error", str(e))
    # --- END: New methods for Exporting ---

if __name__ == "__main__":
    # Set the desired OpenGL format BEFORE creating the application
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    # Request legacy, fixed-function compatible context (OpenGL 2.1, NoProfile)
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.NoProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(0)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)

    # Ensure Qt uses desktop OpenGL before creating the application
    QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseDesktopOpenGL)
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())
