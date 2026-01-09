#!/usr/bin/env python3
"""
Patch script to apply remaining changes to 3D Tiles App V3.py
Sections E, F, G, H, I, J, K
"""
import re

file_path = "3D Tiles App V3 .py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# SECTION E & F: Replace mousePressEvent and mouseMoveEvent
mouse_press_old = r'    def mousePressEvent\(self, event: QtGui\.QMouseEvent\):.*?(?=    def mouseMoveEvent)'
mouse_press_new = '''    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.last_mouse_pos = event.position()
        pixel_ratio = self.devicePixelRatioF()
        screen_x = event.position().x() * pixel_ratio
        screen_y = event.position().y() * pixel_ratio

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # ===== HUMAN REFERENCE TOOL =====
            if self.human_enabled:
                if self.is_click_on_human(screen_x, screen_y):
                    # Start dragging human
                    self.human_dragging = True
                    proj_pos = self.project_world_to_screen(self.human_position_xy[0], self.human_position_xy[1], 
                                                            self.human_anchor_z_at_xy(self.human_position_xy[0], self.human_position_xy[1]))
                    if proj_pos:
                        self.human_drag_offset = (screen_x - proj_pos[0], screen_y - proj_pos[1])
                    self.update()
                    return
                else:
                    # Click on floor places human there
                    if self._cached_modelview is not None and self._cached_projection is not None and self._cached_viewport is not None:
                        try:
                            nx, ny, nz = gluUnProject(screen_x, self._cached_viewport[3] - screen_y, 0.0,
                                                       self._cached_modelview, self._cached_projection, self._cached_viewport)
                            fx, fy, fz = gluUnProject(screen_x, self._cached_viewport[3] - screen_y, 1.0,
                                                       self._cached_modelview, self._cached_projection, self._cached_viewport)
                            rd_x, rd_y, rd_z = fx - nx, fy - ny, fz - nz
                            ray_len = math.hypot(rd_x, math.hypot(rd_y, rd_z))
                            if ray_len > EPSILON:
                                rd_x /= ray_len
                                rd_y /= ray_len
                                rd_z /= ray_len
                                # Snap to anchor plane
                                anchor_z = self.human_anchor_z_at_xy(nx, ny)
                                if abs(rd_z) > EPSILON:
                                    t = (anchor_z - nz) / rd_z
                                    if t > 0:
                                        click_x = nx + rd_x * t
                                        click_y = ny + rd_y * t
                                        self.human_position_xy = (click_x, click_y)
                                        self.update()
                                        return
                        except Exception:
                            pass
            
            # ===== MEASUREMENT TOOL =====
            if self.measurement_enabled:
                if self._cached_modelview is not None and self._cached_projection is not None and self._cached_viewport is not None:
                    try:
                        nx, ny, nz = gluUnProject(screen_x, self._cached_viewport[3] - screen_y, 0.0,
                                                   self._cached_modelview, self._cached_projection, self._cached_viewport)
                        fx, fy, fz = gluUnProject(screen_x, self._cached_viewport[3] - screen_y, 1.0,
                                                   self._cached_modelview, self._cached_projection, self._cached_viewport)
                        
                        # Snap to appropriate plane
                        if self.measurement_mode == 'INSTALLATION_PLANE' and self.tiles:
                            snap_z = self.room_dims.get('target_top_z', 0.0)
                        elif self.measurement_mode == 'ROOM_DIMENSION':
                            snap_z = self.original_slope_func(nx, ny)
                        else:  # SURFACE (future)
                            snap_z = self.original_slope_func(nx, ny)
                        
                        rd_x, rd_y, rd_z = fx - nx, fy - ny, fz - nz
                        if abs(rd_z) > EPSILON:
                            t = (snap_z - nz) / rd_z
                            click_x = nx + rd_x * t
                            click_y = ny + rd_y * t
                            self.add_measurement_point(click_x, click_y, snap_z)
                            self.update()
                            return
                    except Exception:
                        pass
            
            # ===== EXISTING SURFACE SELECTION & TILE/PEDESTAL PICKING =====
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
        
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            # Right-click to clear measurements
            if self.measurement_enabled:
                self.clear_measurements()
                self.update()

    def mouseMoveEvent'''

content = re.sub(mouse_press_old, mouse_press_new, content, flags=re.DOTALL)

# SECTION F: mouseMoveEvent replacement
mouse_move_old = r'    def mouseMoveEvent\(self, event: QtGui\.QMouseEvent\):.*?(?=\n    def wheelEvent)'
mouse_move_new = '''    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        dx = event.position().x() - self.last_mouse_pos.x()
        dy = event.position().y() - self.last_mouse_pos.y()
        buttons = event.buttons()
        
        # ===== HUMAN DRAGGING =====
        if self.human_dragging and self.human_enabled:
            pixel_ratio = self.devicePixelRatioF()
            screen_x = event.position().x() * pixel_ratio
            screen_y = event.position().y() * pixel_ratio
            
            if self._cached_modelview is not None and self._cached_projection is not None and self._cached_viewport is not None:
                try:
                    nx, ny, nz = gluUnProject(screen_x, self._cached_viewport[3] - screen_y, 0.0,
                                               self._cached_modelview, self._cached_projection, self._cached_viewport)
                    fx, fy, fz = gluUnProject(screen_x, self._cached_viewport[3] - screen_y, 1.0,
                                               self._cached_modelview, self._cached_projection, self._cached_viewport)
                    rd_x, rd_y, rd_z = fx - nx, fy - ny, fz - nz
                    ray_len = math.hypot(rd_x, math.hypot(rd_y, rd_z))
                    if ray_len > EPSILON:
                        rd_x /= ray_len
                        rd_y /= ray_len
                        rd_z /= ray_len
                        # Project to anchor plane
                        anchor_z = self.human_anchor_z_at_xy(nx, ny)
                        if abs(rd_z) > EPSILON:
                            t = (anchor_z - nz) / rd_z
                            drag_x = nx + rd_x * t
                            drag_y = ny + rd_y * t
                            self.human_position_xy = (drag_x, drag_y)
                            self.update()
                except Exception:
                    pass
            return
        
        # ===== EXISTING CAMERA CONTROLS =====
        if buttons & QtCore.Qt.MouseButton.LeftButton:
            self.camera_azimuth += dx * 0.35
            self.camera_elevation += dy * 0.35
            self.camera_elevation = max(-89.9, min(89.9, self.camera_elevation))
            self.update()
        elif buttons & QtCore.Qt.MouseButton.MiddleButton:
            pan_speed = 0.0015 * self.camera_distance
            right_rad = math.radians(self.camera_azimuth + 90.0)
            self.room_center_for_rotation.setX(self.room_center_for_rotation.x() - dx * math.cos(right_rad) * pan_speed)
            self.room_center_for_rotation.setY(self.room_center_for_rotation.y() - dx * math.sin(right_rad) * pan_speed)
            self.room_center_for_rotation.setZ(self.room_center_for_rotation.z() + dy * pan_speed)
            self.update()
        self.last_mouse_pos = event.position()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        """Handle mouse release (stop dragging human)."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.human_dragging = False
            self.update()

    def wheelEvent'''

content = re.sub(mouse_move_old, mouse_move_new, content, flags=re.DOTALL)

# Save the updated file
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Sections E & F applied (mouse events)")
