import bisect

import numpy as np
from loguru import logger
from scipy.interpolate import splprep, splev
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsPathItem
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPen, QPainterPath, QColor


class Point(QGraphicsEllipseItem):
    """Class that describes a spline point"""

    def __init__(self, pos, line_thickness=1, point_radius=10, color=None, transparency=255):
        super(Point, self).__init__()
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.transparency = transparency
        self.default_color = get_qt_pen(color, line_thickness, transparency)

        self.setPen(self.default_color)
        # Store center position
        self.center_x, self.center_y = pos[0], pos[1]
        self.setRect(
            pos[0] - self.point_radius * 0.5, 
            pos[1] - self.point_radius * 0.5, 
            self.point_radius, 
            self.point_radius
        )

    def get_coords(self):
        try:
            # Return the center coordinates
            rect = self.rect()
            return rect.x() + rect.width()/2, rect.y() + rect.height()/2
        except RuntimeError:  # Point has been deleted
            return None, None
        
    def get_center(self):
        """Return the exact center of the point"""
        return self.center_x, self.center_y

    def update_color(self):
        self.setPen(QPen(Qt.GlobalColor.transparent, self.line_thickness))

    def reset_color(self):
        self.setPen(self.default_color)

    def update_pos(self, pos):
        """Updates the Point position"""
        # Update center coordinates
        self.center_x, self.center_y = pos.x(), pos.y()
        self.setRect(
            pos.x() - self.point_radius * 0.5, 
            pos.y() - self.point_radius * 0.5, 
            self.point_radius, 
            self.point_radius
        )
        return self.rect()


class Spline(QGraphicsPathItem):
    """Class that describes a spline"""

    def __init__(self, points, n_points, line_thickness=1, color=None, transparency=255, **kwargs):
        super().__init__()
        self.n_points = n_points
        self.knot_points = None
        self.full_contour = None
        self.setPen(get_qt_pen(color, line_thickness, transparency))
        
        # Geometric setup
        self.start_coords = kwargs.get('start_coords', None)
        self.end_coords = kwargs.get('end_coords', None)
        
        # Initialize path
        self.path = QPainterPath()
        self.set_knot_points(points)


    def set_knot_points(self, points):
        try:
            # Ensure points are in the correct format
            if isinstance(points, list) and len(points) == 2:
                x_coords, y_coords = points
            else:
                logger.error(f"Invalid points format: {points}")
                return
            
            # For a closed periodic spline, duplicate the first point at the end
            if len(x_coords) > 0:
                if x_coords[0] != x_coords[-1] or y_coords[0] != y_coords[-1]:
                    x_coords.append(x_coords[0])
                    y_coords.append(y_coords[0])
            
            self.knot_points = [x_coords, y_coords]
            
            # Clear the existing path
            self.path = QPainterPath()
            
            # Start the path at the first point
            if len(x_coords) > 0:
                start_point = QPointF(x_coords[0], y_coords[0])
                self.path.moveTo(start_point)

            self.full_contour = self.interpolate(self.knot_points)
            if self.full_contour[0] is not None:
                # Draw the interpolated spline
                for i in range(1, len(self.full_contour[0])):
                    self.path.lineTo(self.full_contour[0][i], self.full_contour[1][i])

                self.setPath(self.path)
                self.path.closeSubpath()
        except Exception as e:
            logger.error(f"Error setting knot points: {e}")
            self.knot_points = None
            self.full_contour = (None, None)

    def interpolate(self, points):
        """Interpolates the spline points using B-splines"""
        try:
            points_array = np.array(points)
            
            # Check if we have enough points for interpolation
            if points_array.shape[1] < 4:  # Need at least 4 points for periodic spline
                logger.warning(f"Not enough points for spline interpolation: {points_array.shape[1]}")
                return (np.array(points[0]), np.array(points[1]))
            
            # Use smoothing parameter s=0 for exact interpolation through points
            # Use per=1 for periodic (closed) splines
            tck, u = splprep(points_array, u=None, s=0.0, per=1)
            
            # Generate interpolated points
            u_new = np.linspace(u.min(), u.max(), self.n_points)
            x_new, y_new = splev(u_new, tck, der=0)
            
            return (x_new, y_new)
        except Exception as e:
            logger.error(f"Error in spline interpolation: {e}")
            return (np.array(points[0]), np.array(points[1]))

    def update(self, pos, index, path_index=None):
        """Updates the stored spline everytime it is moved"""
        if self.knot_points is None:
            return index
        
        if path_index is not None:
            # Find the closest knot point to the clicked path position
            path_indices = np.zeros(len(self.knot_points[0]))
            for i in range(len(self.knot_points[0])):
                knot_x, knot_y = self.knot_points[0][i], self.knot_points[1][i]
                distances = np.sqrt(
                    (knot_x - self.full_contour[0]) ** 2 + (knot_y - self.full_contour[1]) ** 2
                )
                path_indices[i] = np.argmin(distances)
            
            # Find where to insert the new point
            insert_idx = bisect.bisect_left(path_indices, path_index)
            self.knot_points[0].insert(insert_idx, pos.x())
            self.knot_points[1].insert(insert_idx, pos.y())
            index = insert_idx
        else:
            if index >= len(self.knot_points[0]):
                self.knot_points[0].append(pos.x())
                self.knot_points[1].append(pos.y())
            else:
                self.knot_points[0][index] = pos.x()
                self.knot_points[1][index] = pos.y()
        
        # Ensure the spline remains closed
        if self.knot_points[0][0] != self.knot_points[0][-1] or self.knot_points[1][0] != self.knot_points[1][-1]:
            self.knot_points[0][-1] = self.knot_points[0][0]
            self.knot_points[1][-1] = self.knot_points[1][0]
        
        # Re-interpolate
        self.full_contour = self.interpolate(self.knot_points)
        
        # Rebuild the path
        self.path = QPainterPath(QPointF(self.full_contour[0][0], self.full_contour[1][0]))
        for i in range(1, len(self.full_contour[0])):
            self.path.lineTo(self.full_contour[0][i], self.full_contour[1][i])
        self.setPath(self.path)
        self.path.closeSubpath()

        return index

    def on_path(self, pos):
        if self.full_contour[0] is None:
            return None
        
        x, y = pos.x(), pos.y()
        distances = np.sqrt((self.full_contour[0] - x) ** 2 + (self.full_contour[1] - y) ** 2)
        min_dist = np.min(distances)
        
        if min_dist < 10:  # 10 pixel threshold
            return np.argmin(distances)
        return None

    def get_unscaled_contour(self, scaling_factor):
        if self.full_contour[0] is None:
            return None, None
        return self.full_contour[0] / scaling_factor, self.full_contour[1] / scaling_factor


def get_qt_pen(color, thickness, transparency=255):
    """Create a QPen with the specified color, thickness, and transparency"""
    if isinstance(color, str):
        # Try to get color from Qt.GlobalColor
        try:
            color_enum = getattr(Qt.GlobalColor, color.lower())
            pen_color = QColor(color_enum)
        except AttributeError:
            # Try to parse as hex color
            if color.startswith('#'):
                pen_color = QColor(color)
            else:
                # Default to blue
                pen_color = QColor(Qt.GlobalColor.blue)
    elif isinstance(color, (tuple, list)) and len(color) >= 3:
        # RGB or RGBA tuple
        if len(color) == 3:
            pen_color = QColor(color[0], color[1], color[2])
        else:
            pen_color = QColor(color[0], color[1], color[2], color[3] if len(color) > 3 else 255)
    else:
        # Default to blue
        pen_color = QColor(Qt.GlobalColor.blue)
    
    pen_color.setAlpha(transparency)
    return QPen(pen_color, thickness)