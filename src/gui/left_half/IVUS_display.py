import cv2
import math

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Union, Any

import numpy as np
from loguru import logger
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem
from PyQt6.QtCore import Qt, QLineF, QPointF
from PyQt6.QtGui import QPixmap, QImage, QColor, QFont, QPen
from shapely.geometry import Polygon

from gui.utils.geometry import Point, Spline, get_qt_pen
from gui.right_half.longitudinal_view import Marker
from report.report import compute_polygon_metrics, farthest_points, closest_points
from segmentation.segment import downsample


class ContourType(Enum):
    LUMEN = "lumen"
    EEM = "eem"
    CALCIUM = "calcium"
    BRANCH = "branch"


@dataclass
class ContourConfig:
    """Configuration for a specific contour type"""

    color: Union[
        str, Tuple[int, int, int], Any
    ]  # accept string names ('green'), hex ('#ff00ff'), or RGB tuples (255,0,0)
    thickness: int
    point_radius: int
    point_thickness: int
    alpha: float
    n_points_contour: int
    n_interactive_points: int


class IVUSDisplay(QGraphicsView):
    """
    Displays images and contours and allows the user to add and manipulate contours.
    """

    def __init__(self, main_window):
        super(IVUSDisplay, self).__init__()
        self.main_window = main_window
        config = main_window.config

        self.n_interactive_points = config.display.n_interactive_points
        self.n_points_contour = config.display.n_points_contour
        self.image_size = config.display.image_size
        self.windowing_sensitivity = config.display.windowing_sensitivity
        self.contour_thickness = config.display.contour_thickness
        self.point_thickness = config.display.point_thickness
        self.point_radius = config.display.point_radius

        self.color_contour = getattr(config.display, "color_contour", (255, 255, 255))
        self.alpha_contour = getattr(config.display, "alpha_contour", 255)  # config uses 0..255

        _default_colors = {
            ContourType.LUMEN: getattr(config.display, "color_contour", "green"),
            ContourType.EEM: getattr(config.display, "color_eem", "red"),
            ContourType.CALCIUM: getattr(config.display, "color_calcium", "white"),
            ContourType.BRANCH: getattr(config.display, "color_branch", "green"),
        }

        self.contour_configs = {}
        for ct in ContourType:
            self.contour_configs[ct] = ContourConfig(
                color=_default_colors.get(ct, self.color_contour),
                thickness=self.contour_thickness,
                point_radius=self.point_radius,
                point_thickness=self.point_thickness,
                alpha=self.alpha_contour,
                n_points_contour=self.n_points_contour,
                n_interactive_points=self.n_interactive_points,
            )

        self.graphics_scene = QGraphicsScene(self)
        self.points_to_draw = []
        self.contour_points = []
        self.frame = 0
        self.contour_mode = False
        self.contour_drawn = False
        self.current_contour = None  # entire contour (not only knotpoints), needed for elliptic ratio
        self.lumen_contour = None
        self.new_spline = None
        self.active_point = None
        self.active_point_index = None
        self.measure_index = None
        self.measure_colors = self.main_window.measure_colors
        self.reference_mode = False
        self.active_contour_type: ContourType = ContourType.LUMEN

        self.initial_window_level = 128  # window level is the center which determines the brightness of the image
        self.initial_window_width = 256  # window width is the range of pixel values that are displayed
        self.window_level = self.initial_window_level
        self.window_width = self.initial_window_width

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        image = QGraphicsPixmapItem(QPixmap(self.image_size, self.image_size))
        self.graphics_scene.addItem(image)
        self.setScene(self.graphics_scene)

    def contour_key(self, contour_type: ContourType = None) -> str:
        """Return the string key for the given contour type (defaults to active)."""
        return (contour_type or self.active_contour_type).value

    def get_contour_data(self, contour_type: ContourType = None):
        """Return main_window.data[...] for the given/the active contour type (or None)."""
        key = self.contour_key(contour_type)
        return self.main_window.data.get(key, None)

    def set_active_contour_type(self, contour_type: ContourType):
        """Set active contour type and refresh transient state for editing that contour."""
        if contour_type == self.active_contour_type:
            return
        self.active_contour_type = contour_type

        self.current_contour = None
        self.contour_points = []
        self.active_point = None
        self.active_point_index = None

        self.display_image(update_contours=True, update_image=False, update_phase=False)

    def get_full_contour_list(self, contour_type: ContourType = None):
        """
        Return the list-of-frame full_contours for a contour type.
        If full_contours is still a list (legacy), return it directly.
        """
        # if legacy list
        if not hasattr(self, "full_contours"):
            return None
        if isinstance(self.full_contours, list):
            return self.full_contours
        # dict path (preferred): return list or None
        key = self.contour_key(contour_type)
        return self.full_contours.get(key, None)

    def get_full_contour_for_frame(self, contour_type: ContourType = None, frame: int = None):
        """
        Return the contour (contour_x, contour_y) for a single frame and contour_type.
        Defensive: handles both old list-format and new dict-format.
        """
        frame = self.frame if frame is None else frame
        # try dict style first
        if isinstance(getattr(self, "full_contours", None), dict):
            key = self.contour_key(contour_type)
            contour_list = self.full_contours.get(key)
            if contour_list is None:
                return None
            if 0 <= frame < len(contour_list):
                return contour_list[frame]
            return None
        # fallback to legacy list style
        try:
            return self.full_contours[frame]
        except Exception:
            return None

    def _ensure_main_window_contour_structure(self, key: str):
        """Create basic [ [x per frame], [y per frame] ] structure if missing."""
        if key not in self.main_window.data:
            if hasattr(self, 'images') and self.images is not None:
                nframes = self.images.shape[0]
            else:
                nframes = 0
            self.main_window.data[key] = [[] for _ in range(2)]
            # make per-frame lists for both x/y
            if nframes:
                self.main_window.data[key][0] = [[] for _ in range(nframes)]
                self.main_window.data[key][1] = [[] for _ in range(nframes)]

    def set_data(self, lumen, images):
        """
        Initialize display data. 'lumen' is the legacy argument (first contour),
        but we create entries for all ContourType members in main_window.data
        and prepare self.full_contours dict with per-frame placeholders.
        """
        num_frames = images.shape[0]
        self.image_width = images.shape[1]
        self.scaling_factor = self.image_size / images.shape[1]

        if not hasattr(self.main_window, "data") or self.main_window.data is None:
            self.main_window.data = {}

        self.main_window.data[ContourType.LUMEN.value] = lumen

        # Ensure every contour type has a [ [x per frame], [y per frame] ] structure
        for ct in ContourType:
            key = ct.value
            if key not in self.main_window.data:
                self.main_window.data[key] = [[] for _ in range(2)]
                self.main_window.data[key][0] = [[] for _ in range(num_frames)]
                self.main_window.data[key][1] = [[] for _ in range(num_frames)]
            else:
                # make sure existing entries have per-frame lists of correct length (defensive)
                try:
                    if len(self.main_window.data[key][0]) < num_frames:
                        missing = num_frames - len(self.main_window.data[key][0])
                        self.main_window.data[key][0].extend([[] for _ in range(missing)])
                    if len(self.main_window.data[key][1]) < num_frames:
                        missing = num_frames - len(self.main_window.data[key][1])
                        self.main_window.data[key][1].extend([[] for _ in range(missing)])
                except Exception:
                    self.main_window.data[key] = [[] for _ in range(2)]
                    self.main_window.data[key][0] = [[] for _ in range(num_frames)]
                    self.main_window.data[key][1] = [[] for _ in range(num_frames)]

        self.full_contours = {ct.value: [None] * num_frames for ct in ContourType}

        # For each contour type, try to build a Spline -> unscaled contour (if data exists)
        for ct in ContourType:
            key = ct.value
            contour_data = self.main_window.data.get(key, [[], []])
            for frame in range(num_frames):
                try:
                    if contour_data and contour_data[0][frame]:
                        xs = contour_data[0][frame]
                        ys = contour_data[1][frame]
                        spline = Spline(
                            [xs, ys],
                            self.n_points_contour,
                            self.contour_thickness,
                            self.contour_configs[ct].color if ct in self.contour_configs else self.color_contour,
                            self.contour_configs[ct].alpha if ct in self.contour_configs else self.alpha_contour,
                        )
                        self.full_contours[key][frame] = spline.get_unscaled_contour(scaling_factor=1)
                    else:
                        self.full_contours[key][frame] = None
                except Exception:
                    self.full_contours[key][frame] = None

        self.images = images
        self.main_window.longitudinal_view.set_data(self.images, self.get_full_contour_list(self.active_contour_type))
        self.display_image(update_image=True, update_contours=True, update_phase=True)

    def display_image(self, update_image=False, update_contours=False, update_phase=False):
        """Clears scene and displays current image and contours"""
        image_types = (QGraphicsPixmapItem, Marker)
        if update_image:
            [
                self.graphics_scene.removeItem(item)
                for item in self.graphics_scene.items()
                if isinstance(item, image_types)
            ]
            self.active_point = None
            self.active_point_index = None

            lower_bound = self.window_level - self.window_width / 2
            upper_bound = self.window_level + self.window_width / 2

            normalised_data = np.clip(self.images[self.frame, :, :], lower_bound, upper_bound)
            normalised_data = ((normalised_data - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
            height, width = normalised_data.shape

            if self.main_window.filter == 0:
                normalised_data = cv2.medianBlur(normalised_data, 5)
            elif self.main_window.filter == 1:
                normalised_data = cv2.GaussianBlur(normalised_data, (5, 5), 0)
            elif self.main_window.filter == 2:
                normalised_data = cv2.bilateralFilter(normalised_data, 9, 75, 75)

            if self.main_window.colormap_enabled:
                colormap = cv2.applyColorMap(normalised_data, cv2.COLORMAP_COOL)
                q_image = QImage(colormap.data, width, height, width * 3, QImage.Format.Format_RGB888).scaled(
                    self.image_size, self.image_size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            else:
                q_image = QImage(normalised_data.data, width, height, width, QImage.Format.Format_Grayscale8).scaled(
                    self.image_size, self.image_size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
                )

            image = QGraphicsPixmapItem(QPixmap.fromImage(q_image))
            self.graphics_scene.addItem(image)

            self.main_window.longitudinal_view.update_marker(self.frame)
            marker = Marker(
                int((self.image_width // 2) * self.scaling_factor),
                0,
                int((self.image_width // 2) * self.scaling_factor),
                int(height * self.scaling_factor),
            )
            self.graphics_scene.addItem(marker)

            self.main_window.longitudinal_view.update_marker(self.frame)
            marker = Marker(
                (self.image_width // 2) * self.scaling_factor,
                0,
                (self.image_width // 2) * self.scaling_factor,
                height * self.scaling_factor,
            )
            self.graphics_scene.addItem(marker)

        old_contours = [item for item in self.graphics_scene.items() if not isinstance(item, image_types)]
        [
            self.graphics_scene.removeItem(item)
            for item in self.graphics_scene.items()
            if not isinstance(item, image_types)
        ]  # clear previous scene
        if self.main_window.hide_contours:
            self.main_window.longitudinal_view.hide_lview_contours()
        else:
            if update_contours:
                lumen_data = self.get_contour_data(ContourType.LUMEN)
                if lumen_data and lumen_data[0][self.frame]:
                    # draw lumen and set as lumen_contour, but only set current_contour if lumen is active
                    self.draw_contour(
                        lumen_data,
                        contour_type=ContourType.LUMEN,
                        set_current=(self.active_contour_type == ContourType.LUMEN),
                    )
                else:
                    # no lumen contour for this frame
                    self.lumen_contour = None
                    if self.active_contour_type == ContourType.LUMEN:
                        self.current_contour = None
                        self.contour_points = []

                for ct in ContourType:
                    if ct == ContourType.LUMEN:
                        continue
                    contour_data = self.get_contour_data(ct)
                    if contour_data and contour_data[0][self.frame]:
                        self.draw_contour(contour_data, contour_type=ct, set_current=(ct == self.active_contour_type))

                self.draw_measure()
                self.draw_reference()

                # Compute metrics ALWAYS from lumen_contour (defensive)
                if self.lumen_contour is not None and self.lumen_contour.full_contour[0] is not None:
                    lumen_x, lumen_y = self.lumen_contour.get_unscaled_contour(self.scaling_factor)
                    polygon = Polygon([(x, y) for x, y in zip(lumen_x, lumen_y)])
                    lumen_area, lumen_circumf, _, _ = compute_polygon_metrics(self.main_window, polygon, self.frame)
                    longest_distance, farthest_point_x, farthest_point_y = farthest_points(
                        self.main_window, polygon.exterior.coords, self.frame
                    )
                    shortest_distance, closest_point_x, closest_point_y = closest_points(
                        self.main_window, polygon, self.frame
                    )
                    eem_area, percent_text = self.compute_eem_and_percent_stenosis(self.frame, lumen_area)

                    if not self.main_window.hide_special_points:
                        self.graphics_scene.addLine(
                            QLineF(
                                farthest_point_x[0] * self.scaling_factor,
                                farthest_point_y[0] * self.scaling_factor,
                                farthest_point_x[1] * self.scaling_factor,
                                farthest_point_y[1] * self.scaling_factor,
                            ),
                            QPen(Qt.GlobalColor.yellow, self.point_thickness * 2),
                        )
                        self.graphics_scene.addLine(
                            QLineF(
                                closest_point_x[0] * self.scaling_factor,
                                closest_point_y[0] * self.scaling_factor,
                                closest_point_x[1] * self.scaling_factor,
                                closest_point_y[1] * self.scaling_factor,
                            ),
                            QPen(Qt.GlobalColor.yellow, self.point_thickness * 2),
                        )

                    elliptic_ratio = (longest_distance / shortest_distance) if shortest_distance != 0 else 0
                    self.build_frame_metrics_text(lumen_area, lumen_circumf, elliptic_ratio, longest_distance, shortest_distance, eem_area, percent_text, update_phase)

            else:  # re-draw old elements to put them in foreground
                [self.graphics_scene.addItem(item) for item in old_contours]

        if update_phase:
            if self.main_window.data['phases'][self.frame] == 'D':
                phase = 'Diastole'
                color = QColor(
                    self.main_window.diastole_color[0],
                    self.main_window.diastole_color[1],
                    self.main_window.diastole_color[2],
                )
            elif self.main_window.data['phases'][self.frame] == 'S':
                phase = 'Systole'
                color = QColor(
                    self.main_window.systole_color[0],
                    self.main_window.systole_color[1],
                    self.main_window.systole_color[2],
                )
            else:
                phase = ''
                color = Qt.GlobalColor.white
            self.phase_text = QGraphicsTextItem(phase)
            self.phase_text.setDefaultTextColor(color)
            self.phase_text.setX(self.image_size - self.image_size / 3.75)
            self.phase_text.setFont(QFont('Helvetica', int(self.image_size / 50), QFont.Weight.Bold))
            self.graphics_scene.addItem(self.phase_text)

    def compute_eem_and_percent_stenosis(self, frame: int, lumen_area: float):
        """
        Return (eem_area, percent_stenosis_text).
        Robust to numpy arrays and malformed data structures.
        """
        eem_area = None
        percent_text = "n/a"
        try:
            # Preferred: use prepared full_contours for EEM (display coords)
            eem_full = self.get_full_contour_for_frame(ContourType.EEM, frame)
            if eem_full is not None:
                try:
                    eem_x, eem_y = eem_full
                except Exception:
                    eem_x = eem_y = None

                try:
                    has_eem_coords = (eem_x is not None and len(eem_x) > 0) and (eem_y is not None and len(eem_y) > 0)
                except Exception:
                    has_eem_coords = False

                if has_eem_coords:
                    polygon_eem = Polygon([(float(x), float(y)) for x, y in zip(eem_x, eem_y)])
                    eem_area, _, _, _ = compute_polygon_metrics(self.main_window, polygon_eem, frame)

            # Fallback: use main_window.data (original image coords) and scale to display coords
            if eem_area is None:
                eem_data = self.get_contour_data(ContourType.EEM)
                if eem_data and isinstance(eem_data, (list, tuple)) and len(eem_data) >= 2:
                    xs_orig = ys_orig = None
                    try:
                        # defensive: ensure per-frame lists exist and frame index valid
                        if len(eem_data[0]) > frame:
                            xs_orig = eem_data[0][frame]
                        if len(eem_data[1]) > frame:
                            ys_orig = eem_data[1][frame]
                    except Exception:
                        xs_orig = ys_orig = None

                    try:
                        has_orig = (xs_orig is not None and len(xs_orig) > 0) and (ys_orig is not None and len(ys_orig) > 0)
                    except Exception:
                        has_orig = False

                    if has_orig:
                        xs_scaled = [float(x) * self.scaling_factor for x in xs_orig]
                        ys_scaled = [float(y) * self.scaling_factor for y in ys_orig]
                        polygon_eem = Polygon([(x, y) for x, y in zip(xs_scaled, ys_scaled)])
                        eem_area, _, _, _ = compute_polygon_metrics(self.main_window, polygon_eem, frame)
        except Exception:
            logger.exception("Failed while computing EEM area")

        try:
            if lumen_area is not None and eem_area not in (None, 0):
                percent = (1.0 - (lumen_area / eem_area)) * 100.0
                percent = max(0.0, min(100.0, percent))  # clamp 0..100
                percent_text = f"{round(percent, 2)} %"
        except Exception:
            logger.exception("Failed to compute percent stenosis")

        return eem_area, percent_text
            
    def build_frame_metrics_text(
        self,
        lumen_area,
        lumen_circumf,
        elliptic_ratio,
        longest_distance,
        shortest_distance,
        eem_area,
        percent_stenosis_text,
        update_phase,
    ):
        """
        Build/add a single QGraphicsTextItem with lumen + EEM + percent-stenosis metrics.
        Safely removes any previous metrics item only if it belongs to this scene.
        """
        # remove previous metrics text if present and belongs to our scene (prevents removeItem scene mismatch)
        try:
            prev = getattr(self, "frame_metrics_text", None)
            if prev is not None:
                try:
                    if hasattr(prev, "scene") and prev.scene() is self.graphics_scene:
                        self.graphics_scene.removeItem(prev)
                except Exception:
                    pass
        except Exception:
            pass

        lines = [
            f"Lumen area:\t\t{round(lumen_area, 2)} (mm\N{SUPERSCRIPT TWO})" if lumen_area is not None else "Lumen area:\t\tn/a",
            f"Lumen circ:\t\t{round(lumen_circumf, 2)} (mm)" if lumen_circumf is not None else "Lumen circ:\t\tn/a",
            f"Elliptic ratio:\t\t{round(elliptic_ratio, 2)}" if elliptic_ratio is not None else "Elliptic ratio:\t\tn/a",
            f"Longest distance:\t{round(longest_distance, 2)} (mm)" if longest_distance is not None else "Longest distance:\t\tn/a",
            f"Shortest distance:\t{round(shortest_distance, 2)} (mm)" if shortest_distance is not None else "Shortest distance:\t\tn/a",
            f"EEM area:\t\t{round(eem_area, 2)} (mm\N{SUPERSCRIPT TWO})" if eem_area is not None else "EEM area:\t\tn/a",
            f"Percent stenosis:\t{percent_stenosis_text}",
        ]

        self.frame_metrics_text = QGraphicsTextItem("\n".join(lines))
        self.frame_metrics_text.setFont(QFont("Helvetica", int(self.image_size / 50)))

        self.frame_metrics_text.setPos(5, 5)
        self.graphics_scene.addItem(self.frame_metrics_text)

        if not update_phase:
            try:
                if hasattr(self, "phase_text") and (self.phase_text is not None):
                    self.graphics_scene.addItem(self.phase_text)
            except Exception:
                pass

    def draw_contour(self, contour_data, contour_type: ContourType = None, set_current: bool = False):
        """
        Draw contour_data for the specified contour_type.
        - If set_current is True, this spline becomes self.current_contour (editing target).
        - If contour_type == ContourType.LUMEN we also set self.lumen_contour for metrics.
        """
        if not contour_data or not contour_data[0][self.frame]:
            return

        lumen_x = [point * self.scaling_factor for point in contour_data[0][self.frame]]
        lumen_y = [point * self.scaling_factor for point in contour_data[1][self.frame]]

        ct = contour_type or self.active_contour_type
        cfg = self.contour_configs.get(ct, None)
        thickness = cfg.thickness if cfg else self.contour_thickness
        color = cfg.color if cfg else self.color_contour
        alpha = cfg.alpha if cfg else self.alpha_contour

        spline = Spline([lumen_x, lumen_y], self.n_points_contour, thickness, color, alpha)

        if spline.full_contour[0] is not None:
            knot_points = [
                Point(
                    (spline.knot_points[0][i], spline.knot_points[1][i]),
                    self.point_thickness,
                    self.point_radius,
                    color,
                    alpha,
                )
                for i in range(len(spline.knot_points[0]) - 1)
            ]

            for p in knot_points:
                self.graphics_scene.addItem(p)
            self.graphics_scene.addItem(spline)

            self.full_contours[self.contour_key(ct)][self.frame] = spline.get_unscaled_contour(self.scaling_factor)

            if ct == ContourType.LUMEN:
                self.lumen_contour = spline

            if set_current:
                self.current_contour = spline
                self.contour_points = knot_points
        else:
            logger.warning(f'Spline for frame {self.frame + 1} could not be interpolated for {ct.value}')

    def add_contour(self, point):
        """Creates an interactive contour manually point by point"""

        if self.points_to_draw:
            start_point = self.points_to_draw[0].get_coords()
        else:
            self.contour_drawn = False
            start_point = (point.x(), point.y())

        if start_point[0] is None:  # occurs when Point has been deleted during draw (e.g. by RMB click)
            self.points_to_draw = []
            self.contour_mode = False
            self.main_window.setCursor(Qt.CursorShape.ArrowCursor)
            self.display_image(update_contours=True)
        else:
            if len(self.points_to_draw) > 3:  # start drawing spline after 3 points
                if not self.contour_drawn:
                    self.new_spline = Spline(
                        [
                            [point.get_coords()[0] for point in self.points_to_draw],
                            [point.get_coords()[1] for point in self.points_to_draw],
                        ],
                        self.n_points_contour,
                        self.contour_thickness,
                    )
                    self.graphics_scene.addItem(self.new_spline)
                    self.contour_drawn = True
                else:
                    self.new_spline.update(point, len(self.points_to_draw))

            if len(self.points_to_draw) > 1:
                dist = math.sqrt((point.x() - start_point[0]) ** 2 + (point.y() - start_point[1]) ** 2)

                if dist < 20: # check distance to start point, if close enough, close contour (20 pixels is tested)
                    self.points_to_draw = []
                    if self.new_spline is not None:
                        downsampled = downsample(
                            ([self.new_spline.full_contour[0].tolist()], [self.new_spline.full_contour[1].tolist()]),
                            self.n_interactive_points,
                        )

                        key = self.contour_key()
                        self._ensure_main_window_contour_structure(key)

                        self.main_window.data[key][0][self.frame] = [
                            point / self.scaling_factor for point in downsampled[0]
                        ]
                        self.main_window.data[key][1][self.frame] = [
                            point / self.scaling_factor for point in downsampled[1]
                        ]

                    self.stop_contour()
                    return

            self.points_to_draw.append(Point((point.x(), point.y()), self.point_thickness, self.point_radius))
            self.graphics_scene.addItem(self.points_to_draw[-1])

    def start_contour(self, contour_type: ContourType = None):
        if contour_type is not None:
            self.set_active_contour_type(contour_type)

        self.measure_index = None
        self.main_window.setCursor(Qt.CursorShape.CrossCursor)
        self.contour_mode = True
        self.points_to_draw = []

        key = self.contour_key()
        self._ensure_main_window_contour_structure(key)

        self.main_window.data[key][0][self.frame] = []
        self.main_window.data[key][1][self.frame] = []
        self.display_image(update_contours=True)  # clear previous contour

    def stop_contour(self):
        if self.main_window.image_displayed:
            self.contour_mode = False
            self.main_window.setCursor(Qt.CursorShape.ArrowCursor)
            self.display_image(update_contours=True)

            contour_for_frame = self.get_full_contour_for_frame(self.active_contour_type, self.frame)
            try:
                self.main_window.longitudinal_view.lview_contour(self.frame, contour_for_frame, update=True)
            except Exception as e:
                logger.debug(f"Could not update longitudinal view for frame {self.frame}: {e}")

    def draw_measure(self):
        for index in range(2):
            if (
                self.main_window.data['measures'][self.frame][index] is not None
                and len(self.main_window.data['measures'][self.frame][index]) == 4
            ):
                first_point = QPointF(
                    self.main_window.data['measures'][self.frame][index][0],
                    self.main_window.data['measures'][self.frame][index][1],
                )
                second_point = QPointF(
                    self.main_window.data['measures'][self.frame][index][2],
                    self.main_window.data['measures'][self.frame][index][3],
                )
                self.main_window.data['measures'][self.frame][index] = None
                self.add_measure(first_point, index=index, new=False)
                self.add_measure(second_point, index=index, new=False)

    def add_measure(self, point, index=None, new=True):
        index = index if index is not None else self.measure_index
        new_point = Point((point.x(), point.y()), self.point_thickness, self.point_radius, self.measure_colors[index])
        self.graphics_scene.addItem(new_point)

        if self.main_window.data['measures'][self.frame][index] is None:
            self.main_window.data['measures'][self.frame][index] = [point.x(), point.y()]
        else:  # second point
            self.main_window.data['measures'][self.frame][index] += [point.x(), point.y()]
            line = QLineF(
                self.main_window.data['measures'][self.frame][index][0],
                self.main_window.data['measures'][self.frame][index][1],
                self.main_window.data['measures'][self.frame][index][2],
                self.main_window.data['measures'][self.frame][index][3],
            )
            length = round(line.length() * self.main_window.metadata["resolution"] / self.scaling_factor, 2)
            self.main_window.data['measure_lengths'][self.frame][index] = length
            length_text = QGraphicsTextItem(f'{length} mm')
            length_text.setPos(point.x(), point.y())
            self.graphics_scene.addItem(length_text)
            self.graphics_scene.addLine(line, get_qt_pen(self.measure_colors[index], self.point_thickness))
            if new:
                self.measure_index = None
                self.main_window.setCursor(Qt.CursorShape.ArrowCursor)

    def start_measure(self, index: int):
        if self.contour_mode:
            self.stop_contour()
        self.main_window.data['measures'][self.frame][index] = None
        self.main_window.setCursor(Qt.CrossCursor)
        self.measure_index = index
        self.display_image(update_contours=True)

    def stop_measure(self, index):
        if self.main_window.image_displayed:
            self.measure_index = None
            self.main_window.setCursor(Qt.CursorShape.ArrowCursor)
            self.display_image(update_contours=True)
            self.main_window.longitudinal_view.update_measure(
                self.frame, index, self.main_window['measures'][self.frame][index]
            )

    def draw_reference(self):
        if self.main_window.data['reference'][self.frame] is not None:
            reference_point = self.main_window.data['reference'][self.frame]
            # Convert original coordinates to scaled display coordinates
            scaled_x = reference_point[0] * self.scaling_factor
            scaled_y = reference_point[1] * self.scaling_factor
            reference = Point(
                (scaled_x, scaled_y),
                self.point_thickness,
                self.point_radius,
                self.main_window.reference_color,
            )
            self.graphics_scene.addItem(reference)
            text = QGraphicsTextItem('Reference')
            text.setPos(scaled_x, scaled_y)  # Position text at scaled coordinates
            self.graphics_scene.addItem(text)

    def start_reference(self):
        self.reference_mode = True
        self.main_window.setCursor(Qt.CrossCursor)
        self.main_window.data['reference'][self.frame] = None
        self.display_image(update_contours=True)

    def update_display(self):
        self.display_image(update_image=True, update_contours=True, update_phase=True)

    def set_frame(self, value):
        self.frame = value
        self.current_contour = None
        self.stop_contour()
        if self.measure_index is not None:
            self.stop_measure(self.measure_index)

    def mousePressEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            if self.contour_mode:
                self.add_contour(pos)
            elif self.measure_index is not None:
                self.add_measure(pos)
            elif self.reference_mode:
                pos = self.mapToScene(event.pos())
                original_x = pos.x() / self.scaling_factor
                original_y = pos.y() / self.scaling_factor
                self.main_window.data['reference'][self.frame] = [original_x, original_y]
                self.reference_mode = False
                self.main_window.setCursor(Qt.CursorShape.ArrowCursor)
                self.display_image(update_contours=True)
            else:
                # attempt to switch active contour based on nearest knotpoint across all contour types (new feature)
                try:
                    min_dist = float('inf')
                    nearest_ct = None
                    nearest_idx = None
                    nearest_coord = None
                    threshold_px = 20  # pixels; click must be within this to trigger switch (tested value)

                    for ct in ContourType:
                        contour_data = self.get_contour_data(ct)
                        if not contour_data:
                            continue
                        # defensive: ensure per-frame lists exist
                        try:
                            xs = contour_data[0][self.frame]
                            ys = contour_data[1][self.frame]
                        except Exception:
                            continue
                        if not xs:
                            continue

                        # compare scaled coordinates to click
                        for idx, (x_orig, y_orig) in enumerate(zip(xs, ys)):
                            x = x_orig * self.scaling_factor
                            y = y_orig * self.scaling_factor
                            dist = math.hypot(pos.x() - x, pos.y() - y)
                            if dist < min_dist:
                                min_dist = dist
                                nearest_ct = ct
                                nearest_idx = idx
                                nearest_coord = (x, y)

                    if nearest_ct is not None and min_dist < threshold_px and nearest_ct != self.active_contour_type:
                        self.set_active_contour_type(nearest_ct)
                        self.display_image(update_contours=True)
                except Exception:
                    logger.exception("Error while attempting to switch active contour on click")

                items = self.items(event.pos())
                point = [item for item in items if isinstance(item, Point)]
                spline = [item for item in items if isinstance(item, Spline)]
                if point and point[0] in self.contour_points:
                    self.main_window.setCursor(Qt.CursorShape.BlankCursor)  # remove cursor for precise contour changes
                    # Convert mouse position to item position
                    # https://stackoverflow.com/questions/53627056/how-to-get-cursor-click-position-in-qgraphicsitem-coordinate-system
                    self.active_point_index = self.contour_points.index(point[0])
                    point[0].update_color()
                    self.active_point = point[0]
                elif spline:  # clicked on contour
                    path_index = self.current_contour.on_path(pos)
                    self.main_window.setCursor(Qt.CursorShape.BlankCursor)
                    cfg = self.contour_configs.get(self.active_contour_type)
                    self.active_point = Point(
                        (pos.x(), pos.y()),
                        self.point_thickness,
                        self.point_radius,
                        cfg.color if cfg else self.color_contour,
                        cfg.alpha if cfg else self.alpha_contour,
                    )
                    self.graphics_scene.addItem(self.active_point)
                    self.active_point.update_color()
                    self.active_point_index = self.current_contour.update(pos, self.active_point_index, path_index)

        elif event.buttons() == Qt.MouseButton.RightButton:
            self.mouse_x = event.x()
            self.mouse_y = event.y()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self.active_point_index is not None:
                item = self.active_point
                mouse_position = item.mapFromScene(self.mapToScene(event.pos()))
                new_point = item.update_pos(mouse_position)
                self.current_contour.update(new_point, self.active_point_index)

        elif event.buttons() == Qt.MouseButton.RightButton:
            self.setMouseTracking(True)
            # Right-click drag for adjusting window level and window width
            self.window_level += (event.x() - self.mouse_x) * self.windowing_sensitivity
            self.window_width += (event.y() - self.mouse_y) * self.windowing_sensitivity
            self.display_image(update_image=True)
            self.setMouseTracking(False)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:  # for some reason event.buttons() does not work here
            if self.active_point_index is not None:
                self.main_window.setCursor(Qt.CursorShape.ArrowCursor)
                item = self.active_point
                item.reset_color()

                key = self.contour_key()
                self._ensure_main_window_contour_structure(key)

                self.main_window.data[key][0][self.frame] = [
                    point / self.scaling_factor for point in self.current_contour.knot_points[0]
                ]
                self.main_window.data[key][1][self.frame] = [
                    point / self.scaling_factor for point in self.current_contour.knot_points[1]
                ]
                contour_for_frame = self.get_full_contour_for_frame(self.active_contour_type, self.frame)
                self.display_image(update_contours=True)
                try:
                    self.main_window.longitudinal_view.lview_contour(self.frame, contour_for_frame, update=True)
                except Exception as e:
                    logger.debug(f"Could not update longitudinal view after mouse release for frame {self.frame}: {e}")
                self.active_point_index = None
