import time
import cv2
import numpy as np
import argparse
from abc import ABC, abstractmethod
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment

# --- Point Ordering Strategies ---


class PointOrderer(ABC):
    """Abstract base class for point ordering strategies."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Orders four corner points in a consistent sequence.

        Args:
            pts: A numpy array of shape (4, 2).

        Returns:
            A numpy array of shape (4, 2) with points ordered.
        """
        pass

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        return self.order_points(pts)


class CyclicSortWithCentroidAnchor(PointOrderer):
    """
    Orders points using a robust hybrid method: cyclic sort + stable anchor.
    This is the most reliable method.
    """

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.reshape(4, 2)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts_cyclic = pts[np.argsort(angles)]
        sum_of_coords = pts_cyclic.sum(axis=1)
        top_left_idx = np.argmin(sum_of_coords)
        return np.roll(pts_cyclic, -top_left_idx, axis=0)


class AssignmentOrderer(PointOrderer):
    """
    Orders points by matching them to fixed screen corners.
    Fails with rotation and translation.
    """

    def __init__(self, target_corners: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.target_corners = target_corners

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.reshape(-1, 2)
        D = pairwise_distances(pts, self.target_corners)
        row_ind, col_ind = linear_sum_assignment(D)
        ordered_pts = np.zeros_like(pts)
        ordered_pts[col_ind] = pts[row_ind]
        return ordered_pts


class CentroidOrderer(PointOrderer):
    """
    Orders points based on their quadrant relative to the centroid.
    Can fail with skewed shapes.
    """

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.reshape(4, 2)
        center = pts.mean(axis=0)

        # This can fail if the split isn't 2/2
        top_points = pts[pts[:, 1] < center[1]]
        bottom_points = pts[pts[:, 1] >= center[1]]

        if len(top_points) != 2 or len(bottom_points) != 2:
            # Fallback for degenerate cases to prevent crash
            return pts

        rect = np.zeros((4, 2), dtype=np.float32)
        rect[0] = top_points[np.argmin(top_points[:, 0])]
        rect[1] = top_points[np.argmax(top_points[:, 0])]
        rect[3] = bottom_points[np.argmin(bottom_points[:, 0])]
        rect[2] = bottom_points[np.argmax(bottom_points[:, 0])]
        return rect


class SimpleOrderer(PointOrderer):
    """
    Orders points using sum and difference heuristics.
    Fails with rotation.
    """

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect


# --- Visualization Application ---


class CornerVisualizer:
    def __init__(self, orderer: PointOrderer, fps: int = 30, rot_speed: float = 30.0):
        self.orderer = orderer
        self.fps = fps
        self.rot_speed = rot_speed
        self.rot = 0.0
        self.window_name = f"Corner Ordering: {orderer.__class__.__name__}"

        self.img_bg = np.full((512, 512, 3), 255, np.uint8)
        self.p_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        self.pts_corner = (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * 512).astype(
            np.int32
        )
        self.base_pts = np.array(
            [[130, 190], [380, 150], [400, 300], [100, 300]], dtype=np.float32
        )

    def run(self):
        while True:
            st = time.perf_counter()
            img = self.img_bg.copy()

            # 1. Update and transform points
            center = self.base_pts.mean(axis=0)
            M = cv2.getRotationMatrix2D(tuple(center), self.rot, 1.0)
            pts = cv2.transform(np.array([self.base_pts]), M)[0].astype(np.int32)
            self.rot = (self.rot + self.rot_speed / self.fps) % 360

            # 2. Draw the original polygon
            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=2)
            for i, p in enumerate(pts):
                cv2.circle(
                    img, tuple(p), radius=5, color=self.p_colors[i], thickness=-1
                )

            # 3. Order points and draw matching lines
            pts_sorted = self.orderer(pts).astype(np.int32)
            for i, p_sorted in enumerate(pts_sorted):
                try:
                    og_index = pts.tolist().index(p_sorted.tolist())
                    cv2.line(
                        img,
                        tuple(p_sorted),
                        tuple(self.pts_corner[i]),
                        color=self.p_colors[og_index],
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
                except ValueError:
                    # Point might not be found if ordering fails (e.g., centroid)
                    pass

            # 4. Display the image
            cv2.imshow(self.window_name, img)

            et = time.perf_counter()
            dt_ms = (et - st) * 1000
            wait_time = max(1, int(1000 / self.fps - dt_ms))
            if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


# --- Main Execution ---


def main():
    parser = argparse.ArgumentParser(
        description="Visualize different 4-point ordering algorithms."
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="cyclic",
        choices=["cyclic", "assignment", "centroid", "simple"],
        help="The ordering algorithm to use.",
    )
    args = parser.parse_args()

    # Configuration for the visualizer
    config = {"fps": 30, "rot_speed": 30.0}

    # Factory to create the selected orderer
    pts_corner = (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * 512).astype(np.int32)
    orderer_map = {
        "cyclic": CyclicSortWithCentroidAnchor(),
        "assignment": AssignmentOrderer(target_corners=pts_corner),
        "centroid": CentroidOrderer(),
        "simple": SimpleOrderer(),
    }

    selected_orderer = orderer_map[args.type]

    visualizer = CornerVisualizer(orderer=selected_orderer, **config)
    visualizer.run()


if __name__ == "__main__":
    main()
