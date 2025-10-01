from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances


class PointOrderer(ABC):
    """Abstract base class for point ordering strategies."""

    @abstractmethod
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Orders four corner points in a consistent sequence.

        Args:
            pts: A numpy array of shape (4, 2).

        Returns:
            A numpy array of shape (4, 2) with points ordered.
        """

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

    def __init__(self, target_corners: np.ndarray) -> None:
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
