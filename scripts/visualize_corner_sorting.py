import argparse
import time

import cv2
import numpy as np

from sudoku_extraction.point_order import (
    AssignmentOrderer,
    CentroidOrderer,
    CyclicSortWithCentroidAnchor,
    PointOrderer,
    SimpleOrderer,
)


class CornerVisualizer:
    def __init__(self, orderer: PointOrderer, fps: int = 30, rot_speed: float = 30.0) -> None:
        self.orderer = orderer
        self.fps = fps
        self.rot_speed = rot_speed
        self.rot = 0.0
        self.window_name = f"Corner Ordering: {orderer.__class__.__name__}"

        self.img_bg = np.full((512, 512, 3), 255, np.uint8)
        self.p_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        self.pts_corner = (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * 512).astype(np.int32)
        self.base_pts = np.array([[130, 190], [380, 150], [400, 300], [100, 300]], dtype=np.float32)

    def run(self) -> None:
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
                cv2.circle(img, tuple(p), radius=5, color=self.p_colors[i], thickness=-1)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize different 4-point ordering algorithms.")
    parser.add_argument(
        "--type",
        type=str,
        default="cyclic",
        choices=["cyclic", "assignment", "centroid", "simple"],
        help="The ordering algorithm to use.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the visualization.",
    )

    parser.add_argument(
        "--rot_speed",
        type=float,
        default=30.0,
        help="Rotation speed in degrees per second.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Factory to create the selected orderer
    pts_corner = (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * 512).astype(np.int32)
    orderer_map = {
        "cyclic": CyclicSortWithCentroidAnchor(),
        "assignment": AssignmentOrderer(target_corners=pts_corner),
        "centroid": CentroidOrderer(),
        "simple": SimpleOrderer(),
    }

    selected_orderer = orderer_map[args.type]

    visualizer = CornerVisualizer(
        orderer=selected_orderer,
        fps=args.fps,
        rot_speed=args.rot_speed,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
