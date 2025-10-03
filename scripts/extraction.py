import argparse
import logging

import cv2
import numpy as np

from sudoku_extraction.point_order import CyclicSortWithCentroidAnchor
from sudoku_extraction.sudoku_grid import find_sudoku_grid

logger = logging.getLogger(__name__)


def animate_warp(
    window_name: str,
    image: np.ndarray,
    src_rect: np.ndarray,
    dst_rect: np.ndarray,
    final_size: int,
    duration_sec: float = 1.0,
    fps: int = 30,
) -> None:
    """
    Animates the perspective warp transformation.

    Args:
        window_name: The name of the OpenCV window.
        image: The source image.
        src_rect: The 4 source points for the warp.
        dst_rect: The 4 destination points for the warp.
        final_size: The width and height of the final warped image.
        duration_sec: Duration of the animation in seconds.
        fps: Frames per second for the animation.
    """

    num_steps = int(duration_sec * fps)
    for step in range(1, num_steps + 1):
        alpha = step / num_steps

        # Interpolate between source and destination rectangles
        intermediate_rect = (1 - alpha) * src_rect + alpha * dst_rect

        # Interpolate the size of the output image
        intermediate_w = int((1 - alpha) * image.shape[1] + alpha * final_size)
        intermediate_h = int((1 - alpha) * image.shape[0] + alpha * final_size)

        warp_mat = cv2.getPerspectiveTransform(src_rect, intermediate_rect)
        warped = cv2.warpPerspective(image, warp_mat, (intermediate_w, intermediate_h))

        cv2.imshow(window_name, warped)
        cv2.waitKey(1000 // fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sudoku Grid Detection and Warping")
    parser.add_argument(
        "--image",
        type=str,
        default="images/sudoku_001.jpg",
        help="Path to the input image",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run the Sudoku grid detection and warping."""
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    IMG_PATH = args.image
    WINDOW_NAME = "Sudoku"

    cv2.namedWindow(WINDOW_NAME)

    img = cv2.imread(IMG_PATH)
    if img is None:
        logger.error(f"Error: Could not load image from {IMG_PATH}")
        return

    # --- 1. Find the Sudoku grid in the image ---
    grid_contour = find_sudoku_grid(img)

    if grid_contour is None:
        logger.warning("No Sudoku grid found!")
        cv2.imshow(WINDOW_NAME, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Order the corner points for a consistent transformation
    point_orderer = CyclicSortWithCentroidAnchor()
    rect_src = point_orderer(grid_contour)

    # --- 2. Draw the found contour for visualization ---
    img_annot = img.copy()
    cv2.drawContours(img_annot, [grid_contour], -1, (0, 255, 0), 3)

    # Visualize the ordered points
    for idx, point in enumerate(rect_src):
        cv2.circle(img_annot, tuple(point.astype(int)), 5, (0, 0, 255), -1)
        cv2.putText(
            img_annot,
            str(idx),
            tuple(point.astype(int) + np.array([10, -10])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
    cv2.imshow(WINDOW_NAME, img_annot)
    cv2.waitKey(0)

    # --- 3. Prepare for perspective transformation ---
    # Define the destination as a square. Calculate its side length based on
    # the average width and height of the source rectangle for stability.
    width_top = np.linalg.norm(rect_src[0] - rect_src[1])
    width_bottom = np.linalg.norm(rect_src[3] - rect_src[2])
    height_left = np.linalg.norm(rect_src[0] - rect_src[3])
    height_right = np.linalg.norm(rect_src[1] - rect_src[2])

    side_len = int(max(width_top, width_bottom, height_left, height_right))
    rect_dst = np.array(
        [[0, 0], [side_len - 1, 0], [side_len - 1, side_len - 1], [0, side_len - 1]],
        dtype=np.float32,
    )

    # Ensure points are float32 for cv2 functions
    rect_src = rect_src.astype(np.float32)
    rect_dst = rect_dst.astype(np.float32)

    # --- 4. Animate and perform the final warp ---
    animate_warp(WINDOW_NAME, img, rect_src, rect_dst, side_len)

    # Perform the final, precise warp
    warp_mat = cv2.getPerspectiveTransform(rect_src, rect_dst)
    warped = cv2.warpPerspective(img, warp_mat, (side_len, side_len))

    cv2.imshow(WINDOW_NAME, warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
