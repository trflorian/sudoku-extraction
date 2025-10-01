import argparse
import logging

import cv2
import numpy as np

from sudoku_extraction.point_order import CyclicSortWithCentroidAnchor

logger = logging.getLogger(__name__)


def find_sudoku_grid(
    image: np.ndarray,
    calc_ref_width: int = 640,
    max_saturation: int = 150,
    canny_threshold_1: int = 100,
    canny_threshold_2: int = 255,
    morph_kernel_size: int = 3,
) -> np.ndarray | None:
    """
    Finds the largest square-like contour in an image, likely the Sudoku grid.

    Args:
        image: The input image.
        calc_ref_width: The width to resize the image for processing.
        max_saturation: Maximum saturation value for the mask.
        canny_threshold_1: First threshold for the Canny edge detector.
        canny_threshold_2: Second threshold for the Canny edge detector.
        morph_kernel_size: Size of the morphological operation kernel.

    Returns:
        The contour of the found grid as a numpy array, or None if not found.
    """
    # --- 1. Preprocessing ---
    # Resize for consistent kernel sizes
    ratio = image.shape[0] / image.shape[1]
    img_calc = cv2.resize(image, (calc_ref_width, int(calc_ref_width * ratio)))

    # Use a saturation mask to filter out colored areas, focusing on the grid lines
    hsv = cv2.cvtColor(img_calc, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, lowerb=(0, 0, 0), upperb=(180, max_saturation, 255))
    mask = (hsv[:, :, 1] < max_saturation).astype(np.uint8) * 255

    # Grayscale and apply mask to prepare for edge detection
    gray = cv2.cvtColor(img_calc, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    # --- 2. Edge and Contour Detection ---
    # Canny edge detection
    canny = cv2.Canny(gray, threshold1=canny_threshold_1, threshold2=canny_threshold_2)

    # Dilate to close small gaps in the detected edges
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(morph_kernel_size, morph_kernel_size))
    canny = cv2.morphologyEx(canny, op=cv2.MORPH_DILATE, kernel=kernel, iterations=1)

    # Find external contours
    contours, _ = cv2.findContours(canny, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # --- 3. Filter Contours ---
    contour_candidates: list[np.ndarray] = []
    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.1 * cv2.arcLength(curve=cnt, closed=True)
        approx = cv2.approxPolyDP(curve=cnt, epsilon=epsilon, closed=True)

        # Keep only quadrilaterals (polygons with 4 vertices)
        if len(approx) == 4:
            contour_candidates.append(approx)

    # Filter by area to remove very small or very large shapes
    image_area = img_calc.shape[0] * img_calc.shape[1]
    min_contour_area = 0.1 * image_area
    max_contour_area = 0.9 * image_area

    contour_candidates = [
        cnt for cnt in contour_candidates if min_contour_area < cv2.contourArea(cnt) < max_contour_area
    ]

    if not contour_candidates:
        cv2.imshow("No Contours Found", img_calc)
        cv2.imshow("Canny", canny)
        cv2.waitKey(0)
        return None

    # --- 4. Select Best Candidate and Scale ---
    # Assume the largest valid contour is the Sudoku grid
    # Sort by area in descending order and pick the first one
    best_contour = sorted(contour_candidates, key=cv2.contourArea, reverse=True)[0]

    # Scale the contour points back to the original image's dimensions
    scale_x = image.shape[1] / img_calc.shape[1]
    scale_y = image.shape[0] / img_calc.shape[0]
    scaled_contour = np.array(best_contour * [scale_x, scale_y], dtype=int)

    return scaled_contour


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
