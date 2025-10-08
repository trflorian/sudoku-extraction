import argparse
import logging

import cv2
import numpy as np

from sudoku_extraction.point_order import CyclicSortWithCentroidAnchor
from sudoku_extraction.sudoku_grid import find_sudoku_grid

logger = logging.getLogger(__name__)


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

    # Perform the final, precise warp
    warp_mat = cv2.getPerspectiveTransform(rect_src, rect_dst)
    warped = cv2.warpPerspective(img, warp_mat, (side_len, side_len))

    # threshold and invert the warped image for better visibility
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    # adaptive thresholding to handle varying lighting conditions
    warped_thresh = cv2.adaptiveThreshold(
        warped_gray,
        255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=20,
    )

    # remove intermediate lines between the cells for better digit visibility
    line_thickness = max(1, int(0.02 * side_len))
    for i in range(1, 9):
        line_pos = int(i * warped_thresh.shape[0] / 9)
        cv2.line(warped_thresh, (0, line_pos), (warped_thresh.shape[1], line_pos), 0, line_thickness)
        cv2.line(warped_thresh, (line_pos, 0), (line_pos, warped_thresh.shape[0]), 0, line_thickness)

    # remove lines at border
    cv2.rectangle(warped_thresh, (0, 0), (warped_thresh.shape[1], warped_thresh.shape[0]), 0, line_thickness)

    # extract each cell, and save them as individual images
    cell_images = []
    cell_size = warped_thresh.shape[0] // 9
    for row in range(9):
        for col in range(9):
            cell = warped_thresh[row * cell_size : (row + 1) * cell_size, col * cell_size : (col + 1) * cell_size]
            cell_images.append(cell)

    # pre-process each cell image for MNIST model
    processed_cells = []
    for cell in cell_images:
        # find contours to detect if there is a digit
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # min contour area to filter out noise
        min_contour_area = cell.shape[0] * cell.shape[1] * 0.02
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        if len(contours) == 0:
            # empty cell
            processed_cells.append(np.zeros((28, 28), dtype=np.uint8))
            continue

        # get bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # extract the digit region and resize to fit in 20x20 box
        digit = cell[y : y + h, x : x + w]
        if w > 0 and h > 0:
            scale = 20.0 / max(w, h)
            digit_resized = cv2.resize(digit, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            digit_resized = np.zeros((20, 20), dtype=np.uint8)

        # create a 28x28 image and place the resized digit in the center
        digit_padded = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - digit_resized.shape[1]) // 2
        y_offset = (28 - digit_resized.shape[0]) // 2
        digit_padded[y_offset : y_offset + digit_resized.shape[0], x_offset : x_offset + digit_resized.shape[1]] = (
            digit_resized
        )

        processed_cells.append(digit_padded)

    # re-create the full image with these processed cells for visualization
    border = 2
    reconstructed = np.zeros((9 * 28 + border * 2, 9 * 28 + border * 2), dtype=np.uint8)
    for i, cell in enumerate(processed_cells):
        row = i // 9
        col = i % 9
        reconstructed[row * 28 + border : (row + 1) * 28 + border, col * 28 + border : (col + 1) * 28 + border] = cell
    # draw grid lines
    for i in range(10):
        line_thickness = 2 if i % 3 == 0 else 1
        cv2.line(reconstructed, (border, i * 28 + border), (9 * 28 + border, i * 28 + border), 255, line_thickness)
        cv2.line(reconstructed, (i * 28 + border, border), (i * 28 + border, 9 * 28 + border), 255, line_thickness)

    cv2.imshow("Reconstructed", reconstructed)
    cv2.imshow(WINDOW_NAME, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
