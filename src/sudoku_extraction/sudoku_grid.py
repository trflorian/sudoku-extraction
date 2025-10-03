import cv2
import numpy as np


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
