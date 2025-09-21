import cv2
import numpy as np

img = cv2.imread("images/sudoku_001.jpg")

ratio = img.shape[0] / img.shape[1]

# target_width = 600
target_width = img.shape[1]
img = cv2.resize(img, (target_width, int(target_width * ratio)))

image_area = img.shape[0] * img.shape[1]

# preprocess with grayscale + blur before canny
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# canny edge detection, dilate to close even more gaps
canny = cv2.Canny(blur, 20, 100)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
canny = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel, iterations=1)

# find contours
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# approximate contours to polygons and filter based on area
contour_candidates = []
for cnt in contours:
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    contour_candidates.append(approx)

min_contour_area = 0.1 * image_area
max_contour_area = 0.9 * image_area

contour_candidates = [cnt for cnt in contour_candidates if cv2.contourArea(cnt) > min_contour_area and cv2.contourArea(cnt) < max_contour_area]

cv2.drawContours(img, contour_candidates, -1, (255, 0, 0), 2)

cv2.imshow("Image", img)
cv2.imshow("Canny", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
