import cv2
import numpy as np

# img = cv2.imread("images/sudoku_001.jpg")

class StaticImageCapture:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)

    def read(self):
        return True, self.img.copy()

    def release(self):
        pass

cap = StaticImageCapture("images/sudoku_004.jpg")
# cap = cv2.VideoCapture(0)

window_name = "Rect Detection"
cv2.namedWindow(window_name)
cv2.createTrackbar("th1", window_name, 100, 255, lambda x: None)
cv2.createTrackbar("th2", window_name, 255, 255, lambda x: None)

while True:
    ret, img = cap.read()
    if not ret:
        break

    ratio = img.shape[0] / img.shape[1]

    target_width = 640
    # target_width = img.shape[1]
    img = cv2.resize(img, (target_width, int(target_width * ratio)))

    image_area = img.shape[0] * img.shape[1]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # low saturation filter to remove colors!
    mask = cv2.inRange(hsv, lowerb=(0, 0, 0), upperb=(180, 150, 255))

    # preprocess with grayscale + blur before canny
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = gray

    th1 = cv2.getTrackbarPos("th1", window_name)
    th2 = cv2.getTrackbarPos("th2", window_name)

    # canny edge detection, dilate to close even more gaps
    canny = cv2.Canny(blur, threshold1=th1, threshold2=th2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canny = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel, iterations=1)

    # find contours
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # approximate contours to polygons and filter based on area
    contour_candidates = []
    for cnt in contours:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            contour_candidates.append(approx)

    min_contour_area = 0.1 * image_area
    max_contour_area = 0.9 * image_area

    contour_candidates = [cnt for cnt in contour_candidates if cv2.contourArea(cnt) > min_contour_area and cv2.contourArea(cnt) < max_contour_area]

    cv2.drawContours(img, contour_candidates, -1, (255, 0, 0), 2)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    compsiute = cv2.hconcat([img, cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)])
    cv2.imshow(window_name, compsiute)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()
