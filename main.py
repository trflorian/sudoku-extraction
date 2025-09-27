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

window_name = "Sudoku"
cv2.namedWindow(window_name)
# cv2.createTrackbar("th1", window_name, 100, 255, lambda x: None)
# cv2.createTrackbar("th2", window_name, 255, 255, lambda x: None)
th1 = 100
th2 = 255

selected_candidate = 0

while True:
    ret, img = cap.read()
    if not ret:
        break

    ratio = img.shape[0] / img.shape[1]

    target_width = 640
    # target_width = img.shape[1]
    img_calc = cv2.resize(img, (target_width, int(target_width * ratio)))

    img_annot = img.copy() 

    hsv = cv2.cvtColor(img_calc, cv2.COLOR_BGR2HSV)
    # low saturation filter to remove colors!
    mask = cv2.inRange(hsv, lowerb=(0, 0, 0), upperb=(180, 150, 255))

    # preprocess with grayscale + blur before canny
    gray = cv2.cvtColor(img_calc, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = gray

    # th1 = cv2.getTrackbarPos("th1", window_name)
    # th2 = cv2.getTrackbarPos("th2", window_name)

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

    image_area = img_calc.shape[0] * img_calc.shape[1]
    min_contour_area = 0.1 * image_area
    max_contour_area = 0.9 * image_area

    contour_candidates = [cnt for cnt in contour_candidates if cv2.contourArea(cnt) > min_contour_area and cv2.contourArea(cnt) < max_contour_area]

    # resize candidates back to original image size
    scale_x = img.shape[1] / img_calc.shape[1]
    scale_y = img.shape[0] / img_calc.shape[0]
    contour_candidates = [np.array(cnt * [scale_x, scale_y], dtype=int) for cnt in contour_candidates]
    contours = [np.array(cnt * [scale_x, scale_y], dtype=int) for cnt in contours]

    for i, cand in enumerate(contour_candidates):
        color = (0, 0, 255) if i == selected_candidate else (255, 0, 0)
        cv2.drawContours(img_annot, [cand], -1, color, 2)
    cv2.drawContours(img_annot, contours, -1, (0, 255, 0), 1)

    # composite = cv2.hconcat([img_annot, cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)])
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    canny = cv2.resize(canny, (img.shape[1], img.shape[0]))
    cv2.imshow(window_name, canny)
    cv2.waitKey(0)
    cv2.imshow(window_name, img_annot)
    cv2.waitKey(0)
    break

    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('n'):
    #     selected_candidate = (selected_candidate + 1) % max(1, len(contour_candidates))
    # if key == ord('q') or key == 27:
    #     break

if len(contour_candidates) <= selected_candidate:
    print("No contour candidates found!")
    exit(0)

selected_contour = contour_candidates[selected_candidate]
pts = selected_contour.reshape(4, 2)

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

rect_src = order_points(pts)

# use min of rect_src as square size
side_len = int(min(abs(rect_src[0][0] - rect_src[1][0]), abs(rect_src[0][1] - rect_src[3][1]),
                   abs(rect_src[2][0] - rect_src[3][0]), abs(rect_src[2][1] - rect_src[1][1])))

rect_dst = np.array([[0, 0], [side_len, 0], [side_len, side_len], [0, side_len]], dtype=np.float32)

img_annot = img.copy()

# draw the contour and corner points
cv2.drawContours(img_annot, [selected_contour], -1, (0, 255, 0), 2)

for i, pt in enumerate(rect_src):
    pt = pt.astype(int)
    cv2.circle(img_annot, tuple(pt), 5, (0, 0, 255), -1)
    cv2.putText(img_annot, str(i), tuple(pt - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


cv2.imshow(window_name, img_annot)
cv2.waitKey(0)

# animate the transition into the unwarped rectangle
num_steps = 50
for step in range(1, num_steps + 1):
    alpha = step / num_steps
    intermediate_rect = (1 - alpha) * rect_src + alpha * rect_dst
    intermediate_w = int((1 - alpha) * img.shape[1] + alpha * side_len)
    intermediate_h = int((1 - alpha) * img.shape[0] + alpha * side_len)
    warp_mat = cv2.getPerspectiveTransform(rect_src, intermediate_rect)
    warped = cv2.warpPerspective(img, warp_mat, (intermediate_w, intermediate_h))
    cv2.imshow(window_name, warped)
    cv2.waitKey(1000 // 30)

warp_mat = cv2.getPerspectiveTransform(rect_src, rect_dst)
warped = cv2.warpPerspective(img, warp_mat, (side_len, side_len))
cv2.imshow(window_name, warped)
cv2.waitKey(0)
cv2.destroyAllWindows()