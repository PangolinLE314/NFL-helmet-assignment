import math
from typing import Tuple, List

import cv2
import numpy as np

from homography.line import Line

REJECT_DEGREE_TH = 0.1


def filter_lines(lines):
    """
    Filter lines to get best lines
    :param lines:
    :return:
    """
    final_lines = []

    for line in lines:
        [[x1, y1, x2, y2]] = line

        if x1 != x2:
            m = (y2 - y1) / (x2 - x1)
        else:
            m = 100000000
        c = y2 - m * x2
        theta = math.degrees(math.atan(m))
        # Rejecting lines of slope near to 0 degree or 90 degree and storing others
        if REJECT_DEGREE_TH <= abs(theta) <= (90 - REJECT_DEGREE_TH):
            l = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)  # length of the line
            final_lines.append([x1, y1, x2, y2, m, c, l, theta])
    return final_lines


def remove_extra_lines(final_lines):
    # Removing extra lines
    max_line = 25
    if len(final_lines) > max_line:
        final_lines = sorted(final_lines, key=lambda x: x[6], reverse=True)
        final_lines = final_lines[:max_line]

    return final_lines


def get_lines(img):
    """
    Get lines from image, this will use Hough transform, we focus on white line
    :param img:
    :return:
    """
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lower = np.uint8([0, 150, 0])
    upper = np.uint8([255, 255, 255])
    gray_image = cv2.inRange(image, lower, upper)
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, np.ones((3, 3)))
    # blur_gray_image = cv2.GaussianBlur(gray_image, (15, 15), 1)
    # Generating Edge image
    edge_image = cv2.Canny(gray_image, 40, 100)
    # Finding lines in the image
    lines = cv2.HoughLinesP(edge_image, 1, np.pi / 180, 50, 10, maxLineGap=10)
    # Check if lines found and exit if not.
    if lines is None:
        print("Not enough lines found in the image for Vanishing Point detection.")
        return None
    # Filtering lines wrt angle
    lines = filter_lines(lines)

    v_lines = [line for line in lines if abs(line[7]) > 30 and not (100 < (line[1] + line[3]) / 2 < 600)]
    if len(v_lines) > 0:
        ls = [line[6] for line in v_lines]
        longest_line = v_lines[ls.index(max(ls))]
        theta_longest = abs(longest_line[7])
    else:
        theta_longest = 45

    vertical_lines = [line for line in lines if theta_longest - 25 < abs(line[7]) < theta_longest + 25]
    horizontal_lines = [line for line in lines if
                        abs(line[7]) < theta_longest - 65 or abs((line[7]) >= theta_longest + 45) or abs(line[7]) < 15]

    vertical_lines = remove_extra_lines(vertical_lines)
    horizontal_lines = remove_extra_lines(horizontal_lines)
    return lines, vertical_lines, horizontal_lines


def get_vanishing_point(lines):
    """
    Get vanishing point using RANSAC. We will take combination of 2 lines one by one,
    find their intersection point, and calculate the total error(loss) of that point.
    Error of the point means root of sum of squares of distance of that point from each line.
    :param lines:
    :return:
    """
    vanishing_point = None
    min_error = 100000000000

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            m1, c1 = lines[i][4], lines[i][5]
            m2, c2 = lines[j][4], lines[j][5]
            if m1 != m2:
                x0 = (c1 - c2) / (m2 - m1)
                y0 = m1 * x0 + c1
                err = 0
                for k in range(len(lines)):
                    m, c = lines[k][4], lines[k][5]
                    m_ = (-1 / m)
                    c_ = y0 - m_ * x0
                    x_ = (c - c_) / (m_ - m)
                    y_ = m_ * x_ + c_
                    l = math.sqrt((y_ - y0) ** 2 + (x_ - x0) ** 2)
                    err += l ** 2
                err = math.sqrt(err)
                if min_error > err:
                    min_error = err
                    vanishing_point = [x0, y0]

    return vanishing_point


def get_points_symmetrical_triangle(line1, line2, x):
    """
    Get 2 points in line 1 and line2 to form a symmetrical_triangle
    with vanishing points (intersect of line1 and line2)
    :param line1:
    :param line2:
    :param x:
    :return:
    """
    y = line1.get_y_from_x(x)
    p1_pick = np.int32([x, y])
    d1 = np.linalg.norm(p1_pick - line1.p1)
    p2_pick = line2.get_xy_from_d(d1)
    return p1_pick, p2_pick


def get_points_trapezoid(line1, line2, xs):
    """
    Get 4 points of trapezoid and transform it into a rectangle
    :param line1:
    :param line2:
    :param xs:
    :return:
    """
    p1, p2 = get_points_symmetrical_triangle(line1, line2, xs[0])
    p3, p4 = get_points_symmetrical_triangle(line2, line1, xs[1])
    mid = np.int32((p3 + p4) / 2)
    seg_from_mid = Line(list(mid) + list(p3))

    d_trap_bottom = np.linalg.norm(p1 - p2)
    p3_rec = seg_from_mid.get_xy_from_d(d_trap_bottom / 2)
    p4_rec = seg_from_mid.get_xy_from_d(-d_trap_bottom / 2)

    pts1 = [p1, p2, p3, p4]
    pts2 = [p1, p2, p3_rec, p4_rec]
    # pts1 = order_points(np.int32(pts1))
    # pts2 = order_points(np.int32(pts2))

    return np.int32(pts1), np.int32(pts2)


def rotate(image, angle, warp=False):
    """
    Rotate an image by degree
    :param image: input image, we can add only image_size if we dont want to warp
    :param angle: angle to rotate
    :param warp: warp or not
    :return:
    """
    h, w = image if isinstance(image, tuple) else image.shape[:2]
    cX, cY = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    if warp:
        image_out = cv2.warpAffine(image, M, (nW, nH))
    else:
        image_out = None
    M_rotate = np.vstack((M, np.float32([0, 0, 1])))
    return M_rotate, image_out


def perspective_warp(image: np.ndarray, transform: np.ndarray, warp=False):
    """
    Warp and image with Perspective transform
    :param image: input image, we can add only image_size if we dont want to warp
    :param transform:
    :param warp: warp or not
    :return:
    """
    h, w = image if isinstance(image, tuple) else image.shape[:2]
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, transform)
    xmin = math.floor(corners_aft[:, 0, 0].min())
    ymin = math.floor(corners_aft[:, 0, 1].min())
    xmax = math.ceil(corners_aft[:, 0, 0].max())
    ymax = math.ceil(corners_aft[:, 0, 1].max())
    # x_adj = math.floor(xmin - corners_aft[0, 0, 0])
    # y_adj = math.floor(ymin - corners_aft[0, 0, 1])
    translate = np.eye(3)
    translate[0, 2] = -xmin
    translate[1, 2] = -ymin
    corrected_transform = np.matmul(translate, transform)
    new_w = min(math.ceil(xmax - xmin), 3000)
    new_h = min(math.ceil(ymax - ymin), 3000)
    if warp:
        warped_image = cv2.warpPerspective(image, corrected_transform, (new_w, new_h))
    else:
        warped_image = None
    return corrected_transform, warped_image


def trapezoid_transform(line1: Line, line2: Line, img=None, warp=False):
    """
    Transform image using trapezoid to make all lines parallel (vanishing point go to inf)
    :param line1:
    :param line2:
    :param warp: warp or not
    :param img: input image to warp

    :return:
    """
    pts1, pts2 = get_points_trapezoid(line1, line2, [line1.segment[0], line2.segment[2]])
    M_trapezoid = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    if warp:
        h, w = img.shape[:2]
        warped_image = cv2.warpPerspective(img, M_trapezoid, (w, h))
    else:
        warped_image = None
    return M_trapezoid, warped_image, pts1, pts2


def rotate_transform(img, pts2, warp=False):
    mid1 = (pts2[0] + pts2[1]) / 2
    mid2 = (pts2[2] + pts2[3]) / 2
    mid_all = Line(list(mid1) + list(mid2))
    M_rotate, warped_img = rotate(img, 180 - mid_all.angle * 180 / np.pi, warp=warp)
    return M_rotate, warped_img


def pick_two_good_lines(custom_lines: List[Line], thres_length: int = 720, thres_gap: int = 500) -> Tuple[Line, Line]:
    """
    Pick two good lines to calculate transform.
    Good line is line with max distance, len > thres_length and distance < thres_max
    :param custom_lines: List of lines
    :param thres_length: threshold for length
    :param thres_gap: threshold for max distance
    :return:
    """
    line_gap = np.zeros((len(custom_lines), len(custom_lines)))
    for i, custom_line1 in enumerate(custom_lines):
        for j, custom_line2 in enumerate(custom_lines):
            if i >= j:
                continue
            if Line(custom_line1.segment).length < thres_length or Line(custom_line2.segment).length < thres_length:
                continue
            line_gap[i, j] = distance_line2line(custom_lines[i], custom_lines[j])
            if line_gap[i, j] >= thres_gap:
                line_gap[i, j] = 0

    result = np.where(line_gap == np.amax(line_gap))
    line1 = custom_lines[int(result[0][0])]
    line2 = custom_lines[int(result[1][0])]
    return line1, line2


def plot_point(image, p, color=(255, 0, 0), label=None):
    cv2.circle(image, tuple(p), 15, color, -1)
    if label is not None:
        cv2.putText(image, label, (p[0], p[1] - 2), 0, 3, color, thickness=5, lineType=cv2.LINE_AA)
    return image


def distance_line2line(line1: Line, line2: Line, check_intersec=True):
    """
    Distance between 2 segment
    :param check_intersec:
    :param line1:
    :param line2:
    :return:
    """
    seg1 = Line(line1.segment)
    seg2 = Line(line2.segment)
    d1 = np.linalg.norm(np.int32(seg1.p1) - np.int32(seg2.p1))
    d2 = np.linalg.norm(np.int32(seg1.p2) - np.int32(seg2.p2))
    if check_intersec:
        is_intersect = intersect(seg1.p1, seg1.p2, seg2.p1, seg2.p2)
        if is_intersect:
            return 0
    return min(d1, d2)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def transform_from_vertical_lines(line1, line2, img, warp=False):
    h, w = img.shape[:2]
    p1 = line1.horizontal_intersect(0)
    p2 = line1.horizontal_intersect(h - 1)
    p3 = line2.horizontal_intersect(0)
    p4 = line2.horizontal_intersect(h - 1)
    d_top = (p3 - p1)[0]
    d_bottom = (p4 - p2)[0]
    scale = 1.0 - (d_top / d_bottom)
    pts1 = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    pts2 = np.float32([[0, 0], [w - 1, 0], [(w - 1) - scale * (w / 2), h - 1], [scale * (w / 2), h - 1]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    pts = np.float32([p1, p3, p4, p2]).reshape(-1, 1, 2)
    pts2 = cv2.perspectiveTransform(pts, M).reshape(4, 2)
    M_rotate, _ = rotate_transform(img, pts2)
    M_full = M_rotate @ M
    M_correct, image_out = perspective_warp(img, M_full, warp=warp)
    return M_correct, image_out


def transform_from_horizontal_lines(line1, line2, h_lines, M_v, image=None, warp=False):
    horizontal_lines = [Line(line) for line in h_lines]
    horizontal_ls = [line.length for line in horizontal_lines]
    index_max_len = np.where(horizontal_ls == max(horizontal_ls))[0][0]

    good_line = horizontal_lines[index_max_len]
    pts_h = np.vstack((good_line.p1, good_line.p2))

    pts1 = np.float32(line1.segment).reshape(2, -1)
    pts2 = np.float32(line2.segment).reshape(2, -1)

    pts = np.vstack((np.vstack((pts1, pts2)), pts_h)).reshape(-1, 1, 2)
    pts_out = cv2.perspectiveTransform(pts, M_v).reshape(6, 2)

    p1, p2, p3, p4, p1_ver, p2_ver = pts_out
    p3 = p2_ver - p1_ver + p1
    p4 = p2_ver - p1_ver + p2

    pts = np.float32([p1, p2, p3, p4])
    p3_new = np.float32([p3[0], p1[1]])
    p4_new = np.float32([p4[0], p2[1]])
    pts_rec = np.float32([p1, p2, p3_new, p4_new])

    M_horizontal = cv2.getPerspectiveTransform(pts, pts_rec)
    if warp:
        warped_img = cv2.warpPerspective(image, M_horizontal, image.shape[:2])
    else:
        warped_img = None
    return M_horizontal, warped_img
