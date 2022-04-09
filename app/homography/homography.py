import os
from pathlib import Path

import cv2
import numpy as np

from visualize import vis_utils
from homography import transform_calculator
from homography.line import Line


def find_homography_lines(img,
                          rotate=False,
                          warp=False,
                          vis=False,
                          write=False,
                          output_name=None):
    """
    Find homography transform using line detection
    :param output_name:
    :param img:
    :param rotate:
    :param warp:
    :param vis:
    :param write:
    :return:
    """
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = img.shape[:2]
    all_line, v_lines, h_lines = transform_calculator.get_lines(img)

    if output_name is not None:
        output_name_base = output_name.rsplit('.', 1)[0]
    else:
        output_name_base = 'test'

    if vis:
        # Drawing org lines
        # for org_line in all_line:
        #     Line(org_line).draw(img, show_segment=False, long=False, line_color=(244, 0, 123))

        for org_line in h_lines:
            Line(org_line).draw(img, show_segment=False, long=False, line_color=(0, 0, 255))

        for org_line in v_lines:
            Line(org_line).draw(img, show_segment=False, long=False)

    vis_utils.show_img(img, output_name_base + '_line_detections.jpg', show=vis, write=write)

    thetas = [line[7] for line in v_lines]
    already_vertical = False

    if np.count_nonzero(abs(np.float32(thetas)) > 88) > 10:  # many vertical lines
        already_vertical = True
        vanishing_point = [w / 2, 1000000000000]
        custom_lines = [Line(vanishing_point + [line[2], line[3]]) for line in v_lines]

        # custom_lines = [Line(line) for line in v_lines]
        # custom_lines = [Line(line.get_segment(0,w-1,0,h-1)) for line in custom_lines]
    else:
        vanishing_point = transform_calculator.get_vanishing_point(v_lines)
        custom_lines = [Line(vanishing_point + [line[2], line[3]]) for line in v_lines]
    for line in custom_lines:
        line.get_segment(0, w - 1, 0, h - 1)
    line1, line2 = transform_calculator.pick_two_good_lines(custom_lines, thres_length=h)

    if vis:
        # Drawing long lines
        for custom_line in custom_lines:
            custom_line.draw(img, show_segment=False)
        for custom_line in [line1, line2]:
            custom_line.draw(img, show_segment=False, line_color=(0, 100, 230))

    vis_utils.show_img(img, output_name_base + '_line_filter.jpg', show=vis, write=write)

    if not already_vertical:
        M_v, _ = transform_calculator.transform_from_vertical_lines(line1=line1, line2=line2, img=img, warp=False)
    else:
        M_v = np.eye(3)

    M_h, _ = transform_calculator.transform_from_horizontal_lines(line1, line2, h_lines, M_v)

    M_final = np.float32(M_h) @ np.float32(M_v)
    M_final_correct, img_final = transform_calculator.perspective_warp(img, M_final, warp=warp)
    vis_utils.show_img(img_final, output_name_base + '_line_homography_warped.jpg', show=vis, write=write)

    return M_final_correct, img_final


def find_homography_lines_video(video_dir, video_name, output_file=None):
    """
    Find homograhies for all frames in a video
    :param video_dir:
    :param video_name:
    :param output_file: output homographies to that file
    :return:
    """
    if output_file is not None and Path(output_file).exists():
        line_homographies = np.load(output_file)
    else:
        video_path = f'{video_dir}/{video_name}.mp4'
        vidcap = cv2.VideoCapture(video_path)
        line_homographies = []
        rotate = False
        while True:
            it_worked, img = vidcap.read()
            if not it_worked:
                break
            line_homography, _ = find_homography_lines(img, rotate=rotate, vis=False)
            line_homographies.append(line_homography)
        if output_file is not None:
            os.makedirs(str(Path(output_file).parent), exist_ok=True)
            with open(output_file, 'wb') as f:
                np.save(f, line_homographies)
    return line_homographies
