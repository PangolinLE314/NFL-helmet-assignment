import os
import random
import subprocess
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pylab as plt
import numpy as np


def compute_color_for_id(label: int) -> Tuple:
    """
    Compute color from label
    :param label: id of the color
    :return Tuple represent the color
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def plot_one_box(x, im, box_format='xywh', color=None, label=None, line_thickness=3):
    """
    Plot 1 box in the image
    :param x: box in the format xywh (center + wh) or tlbr (tl + br)
    :param im: im to draw
    :param box_format: format of the box 'xyxy' or 'tlbr'
    :param color: color of the det
    :param label: text to put
    :param line_thickness: line_thickness
    :return:
    """
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    if box_format == 'xywh':
        x1, y1, w, h = x
        c1 = (int(x1 - w / 2), int(y1 - h / 2))
        c2 = (int(x1 + w / 2), int(y1 + h / 2))
    else:
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return im


def plot_boxes(boxes, im, box_format='xywh', color=None, label=None, line_thickness=3):
    """
    Plot multiple boxes
    :param boxes: box in the format xywh (center + wh) or tlbr (tl + br)
    :param im: im to draw
    :param box_format: format of the box 'xyxy' or 'tlbr'
    :param color: color of the det
    :param label: text to put
    :param line_thickness: line_thickness
    :return:
    """
    for box in boxes:
        im = plot_one_box(box, im, box_format, color, label, line_thickness)
    return im


def plot_boxes_with_track_ids(ds_res, im, box_format='xywh'):
    """
    Plot boxes_with_track_ids result
    :param ds_res: result of boxes_with_track_ids [[l,t,r,b,track_id]]
    :param im: image to draw
    :param box_format:
    :return:
    """
    if len(ds_res) == 0:
        return im

    boxes = ds_res[:, 0:4]
    track_ids = ds_res[:, 4]
    for box, track_id in zip(boxes, track_ids):
        try:
            track_id = int(track_id)
            if track_id > 5000:  # un tracked trackid
                track_id = 'unknown'
                raise ValueError
            color = compute_color_for_id(track_id)
        except ValueError:
            color = (255, 0, 0)
        if str(track_id) != 'unknown':
            plot_one_box(box, im, color=color, box_format=box_format, label=str(track_id))
    return im


def plot_one_frame_in_video_data(video_data_in, image, show_track=False):
    xywhs = video_data_in[['x', 'y', 'width', 'height']].values
    attribute = 'cluster'
    if len(xywhs) > 0:
        if show_track and attribute in video_data_in:  # show track
            track_ids = video_data_in[[attribute]].values
            det_with_track_ids = np.hstack((xywhs, track_ids))
            image = plot_boxes_with_track_ids(det_with_track_ids, image, box_format='xywh')
        else:  # only show det
            image = plot_boxes(xywhs, image, color=(255, 0, 0), label='det')

    return image


def create_football_field(
        linenumbers=True,
        endzones=True,
        highlight_line=False,
        highlight_line_number=10,
        highlighted_name="Line of Scrimmage",
        fifty_is_los=False,
        figsize=(12, 6.33),
        field_color="lightgreen",
        ez_color='forestgreen',
        ax=None,
):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle(
        (0, 0),
        120,
        53.3,
        linewidth=0.1,
        edgecolor="r",
        facecolor=field_color,
        zorder=0,
    )

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='black')

    if fifty_is_los:
        ax.plot([60, 60], [0, 53.3], color="gold")
        ax.text(62, 50, "<- Player Yardline at Snap", color="gold")
    # Endzones
    if endzones:
        ez1 = patches.Rectangle(
            (0, 0),
            10,
            53.3,
            linewidth=0.1,
            edgecolor="black",
            facecolor=ez_color,
            alpha=0.6,
            zorder=0,
        )
        ez2 = patches.Rectangle(
            (110, 0),
            120,
            53.3,
            linewidth=0.1,
            edgecolor="black",
            facecolor=ez_color,
            alpha=0.6,
            zorder=0,
        )
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    ax.axis("off")
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            ax.text(
                x,
                5,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,  # fontname='Arial',
                color="black",
            )
            ax.text(
                x - 0.95,
                53.3 - 5,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,  # fontname='Arial',
                color="black",
                rotation=180,
            )
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color="black")
        ax.plot([x, x], [53.0, 52.5], color="black")
        ax.plot([x, x], [22.91, 23.57], color="black")
        ax.plot([x, x], [29.73, 30.39], color="black")

    if highlight_line:
        hl = highlight_line_number + 10
        ax.plot([hl, hl], [0, 53.3], color="yellow")
        ax.text(hl + 2, 50, "<- {}".format(highlighted_name), color="yellow")

    border = patches.Rectangle(
        (-5, -5),
        120 + 10,
        53.3 + 10,
        linewidth=0.1,
        edgecolor="orange",
        facecolor="white",
        alpha=0,
        zorder=0,
    )
    ax.add_patch(border)
    ax.set_xlim((0, 120))
    ax.set_ylim((0, 53.3))
    return ax


def show_img(image, img_name=None, show=False, write=False):
    """
    Show image or save image to output
    :param image:
    :param img_name:
    :param show:
    :param write:
    :return:
    """
    if show:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
    if write:
        if img_name is not None:
            os.makedirs(str(Path(img_name).parent), exist_ok=True)
            cv2.imwrite(img_name, image)


def write_mp4(img_folder, output_name):
    """
    Write a list of image in an folder to mp4 video
    :param img_folder:
    :param output_name:
    :return:
    """
    if os.path.exists(output_name):
        os.remove(output_name)
    else:
        os.makedirs(Path(output_name).parent, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            f'{img_folder}/%*.jpg',
            "-crf",
            "0",
            "-preset",
            "slow",
            "-vcodec",
            "libx264",
            output_name,
        ]
    )
    return output_name
