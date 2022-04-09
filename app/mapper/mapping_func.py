import math

import cv2
import numpy as np
import pandas as pd
import scipy


def get_nearest_frame(frame_id: int, player_tracking: pd.DataFrame) -> int:
    """
    Get the nearest estimated frame in the tracking data
    :param frame_id: frame_id of video
    :param player_tracking: tracking data
    :return:
    """

    available_frames = player_tracking.est_frame.unique()
    shift = min(abs(available_frames - frame_id))
    plus_frame = frame_id + shift
    minus_frame = frame_id - shift
    nearest_frame = plus_frame if plus_frame in available_frames else minus_frame
    return nearest_frame


def hungarian_matching(baseline_helmets: pd.DataFrame, player_tracking: pd.DataFrame, maxcost=10000):
    """
    Hungarian algorithm to match baseline_helmets and player_tracking (projected in image plane)
    :param baseline_helmets: baseline_helmets
    :param player_tracking: player_tracking
    :param maxcost: max cost
    :return:
    """
    warped_pt_pts = list(zip(player_tracking['xc'], player_tracking['yc']))
    bh_pts = list(zip(baseline_helmets['x'], baseline_helmets['y']))
    confidence = baseline_helmets['conf']

    match_cost = [[c * np.linalg.norm(np.float32(pt1) - np.float32(pt2)) ** 2 for pt1 in warped_pt_pts] for (pt2, c) in
                  zip(bh_pts, confidence)]
    const_cost = [[c * maxcost for _ in bh_pts] for c in confidence]
    cost_matrix = np.concatenate([match_cost, const_cost], axis=1)
    idxs1, idxs2 = scipy.optimize.linear_sum_assignment(cost_matrix)
    cost = cost_matrix[idxs1, idxs2].sum() / len(idxs1)
    try:
        idxs1, idxs2 = np.array(
            [[idx1, idx2] for idx1, idx2 in zip(idxs1, idxs2) if idx2 < len(warped_pt_pts)]).transpose()
    except Exception as e:
        print(f'Exection at hungarian_matching: {e}')
        idxs1, idxs2 = [], []
    labels = player_tracking.iloc[idxs2].player.tolist().copy()
    baseline_helmets.loc[:, 'label'] = 'unknown'
    baseline_helmets.iloc[idxs1, baseline_helmets.columns.get_loc('label')] = labels

    return baseline_helmets, cost


def apply_matrix(M, player_tracking: pd.DataFrame):
    """
    Apply the homography to transform from player_tracking (floor_plane) to baseline helmets (image plane).
    :param M: Matrix
    :param player_tracking: tracked helmets
    :return: This function will add 2 more columns 'xc' and 'yc' to player_tracking
    """
    all_src = np.float32([[player_tracking['x'], player_tracking['y']]]).transpose().reshape(-1, 1, 2)
    tr_src = cv2.perspectiveTransform(all_src, M).transpose()

    player_tracking.loc[:, 'xc'] = tr_src[0, 0]
    player_tracking.loc[:, 'yc'] = tr_src[1, 0]
    return player_tracking


def find_init_matrix(baseline_helmets: pd.DataFrame,
                     player_tracking: pd.DataFrame,
                     view=None,
                     use_confidence=True,
                     flip=False,
                     use_line=False):
    """
    Find Center Reduce matrix
    If adapt_to_view = Endzone  switch x and y coordinates
    If use_confidence, weight normalisation with confidence score

    """
    bh = baseline_helmets.copy()
    if use_line:
        bh.x = bh.x_l
        bh.y = bh.y_l

    th = player_tracking.copy()

    # Center matrix : Align centroid
    if use_confidence:
        bh_centroid = np.array([np.average(bh.x, weights=bh.conf), np.average(bh.y, weights=bh.conf)])
    else:
        bh_centroid = np.array([np.average(bh.x), np.average(bh.y)])

    th_centroid = np.array([np.average(th.x), np.average(th.y)])

    C1 = np.float32([[1, 0, -th_centroid[0]],
                     [0, 1, -th_centroid[1]],
                     [0, 0, 1]])

    C2 = np.float32([[1, 0, bh_centroid[0]],
                     [0, 1, bh_centroid[1]],
                     [0, 0, 1]])

    bh.loc[:, 'd'] = bh.apply(lambda x: math.sqrt((x.x - bh_centroid[0]) ** 2
                                                  + (x.y - bh_centroid[1]) ** 2), axis=1)
    if use_confidence:
        bh_std = np.average(bh.d, weights=bh.conf)
    else:
        bh_std = np.average(bh.d)

    th.loc[:, 'd'] = th.apply(lambda x: math.sqrt((x.x - th_centroid[0]) ** 2
                                                  + (x.y - th_centroid[1]) ** 2), axis=1)

    th_std = np.average(th.d)

    ratio = bh_std / th_std

    R = np.float32([[ratio, 0, 0],
                    [0, ratio, 0],
                    [0, 0, 1]])

    if view == 'Endzone':
        A = np.float32([[0, 1.4, 0],
                        [0.7, 0, 0],
                        [0, 0, 1]])

    if view == 'Sideline':
        A = np.float32([[1.4, 0, 0],
                        [0, -0.7, 0],
                        [0, 0, 1]])

    F = np.float32([[-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]])

    if not flip:
        init_matrix = C2 @ A @ R @ C1
    else:
        init_matrix = C2 @ F @ A @ R @ C1
    return init_matrix


def find_M(baseline_helmets: pd.DataFrame, player_tracking: pd.DataFrame):
    """
    Find the homography between 2 set of points: baseline_helmets and player_tracking
    :param baseline_helmets:
    :param player_tracking:
    :return:
    """
    conf_th = 0.7
    left = baseline_helmets[['x', 'y', 'conf', 'label']].set_index('label')
    right = player_tracking[['x', 'y', 'player']].set_index('player')
    merged = left.join(right, how='inner', rsuffix='_r')
    merged = merged[merged['conf'] > conf_th]
    src = np.float32([merged['x_r'], merged['y_r']]).transpose().reshape(-1, 1, 2)
    dst = np.float32([merged['x'], merged['y']]).transpose().reshape(-1, 1, 2)
    if len(src) == 3:
        M = cv2.getAffineTransform(src, dst)
        M = np.vstack((M, [0, 0, 1]))
    elif len(src) < 3:
        raise Exception('not enough input pts for homography mapping')
    else:
        M, _ = cv2.findHomography(src, dst)
    return M


def find_homography_with_label(gt_helmets_frame, player_tracking_frame):
    """
    Find homography between gt and player_tracking data
    :param gt_helmets_frame:
    :param player_tracking_frame:
    :return:
    """
    player_tracking_frame = player_tracking_frame[['player', 'x', 'y']]
    gt_helmets_frame = gt_helmets_frame[['label', 'x', 'y']]
    dfinal = player_tracking_frame.merge(gt_helmets_frame, how='inner', left_on='player', right_on='label')

    pts_player_tracking = dfinal[['x_x', 'y_x']].values
    pts_gt_helmets = dfinal[['x_y', 'y_y']].values

    H, _ = cv2.findHomography(pts_player_tracking, pts_gt_helmets)
    pts_pt = pts_player_tracking.reshape(-1, 1, 2)
    pts_pt_warped = cv2.perspectiveTransform(pts_pt, H).reshape(len(pts_pt), 2)
    return H, pts_gt_helmets, pts_pt_warped


def apply_line_homography_matrix(M, baseline_helmets: pd.DataFrame):
    """
    Apply the homography to transform from baseline_helmets (image_plane).
    :param baseline_helmets:
    :param M: Matrix
    :param baseline_helmets: tracked helmets
    :return: This function will add 2 more columns 'xc' and 'yc' to player_tracking
    """
    all_src = np.float32([[baseline_helmets['x'], baseline_helmets['y']]]).transpose().reshape(-1, 1, 2)
    tr_src = cv2.perspectiveTransform(all_src, M).transpose()

    baseline_helmets.loc[:, 'x_l'] = tr_src[0, 0]
    baseline_helmets.loc[:, 'y_l'] = tr_src[1, 0]
    return baseline_helmets
