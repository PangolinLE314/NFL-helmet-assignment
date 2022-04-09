import os
import sys

ROOT_DIR = os.environ.get("ROOT_DIR", f'{os.getcwd()}/app')
sys.path.append(
    f'{ROOT_DIR}/modules/deepsort/deep_sort_pytorch')
import cv2
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal
from deep_sort import DeepSort
from tqdm.auto import tqdm
from utils.parser import get_config
from visualize import vis_utils
from pathlib import Path


class Tracker(object):
    def __init__(self, deepsort_config='../configs/deepsort.yaml', debug=False, input_file=None):
        """
        Init the tracker with a config for deepsort
        :param deepsort_config: deepsort configs
        :param debug: debug mode
        """
        self.deepsort_config = deepsort_config
        self.debug = debug
        self.cur_untrack_id = 10000
        self.input_file = input_file

    def run_deepsort(self, video_data, video_dir):
        """
        Run deepsort for NFL helmet data
        :param video_data: df for video
        :param video_dir: video_directory
        :return:
        """
        cfg = get_config()
        cfg.merge_from_file(self.deepsort_config)
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST,
                            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE,
                            n_init=cfg.DEEPSORT.N_INIT,
                            nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        video_data = video_data.sort_values('frame').reset_index(drop=True)
        ds = []
        myvideo = video_data.video.unique()[0]
        cap = cv2.VideoCapture(f'{video_dir}/{myvideo}.mp4')

        for frame, video_data_in in tqdm(video_data.groupby(['frame']), total=video_data['frame'].nunique()):
            xywhs = video_data_in[['x', 'y', 'width', 'height']].values
            success, image = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            confs = np.ones([len(video_data_in), ])
            clss = np.zeros([len(video_data_in), ])
            try:
                outputs = deepsort.update(xywhs, confs, clss, image)
            except Exception as e:
                outputs = []
                print(f'Error {e} : Skipped')
            deepsort_out = pd.DataFrame(outputs, columns=['left', 'top', 'right', 'bottom', 'cluster', 'class'])

            video_data_in = self.merge_deepsort_result(deepsort_out, video_data_in)
            ds.append(video_data_in)

            if self.debug:
                image_vis = vis_utils.plot_one_frame_in_video_data(video_data_in, image, show_track=True)
                cv2.imshow("test", image_vis)
                cv2.waitKey(1)
        dout = pd.concat(ds)
        return dout

    @staticmethod
    def merge_deepsort_result(deepsort_out, video_data_in):
        """
        Merge deepsort result with video data result using eucledian distance
        :param deepsort_out: output of deepsort
        :param video_data_in: video data in
        :return:
        """
        if len(deepsort_out) == 0:
            video_data_in.loc[:, 'cluster'] = 'unknown'
            return video_data_in

        forfit = 10
        deepsort_pts = list(zip(deepsort_out['left'], deepsort_out['top']))
        video_pts = list(zip(video_data_in['left'], video_data_in['top']))

        # Compute matching with Hungarian algorithm in both sides
        match_cost = np.array(
            [[np.linalg.norm(np.float32(pt1) - np.float32(pt2)) for pt1 in deepsort_pts] for pt2 in video_pts])
        trash_cost = np.array([[forfit for _ in deepsort_pts] for _ in deepsort_pts])

        cost_matrix = np.concatenate([match_cost, trash_cost], axis=0)
        idxs1, idxs2 = scipy.optimize.linear_sum_assignment(cost_matrix)

        idxs1, idxs2 = np.array(
            [[idx1, idx2] for idx1, idx2 in zip(idxs1, idxs2) if idx1 < len(video_pts)]).transpose()
        labels = list(deepsort_out.iloc[idxs2]['cluster'].copy())
        video_data_in.loc[:, 'cluster'] = 'unknown'
        video_data_in.iloc[idxs1, video_data_in.columns.get_loc('cluster')] = labels
        return video_data_in

    def run_deepsort_on_baseline_helmets(self, dbh: pd.DataFrame, video_dir: str, compute_only=None):
        """
        Run deepsort on baseline helmets data
        :param dbh:
        :param video_dir: video dir
        :param compute_only: list of id to compute only, if None -> compute all
        :return:
        """
        outs = []
        for i, (video_name, video_data) in enumerate(dbh.groupby('video')):
            if compute_only is not None:
                if i not in compute_only:
                    continue
            print(f'Run deepsort on video :{video_name}')
            out = self.run_deepsort(video_data, video_dir)
            outs.append(out)
        data_baseline_helmets = pd.concat(outs).copy()
        return data_baseline_helmets

    def asign_id_for_untracked_dets(self, data_baseline_helmets: pd.DataFrame, dbh_low: pd.DataFrame):
        """
        asign_id_for_untracked_dets
        :param data_baseline_helmets:
        :param dbh_low:
        :return:
        """
        data_baseline_helmets = pd.concat([data_baseline_helmets, dbh_low]).sort_values(
            ['video', 'frame', 'conf']).reset_index(drop=True)

        data_baseline_helmets.cluster = data_baseline_helmets.cluster.apply(
            lambda x: self.increase_id() if x == 'unknown' else x)
        return data_baseline_helmets

    def increase_id(self):
        self.cur_untrack_id = self.cur_untrack_id + 1
        return self.cur_untrack_id

    def run_deepsort_all(self, dbh_high, dbh_low, video_dir, compute_only=None, output_file=None):
        """
        Run deepsort for a dataset
        :param dbh_high:
        :param dbh_low:
        :param video_dir:
        :param compute_only:
        :param output_file:
        :return:
        """
        if self.input_file is not None and Path(self.input_file).exists():
            data_baseline_helmets = pd.read_csv(self.input_file)
        else:
            data_baseline_helmets = self.run_deepsort_on_baseline_helmets(dbh_high, video_dir,
                                                                          compute_only=compute_only)
            if compute_only is not None:
                list_videos = data_baseline_helmets.video.unique()[compute_only]
                dbh_low_only_compute = dbh_low[dbh_low.video.isin(list_videos).copy()]
            else:
                dbh_low_only_compute = dbh_low
            data_baseline_helmets = self.asign_id_for_untracked_dets(data_baseline_helmets, dbh_low_only_compute)
            if output_file is not None:
                os.makedirs(str(Path(output_file).parent), exist_ok=True)
                data_baseline_helmets.to_csv(output_file)
        list_videos = data_baseline_helmets.video.unique()

        return data_baseline_helmets, list_videos
