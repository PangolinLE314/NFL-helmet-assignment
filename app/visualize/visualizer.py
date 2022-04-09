import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from decord import VideoReader, cpu

from mapper import mapping_func
from visualize import vis_utils, vis_match


class Visualizer(object):
    """
    Class to visualize the data
    """

    def __init__(self, input_video_root: str, output_video_root=None):
        self.input_video_root = input_video_root
        self.output_video_root = output_video_root
        self.cap = None
        self.current_video = None
        self.ground_player_tracking = None
        self.vis_dir = str(Path(output_video_root).joinpath('visualization'))
        os.makedirs(self.vis_dir, exist_ok=True)

    def sort_player_tracking_by_time(self, player_tracking):
        """
        Sort player tracking by time
        :param player_tracking:
        :return:
        """
        ground_player_tracking = player_tracking.copy()
        ground_player_tracking["track_time_count"] = (
            ground_player_tracking.sort_values("time").groupby("game_play")["time"].rank(method="dense").astype("int")
        )
        self.ground_player_tracking = ground_player_tracking

    def read_video(self, video_name, video_extension='mp4'):
        """
        Read video using decord
        :param video_name: video name
        :param video_extension: video extension
        :return:
        """
        if video_name != self.current_video:
            full_video_name = str(Path(self.input_video_root).joinpath(f'{video_name}.{video_extension}'))
            self.cap = VideoReader(full_video_name, ctx=cpu(0))
            self.current_video = video_name

    def get_frame(self, video_name, frame_id, mode='video'):
        if mode == 'video':
            self.read_video(video_name=video_name)
            self.cap.seek(0)
            frame = self.cap[frame_id - 1].asnumpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif mode == 'image':
            img_dir = Path(self.input_video_root).joinpath(video_name).joinpath('{:06d}.jpg'.format(frame_id))
            frame = cv2.imread(str(img_dir))
        else:
            raise ValueError
        return frame

    def show_raw_video(self, video_name):
        """
        Show raw video
        :param video_name: video name
        :return:
        """
        self.read_video(video_name)
        self.cap.seek(0)
        for i in range(len(self.cap)):
            frame = self.cap.next().asnumpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.show_img(frame, window_name='Raw video')
        cv2.destroyWindow('Raw video')

    def show_baseline_helmets_one_frame(self, data_baseline_helmets: pd.DataFrame,
                                        video_name: str,
                                        frame_number: int,
                                        show=False,
                                        write=False,
                                        output_name_prefix=''):
        """
        Show baseline helmets detections in one particular frame
        :param data_baseline_helmets:
        :param video_name
        :param frame_number:
        :param show
        :param write
        :param output_name_prefix:
        :return:
        """
        video_frame = f'{video_name}_{frame_number}'
        video_data_in = data_baseline_helmets[data_baseline_helmets['video_frame'] == video_frame]
        self.read_video(video_name)
        self.cap.seek(0)
        image = self.cap[frame_number - 1].asnumpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = vis_utils.plot_one_frame_in_video_data(video_data_in, image, show_track=False)
        output_name = str(Path(self.vis_dir).joinpath('{}_{}_{:06d}.jpg'.format(
            video_name,
            output_name_prefix,
            int(frame_number))))
        self.show_img(image, show=show, write=write, output_name=output_name)

    @staticmethod
    def show_img(img, show=True, window_name='Test', write=False, output_name=None):
        """
        Show or write image
        :param img: image to write or show
        :param show: flag to show
        :param window_name: window name
        :param write: flag to write
        :param output_name: fullpath of img to write
        :return:
        """
        if show:
            cv2.imshow(window_name, img)
            cv2.waitKey(10)
        if write:
            cv2.imwrite(output_name, img)

    def show_baseline_helmets_all(self,
                                  data_baseline_helmets: pd.DataFrame,
                                  video_name: str,
                                  show=True,
                                  write=False,
                                  show_track=False,
                                  output_name_prefix=None):
        """
        Visualize the baseline helmets detection
        :param data_baseline_helmets:
        :param video_name
        :param show show
        :param write
        :param show_track
        :param output_name_prefix:
        :return:
        """
        data_baseline_helmets_show = data_baseline_helmets[data_baseline_helmets.video == video_name].copy()
        video_data = data_baseline_helmets_show.sort_values('frame').reset_index(drop=True)
        myvideos = video_data.video.unique()
        for myvideo in myvideos:
            self.read_video(myvideo)
            self.cap.seek(0)
            if self.output_video_root is not None and output_name_prefix is not None:
                full_name = str(Path(self.vis_dir).joinpath(myvideo).joinpath(output_name_prefix))
                os.makedirs(full_name, exist_ok=True)
            else:
                write = False
                full_name = ''
            for frame, video_data_in in video_data.groupby(['frame']):
                videoframe = video_data_in.frame.unique()[0]
                full_img_name = full_name + '/{:06d}.jpg'.format(videoframe)
                image = self.cap.next().asnumpy()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                vis_utils.plot_one_frame_in_video_data(video_data_in, image, show_track)

                self.show_img(image, show=show,
                              window_name="Baseline Helmets",
                              write=write,
                              output_name=full_img_name)

            if write:
                vis_utils.write_mp4(full_name, str(Path(self.vis_dir).joinpath(f'{myvideo}_{output_name_prefix}.mp4')))

    def visualize_player_tracking(self, player_tracking: pd.DataFrame,
                                  video_name: str,
                                  time_count: int,
                                  show=False,
                                  zoom=False):
        """
        Visualize
        :param player_tracking:
        :param video_name:
        :param time_count:
        :param show:
        :param zoom:
        :return:
        """
        if self.ground_player_tracking is None:
            self.sort_player_tracking_by_time(player_tracking)

        ax = vis_utils.create_football_field()
        game_id, play_id, view = video_name.split('_')
        game_play = game_id + '_' + play_id
        track_time_count = time_count
        example_tracks = self.ground_player_tracking.query(
            "game_play == @game_play and track_time_count == @track_time_count")
        for team, d in example_tracks.groupby("team"):
            ax.scatter(d["x"], d["y"], label=team, s=65, lw=1, edgecolors="black", zorder=5)
        ax.legend().remove()
        ax.set_title(f"Tracking data for {game_play}: at time_count = {time_count}", fontsize=15)

        if zoom:
            example_tracks = self.ground_player_tracking.query(
                "game_play == @game_play")
            x_min = min(example_tracks['x'])
            y_min = min(example_tracks['y'])

            x_max = max(example_tracks['x'])
            y_max = max(example_tracks['y'])

            ax.set_xlim([x_min - 5, x_max + 5])
            ax.set_ylim([y_min - 5, y_max + 5])

        if show:
            plt.show()

    def animate(self, player_tracking, video_name, output_name="test.html", write=False):
        """
        Animate using plotly and save to output
        :param player_tracking:
        :param video_name:
        :param output_name:
        :param write:
        :return:
        """
        self.sort_player_tracking_by_time(player_tracking)
        game_id, play_id, view = video_name.split('_')
        game_play = game_id + '_' + play_id

        folder_out = Path(self.vis_dir)
        folder_out.mkdir(exist_ok=True)
        html_out = folder_out.joinpath(f'{video_name}_{output_name}')
        vis_match.vis_video(self.ground_player_tracking, game_play, html_out=html_out, write=write)

    def show_example_gt(self, video_name):
        base_dir = str(Path(self.input_video_root).parent)
        example_video = f'{base_dir}/train/{video_name}.mp4'
        tr_helmets = pd.read_csv(f'{base_dir}/train_baseline_helmets.csv')
        labels = pd.read_csv(f'{base_dir}/train_labels.csv')
        output_video = vis_match.video_with_baseline_boxes(example_video,
                                                           tr_helmets, labels)
        os.system(f'mv {output_video} {self.vis_dir}')

    def show_player_tracking_projection_one_frame(self, video_name,
                                                  frame_id,
                                                  helmets_vid,
                                                  player_tracking_vid,
                                                  show=False,
                                                  write=False,
                                                  output_name_prefix=""
                                                  ):
        """
        Find homography to transform data player tracking and project on to image plane
        :param video_name:
        :param frame_id:
        :param helmets_vid:
        :param player_tracking_vid:
        :param show
        :param write
        :param output_name_prefix
        :return:
        """
        image = self.get_frame(video_name, frame_id, mode='video')
        nearest_frame_id = mapping_func.get_nearest_frame(frame_id, player_tracking_vid)
        player_tracking_frame = player_tracking_vid[player_tracking_vid.est_frame == nearest_frame_id]
        helmets_frame = helmets_vid[helmets_vid.frame == frame_id]

        H, pts_helmets, pts_pt_warped = mapping_func.find_homography_with_label(helmets_frame, player_tracking_frame)

        for pt in pts_helmets:
            image = cv2.circle(image, (int(pt[0]), int(pt[1])), 5, color=(255, 0, 0), thickness=5)
        for pt in pts_pt_warped:
            image = cv2.circle(image, (int(pt[0]), int(pt[1])), 5, color=(0, 255, 0), thickness=5)

        folder_out = Path(self.vis_dir).joinpath(video_name)
        folder_out.mkdir(exist_ok=True)
        output_name = str(folder_out.joinpath('{}_{:06d}.jpg'.format(output_name_prefix, int(frame_id))))
        self.show_img(image, window_name="Projection", show=show, write=write, output_name=output_name)
        return image

    def create_final_video_result(self, submission, sub_labels, video_id=0):
        from modules.helmet_assignment.helmet_assignment.video import video_with_predictions
        submission['video'] = submission['video_frame'].str.split('_').str[:3].str.join('_') + '.mp4'
        debug_videos = submission['video'].unique()
        video_name = debug_videos[video_id].split('.')[0]
        # Create video showing predictions for one of the videos.
        video_out = video_with_predictions(
            f'{self.input_video_root}/{debug_videos[video_id]}', sub_labels, freeze_impacts=False)

        folder_out = Path(self.vis_dir)
        folder_out.mkdir(exist_ok=True)

        os.system(f'mv {video_out} {str(folder_out)}')
