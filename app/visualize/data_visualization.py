import copy
from typing import List

import pandas as pd

from data_io.nfl_dataloader import NFLDataloader
from visualize.visualizer import Visualizer
from homography.test_homography import test_sample_image


def visualize_raw_data(vis: Visualizer, nfl_data: NFLDataloader,
                       video_id=1,
                       frame_number=154,
                       show=False,
                       write=True):
    """
    Visualize data raw
    :param vis: Visualizer
    :param nfl_data: dataloader
    :param video_id: id of video to show
    :param frame_number: frame number of the video to show
    :param show: show or not
    :param write: write to file or not
    :return:
    """
    list_video = sorted(list(nfl_data.data_baseline_helmets.video.unique()))
    vis_video_name = list_video[video_id]
    vis.show_baseline_helmets_one_frame(nfl_data.data_baseline_helmets,
                                        video_name=vis_video_name,
                                        frame_number=frame_number,
                                        write=write,
                                        show=show,
                                        output_name_prefix='test_baseline_helmets')

    vis.show_baseline_helmets_all(nfl_data.data_baseline_helmets,
                                  video_name=vis_video_name,
                                  show=show,
                                  write=write,
                                  show_track=False,
                                  output_name_prefix='baseline_helmets')

    vis.visualize_player_tracking(nfl_data.data_player_tracking,
                                  vis_video_name,
                                  10,
                                  show=show,
                                  zoom=True)
    vis.animate(nfl_data.data_player_tracking, vis_video_name, 'player_tracking.html', write=write)

    vis.show_example_gt(vis_video_name)


def visualize_result(vis: Visualizer,
                     mappers: List,
                     list_videos: List,
                     video_id: int,
                     frame_number: int,
                     submission: pd.DataFrame,
                     sub_labels: pd.DataFrame,
                     show=False,
                     write=True):
    """
    Visualize and write result to file
    :param vis:
    :param mappers:
    :param list_videos:
    :param video_id:
    :param frame_number:
    :param data_baseline_helmets
    :param submission:
    :param sub_labels:
    :param show:
    :param write:
    :return:
    """
    test_sample_image()
    video_name = list_videos[video_id]
    frame = vis.get_frame(video_name, frame_number, mode='video')
    frame2 = copy.deepcopy(frame)

    vis.show_baseline_helmets_all(mappers[video_id].baseline_helmets,
                                  video_name,
                                  show=show,
                                  write=write,
                                  show_track=True,
                                  output_name_prefix='deepsort_tracking')

    mappers[video_id].vis_line_homography_projection(frame2,
                                                     frame_number,
                                                     vis_bh=True,
                                                     vis_th=True,
                                                     show=show,
                                                     write=write,
                                                     output_root=vis.vis_dir)

    frame_draw = mappers[video_id].vis_data_frame_cv(frame,
                                                     frame_number,
                                                     vis_bh=True,
                                                     vis_th=True,
                                                     show=show,
                                                     write=write,
                                                     output_root=vis.vis_dir)
    vis.create_final_video_result(submission, sub_labels, video_id=video_id)

#

# video_name = f'57906_000718_Sideline'
# nfl_test = NFLDataloader(mode='test')
# vis = Visualizer(input_video_root=nfl_test.video_dir, output_video_root='../output')
# frame_number = 100
# visualize_raw_data(vis, nfl_test, video_name, frame_number, show=False, write=True)
# player_tracking_vid, _, gt_helmets_vid = nfl_train.get_vid_data(video_name)
#

# output_video_root = '../output'
# vis = Visualizer(nfl_train.video_dir, output_video_root)
# vis.show_player_tracking_projection_one_frame(video_name,
#                                               200,
#                                               gt_helmets_vid,
#                                               player_tracking_vid,
#                                               write=True,
#                                               output_name_prefix='projection_gt')
# nfl_train = NFLDataloader(mode='train')
# player_tracking_vid, _, gt_helmets_vid = nfl_train.get_vid_data(video_name)
#
# output_video_root = '../output'
# vis = Visualizer(nfl_train.video_dir, output_video_root)
# vis.show_player_tracking_projection_one_frame(video_name,
#                                               200,
#                                               gt_helmets_vid,
#                                               player_tracking_vid,
#                                               write=True,
#                                               output_name_prefix='projection_gt')
