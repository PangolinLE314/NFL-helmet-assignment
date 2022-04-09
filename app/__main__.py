import os
import sys

sys.path.insert(0, f'{os.getcwd()}/app')
sys.path.insert(0, f'{os.getcwd()}/app/modules')
from visualize import data_visualization
from data_io import post_processing
from homography import homography
from data_io.nfl_dataloader import NFLDataloader
from tracker.tracker import Tracker
from mapper.video_mapper import VideoMapper
from visualize.visualizer import Visualizer

if __name__ == '__main__':
    # Get env variable (useful for docker solution)
    video_id = int(os.environ.get('VIDEO_ID', 1))
    frame_number = int(os.environ.get('FRAME_NUMBER', 154))
    visualize_raw = os.environ.get('VISUALIZE_RAW', 'True').lower() in ['1', 'true']
    visualize_result = os.environ.get('VISUALIZE_RESULT', 'True').lower() in ['1', 'true']
    output_video_root = 'output'
    input_data_root = 'input'

    nfl_test = NFLDataloader(f'{input_data_root}/nfl-health-and-safety-helmet-assignment', mode='test')
    vis = Visualizer(nfl_test.video_dir, output_video_root)

    # Visualize raw data
    if visualize_raw:
        data_visualization.visualize_raw_data(vis, nfl_test, video_id, frame_number, show=False, write=True)

    # Run deepsort to track baseline helmets
    deepsort_config = 'configs/deepsort.yaml'
    deepsort_res_file = f'{output_video_root}/deepsort/deepsort_baseline_helmets.csv'
    nfl_tracker = Tracker(deepsort_config=deepsort_config, debug=False, input_file=deepsort_res_file)
    compute_only = None
    data_baseline_helmets, list_videos = nfl_tracker.run_deepsort_all(nfl_test.dbh_high,
                                                                      nfl_test.dbh_low,
                                                                      nfl_test.video_dir,
                                                                      compute_only=compute_only,
                                                                      output_file=deepsort_res_file)
    list_game_plays = list(map(lambda x: x.rsplit('_', 1)[0], list(list_videos)))
    data_player_tracking = nfl_test.data_player_tracking[
        nfl_test.data_player_tracking.game_play.isin(list_game_plays)].copy()

    # Run mappers to map baseline helmets and player tracking
    mappers = []
    for video_name in list_videos:
        print(f"Start processing video:{video_name}")
        if 'Sideline' in video_name:
            # Run homography finding using lines
            use_line = True
            line_homographies_file = f'{output_video_root}/line_homographies/line_homographies_{video_name}.npy'
            line_homographies = homography.find_homography_lines_video(nfl_test.video_dir,
                                                                       video_name,
                                                                       output_file=line_homographies_file)
        else:
            use_line = False
            line_homographies = None

        mapper = VideoMapper(video_name,
                             data_baseline_helmets,
                             data_player_tracking,
                             use_line=use_line,
                             line_homographies=line_homographies)
        mapper.run_all()
        mappers.append(mapper)

    # Run submission preparation and scorer
    baseline_helmets_all = [mapper.baseline_helmets for mapper in mappers]
    score, submission, sub_labels = post_processing.prepare_submission(baseline_helmets_all,
                                                                       nfl_test.sample_submission,
                                                                       output_video_root=output_video_root,
                                                                       data_labels=nfl_test.data_labels
                                                                       )

    # Visualize result including
    if visualize_result:
        data_visualization.visualize_result(vis,
                                            mappers,
                                            list_videos,
                                            video_id,
                                            frame_number,
                                            submission,
                                            sub_labels,
                                            show=False,
                                            write=True)

    print(f'Code finished successfully. Score = {score}')
