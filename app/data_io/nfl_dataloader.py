from pathlib import Path

import pandas as pd


class NFLDataloader(object):
    """
    Class to load data from NFL Safety
    """

    def __init__(self, root: str = 'input/nfl-health-and-safety-helmet-assignment', mode: str = 'test'):
        """
        Init the class
        :param root: root_dir contains the nfl dataset
        :param mode: can be 'train' or 'test'
        """
        self.root = root
        self.mode = mode
        self.data_baseline_helmets = pd.read_csv(f'{root}/{mode}_baseline_helmets.csv')
        self.data_player_tracking = pd.read_csv(f'{root}/{mode}_player_tracking.csv')
        self.data_labels = pd.read_csv(f'{root}/train_labels.csv')
        if mode == 'train':
            self.data_labels = self.process_labels(self.data_labels)
        self.video_dir = f'{root}/{mode}/'
        self.data_baseline_helmets = self.process_baseline(self.data_baseline_helmets)
        self.data_player_tracking = self.add_track_features(self.data_player_tracking)
        self.dbh_high = self.data_baseline_helmets[self.data_baseline_helmets.conf > 0.4]
        self.dbh_low = self.data_baseline_helmets[self.data_baseline_helmets.conf <= 0.4]
        self.dbh_low = self.dbh_low.assign(cluster='unknown')
        self.sample_submission = pd.read_csv(f'{root}/sample_submission.csv')

    @staticmethod
    def process_baseline(baseline_df):
        baseline_df['gameKey'] = baseline_df['video_frame'].apply(lambda x: int(x.split('_')[0]))
        baseline_df['playID'] = baseline_df['video_frame'].apply(lambda x: int(x.split('_')[1]))
        baseline_df['view'] = baseline_df['video_frame'].apply(lambda x: x.split('_')[2])
        baseline_df['frame'] = baseline_df['video_frame'].apply(lambda x: int(x.split('_')[3]))
        baseline_df['video'] = baseline_df['video_frame'].str.split('_').str[:3].str.join('_')
        baseline_df['x'] = baseline_df.apply(lambda x: x.left + x.width / 2, axis=1)
        baseline_df['y'] = baseline_df.apply(lambda x: x.top + x.height / 2, axis=1)
        baseline_df['label'] = 'unknown'
        return baseline_df

    @staticmethod
    def process_labels(labels_df):
        labels_df['x'] = labels_df.apply(lambda x: x.left + x.width / 2, axis=1)
        labels_df['y'] = labels_df.apply(lambda x: x.top + x.height / 2, axis=1)
        return labels_df

    @staticmethod
    def add_track_features(tracks, fps=59.94, snap_frame=10):
        """
        Add column features helpful for syncing with video data.
        :param tracks: track_features
        :param fps: fps
        :param snap_frame: snap frame
        :return:
        """
        tracks = tracks.copy()
        tracks["game_play"] = (
                tracks["gameKey"].astype("str")
                + "_"
                + tracks["playID"].astype("str").str.zfill(6)
        )
        tracks["time"] = pd.to_datetime(tracks["time"])
        snap_dict = (
            tracks.query('event == "ball_snap"').groupby("game_play")["time"].first().to_dict()
        )
        tracks["snap"] = tracks["game_play"].map(snap_dict)
        tracks["isSnap"] = tracks["snap"] == tracks["time"]
        tracks["team"] = tracks["player"].str[0].replace("H", "Home").replace("V", "Away")
        tracks["snap_offset"] = (tracks["time"] - tracks["snap"]).astype(
            "timedelta64[ms]"
        ) / 1000
        # Estimated video frame
        tracks["est_frame"] = (
            ((tracks["snap_offset"] * fps) + snap_frame).round().astype("int")
        )
        return tracks

    def get_vid_data(self, video_name):
        """
        Get data frame for 1 video
        :param video_name:
        :return:
        """
        game_play = Path(video_name.rsplit('_', 1)[0]).name
        mp4_name = Path(video_name).name + '.mp4'
        player_tracking_vid = self.data_player_tracking.loc[self.data_player_tracking.game_play == game_play].copy()
        if mp4_name in self.data_labels.video.values:
            gt_helmets_vid = self.data_labels.loc[self.data_labels.video == mp4_name].copy()
        else:
            gt_helmets_vid = None
        baseline_helmets_vid = self.data_baseline_helmets.loc[self.data_baseline_helmets.video == mp4_name].copy()
        return player_tracking_vid, baseline_helmets_vid, gt_helmets_vid
