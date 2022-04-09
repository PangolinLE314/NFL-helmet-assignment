import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd

from mapper import mapping_func
from mapper.frame_mapper import Matcher2d
from visualize import vis_utils


class VideoMapper(object):
    """
    Class to map 2d baseline detected helmets and tracked helmets
    """

    def __init__(self, video_name, data_baseline_helmets, data_player_tracking, use_line=False, line_homographies=None):
        """
        Init the Mapper with playID and view
        :param video_name: input video_name
        :param data_baseline_helmets: baseline helmets tracked by deepsort
        :param data_player_tracking: data player tracking provided by the dataset
        :param line_homographies calculated by lines
        """
        # Get a copy of each helmets for this video
        game_id, play_id, view = video_name.split('_')
        play_id = int(play_id)
        dpt = data_player_tracking.groupby(['playID'])
        dbh = data_baseline_helmets.groupby(['playID', 'view'])
        player_tracking = dpt.get_group(play_id).reset_index(drop=True).copy()
        baseline_helmets = dbh.get_group((play_id, view)).reset_index(drop=True).copy()
        self.baseline_helmets = baseline_helmets
        self.player_tracking = player_tracking
        self.view = view

        self.use_line = use_line

        # Create a Matcher2d for each frame
        tracking_helmets_group = player_tracking.groupby('est_frame')
        baseline_helmets_group = baseline_helmets.groupby('frame')
        self.matcher2ds = []
        self.frame_ids = []
        for frame_id in self.baseline_helmets.frame.unique():
            self.frame_ids.append(frame_id)
            nearest_frame_id = mapping_func.get_nearest_frame(frame_id, player_tracking)
            th = tracking_helmets_group.get_group(nearest_frame_id).copy()
            bh = baseline_helmets_group.get_group(frame_id).copy()
            if line_homographies is not None:
                line_homography = line_homographies[self.frame_ids.index(frame_id)]
            else:
                line_homography = None
            self.matcher2ds.append(Matcher2d(bh, th, view, line_homography=line_homography))

        self.frames_mx = []

    def animate(self):
        """
        Animate points
        :return:
        """
        fig, ax = plt.subplots(figsize=(15, 15))

        point_th, = ax.plot([], [], ls="none", marker="o", color='blue')
        point_bh, = ax.plot([], [], ls="none", marker="o", color='orange')
        links = [ax.plot([], [], color='green') for _ in range(22)]

        def anim(k):
            i = min(k, len(self.matcher2ds))
            matcher2d = self.matcher2ds[k]
            x, y = matcher2d.get_bh_xy(conf_th=0.2)
            point_bh.set_data(x, y)
            x, y = matcher2d.get_th_xy()
            point_th.set_data(x, y)

            if 'label' in matcher2d.bh.columns:
                left = matcher2d.bh[['x', 'y', 'label']].set_index('label')
                right = matcher2d.th[['xc', 'yc', 'player']].set_index('player')
                merged = left.join(right, how='inner')
                pts = [([row.x, row.xc], [row.y, row.yc]) for _, row in merged.iterrows()]

                for link in links:
                    link[0].set_data([], [])
                for link, pt in zip(links, pts):
                    link[0].set_data(pt[0], pt[1])

            return point_bh, point_th

        ax.set_xlim([-10, 1400])
        ax.set_ylim([-10, 800])

        ani = animation.FuncAnimation(fig=fig, func=anim, frames=range(len(self.matcher2ds)), interval=50,
                                      blit=True)
        return ani

    def projection(self, flip=False, use_line=False):
        """
        Compute transformation on tracking data for each frame
        :param use_line:
        :param flip: True to flip image otherwise false
        :return:
        """
        for matcher2d in self.matcher2ds:
            matcher2d.projection(flip=flip, use_line=use_line)

    def apply_line_homography(self):
        """
        apply line homography
        :return:
        """
        for matcher2d in self.matcher2ds:
            matcher2d.apply_line_homography()

    def homography(self):
        """

        :return:
        """
        for matcher2d in self.matcher2ds:
            matcher2d.homography()

    def apply_best_matrix(self):
        for matcher2d in self.matcher2ds:
            matcher2d.apply_best_matrix()

    def match(self):
        for matcher2d in self.matcher2ds:
            matcher2d.match()

    def foward_repair(self):
        # Apply previous homography on current frame
        cur_M = self.matcher2ds[0].matrix
        for matcher2d in self.matcher2ds:
            matcher2d.homography(cur_M)
            matcher2d.match()
            matcher2d.homography()
            matcher2d.match()
            cur_M = matcher2d.best_matrix

    def backward_repair(self):
        # Apply next homography on current frame
        cur_M = self.matcher2ds[-1].matrix
        for matcher2d in reversed(self.matcher2ds):
            matcher2d.homography(cur_M)
            matcher2d.match()
            matcher2d.homography()
            matcher2d.match()
            cur_M = matcher2d.best_matrix

    # Map procedures
    def update_map(self):
        self.baseline_helmets = pd.concat([matcher2d.bh for matcher2d in self.matcher2ds])
        self.frames_mx = [matcher2d.best_matrix for matcher2d in self.matcher2ds]

    # Track procedures # Track on the top occuring label for each cluster
    def cluster_count_track(self):
        # Find the top occuring label for each cluster
        sortlabel_map = self.baseline_helmets.groupby('cluster')['label'].value_counts() \
            .sort_values(ascending=False).to_frame() \
            .rename(columns={'label': 'label_count'}) \
            .reset_index() \
            .groupby(['cluster']) \
            .first()['label'].to_dict()

        # Find the # of times that label appears for the deepsort_cluster.
        sortlabelcount_map = self.baseline_helmets.groupby('cluster')['label'].value_counts() \
            .sort_values(ascending=False).to_frame() \
            .rename(columns={'label': 'label_count'}) \
            .reset_index() \
            .groupby(['cluster']) \
            .first()['label_count'].to_dict()

        # Find the total # of label for each deepsort_cluster.
        sortlabeltotal_map = self.baseline_helmets.groupby('cluster')['label'].value_counts() \
            .sort_values(ascending=False).to_frame() \
            .rename(columns={'label': 'label_count'}) \
            .reset_index() \
            .groupby(['cluster']) \
            .sum()['label_count'].to_dict()

        sortlabelconf_map = {k: (sortlabelcount_map[k] / sortlabeltotal_map[k]) for k in sortlabeltotal_map}

        self.baseline_helmets['label_cluster'] = self.baseline_helmets['cluster'].map(sortlabel_map)
        self.baseline_helmets['cluster_count'] = self.baseline_helmets['cluster'].map(sortlabelcount_map)
        self.baseline_helmets['cluster_conf'] = self.baseline_helmets['cluster'].map(sortlabelconf_map)

        # Merge baseline_helmets with the tracking clusters infos
        for _, example in self.baseline_helmets.groupby('video_frame'):
            example['cluster_score'] = example.apply(lambda x: x.cluster_count * x.cluster_conf ** 3, axis=1)
            example.sort_values('cluster_score', ascending=False, inplace=True)
            assigned = set()
            for idx, row in example.iterrows():
                if row.label_cluster not in assigned or row.label_cluster == 'unknown':
                    assigned.add(row.label_cluster)
                    self.baseline_helmets.loc[idx, 'label'] = row.label_cluster
                elif row.label not in assigned:
                    assigned.add(row.label)
                    self.baseline_helmets.loc[idx, 'label'] = row.label
                else:
                    self.baseline_helmets.loc[idx, 'label'] = 'unknown'

    def update_track(self):
        # Group on frame
        thg = self.player_tracking.groupby('est_frame')
        bhg = self.baseline_helmets.groupby('frame')

        self.matcher2ds = []
        for frameID in self.baseline_helmets.frame.unique():
            nearest_frameID = mapping_func.get_nearest_frame(frameID, self.player_tracking)
            th = thg.get_group(nearest_frameID).copy()
            bh = bhg.get_group(frameID).copy()
            self.matcher2ds.append(Matcher2d(bh, th, self.view))

    def vis_data_frame(self, frame_number):
        """
        Visualize data at ith frame using matplot lib
        :return:
        """
        index = self.frame_ids.index(frame_number)
        matcher2d = self.matcher2ds[index]

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

        x_bh, y_bh = matcher2d.get_bh_xy(conf_th=0.0)
        x_th, y_th = matcher2d.get_th_xy()
        ax.scatter(x_bh, y_bh, marker="o", color='blue', s=50)
        ax.scatter(x_th, y_th, marker="o", color='orange', s=50)
        # point_bh, = ax.plot([], [], ls="none", marker="o", color='orange')
        # links = [ax.plot([], [], color='green') for _ in range(22)]
        return fig

    def vis_data_frame_cv(self, image, frame_number,
                          vis_bh=True,
                          vis_th=True,
                          show=False,
                          write=False,
                          output_root=None):
        """
        Visualize data at a frame using opencv
        :param image:
        :param frame_number:
        :param vis_bh:
        :param vis_th:
        :param output_root:
        :param write:
        :param show:
        :return:
        """
        index = self.frame_ids.index(frame_number)
        matcher2d = self.matcher2ds[index]
        x_bh, y_bh = matcher2d.get_bh_xy(conf_th=0.0)
        x_th, y_th = matcher2d.get_th_xy()

        if vis_bh:
            for x, y in zip(x_bh, y_bh):
                image = cv2.circle(image, (int(x), int(y)), 5, color=(255, 0, 0), thickness=5)

        if vis_th:
            for x, y in zip(x_th, y_th):
                image = cv2.circle(image, (int(x), int(y)), 5, color=(0, 255, 0), thickness=5)

        vis_utils.show_img(image, show=show, write=write,
                           img_name='{}/position_image_{:06d}.jpg'.format(output_root, frame_number))
        return image

    def run_all(self):
        if self.use_line:
            self.apply_line_homography()
            print("using line")
        self.projection(flip=False, use_line=self.use_line)
        self.match()
        self.projection(flip=True, use_line=self.use_line)
        self.match()
        self.homography()
        self.match()
        self.foward_repair()
        self.backward_repair()
        self.update_map()
        self.cluster_count_track()
        self.homography()
        self.apply_best_matrix()

    def vis_line_homography_projection(self, image,
                                       frame_number,
                                       vis_bh=True,
                                       vis_th=True,
                                       show=False,
                                       write=False,
                                       output_root=None):
        """
        Visualize homography projection
        :param image:
        :param frame_number:
        :param vis_bh
        :param vis_th
        :param show:
        :param write
        :param output_root
        :return:
        """
        index_frame = self.frame_ids.index(frame_number)
        baseline_helmets = self.matcher2ds[index_frame].bh
        player_tracking = self.matcher2ds[index_frame].th
        line_homography = self.matcher2ds[index_frame].line_homography
        h, w = image.shape[:2]
        img_final = cv2.warpPerspective(image, line_homography, (w, h))
        if vis_bh:
            for x, y in zip(baseline_helmets.x_l, baseline_helmets.y_l):
                img_final = cv2.circle(img_final, (int(x), int(y)), 5, color=(255, 0, 0), thickness=5)
        if vis_th:
            for x, y in zip(player_tracking.x_l, player_tracking.y_l):
                img_final = cv2.circle(img_final, (int(x), int(y)), 5, color=(0, 255, 0), thickness=5)
        vis_utils.show_img(img_final, show=show,
                           write=write,
                           img_name='{}/bird_view_image_{:06d}.jpg'.format(output_root, frame_number))

# input_video_root = '/home/tracking/Documents/Personal/SkillCorner/input/nfl-health-and-safety-helmet-assignment/test'
# output_video_root = '../output'
# vis = Visualizer(input_video_root, output_video_root)
#
# video_name = '57906_000718_Endzone'
# self = VideoMapper(video_name, dpt=dpt, dbh=dbh)
# self.projection(flip=True)
# self.match()
# self.projection(flip=False)
# self.match()
# self.homography()
# self.match()
# self.foward_repair()
# self.backward_repair()
# plt.show()
# self.update_map()
# self.cluster_count_track()
# self.update_track()
# self.homography()
#
#
# frame_number = 50
# frame = vis.get_frame(video_name, frame_number)
# self.vis_data_frame(frame_number)
# frame_draw = self.vis_data_frame_cv(frame, frame_number)
# # vis.show_baseline_helmets_one_frame(data_baseline_helmets,
# #                                     video_name,
# #                                     frame_number,
# #                                     write=True,
# #                                     output_name='test.jpg')
# plt.imshow(frame_draw)
# plt.show()
#
# # self.match()
# # ani = self.animate()
# # f = "test.gif"
# # writergif = animation.PillowWriter(fps=30)
# # ani.save(f, writer=writergif)
