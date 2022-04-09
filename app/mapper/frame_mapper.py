import numpy as np

from mapper import mapping_func


class Matcher2d(object):
    """
    Class to map 2d from floor plane to image plane. This will use homography and then calculate the cost.
    """

    def __init__(self, bh, th, view, line_homography=None, cost=np.inf):
        """
        Init the class with baseline helmets and tracking helmets
        :param bh: baseline helmets
        :param th: tracking helmets
        :param view: view of the video
        :param homography_matrix: homography between image plane and floor plane
        :param cost: the reproject cost
        """
        self.bh = bh
        # Transformation matrix from x,y (floor plane) to xc, yc (image plane)
        if line_homography is None:
            self.line_homography = np.eye(3, dtype=np.float32)
            self.inv_line_homography = np.eye(3, dtype=np.float32)
            self.use_lines = False
        else:
            self.line_homography = line_homography
            self.inv_line_homography = np.linalg.inv(self.line_homography)
            self.use_lines = True

        self.matrix = np.eye(3)
        self.th = mapping_func.apply_matrix(self.matrix, th)

        # Initialise best arguments
        self.view = view
        self.cost = cost
        self.best_matrix = self.matrix

    def apply_line_homography(self):
        """
        Apply line homography to baseline helmets
        :return:
        """
        self.bh = mapping_func.apply_line_homography_matrix(self.line_homography, self.bh)

    def projection(self, flip=False, use_line=False):
        """
        Find the init matrix by scale and rotate follow view
        :param flip: flip or not
        :param use_line: use homography computed by line or not
        :return:
        """

        CR = mapping_func.find_init_matrix(baseline_helmets=self.bh,
                                           player_tracking=self.th,
                                           view=self.view,
                                           flip=flip,
                                           use_line=use_line)
        if use_line:
            self.matrix = self.inv_line_homography @ CR
            self.th = mapping_func.apply_line_homography_matrix(CR, self.th.copy())
        else:
            self.matrix = CR
        self.th = mapping_func.apply_matrix(self.matrix, self.th.copy())

    def homography(self, M=None):
        """
        Find homography using 2 set of points: baseline helmets tracking helmets
        :param M: if M is None then find and apply, otherwise apply M to tracking helmets
        :return:
        """
        if M is None:
            try:
                M = mapping_func.find_M(self.bh.copy(), self.th.copy())
            except Exception as e:
                # print(432434, e)
                return
        self.matrix = M
        self.th = mapping_func.apply_matrix(M, self.th.copy())

    def match(self):
        """
        Match baseline helmets and tracked helmets
        :return:
        """
        bh, cost = mapping_func.hungarian_matching(self.bh.copy(), self.th.copy())
        if cost < self.cost:
            bh['map_cost'] = cost
            self.cost = cost
            self.bh = bh
            self.best_matrix = self.matrix

    def apply_best_matrix(self):
        self.th = mapping_func.apply_matrix(self.best_matrix, self.th.copy())

    # For visualisation
    def get_bh_xy(self, conf_th=0):
        bh = self.bh.query('conf > @conf_th')
        return bh.x.to_list(), bh.y.to_list()

    def get_th_xy(self):
        return self.th.xc.to_list(), self.th.yc.to_list()
