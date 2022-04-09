import copy
import math

import cv2
import numpy as np


class Line(object):
    """
    Custom line object
    """

    def __init__(self, line):
        """
        Init the line with x1,y1 and x2,y2
        """
        self.line = [line[i] for i in range(4)]
        self.x1, self.y1, self.x2, self.y2 = list(map(int, self.line))
        self.segment = copy.deepcopy(self.line)

    @property
    def m(self):
        """
        Get slope of the line
        :return:
        """
        return (self.y2 - self.y1) / ((self.x2 - self.x1) + 0.0000001)

    @property
    def p1(self):
        """
        First point
        :return:
        """
        return np.int32([self.x1, self.y1])

    @property
    def p2(self):
        """
        Second point
        :return:
        """
        return np.int32([self.x2, self.y2])

    @property
    def angle(self):
        return math.atan2(self.p1[0] - self.p2[0], self.p1[1] - self.p2[1])

    @property
    def length(self):
        """
        length of the line
        :return:
        """
        return np.linalg.norm(self.p1 - self.p2)

    def get_y_from_x(self, x):
        """
        Get y position given x
        :param x:
        :return:
        """
        return int(self.m * (x - self.x1) + self.y1)

    def get_x_from_y(self, y):
        """
        Get x position given y
        :param y:
        :return:
        """
        return int((y - self.y1) / self.m + self.x1)

    def get_xy_from_d(self, d):
        """
        Get x and y given the length
        :param d:
        :return:
        """
        scale = d / self.length
        x = scale * (self.x2 - self.x1) + self.x1
        y = scale * (self.y2 - self.y1) + self.y1
        return np.int32([x, y])

    def vertical_intersect(self, x):
        """
        Find vertical intersect
        :param x:
        :return:
        """
        return np.int32([x, self.get_y_from_x(x)])

    def horizontal_intersect(self, y):
        """
        Find horizontal intersect
        :param y:
        :return:
        """
        return np.int32([self.get_x_from_y(y), y])

    @property
    def positive_p(self):
        return self.horizontal_intersect(10000)

    @property
    def negative_p(self):
        return self.horizontal_intersect(-10000)

    def get_segment(self, x_min, x_max, y_min, y_max):
        """
        Get segment limit by a ROI
        :param x_min:
        :param x_max:
        :param y_min:
        :param y_max:
        :return:
        """
        h1 = self.horizontal_intersect(y_min)
        h2 = self.horizontal_intersect(y_max)
        v1 = self.vertical_intersect(x_min)
        v2 = self.vertical_intersect(x_max)
        ps = []
        for p in [h1, h2, v1, v2]:
            if self.check_out_side(p, x_min, x_max, y_min, y_max):
                ps.append(p)
        if len(ps) != 2:
            pass
        else:
            if ps[0][1] <= ps[1][1]:
                self.segment = list(ps[0]) + list(ps[1])
            else:
                self.segment = list(ps[1]) + list(ps[0])
        return self.segment

    @staticmethod
    def check_out_side(p, x_min, x_max, y_min, y_max):
        """
        Check if a point is inside a ROI or not
        :param p:
        :param x_min:
        :param x_max:
        :param y_min:
        :param y_max:
        :return:
        """
        x, y = p
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
        else:
            return False

    def draw(self, image, long=True, show_segment=False, line_color=(0, 255, 0)):
        if long:
            cv2.line(image, self.negative_p, self.positive_p, line_color, 2)
        else:
            cv2.line(image, tuple(self.p1), tuple(self.p2), line_color, 2)
        if show_segment:
            cv2.circle(image, (self.segment[0], self.segment[1]), 15, (0, 0, 255), -1)
            cv2.circle(image, (self.segment[2], self.segment[3]), 15, (0, 255, 255), -1)
        return image
