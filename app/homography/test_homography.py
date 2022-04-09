from pathlib import Path

import cv2

from homography import homography


def test_sample_image(
        img_path='input/nfl-health-and-safety-helmet-assignment/images/57512_000484_Sideline_frame1059.jpg',
        show=False,
        write=False):
    """
    Test line homography with 1 image
    :param img_path:
    :param show:
    :param write:
    :return:
    """
    try:
        img_name = Path(img_path).name
        img = cv2.imread(img_path)
        rotate = False
        M_final_correct, img_final = homography.find_homography_lines(img,
                                                                      rotate=rotate,
                                                                      vis=show,
                                                                      warp=True,
                                                                      write=write,
                                                                      output_name=f'output/visualization/{img_name}')
    except AttributeError as e:
        print(f'Exception in test_sample_image, {e}')
        img_final = np.ones((100, 100, 3))
    return img_final
