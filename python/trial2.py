import cv2

import matplotlib.pyplot as plt
import numpy as np
from planarH import computeH_ransac
from opts import get_opts
from matchPics import matchPics

opts = get_opts()

def blending(img1,img2):
    matches, locs1, locs2 = matchPics(img1, img2, opts)
    x1 = locs1[matches[:, 0], 0:2]
    x2 = locs2[matches[:, 1], 0:2]
    H = computeH_ransac(x1, x2, opts)
    height_img1 = x1.shape[0]
    width_img1 = x1.shape[1]
    width_img2 = x2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 +width_img2

    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    mask1 = create_mask(x1,x2,version='left_image')
    panorama1[0:x1.shape[0], 0:x2.shape[1], :] = x1
    panorama1 *= mask1
    mask2 = create_mask(img1,img2,version='right_image')
    panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
    result=panorama1+panorama2

    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    return final_result


def create_mask(img1,img2,version):
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 +width_img2
    offset = int(800 / 2)
    barrier = img1.shape[1] - int(800 / 2)
    mask = np.zeros((height_panorama, width_panorama))
    if version== 'left_image':
        mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
        mask[:, :barrier - offset] = 1
    else:
        mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
        mask[:, barrier + offset:] = 1
    return cv2.merge([mask, mask, mask])

def main():
    img1 = cv2.imread('../data/pano_left.jpg')
    img2 = cv2.imread('../data/pano_right.jpg')

    final = blending(img1,img2)
    cv2.imwrite('panoramaaaa.jpg', final)

if __name__ == '__main__':
    main()