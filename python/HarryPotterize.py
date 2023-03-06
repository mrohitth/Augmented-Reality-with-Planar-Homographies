import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts


# Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
from helper import plotMatches
import matplotlib.pyplot as plt

# Q2.2.4

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

def warpImage(opts):
    matches,locs1,locs2 = matchPics(cv_cover, cv_desk, opts)

    a = (cv_cover.shape[1], cv_cover.shape[0])
    hp_cover_new = cv2.resize(hp_cover, a)

    # plotMatches(cv_desk, cv_cover,  matches, locs1, locs2)

    x1 = locs1[matches[:, 0], 0:2]
    x2 = locs2[matches[:, 1], 0:2]

    bestH2to1, inliers = computeH_ransac(x1, x2, opts)

    # hp_cover = cv2.resize(hp_cover, cv_cover.shape[1],cv_cover.shape[0])
    composite_image = compositeH(bestH2to1, hp_cover_new, cv_desk)
    plt.imshow(composite_image)
    plt.show()
    cv2.imwrite('../python/images/warped_image2500_40.jpeg', composite_image)
    # pass



if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


