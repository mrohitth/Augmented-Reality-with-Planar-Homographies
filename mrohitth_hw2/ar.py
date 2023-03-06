import numpy as np
import cv2
from helper import loadVid
from matchPics import matchPics
from PIL import Image


#Import necessary functions
from opts import get_opts
from planarH import computeH
from planarH import computeH_ransac
from planarH import compositeH

#Write script for Q3.1
opts = get_opts()

book = loadVid('../data/book.mov')
source = loadVid('../data/ar_source.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')


def video(cv_cover, frame, arr, opts):
    m, locs1, locs2 = matchPics(cv_cover, frame, opts)
    x1 = locs1[m[:, 0], 0:2]
    x2 = locs2[m[:, 1], 0:2]
    
    H2to1, inliers = computeH_ransac(x1, x2, opts)
    arr = arr[45:310,:,:]
    cover_width = cv_cover.shape[1]
    width = int(arr.shape[1]/arr.shape[0]) * cv_cover.shape[0]

    r_ar = cv2.resize(arr, (width,cv_cover.shape[0]), interpolation = cv2.INTER_AREA)
    h, w, d = r_ar.shape
    cropped_ar = r_ar[:, int(w/2) - int(cover_width/2) : int(w/2) +  int(cover_width/2), :]
    
    r = compositeH(H2to1, cropped_ar, frame)
    
    return r

a, b, c = book[1].shape
out = cv2.VideoWriter('arrr.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25, (b, a))

for i in range(source.shape[0]):
    frame = book[i]
    ar = source[i]
    print(i)
    final_vid = video(cv_cover, frame, ar, opts)
    out.write(final_vid)

cv2.destroyAllWindows()
out.release()

#######################################################################

