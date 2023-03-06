# import cv2
# import numpy as np
# from scipy.ndimage.morphology import distance_transform_edt
# from planarH import computeH
# from opts import get_opts
# from matchPics import matchPics
# from planarH import computeH_ransac
# import os


# def imageStitching(im1, im2, H2to1):
#     '''
#     Returns a panorama of im1 and im2 using the given homography matrix. 
#     Warps img2 into img1 reference frame using the provided warpH() function
#     INPUT
#         im1 and im2 - two images for stitching
#         H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
#                  equation
#     OUTPUT
#         Blends img1 and warped img2 and outputs the panorama image
#     '''
    
#     pano_im_W = im1.shape[1] + im2.shape[1]
#     pano_im_H = im2.shape[0]
#     im1_H, im1_W = im1.shape[:2]
#     pano_im = cv2.warpPerspective(im2, H2to1, (pano_im_W, pano_im_H))
#     cv2.imwrite('warp.jpg',pano_im)
#     for x in range(im1_W):
#         for y in range(im1_H):
#             '''
#             if pano_im[y,x].sum() < warp_im1[y,x].sum():
#                 pano_im[y,x] = warp_mask1[y,x] * warp_im1[y,x] + (1-warp_mask1[y,x]) * warp_im2[y,x]
#             '''
#             if pano_im[y,x].sum() < im1[y,x].sum():
#                 pano_im[y,x] = im1[y,x]
#     #pano_im[0:im1.shape[0], 0:im1.shape[1]] = im1

#     #mask = np.zeros((im1.shape[0], im1.shape[1]))
#     '''
#     mask = np.zeros((6, 6))
#     mask[0,:] = 1
#     mask[-1,:] = 1
#     mask[:,0] = 1
#     mask[:,-1] = 1
#     mask = distance_transform_edt(1-mask)
#     mask = mask/(mask.max(0) + 0.00001)
 
#     print(mask)
#     '''
#     #cv2.imwrite('mask.jpg', mask)

#     cv2.imwrite('pano_im.jpg',pano_im)

#     return pano_im

# opts = get_opts()

# im1 = cv2.imread('../data/grail00.jpg')
# im2 = cv2.imread('../data/grail01.jpg')
# m, locs1, locs2 = matchPics(im1, im2, opts)
# x1 = locs1[m[:, 0], 0:2]
# x2 = locs2[m[:, 1], 0:2]
    
# H2to1, inliers = computeH_ransac(x1, x2, opts)
# # H2to1 = computeH(im1, im2)


# pano_im = imageStitching(im1, im2, H2to1)

####################################################################


# import nbimporter
import cv2

import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage.morphology import distance_transform_edt

# from q2 import briefLite,briefMatch,plotMatches
# from q3 import computeH_ransac
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac
from opts import get_opts

opts = get_opts()

def Panorama(img1, img2, H2to1):
    # print(H2to1)
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 +width_img2

    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    mask1 = create_mask(img1,img2,version='left_image')
    panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
    panorama1 *= mask1
    mask2 = create_mask(img1,img2,version='right_image')
    panorama2 = cv2.warpPerspective(img2, H2to1, (width_panorama, height_panorama))*mask2
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
    offset = int(800/ 2)
    barrier = img1.shape[1] - int(800 / 2)
    mask = np.zeros((height_panorama, width_panorama))
    if version== 'left_image':
        mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
        mask[:, :barrier - offset] = 1
    else:
        mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
        mask[:, barrier + offset:] = 1
    return cv2.merge([mask, mask, mask])   

    # def blendmask(im):
    #     mask = np.ones((im.shape[0], im.shape[1]))
    #     mask[0,:] = 0
    #     mask[im.shape[0]-1,:] = 0
    #     mask[:, 0] = 0
    #     mask[:, im.shape[1]-1] = 0
    #     mask = distance_transform_edt( mask)
    #     mask = mask / np.max(mask)
    #     return mask
    
    # h1, w1, _ = im1.shape
    # h2, w2, _ = im2.shape
    # lefttop = np.array([0,0,1]).T
    # righttop = np.array([w2-1,0,1]).T
    # rightbot = np.array([w2-1,h2-1,1]).T
    # leftbot = np.array([0,h2-1,1]).T
    
    # proj_lt = H2to1@lefttop
    # proj_lt /=proj_lt[2]
    
    # proj_rt = H2to1@righttop
    # proj_rt /=proj_rt[2]
    
    # proj_rb = H2to1@rightbot
    # proj_rb /=proj_rb[2]
    
    # proj_lb = H2to1@leftbot
    # proj_lb /=proj_lb[2]
   
    
    # tx=0
    # ty=0
    
    # tx = int(max((-proj_lt[0], -proj_lb[0], 0)))
    # ty = int(max((-proj_lt[1], -proj_rt[1], 0)))
    
    # W = max(proj_rb[0], proj_rt[0]).astype(int) + tx
    # H = max(proj_lb[1], proj_rb[1]).astype(int) + ty
    
    # M = np.array([[1, 0, tx], [0 , 1, ty], [0, 0, 1]]).astype(float)
    
    # img_wr = cv2.warpPerspective(im2, M @ H2to1 , (W, H))
    # img_wr = img_wr/255 
   
    # I = np.identity(3)
    
    # img_wl = cv2.warpPerspective(im1, M @ I , (W, H))
    # img_wl = img_wl/255
    
    
    # mask_r = blendmask(im2)
    # wmask_r = cv2.warpPerspective(mask_r, M @ H2to1 , (W, H))
    # mask_l = blendmask(im1)
    # wmask_l = cv2.warpPerspective(mask_l, M @ I , (W, H))

    # sum_mask = wmask_r + wmask_l
    
    
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     wmask_r = wmask_r / sum_mask
    #     wmask_r[np.isnan(wmask_r)] = 0
    #     wmask_l = wmask_l / sum_mask
    #     wmask_l[np.isnan(wmask_l)] = 0

        
        
    # wmask_l = np.expand_dims(wmask_l, axis = 2)
    # wmask_l = np.tile(wmask_l, (1,1,3))
    
    # wmask_r = np.expand_dims(wmask_r, axis = 2)
    # wmask_r = np.tile(wmask_r, (1,1,3))
    
    # img_pano = img_wr * wmask_r + img_wl * wmask_l
    # img_pano[np.isnan(img_pano)] = 0

    
    # return img_pano

im1 = cv2.imread('../data/pano_left.jpg')
im2 = cv2.imread('../data/pano_right.jpg')
    
matches, locs1, locs2 = matchPics(im1, im2, opts)
x1 = locs1[matches[:, 0], 0:2]
x2 = locs2[matches[:, 1], 0:2]

bestH2to1, inliers = computeH_ransac(x1, x2, opts)
bestH2to1 = [[ 7.08752959e-01, -4.12962910e-03,  3.61145407e+02],
 [-9.24058323e-02,  8.92643796e-01,  3.26116538e+01],
 [-2.02935641e-04,  1.09027873e-05,  1.00000000e+00]]
img_pano2 = Panorama(im1, im2, bestH2to1)
cv2.imwrite('panoramaaaaa.jpg', img_pano2)
# plt.figure(figsize = (20,10))

# if(img_pano2 is not None):
#     img_pano2 = cv2.cvtColor(np.float32(img_pano2), cv2.COLOR_BGR2RGB)
# plt.imshow(img_pano2)
# plt.show()

