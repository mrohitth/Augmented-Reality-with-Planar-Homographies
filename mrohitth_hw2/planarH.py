import numpy as np
import cv2


def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points

    hmm = []
    for i in range(x1.shape[0]):
        hmm.append([-x2[i, 0], -x2[i, 1], -1, 0, 0, 0, x1[i, 0]*x2[i, 0], x1[i, 0]*x2[i, 1], x1[i, 0]])
        hmm.append([0, 0, 0, -x2[i, 0], -x2[i, 1], -1, x1[i, 1]*x2[i, 0], x1[i, 1]*x2[i, 1], x1[i, 1]])

    u, s, v = np.linalg.svd(np.asarray(hmm))
    H2to1 = v[-1, :].reshape(3,3)
    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points
    x1_centre = [np.mean(x1[:,0]), np.mean(x1[:,1])]
    x2_centre = [np.mean(x2[:,0]), np.mean(x2[:,1])]

    #Shift the origin of the points to the centroid
    p1, p2 = []
    for i in range(x1.shape[0]):
        p1[i] = np.sqrt((x1[i,0] - x1_centre[0])**2 + (x1[i,1] - x1_centre[1])**2)
        p2[i] = np.sqrt((x2[i,0] - x2_centre[0])**2 + (x2[i,1] - x2_centre[1])**2)

    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_norm = np.sqrt(2)/(np.amax(p1))
    x2_norm = np.sqrt(2)/(np.amax(p2))



    mat1, mat2, mat3, mat4 = np.eye(3)
    for i in range(0, 2):
        mat1[i, i] = x1_norm
        mat2[i, i] = x2_norm

        mat3[i, 2] = -x1_centre[i]
        mat4[i, 2] = -x2_centre[i]

    #Similarity transform 1
    T1 = mat1@mat3

    #Similarity transform 2
    T2 = mat2@mat4

    #Compute homography
    x1_homography = np.vstack((x1.T, np.ones((x1.shape[0]))))
    x2_homography = np.vstack((x2.T, np.ones((x2.shape[0]))))

    xx = T1@x1_homography
    xx = xx/xx[2, :]
    xx = xx.T[:, 0:2]

    xy = T2@x2_homography
    xy = xy/xy[2, :]
    xy = xy.T[:, 0:2]

    H = computeH(xx, xy)    

    #Denormalization
    H1 = np.dot(np.linalg.inv(T1), H)
    H2to1 = np.dot(H1, T2)
    
    return H2to1




def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
    max_inliers = -1
    m1 = np.hstack((locs1, np.ones((locs1.shape[0], 1))))
    m2 = np.hstack((locs2, np.ones((locs2.shape[0], 1))))
    

    for i in range(max_iters):
        index_r = np.random.randint(locs1.shape[0], size=4)
        p1 = locs1[index_r]
        p2 = locs2[index_r]

        H = computeH(p1, p2)

        m = np.matmul(H, m2.T)
        m = m.T

        d1 = np.expand_dims(m[:, -1], axis=1)
        d2 = ((m/d1) - m1)
        d2 = np.linalg.norm(d2, axis = 1)
        inlier = np.where(d2<inlier_tol, 1, 0)
        if(np.sum(inlier) > max_inliers):
            max_inliers = np.sum(inlier)
            inliers = inlier
        index_x = np.where(inliers == 1)
        bestH2to1 = computeH(locs1[index_x[0], :], locs2[index_x[0], :])    

    return bestH2to1, inliers



def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
     
    mask=np.ones(template.shape)
    template=cv2.transpose(template)
    mask=cv2.transpose(mask)

    #Create mask of same size as template
    mask1 = cv2.warpPerspective(mask, np.linalg.inv(H2to1), (img.shape[0],  img.shape[1]))
    
    #Warp mask by appropriate homography
    warp_mask = cv2.transpose(mask1)

    #Warp template by appropriate homography
    warp_template = cv2.warpPerspective(template, np.linalg.inv(H2to1), (img.shape[1], img.shape[1]))  #check and confirm
    
    #Use mask to combine the warped template and the image
    template = cv2.transpose(warp_template)
    img[np.nonzero(warp_mask)] = template[np.nonzero(warp_mask)]
    composite_img = img

    return composite_img


