import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from helper import plotMatches

#Q2.1.6

def rotTest(opts):

    #Read the image and convert to grayscale, if necessary
    opts = get_opts()
    image = cv2.imread('../data/cv_cover.jpg')
    hist_match = list()
    
    for i in range(36):

        #Rotate Image
        img_rotate = rotate(image, 10*(i+1))
        matches, locs1, locs2 = matchPics(image, img_rotate, opts)

       
    
        #Update histogram
        plotMatches(image, img_rotate, matches, locs1, locs2)
        hist_match.append(len(matches))

    
    print(hist_match)
    # pass 
 

    #Display histogram
    plt.hist(hist_match)
    plt.ylabel("Number of Matches")
    plt.show()

if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
