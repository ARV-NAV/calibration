import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import calibration_util

def undistortStereo(imgL, imgR):
    # Get distortion matrix and vectors, make new optimal camera mtx for our
    # specific w/h
    mtxL, distL = calibration_util.get_L_mtx_dist()
    mtxR, distR = calibration_util.get_R_mtx_dist()
    h, w = imgL.shape[:2]
    #newcmtxL, roiL = cv.getOptimalNewCameraMatrix(mtxL, distL, (w,h), 1, (w,h))
    #newcmtxR, roiR = cv.getOptimalNewCameraMatrix(mtxR, distR, (w,h), 1, (w,h))

    # undistort L and R
    imgL = cv.undistort(imgL, mtxL, distL, None)
    imgR = cv.undistort(imgR, mtxL, distL, None)
    # crop to the smallest image
    #if roiL[3] < roiR [3]:
    #    roi = roiL
    #else:
    #    roi = roiR

    #x, y, w, h = roi
    #imgL = imgL[y:y+h, x:x+w]
    #imgR = imgR[y:y+h, x:x+w]

    #cv.imshow("L", imgL)
    #cv.waitKey(0)
    #cv.imshow("R", imgR)
    #cv.waitKey(0)

    return imgL, imgR

# Open left and right images as 8 bit grayscale
#imgLargeL = cv.imread('./left/left_02.ppm',0)
#imgLargeR = cv.imread('./right/right_02.ppm',0)
#imgL = cv.imread('imgL.png',0)
#imgR = cv.imread('imgR.png',0)

#Pull frames from camera
capL = cv.VideoCapture(2)
capR = cv.VideoCapture(4)


# SADWindowSize must be odd, be within 5..255 and be not larger than image width or height in function 'compute'...
# See here for info: https://docs.opencv.org/master/d2/d85/classcv_1_1StereoSGBM.html#details
stereo = cv.StereoSGBM_create(
        numDisparities=32,
        blockSize=5,
        P1=600,
        P2=2400,
        uniquenessRatio=8,
        speckleWindowSize=150,
        speckleRange=2)

stereo.setSpeckleRange(100)
stereo.setPreFilterCap(63)
while True:
    capL.grab()
    capR.grab()

    retL, imgLargeL = capL.retrieve()
    retR, imgLargeR = capR.retrieve()

    # print(imgLargeL.shape[0], imgLargeL.shape[1])
    #undistort the images
    #imgLargeL, imgLargeR = undistortStereo(imgLargeL, imgLargeR)

    # Resize the images
    dsize = (int (imgLargeL.shape[1] * 0.4), int (imgLargeL.shape[0] * 0.4))
    imgL = cv.resize(imgLargeL, dsize)
    imgR = cv.resize(imgLargeR, dsize)

    # Instead, crop to the center of the images
    #h, w = imgLargeL.shape[:2]
    #imgL = imgLargeL[int(h/3):int(2*(h/3)), int(w/3):int(2*(w/3))]
    #imgR = imgLargeR[int(h/3):int(2*(h/3)), int(w/3):int(2*(w/3))]


    imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    disparity = stereo.compute(imgL,imgR)
    # print(disparity)

    # Code to save matplotlib image
    # plt.imshow(disparity,'gray')
    # plt.savefig('gray.png')
    # print("Max is", disparity.max())

    # norm_coeff = 255 / disparity.max()
    # A better way is to use the cv.normalize rather than manually normalizing
    norm_image = cv.normalize(disparity, None, alpha = 0, beta = 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

    bothTop = np.hstack((imgL, imgR))

    darkImg = np.zeros((imgL.shape), np.uint8)
    bothBottom = np.hstack((norm_image,darkImg))

    allImgs = np.vstack((bothTop,bothBottom))
    cv.imshow("all",allImgs)

    #cv.imshow("gray", disparity*norm_coeff / 255)
    # Wait untill q is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv.destroyAllWindows()
