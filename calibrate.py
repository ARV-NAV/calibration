import numpy as np
import cv2 as cv
import glob

CHESSBOARD_ROWS = 9
CHESSBOARD_COLS = 6
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# Lets see what this looks like...
print(criteria)
objp = np.zeros((CHESSBOARD_COLS*CHESSBOARD_ROWS,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_ROWS,0:CHESSBOARD_COLS].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.ppm')
dimsOfImage = 0
# gray = 
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Get dimensions of gray
    dimsOfImage = gray.shape[::-1]

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (CHESSBOARD_ROWS, CHESSBOARD_COLS), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (CHESSBOARD_ROWS, CHESSBOARD_COLS), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)
    else:
        print("No chessboard found :(")
cv.destroyAllWindows() 

# calibrate camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, dimsOfImage, None, None)

print(mtx)
print(dist)
