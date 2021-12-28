'''
This program calibrates the camera usina a picture of a chessboard
    -Need to have a folder of 10+ imaages of the chessboard taken under different positions and lighting
    conditions
    -Mostly copied/inspired from the post at:
        https://automaticaddison.com/how-to-perform-pose-estimation-using-an-aruco-marker/
'''


import cv2 as cv
import numpy as np
import glob2 as glob
import os

folderPath = "C:\\Users\\bgonzalez\\OneDrive - Mustang Plumbing\\Documents\\Calibration_Images\\"

#chessboard dimensions
squaresNumX = 10
squaresNumY = 7
nX = squaresNumX - 1 #number of interior corners in x dir
nY = squaresNumY - 1 #number of interior corners in y dir
square_size = 0.025 #TODO: add a real mezzy for this! It is the size of each square in meters on chessboard

#Set termination criteria. Stop when we:
    #hit a certain accuracy (0.001)
    #hit a certain number of iterations (30)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#Define real world coordinates for points in 3D coordinate frame
    #object points are (0,0,0) - (5,8,0)
objectPoints3D = np.zeros((nX*nY, 3), np.float32)

#X and Y coordinates
objectPoints3D[:,:2] = np.mgrid[0:nY, 0:nX].T.reshape(-1, 2)
objectPoints3D = objectPoints3D * square_size

#store vectors of 3D points in REAL WORLD SPACE
objectPoints = []

#store vectors of 3D points in CAMERA SPACE
imagePoints = []

def main():
    images = glob.glob(os.path.join(folderPath, '*.jpg'))

    for imageFile in images:
        #load the image
        image = cv.imread(imageFile)
        #convert to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #find corners
        success, corners = cv.findChessboardCorners(gray, (nY,nX), None)

        if success:
            print("success")
            #append object points
            objectPoints.append(objectPoints3D)
            #find more exact corner pixels
            corners2 = cv.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            #append image points
            imagePoints.append(corners2)
            #draw the corners
            cv.drawChessboardCorners(image,(nX,nY),corners2, success)
            #display the image
                #used for testing
            cv.imshow("Image", image)
            #display window for a short period
            cv.waitKey(1000)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints,
                                                          imagePoints,
                                                          gray.shape[::-1],
                                                          None,
                                                          None)
        #save parameters to a file
        cvFile = cv.FileStorage(folderPath+'calibration_chessboard.yaml', cv.FILE_STORAGE_WRITE)
        cvFile.write('K', mtx)
        cvFile.write('D', dist)
        cvFile.release()

        print("Camera Matrix: ")
        print(mtx)

        print("\n Distortion Coefficient:")
        print(dist)

        cv.destroyAllWindows()

if __name__ == '__main__':
    print(__doc__)
    main()

