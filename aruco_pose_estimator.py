import sys

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

#dictionary used to generate the aruco marker originally (found in generateAruco file
arucoDictionaryName = "DICT_ARUCO_ORIGINAL"

#The built in aruco dictionaries (from the aruco module within openCV)
ARUCO_DICT = {
  "DICT_4X4_50": cv.aruco.DICT_4X4_50,
  "DICT_4X4_100": cv.aruco.DICT_4X4_100,
  "DICT_4X4_250": cv.aruco.DICT_4X4_250,
  "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
  "DICT_5X5_50": cv.aruco.DICT_5X5_50,
  "DICT_5X5_100": cv.aruco.DICT_5X5_100,
  "DICT_5X5_250": cv.aruco.DICT_5X5_250,
  "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
  "DICT_6X6_50": cv.aruco.DICT_6X6_50,
  "DICT_6X6_100": cv.aruco.DICT_6X6_100,
  "DICT_6X6_250": cv.aruco.DICT_6X6_250,
  "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
  "DICT_7X7_50": cv.aruco.DICT_7X7_50,
  "DICT_7X7_100": cv.aruco.DICT_7X7_100,
  "DICT_7X7_250": cv.aruco.DICT_7X7_250,
  "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
  "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL #TODO: if this doesn't work, change to the library used to create
                                                    #the aruco
}

#the side length of the printed aruco marker
arucoSideLength = 0.0785 #TODO: add real mezzy for this! Don't cheap out either, make sure to use calipers

#the .yaml file generated from the calibration file holding the matricies for calibration
cameraCalibrationParametersFilename = 'calibration_chessboard.yaml'

def eulerFromQuaternion(x, y, z, w):
    """
    Creates euler angles from a quaternion input
        Note on how quaternions work (bc I am dumb and am trying to figure it out):

            Quaternions take advantage of the concept derived from Eulers Rotation Theorem. This theorem states that all
            rotations, or compositions of rotations, about multiple axes where at least one point remains fixed in 3D
            space are equivelent to one rotation about an axis that runs through the fixed point. This axis is called
            the Euler's Axis, and can be expressed as a vector in whatever coordinate space is convenient.

            A quaternion is a way of expressing this in a convenient and easy to understand way. It is expressed as
            a vector representing the Euler's Axis, and a scalar proportional to the rotation about the Euler's Axis.
            The scalar is equal to cos(theta/2) + sin(theta/2), where theta is the angle about the Euler's Axis
            IN RADIANS. It is preferred over rotation matrices in many cases because it is visually intuitive to
            understand the direction and degree of rotation.

            Quaternions are very similar to the axis-angle representation of rotations, in which the Euler's Rotation
            Theorem is also applied, but Quaternions are easier to apply onto a 3-D point.

    :param x: The x component of the Euler Axis
    :param y: The y component of the Euler Axis
    :param z: The z component of the Euler Axis
    :param w: cos(theta/2) + sin(theta/2), where theta is the rotation about the euler's axis in radians.

    :return: Euler's Angles:
                 rotation about the x axis (rx)
                 rotation about the y axis (ry)
                 rotation about the z axis (rz)
    """

    #projection of the euler's axis onto xy plane represented as coordinates t0,t1 I think?
    t0 = +2.0*(w*x - y*z)
    t1 = +1.0 - 2.0(x*x + y*y)
    #atan2 returns the angle between the positive x-axis and some point (y,x)
    #unilke atan, atan2 considers the sign of x and y
    rx = math.atan2(t0, t1)

    #neato, this is a code expression of a piece-wise function I think. Much more intuitive this way, but is there
    #a way to solve this without a piece-wise?
    t2 = +2.0*(w*y - z*x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    ry = math.asin(t2)

    #follows the same pattern as the calculation for rx. Notice the w*axis of rotation + other two axes multiplied
    #together, then the squaring. Probably some right-hand rule reasoning to what is chosen for what
    t3 = +2.0*(w*z + x*y)
    t4 = +1.0 - 2.0*(y*y + z*z)
    rz = math.atan2(t3, t4)

    return rx, ry, rz #in radians

def main():
    """
    main method of the program. This is where the detecting of the aruco's will happen.
    :return: nothing
    """

    #check that there is a valid aruco marker referenced in the dictionary
        #(not actually checking the image yet, just seeing if we're asking for something dumb)
    if ARUCO_DICT.get(arucoDictionaryName, None) is None:
        print("No markers found")
        sys.exit(0)

    #Retrieve the camera parameters from the .yaml file
        #save the parameters as 'mtx' and 'dst'
    cvFile = cv.FileStorage(cameraCalibrationParametersFilename, cv.FILE_STORAGE_READ)
    mtx = cvFile.getNode('K').mat()
    dst = cvFile.getNode('D').mat()
    cvFile.release()

    #retrieve the aruco dictionary and create the parameters for it (opencv does this with a method)
    print("Detecting markers...")
    thisArucoDictionary = cv.aruco.Dictionary_get(ARUCO_DICT[arucoDictionaryName])
    thisArucoParameters = cv.aruco.DetectorParameters_create()

    #start up the video stream
    cap = cv.VideoCapture(0)

    while(True):
        #run this loop on each frame
        #returns a boolean, then the actual frame
        ret, frame = cap.read()

        #detect the aruco markers in the frame (applying our calibration matrices as well)
        (corners, markerIDs, rejected) = cv.aruco.detectMarkers(
            frame, thisArucoDictionary, parameters=thisArucoParameters, cameraMatrix=mtx, distCoeff=dst
        )

        #if we actually see a marker, run this snippet
        if markerIDs is not None:

            #draw the square around detected markers
            cv.aruco.drawDetectedMarkers(frame, corners, markerIDs, borderColor=(255, 0, 255))
            #NOTE: if this line ^ breaks things, remove borderColor, it defaults to none

            #get the rotation and translation vectors of the aruco
                #opencv actually has a method for this, completley cracked
            rvecs, tvecs, objPoints = cv.aruco.estimatePoseSingleMarkers(
                corners,
                arucoSideLength,
                mtx,
                dst)

            """
            Now we actually print and solve for the pose of the marker. This is provided in the camera frame, with:
                x-axis pointing right
                y-axis pointing down
                z-axis pointing away
            There is not an entirely simple way to adjust this frame, it involves more rotation and translation math, 
            either through futzing about with quaternions or rotation matricies. The whole process would be reliant
            on the camera being in a fixed, known poisition, so no. For now, it is easiest to represent pose relative 
            to the camera.
            """

            #iterate through all of the found markers
            for i, marker_id in enumerate(markerIDs):

                #first, store the translation info
                    #position relative to camera
                transformTranslationX = tvecs[i][0][0]
                transformTranslationY = tvecs[i][0][1]
                transformTranslationZ = tvecs[i][0][2]

                #next, store the rotation camera
                    #again, relative to the camera
                #this numpy method is weird, it pretty much makes an identity matrix of size nxn by default
                rotationMatrix = np.eye(4)
                #NOTE: This function uses Rodrigues' formula to convert from a rotation vector to a rotation matrix
                rotationMatrix[0:3][0:3] - cv.Rodrigues(np.array(rvecs[i][0]))[0]

