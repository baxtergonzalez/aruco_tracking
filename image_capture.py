'''
This program captures the frames of a video feed and saves them as .jpg's. The specific goal in it's design was to
capture calibration images for aruco pose estimation.
    -by default, takes 25 pictures
'''

import cv2 as cv
import os
import time

#name of the folder/path where all of the photos are stored
folderPath = "C:\\Users\\bgonzalez\\OneDrive - Mustang Plumbing\\Documents\\Calibration_Images\\"

def main():
    #starts up the camera
    cap = cv.VideoCapture(0)
    #creates a folder to store the images if it doesn't exist already
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    #the total number of pictures that will be taken
    totalPics = 25
    #simple iterator
    picNum = 1
    #one secomd delay to allow the camera to compensate/zoom/focus
    time.sleep(1)

    #loop through and capture photos from the feed
    while picNum <= totalPics:
        #grab the current frame from the video reference
        success, img = cap.read(0)
        #save the puicture to the folder and name it appropriately
        cv.imwrite(folderPath + f"\\Image_{picNum}.jpg", img)
        print(f"Image: {picNum} captured")
        #delay 1 second before the next picture is taken
        time.sleep(1)
        #increase iterator by one
        picNum = picNum + 1

if __name__ == '__main__':
    print("Initialized...")
    main()
