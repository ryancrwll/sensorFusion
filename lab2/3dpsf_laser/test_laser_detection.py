#!/usr/bin/env python
import numpy as np
import cv2
import os
import argparse
import glob
import my_library
from utils import list_image_files

def test_laser_detection():
    # Handle input parameters
    parser = argparse.ArgumentParser(description="Check the laser detection method")
    parser.add_argument('--images_dir', '-i', dest='imgs_dir', action='store', type=str, required=True,
                        help="Folder containing the input images")    
    parser.add_argument('--min_red', '-t', dest='min_red', action='store', type=int, default=128,
                        help="Threshold, minimum red value to consider as the laser")
    param = parser.parse_args()

    # List the files in the specified input folder
    files = list_image_files(param.imgs_dir)
    files.sort() # Just to list the files in order
    if not files:
        print('[ERROR] No images found in the specified folder')
        return

    # Run over all the images in the folder and detect the laser
    for file in files:
        print('- Detecting laser in image %s... ' % file)
        
        # Load the image
        img = cv2.imread(file)        
        if img is None:
            print("[WARNING] Unable to load image %s!", file)
            continue            
        
        rimg = img[:, :, 2].copy()
        rimg[np.where(rimg < param.min_red)] = 0
        cv2.namedWindow("Segmented Red Channel", cv2.WINDOW_NORMAL)    
        cv2.imshow("Segmented Red Channel", rimg)
        
        # Detect the laser
        laser_pts = my_library.detect_laser(img, param.min_red)
        #laser_pts = laser_processing.detect_laser_max(img, param.min_red)
        
        # Show the detection on screen
        img_laser = img.copy()
        cv2.namedWindow("Laser Detection", cv2.WINDOW_NORMAL)    
        for pt in laser_pts:
            cv2.drawMarker(img_laser, (int(pt[0]), int(pt[1])), color=(255, 0, 0), markerSize=5, markerType=cv2.MARKER_CROSS, thickness=1)
        cv2.imshow("Laser Detection", img_laser)
        print("    - Showing laser detection, press any key to continue")
        cv2.waitKey()

if __name__ == '__main__':
    test_laser_detection()
