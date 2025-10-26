#!/usr/bin/env python
import numpy as np
import cv2
import os
import argparse
import glob
import my_library as lib
from utils import list_image_files

def test_camera_pose_charuco():
    # Handle input parameters
    parser = argparse.ArgumentParser(description="Tests the camera pose estimation from a ChAruco pattern")
    parser.add_argument('--images_dir', '-i', dest='imgs_dir', action='store', type=str, required=True,
                        help="Folder containing the input images")
    parser.add_argument('--squares_x', dest='squares_x', action='store', type=int, required=True,
                        help='Number of squares in the X direction of the Charuco pattern')
    parser.add_argument('--squares_y', dest='squares_y', action='store', type=int, required=True,
                        help='Number of squares in the Y direction of the Charuco pattern')
    parser.add_argument('--square_size', dest='square_size', action='store', type=float, required=True,
                        help='Side length of a square of the pattern')
    parser.add_argument('--marker_size', dest='marker_size', action='store', type=float, required=True,
                        help='Side length of an Aruco marker of the pattern')
    parser.add_argument('--dictionary_id', dest='dict_id', action='store', type=int, default=cv2.aruco.DICT_4X4_50,
                        help='Aruco markers'' Dictionary ID')
    parser.add_argument('--calib_file', '-c', dest='calib_file', type=str, action='store', required=True,
                        help='The camera calibration file')
    param = parser.parse_args()

    # Load the camera calibration
    cv_file = cv2.FileStorage(param.calib_file, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("camera_matrix").mat()
    dist_coeffs = cv_file.getNode("distortion_coefficients").mat()

    # Create the charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(param.dict_id)
    board = cv2.aruco.CharucoBoard([param.squares_x, param.squares_y], param.square_size, param.marker_size, dictionary)    

    # List the files in the specified input folder
    files = list_image_files(param.imgs_dir)
    files.sort() # Just to list the files in order
    if not files:
        print('[ERROR] No images found in the specified folder')
        return

    # Run over all the images in the folder and try to detect the chessboard corners on each image
    for file in files:
        print('- Detecting corners on image %s... ' % file)
        
        # Load the image in grayscale
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)        
        if img is None:
            print("[WARNING] Unable to load image %s!", file)
            continue            
        
        # Image used for visualization (a copy of the input image, but with 3 channels to be able to draw the markers in color)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # pose_computed, c_Tf_w = lib.charuco_pose_estimation(img, board, dictionary, camera_matrix, dist_coeffs)
        # rvec, _ = cv2.Rodrigues(c_Tf_w[0:3, 0:3])
        # tvec = c_Tf_w[0:3, 3].squeeze()

        # if pose_computed == True:
        #     cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, 0.1)  # axis length 100 can be changed according to your requirement
        #     cv2.namedWindow("Charuco detection", cv2.WINDOW_NORMAL)
        #     cv2.imshow("Charuco detection", vis)
        #     print('    - Showing pose estimation for image ' + str(os.path.basename(file)) + ', press any key to continue...')
        #     cv2.waitKey()
        # else:
        #     print('Pattern not detected or pose not correctly estimated')

        pose_computed, c_Tf_w = lib.charuco_pose_estimation(img, board, dictionary, camera_matrix, dist_coeffs)

        if pose_computed and c_Tf_w is not None:
            rvec, _ = cv2.Rodrigues(c_Tf_w[0:3, 0:3])
            tvec = c_Tf_w[0:3, 3].squeeze()

            cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            cv2.namedWindow("Charuco detection", cv2.WINDOW_NORMAL)
            cv2.imshow("Charuco detection", vis)
            print('    - Showing pose estimation for image ' + str(os.path.basename(file)) + ', press any key to continue...')
            cv2.waitKey()
        else:
            print('Pattern not detected or pose not correctly estimated')


if __name__ == '__main__':
    test_camera_pose_charuco()