#!/usr/bin/env python
import numpy as np
import cv2
import os
import argparse
import glob
from utils import list_image_files

def monocular_calib_charuco():
    # Handle input parameters
    parser = argparse.ArgumentParser(description="Monocular camera calibration using a ChAruco pattern")
    parser.add_argument('--images_dir', '-i', dest='imgs_dir', action='store', type=str, default="./images", required=True,
                        help='Folder containing the input images')
    parser.add_argument('--squares_x', dest='squares_x', action='store', type=int, required=True,
                        help='Number of squares in the X direction of the Charuco pattern')
    parser.add_argument('--squares_y', dest='squares_y', action='store', type=int, required=True,
                        help='Number of squares in the Y direction of the Charuco pattern')
    parser.add_argument('--square_size', dest='square_size', action='store', type=float, required=True,
                        help='Side length of a square of the pattern')
    parser.add_argument('--marker_size', dest='marker_size', action='store', type=float, required=True,
                        help='Side length of an Aruco marker of the pattern')
    parser.add_argument('--dictionary_id', dest='dict_id', action='store', type=int, default=cv2.aruco.DICT_4X4_100,
                        help='Aruco markers'' Dictionary ID')    
    parser.add_argument('--out_file', '-o', dest='out_calib_file', action='store', type=str, default="calib.yaml",
                        help='Output calibration file containing the intrinsic parameters of the camera')
    parser.add_argument('--debug_dir', dest='debug_dir', action='store', type=str, default="",
                        help='If not empty, it will generate the specified directory and debug data will be stored there')
    param = parser.parse_args()

    # Create the output debug folder, if needed
    if param.debug_dir and not os.path.isdir(param.debug_dir):
        os.makedirs(param.debug_dir)

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
    all_corners = []
    all_corner_ids = []
    images_used = []
    for file in files:
        print('- Detecting corners on image %s... ' % file)

        # Load the image in grayscale
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)        
        if img is None:
            print("[WARNING] Unable to load image %s!", file)
            continue            

        # Detect the Aruco markers
        marker_corners, marker_ids, rejected_markers = cv2.aruco.detectMarkers(img, dictionary)

        if len(marker_corners) > 0:
            # Interpolate the corners near the detected markers
            num_detected, corners, corner_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board)
            if num_detected > max(param.squares_x, param.squares_y): # Special check: if only a row/column of the corners is detected, all points will lie in a line, and so the calibration process will fail with an error (pnp cannot be estimated from collinear points)
                all_corners.append(corners)
                all_corner_ids.append(corner_ids)
                images_used.append(file)

                if param.debug_dir:
                    # Draw the detected markers and corners
                    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
                    for corner in corners:            
                        cv2.drawMarker(vis, (int(corner[0, 0]), int(corner[0, 1])), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 5, 1)
                    name = os.path.splitext(os.path.basename(file))[0]
                    outfile = os.path.join(param.debug_dir, name + '_detected_charuco_board.png')
                    cv2.imwrite(outfile, vis)

    # Get the size of an image in the dataset (assumed all to be the same size!)
    h, w = img.shape[:2]

    # Compute the camera parameters given all the samples collected
    print('- Found pattern in %d/%d images' % (len(images_used), len(files)))
    print('- Calibrating...')
    error, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_corners, all_corner_ids, board, (w, h), None, None)

    # Print some results on screen
    print('- Calibration done:')
    print("    - Mean reprojection error:", error)
    print("    - Camera matrix:\n", K)
    print("    - Distortion parameters: ", dist.ravel())

    # Save the parameters to a file
    cv_file = cv2.FileStorage(param.out_calib_file, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", K)
    cv_file.write("distortion_coefficients", dist)
    cv_file.write("image_width", w)
    cv_file.write("image_height", h)
    cv_file.write("mean_error", error)
    cv_file.release() 

    if param.debug_dir:
        print('- Generating reprojection images...')
        # Compute the reprojection images
        for ind in range(len(images_used)):
            file = images_used[ind]
            name = os.path.splitext(os.path.basename(file))[0]
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            outfile = param.debug_dir + "/" + name + '_detection(+)_vs_reprojection(x).png'
            reprojected_pts, _ = cv2.projectPoints(board.getChessboardCorners(), rvecs[ind], tvecs[ind], K, dist)
            reprojected_pts = reprojected_pts.squeeze() # Remove singleton dimension to get an Nx2 matrix (cv2.projectPoints returns an Nx1x2 matrix, which is not very intuitive...)
            detected_pts = all_corners[ind]            
            detected_pts = detected_pts.squeeze() # Remove singleton dimension also here
            # Show both detected and reprojected points in the image with different markers
            for pt in detected_pts:            
                cv2.drawMarker(vis, (int(pt[0]), int(pt[1])), color=(255, 0, 0), markerSize=5, markerType=cv2.MARKER_CROSS, thickness=1)
            for pt in reprojected_pts:
                cv2.drawMarker(vis, (int(pt[0]), int(pt[1])), color=(0, 255, 0), markerSize=5, markerType=cv2.MARKER_TILTED_CROSS, thickness=1)
            cv2.imwrite(outfile, vis)

if __name__ == '__main__':
    monocular_calib_charuco()
