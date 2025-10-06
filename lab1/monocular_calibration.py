#!/usr/bin/env python
import numpy as np
import cv2
import os
import argparse
import glob

def list_image_files(images_dir):
    ext = ['png', 'jpg', 'gif'] 
    files = []
    [files.extend(glob.glob(images_dir + '/*.' + e)) for e in ext]
    return files

def monocular_calib():
    # Handle input parameters
    parser = argparse.ArgumentParser(description="Monocular camera calibration")
    parser.add_argument('--row_corners', dest='pattern_row_corners', action='store', type=int, required=True,
                        help='Number of internal corners in a row of the chessboard pattern')
    parser.add_argument('--col_corners', dest='pattern_col_corners', action='store', type=int, required=True,
                        help='Number of internal corners in a column of the chessboard pattern')
    parser.add_argument('--squares_size', dest='square_size', action='store', type=float, required=True,
                        help='Side length of the squares of the chessboard pattern')
    parser.add_argument('--images_dir', '-i', dest='imgs_dir', action='store', type=str, default="./images", required=True,
                        help='Folder containing the input images')
    parser.add_argument('--out_calib', '-o', dest='out_calib_file', action='store', type=str, required=True,
                        help='Output calibration file containing the intrinsic parameters of the camera')
    parser.add_argument('--debug_dir', dest='debug_dir', action='store', type=str, default="",
                        help='If not empty, it will generate the specified directory and debug data will be stored there')
    param = parser.parse_args()

    # Create the output debug folder, if needed
    if param.debug_dir and not os.path.isdir(param.debug_dir):
        os.makedirs(param.debug_dir)

    # Create the corner points of the chessboard of the calibration pattern
    pattern_size = (param.pattern_row_corners, param.pattern_col_corners)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= param.square_size

    # List the files in the specified input folder
    files = list_image_files(param.imgs_dir)
    files.sort() # Just to list the files in order
    if not files:
        print('[ERROR] No images found in the specified folder')
        return

    # Run over all the images in the folder and try to detect the chessboard corners on each image
    obj_points = []
    img_points = []
    h, w = 0, 0
    images_used = []
    for file in files:
        print('- Detecting corners on image %s... ' % file, end='')

        # Load the image in gray scale
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("[WARNING] Unable to load image %s!", file)
            continue

        # Try to find the chessboard on the current image
        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            print('Chessboard detected')

            # If found, refine the corners to subpixel accuracy
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

            images_used.append(file)

            if param.debug_dir:
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)
                name = os.path.splitext(os.path.basename(file))[0]
                outfile = param.debug_dir + "/" + name + '_detected_chessboard.png'
                cv2.imwrite(outfile, vis)
        else:
            print('Chessboard NOT detected!')
            continue

        # Add the detected corners and a set of reference 3D points to the lists
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

    # Compute the camera parameters given all the samples collected
    print('- Found pattern in %d/%d images' % (len(images_used), len(files)))
    print('- Calibrating...')
    error, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    # Print some results on screen
    print('- Calibration done:')
    print("    - Mean reprojection error:", error)
    print("    - Camera matrix:\n", K)
    print("    - Distortion parameters: ", dist.ravel())

    # Save the parameters to a file
    cv_file = cv2.FileStorage(param.out_calib_file, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", K)
    cv_file.write("distortion_coefficients", dist)
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
            reprojected_pts, _ = cv2.projectPoints(pattern_points, rvecs[ind], tvecs[ind], K, dist)
            reprojected_pts = reprojected_pts.squeeze() # Remove singleton dimension to get an Nx2 matrix (cv2.projectPoints returns an Nx1x2 matrix, which is not very intuitive...)
            detected_pts = img_points[ind]
            # Show both detected and reprojected points in the image with different markers
            for pt in detected_pts:
                cv2.drawMarker(vis, (int(pt[0]), int(pt[1])), color=(255, 0, 0), markerSize=5, markerType=cv2.MARKER_CROSS, thickness=1)
            for pt in reprojected_pts:
                cv2.drawMarker(vis, (int(pt[0]), int(pt[1])), color=(0, 255, 0), markerSize=5, markerType=cv2.MARKER_TILTED_CROSS, thickness=1)
            cv2.imwrite(outfile, vis)


if __name__ == '__main__':
    monocular_calib()
