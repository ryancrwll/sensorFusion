#!/usr/bin/env python
import numpy as np
import cv2
import os
import argparse
import glob
import warnings

def list_image_files(images_dir):
    ext = ['png', 'jpg', 'gif'] 
    files = []
    [files.extend(glob.glob(images_dir + '/*.' + e)) for e in ext]
    return files

def main():
    parser = argparse.ArgumentParser(description="Stereo camera calibration")
    parser.add_argument('--row_corners', dest='pattern_row_corners', action='store', type=int, required=True,
                        help='Number of internal corners in a row of the chessboard pattern')
    parser.add_argument('--col_corners', dest='pattern_col_corners', action='store', type=int, required=True,
                        help='Number of internal corners in a column of the chessboard pattern')
    parser.add_argument('--squares_size', dest='square_size', action='store', type=float, required=True,
                        help='Side length of the squares of the chessboard pattern')
    parser.add_argument('--left_calib_file', '-l', dest='left_calib_file', action='store', type=str, required=True,
                        help='Calibration file for the left camera')
    parser.add_argument('--right_calib_file', '-r', dest='right_calib_file', action='store', type=str, required=True,
                        help='Calibration file for the right camera')
    parser.add_argument('--left_images_dir', '-li', dest='left_imgs_dir', action='store', type=str, required=True,                        
                        help='Folder containing the input images of the left camera')
    parser.add_argument('--right_images_dir', '-ri', dest='right_imgs_dir', action='store', type=str, required=True,
                        help='Folder containing the input images of the right camera')    
    parser.add_argument('--out_file', '-o', dest='out_calib_file', action='store', type=str, required=True,
                        help='Output calibration file containing the extrinsic parameters of the stereo pair')    
    parser.add_argument('--debug_dir', dest='debug_dir', action='store', type=str, default="",
                        help='If not empty, it will generate the specified directory and debug data will be stored there')
    param = parser.parse_args()

    # Create the output debug folder, if needed
    if param.debug_dir:
        left_debug_dir = param.debug_dir + "/left"
        right_debug_dir = param.debug_dir + "/right"
        if not os.path.isdir(param.debug_dir):
            os.makedirs(param.debug_dir)
        if not os.path.isdir(left_debug_dir):
            os.makedirs(left_debug_dir)
        if not os.path.isdir(right_debug_dir):
            os.makedirs(right_debug_dir)

    # Create the corner points of the chessboard of the calibration pattern
    pattern_size = (param.pattern_row_corners, param.pattern_col_corners)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= param.square_size

    # Load the monocular calibrations
    left_cv_file = cv2.FileStorage(param.left_calib_file, cv2.FILE_STORAGE_READ)
    left_K = left_cv_file.getNode("camera_matrix").mat()
    left_dist = left_cv_file.getNode("distortion_coefficients").mat()
    right_cv_file = cv2.FileStorage(param.right_calib_file, cv2.FILE_STORAGE_READ)
    right_K = right_cv_file.getNode("camera_matrix").mat()
    right_dist = right_cv_file.getNode("distortion_coefficients").mat()
    
    obj_points = []
    left_img_points = []
    right_img_points = []
    h, w = 0, 0
    num_images_used = 0

    # Read the image files
    left_files = list_image_files(param.left_imgs_dir)
    left_files.sort() # Sort the file names, they need to coincide for left/right
    right_files = list_image_files(param.right_imgs_dir)
    right_files.sort()
    if len(left_files) != len(right_files):
        print('[ERROR] The number of files is different. There should be the same number of images for the left/right sets.')
        return
    if not left_files:
        print('[ERROR] No images found in the specified folders')
        return

    pairs_files = zip(left_files, right_files)
    pair_ind = 0
    for left_file, right_file in pairs_files:
        print('- Detecting corners on image pair %d: ' % pair_ind)
        pair_ind = pair_ind + 1

        # Load the images of the pair
        left_img = cv2.imread(left_file, 0)
        if left_img is None:
            print('[WARNING] Unable to load left image %s! Continuing with the next pair...' % left_file)
            continue
        right_img = cv2.imread(right_file, 0)
        if right_img is None:
            print('[WARNING] Unable to load right image %s! Continuing with the next pair...' % right_file)
            continue

        # Try to detect the corners in the left image
        h, w = left_img.shape[:2]
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)

        print('    - Finding chessboard on left image %s...' % left_file, end='')
        left_found, left_corners = cv2.findChessboardCorners(left_img, pattern_size)
        if left_found:
            print('Chessboard detected')

            # If found, refine the corners to subpixel accuracy
            cv2.cornerSubPix(left_img, left_corners, (5, 5), (-1, -1), term)

            if param.debug_dir:
                vis = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis, pattern_size, left_corners, left_found)
                name = os.path.splitext(os.path.basename(left_file))[0]
                outfile = left_debug_dir + "/" + name + '_detected_chessboard.png'
                cv2.imwrite(outfile, vis)
        else:
            print('Chessboard NOT detected!')
            continue

        print('    - Finding chessboard on right image %s...' % right_file, end='')
        right_found, right_corners = cv2.findChessboardCorners(right_img, pattern_size)
        if right_found:
            print('Chessboard detected')

            # If found, refine the corners to subpixel accuracy
            cv2.cornerSubPix(right_img, right_corners, (5, 5), (-1, -1), term)

            if param.debug_dir:
                vis = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis, pattern_size, right_corners, right_found)
                name = os.path.splitext(os.path.basename(right_file))[0]
                outfile = right_debug_dir + "/" + name + '_detected_chessboard.png'
                cv2.imwrite(outfile, vis)
        else:
            print('Chessboard NOT detected!')
            continue

        # Add the detected corners to the list
        left_img_points.append(left_corners.reshape(-1, 2))
        right_img_points.append(right_corners.reshape(-1, 2))
        obj_points.append(pattern_points)

        num_images_used = num_images_used + 1

    # Compute the camera parameters given all the samples collected
    print('- Found pattern in %d/%d image pairs' % (num_images_used, len(left_files)))
    print('- Calibrating...')
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    flags = (cv2.CALIB_FIX_INTRINSIC)
    error, left_camera_calib, left_dist_coefs, right_camera_calib, right_dist_coefs, R, T, E, F = cv2.stereoCalibrate(obj_points, left_img_points, right_img_points, left_K, left_dist, right_K, right_dist, (w, h), criteria=criteria, flags=flags)

    # Print some results on screen
    print('- Stereo calibration done:')
    print('    - Error: ', error)
    print('    - R:\n', R)
    print('    - T:\n', T)

    # Save the parameters to a file
    cv_file = cv2.FileStorage(param.out_calib_file, cv2.FILE_STORAGE_WRITE)
    cv_file.write("image_width", w)
    cv_file.write("image_height", h)
    cv_file.write("stereo_rotation", R)
    cv_file.write("stereo_translation", T)
    cv_file.write("left_camera_matrix", left_camera_calib)
    cv_file.write("left_distortion_coefficients", left_dist_coefs)
    cv_file.write("right_camera_matrix", right_camera_calib)
    cv_file.write("right_distortion_coefficients", right_dist_coefs)
    cv_file.write("essential_matrix", E)
    cv_file.write("fundamental_matrix", F)
    cv_file.release()


if __name__ == '__main__':
    main()
