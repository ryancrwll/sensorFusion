#!/usr/bin/env python
import argparse
import os
import cv2
import numpy as np
import glob


def list_image_files(images_dir):
    ext = ['png', 'jpg', 'gif'] 
    files = []
    [files.extend(glob.glob(images_dir + '/*.' + e)) for e in ext]
    return files

# Main function
def main():
    # Parameters
    parser = argparse.ArgumentParser(description="Rectifies a stereo dataset")
    parser.add_argument('--left_images_dir', '-li', dest='left_imgs_dir', action='store', type=str, required=True,                        
                        help='Folder containing the input images of the left camera')
    parser.add_argument('--right_images_dir', '-ri', dest='right_imgs_dir', action='store', type=str, required=True,
                        help='Folder containing the input images of the right camera')    
    parser.add_argument('--stereo_calib', '-c', action='store', dest='calib_file', default='./camera_calib.yaml', type=str, required=True,
                        help='Stereo camera calibration file (in OpenCV''s YAML format)')
    parser.add_argument('--out_left_images_dir', '-lo', action='store', dest='left_out_dir', type=str, required=True,
                        help='Output directory for the rectified images of the left camera')
    parser.add_argument('--out_right_images_dir', '-ro', action='store', dest='right_out_dir', type=str, required=True,
                        help='Output directory for the rectified images of the right camera')
    parser.add_argument('--out_calib', action='store', dest='out_calib_file', type=str, required=True, 
                        help='File where the new stereo calibration parameters corresponding to the rectified setup will be stored')
    parser.add_argument('--alpha', action='store', dest='alpha', default=0., type=float,
                        help='Rectification scale parameter, ranging from 0 to 1 (0=full crop, 1=no crop)')

    param = parser.parse_args()

    # Read the files in the stereo dataset
    left_files = list_image_files(param.left_imgs_dir)
    right_files = list_image_files(param.right_imgs_dir)

    if len(left_files) != len(right_files):
        print("[WARNING] The number of left/right images must be the same! Rectification will proceed, but check for synchronization issues!")

    # Read the left/right camera calibration parameters
    stereo_calib_cv = cv2.FileStorage(param.calib_file, cv2.FILE_STORAGE_READ)
    stereo_calib = {
        "image_width": int(stereo_calib_cv.getNode("image_width").real()),
        "image_height": int(stereo_calib_cv.getNode("image_height").real()),
        "left_camera_matrix": stereo_calib_cv.getNode("left_camera_matrix").mat(),
        "left_distortion_coefficients": stereo_calib_cv.getNode("left_distortion_coefficients").mat(),
        "right_camera_matrix": stereo_calib_cv.getNode("right_camera_matrix").mat(),
        "right_distortion_coefficients": stereo_calib_cv.getNode("left_distortion_coefficients").mat(),
        "stereo_rotation": stereo_calib_cv.getNode("stereo_rotation").mat(),
        "stereo_translation": stereo_calib_cv.getNode("stereo_translation").mat(),
    }

    # Create the output dirs
    if not os.path.exists(param.left_out_dir):
        os.makedirs(param.left_out_dir)
    if not os.path.exists(param.right_out_dir):
        os.makedirs(param.right_out_dir)

    # Pre-compute rectification parameters
    print("- Pre-computing the rectification parameters...")
    im_size = (stereo_calib["image_width"], stereo_calib["image_height"])
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(stereo_calib["left_camera_matrix"],
                                                      stereo_calib["left_distortion_coefficients"],
                                                      stereo_calib["right_camera_matrix"],
                                                      stereo_calib["right_distortion_coefficients"], im_size,
                                                      stereo_calib["stereo_rotation"],
                                                      stereo_calib["stereo_translation"], alpha=param.alpha)
    left_maps = cv2.initUndistortRectifyMap(stereo_calib["left_camera_matrix"],
                                            stereo_calib["left_distortion_coefficients"],
                                            R1, P1, im_size, cv2.CV_16SC2)
    right_maps = cv2.initUndistortRectifyMap(stereo_calib["right_camera_matrix"],
                                             stereo_calib["right_distortion_coefficients"],
                                             R2, P2, im_size, cv2.CV_16SC2)

    # Rectify all the images
    for i in range(len(left_files)):
        path, ext = os.path.splitext(os.path.basename(left_files[i]))
        full_out_img_path = os.path.join(param.left_out_dir, path + ext.lower())
        print("- Rectifying left image " + left_files[i])
        iml = cv2.imread(left_files[i])
        iml_rect = cv2.remap(iml, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
        cv2.imwrite(full_out_img_path, iml_rect)

    for i in range(len(right_files)):
        path, ext = os.path.splitext(os.path.basename(right_files[i]))
        full_out_img_path = os.path.join(param.right_out_dir, path + ext.lower())
        print("- Rectifying right image " + right_files[i])
        imr = cv2.imread(right_files[i])
        imr_rect = cv2.remap(imr, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
        cv2.imwrite(full_out_img_path, imr_rect)

    # Save the new camera parameters
    cv_file = cv2.FileStorage(param.out_calib_file, cv2.FILE_STORAGE_WRITE)
    cv_file.write("left_R", R1)
    cv_file.write("right_R", R2)
    cv_file.write("left_P", P1)
    cv_file.write("right_P", P2)
    cv_file.write("Q", Q)
    cv_file.release()

if __name__ == '__main__':
    main()
