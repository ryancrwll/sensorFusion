#!/usr/bin/env python
import argparse
import my_library as lib
import cv2
import glob
import numpy as np
import open3d as o3d
import drawing
import utils

def laser_plane_calib():
    # Handle input parameters
    parser = argparse.ArgumentParser(description="Laser plane calibration")
    parser.add_argument('--images_dir', '-i', dest='imgs_dir', action='store', type=str, required=True,
                        help="Folder containing the input images")
    parser.add_argument('--camera_calib', '-c', dest='calib_file', action='store', type=str, required=True,
                        help='The monocular camera calibration')
    parser.add_argument('--out_file', '-o', dest='out_plane_file', action='store', type=str, required=True,
                        help='The output laser plane calibration file')
    parser.add_argument('--min_red', '-t', dest='min_red', action='store', type=int, default=128,
                        help="Threshold, minimum red value to consider as the laser")
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
    param = parser.parse_args()

    # Load the camera calibration
    cv_file = cv2.FileStorage(param.calib_file, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("camera_matrix").mat()
    dist_coeffs = cv_file.getNode("distortion_coefficients").mat()

    # Create the charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(param.dict_id)
    # NOTE: Enis old opencv version 
    # board = cv2.aruco.CharucoBoard_create(param.squares_x, param.squares_y, param.square_size, param.marker_size, dictionary)    
    board = cv2.aruco.CharucoBoard((param.squares_x, param.squares_y),
                               param.square_size,
                               param.marker_size,
                               dictionary)

    # List the files in the specified input folder
    files = utils.list_image_files(param.imgs_dir)
    files.sort() # Just to list the files in order
    if not files:
        print("[ERROR] No images found in the specified folder")
        return

    # Run over all the images in the folder and detect the laser
    all_pts_list = []
    for file in files:
        print("- Detecting laser in image %s... " % file)
        
        # Load the image
        img = cv2.imread(file)        
        if img is None:
            print("[WARNING] Unable to load image %s!" % file)
            continue            

        # Detect the laser
        laser_pts = lib.detect_laser(img, param.min_red)

        # Convert the points detected on the image to rays emanating from the camera (un-project them)
        rays_dir = lib.get_normalized_coordinates(laser_pts, camera_matrix, dist_coeffs)

        # Compute the pose of the pattern with respect to the camera
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pose_computed, c_Tf_w = lib.charuco_pose_estimation(img, board, dictionary, camera_matrix, dist_coeffs)

        # Extract the plane corresponding to the pattern from the computed transform
        pattern_plane = lib.extract_xy_plane_from_charuco_pose(c_Tf_w)

        # Compute the intersection between the pattern plane and the rays
        rays_origin = np.zeros_like(rays_dir) # The origin of the rays is zero, because we are in the camera frame for all these computations
        int_pts = lib.rays_plane_intersection(pattern_plane, rays_origin, rays_dir)
        all_pts_list.append(int_pts)

    # Convert the list of arrays to a single numpy array
    all_pts = np.concatenate(all_pts_list)

    # Fit a plane to the detected points
    laser_plane = lib.fit_plane(all_pts)
    
    # Save the laser plane
    cv_file = cv2.FileStorage(param.out_plane_file, cv2.FILE_STORAGE_WRITE)
    cv_file.write('A', laser_plane[0])
    cv_file.write('B', laser_plane[1])
    cv_file.write('C', laser_plane[2])
    cv_file.write('D', laser_plane[3])
    cv_file.release()   

    # Draw the calibration results
    # - The camera frame
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # - The laser/calibration plane intersection points
    ints_pcd = o3d.geometry.PointCloud()
    ints_pcd.points = o3d.utility.Vector3dVector(all_pts)
    plane_model, inliers = ints_pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    # - The calibrated plane
    plane_mesh = drawing.create_plane_mesh(laser_plane, scale=2)
    print("- Showing the results (camera frame + 3D calibration points + calibrated laser plane). Press 'q' or 'ESC' to finish...")
    o3d.visualization.draw_geometries([mesh_frame, ints_pcd, plane_mesh], mesh_show_back_face=True)    
    

if __name__ == '__main__':
    laser_plane_calib()
