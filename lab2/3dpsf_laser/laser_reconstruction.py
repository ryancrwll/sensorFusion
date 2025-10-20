import argparse
import numpy as np
import cv2
import open3d as o3d
import my_library as lib
import utils


def laser_reconstruction():

    parser = argparse.ArgumentParser(description="Laser plane calibration")
    parser.add_argument('--images_dir', '-i', dest='imgs_dir', action='store', type=str, required=True,
                        help="Folder containing the input images")
    parser.add_argument('--camera_calib', '-c', dest='calib_file', action='store', type=str, required=True,
                        help='The monocular camera calibration file')
    parser.add_argument('--laser_calib', '-l', dest='laser_calib', action='store', type=str, required=True,
                        help='The laser plane calibration file')
    parser.add_argument('--out_pts', '-o', dest='out_point_set_file', action='store', type=str, required=True,
                        help='Output reconstructed point set file')                        
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
    cv_file.release()

    # Load the laser plane calibration
    cv_file = cv2.FileStorage(param.laser_calib, cv2.FILE_STORAGE_READ)
    A = cv_file.getNode("A").real()
    B = cv_file.getNode("B").real()
    C = cv_file.getNode("C").real()
    D = cv_file.getNode("D").real()
    laser_plane = (A, B, C, D)
    cv_file.release()

    # Create the charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(param.dict_id)
    board = cv2.aruco.CharucoBoard_create(param.squares_x, param.squares_y, param.square_size, param.marker_size, dictionary)    

    # List the input images
    files = utils.list_image_files(param.imgs_dir)
    files.sort() # Just to list the files in order
    if not files:
        print('[ERROR] No images found in the specified folder')
        return

    # Detect and reconstruct the laser on each image
    all_pts_list = []
    for file in files:
        print('- Reconstructing laser in image %s... ' % file)
        
        # Load the image
        img = cv2.imread(file)        
        if img is None:
            print("[WARNING] Unable to load image %s!", file)
            continue     

        ### Exercise 9: Implement the the missing parts of the laser reconstruction
        ###             The remaining comments in this loop describe the steps that must be filled to achieve a reconstruction

        # Detect the laser in the image
        

        # Convert the detected image points to rays emanating from the camera (un-project them)
        

        # Intersect them with the laser plane
        

        # Compute the pose of the pattern with respect to the camera
        

        # We want the opposite transform: the one passing points from the camera frame to the world frame
        

        # This intersection is within the camera frame, transform the points to the world frame given the pose estimated from the charuco pattern, store the value in int_pts_w variable, of shape num_pts x 3
        

        # Add the points of this profile to the global list of 3D points (all_pts_list variable)
        all_pts_list.append(int_pts_w)

    # Draw the reeconstructed points
    all_pts = np.concatenate(all_pts_list, axis=0)
    ints_pcd = o3d.geometry.PointCloud()
    ints_pcd.points = o3d.utility.Vector3dVector(all_pts)
    o3d.visualization.draw_geometries([ints_pcd], mesh_show_back_face=True)
    print("- Showing the 3D reconstruction. Press 'q' or 'ESC' to finish...")
    o3d.io.write_point_cloud(param.out_point_set_file, ints_pcd)

if __name__ == '__main__':
    laser_reconstruction()
