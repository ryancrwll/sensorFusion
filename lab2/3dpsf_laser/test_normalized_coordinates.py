import argparse
import cv2
import my_library as lib
import numpy as np

def test_normalized_coordinates():
    # Handle input parameters
    parser = argparse.ArgumentParser(description="Check the get_normalized_coordinates method")
    parser.add_argument('--camera_calib', '-c', dest='calib_file', action='store', type=str, required=True,
                        help="The camera calibration with which we will test the back-projection method")
    parser.add_argument('--num_samples', '-n', dest='num_samples', action='store', type=int, default=100,
                        help="The number of random sample points used for the test")
    parser.add_argument('--tolerance', '-t', dest='tolerance', action='store', type=float, default=1,
                        help="The tolerance for the difference check used in our test")
    param = parser.parse_args()

    # Load the camera calibration
    cv_file = cv2.FileStorage(param.calib_file, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("camera_matrix").mat()
    dist_coeffs = cv_file.getNode("distortion_coefficients").mat()
    image_width = int(cv_file.getNode("image_width").real())
    image_height = int(cv_file.getNode("image_height").real())
    cv_file.release()

    # Create a set of random 2D points in the image plane
    i_x = np.random.randint(0, image_width, (param.num_samples, 1)).astype(float)
    i_y = np.random.randint(0, image_height, (param.num_samples, 1)).astype(float)
    image_pts = np.append(i_x, i_y, axis=1)

    # Back-project these points (the results are rays originating from the camera)
    rays_dir = lib.get_normalized_coordinates(image_pts, camera_matrix, dist_coeffs)

    # Give the points a random depth
    depth = np.random.uniform(1.0, 100.0, (param.num_samples, 1))
    pts_3d = rays_dir*depth

    # Now, if everything works as expected, projecting those 3D points using the calibration should lead to the same 2D image points
    image_pts_2, _ = cv2.projectPoints(pts_3d, (0,0,0), (0,0,0), camera_matrix, dist_coeffs)
    image_pts_2 = image_pts_2.squeeze() # Remove singleton dimension to get an Nx2 matrix (cv2.projectPoints returns an Nx1x2 matrix, which is not very intuitive...)

    # Compare the 2D points sampled at the beginning with the points that we back-projected and then projected again
    for pt_or, pt_test in zip(image_pts, image_pts_2):
        diff = np.sum(np.abs(pt_or-pt_test))
        if diff > param.tolerance:            
            print("[ERROR] at least one of the back-projected --> projected points do not match the original pixel position")
            print("absolute difference = %f (> tolerance = %f)" % (diff, param.tolerance))
            return
    
    print("The back-projection method seems to work correctly!")


if __name__ == '__main__':
    test_normalized_coordinates()