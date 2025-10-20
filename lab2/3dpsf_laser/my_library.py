import cv2
import numpy as np


def charuco_pose_estimation(img, board, dictionary, camera_matrix, dist_coeffs):
    """
    Detects the pose of a ChAruco pattern given a calibration.

    Parameters
        - img: the input image.
        - board: the description of the ChAruco board.
        - dictionary: the dictionary ID of the board.
        - camera_matrix: camera matrix.
        - dist_coeffs: lens distortion coefficients.
    Returns:
        - A boolean, indicating if the pattern was correctly detected.
        - The pose of the camera, in the form of a homogeneous transform that transforms 
        points from the board coordinate system to the camera coordinate system: c_Tf_w
    """

    # Detect the Aruco markers
    marker_corners, marker_ids, rejected_markers = cv2.aruco.detectMarkers(img, dictionary)

    if len(marker_corners) > 0:
        # Interpolate the corners near the detected markers
        num_detected, corners, corner_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board)
        if num_detected > 0:
            # Estimate the pose using the calibration and the board description
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(corners, corner_ids, board, camera_matrix, dist_coeffs, None, None)
            if retval:
                R, _ = cv2.Rodrigues(rvec)
                c_Tf_w = np.eye(4,4)
                c_Tf_w[0:3, 0:3] = R
                c_Tf_w[0:3, 3] = tvec.ravel()  
                return retval, c_Tf_w
    return False, None

def extract_xy_plane_from_charuco_pose(c_Tf_w):
    """
    Extracts the A, B, C, D parameters of the XY plane of the calibration pattern.

    Parameters:
        - c_Tf_w: the pose of the pattern, output of charuco_pose_estimation function.
    Returns:
        - A 4-elements tuple containing the parameters of the plane in normal form Ax+By+Cz+D=0
    """
    normal = c_Tf_w[0:3, 2]
    A = normal[0]
    B = normal[1]
    C = normal[2]
    t = c_Tf_w[0:3, 3]
    D = -np.dot(normal, t)

    return (A, B, C, D)

def detect_laser(img, min_red):
    """
    Gets a list of 2D image points that correspond to the profile of the laser using the COG method.

    Parameters:
        - img: the input image, in OpenCV format (remember that OpenCV uses BGR by default!).
        - min_red: the threshold parameter, all pixels with a value of red below this value 
        will not be taken into account
    Returns
        - A Numpy array of size num_pts x 2
    """
    pass

def fit_plane(points):
    """
    Fits a plane to a set of points. 
    
    Parameters:
        - points: a numpy array of size num_pts x 3 containing the sample points on the plane to fit.
    Returns:
        - A 4-elements tuple containing the parameters of the plane in normal form Ax+By+Cz+D=0.
    """
    pass


def get_normalized_coordinates(p2ds, camera_matrix, dist_coeffs):
    """
    Compute the normalized coordinates of a set of 2D image points by inverting the projection using the camera calibration

    Parameters:
        - p2ds: 2D image points, a Numpy array of shape num_pts x 2
        - camera_matrix: 3x3 camera matrix describing the perspective/pinhole projection.
        - dist_coeffs: an array of coefficients of the lens distortion model.
    Returns:
        - normalized coordinates, a Numpy array of size num_pts x 3 where the 3rd column is a vector of ones.
    """
    pass
    

def rays_plane_intersection(plane, rays_origin, rays_direction, epsilon=1e-6):
    """
    Find the intersection (if any) between a set of rays and a plane.

    Parameters:
        - plane: a tuple with (A, B, C, D) parameters of the plane in normal form Ax+By+Cz+D=0.
        - rays_origin: origins of the rays, as a numpy array of size num_rays x 3.
        - rays_direction: direction vectors of the rays, as a numpy array of size num_rays x 3.
        - epsilon: floating value that we consider the minimum.
    Returns:
        - intersections: intersection points between the input rays and plane, of shape num_rays x 3. 
        In case there is no intersection, we mark it with an infinite point: [np.Inf, np.Inf, np.Inf]
    """
    pass