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
    pts = []

    for y in range(img.shape[0]):                   # loop over rows
        red_row = img[y, :, 2]                      # extract red channel along row y
        valid_idx = np.where(red_row >= min_red)[0] # mask of pixels with enough red
        if len(valid_idx) == 0:                     # if no valid pixels found
            continue

        red_vals = red_row[valid_idx]               # red values of the valid pixels
        cog_x = np.sum(valid_idx * red_vals) / np.sum(red_vals)
        pts.append([cog_x, y])                      # (x, y) format

    return np.array(pts)

def fit_plane(points):
    """
    Fits a plane to a set of points. 
    
    Parameters:
        - points: a numpy array of size num_pts x 3 containing the sample points on the plane to fit.
    Returns:
        - A 4-elements tuple containing the parameters of the plane in normal form Ax+By+Cz+D=0.
    """
    pass
    n = points.shape[0]                             # number of points

    centroid = 1/n * np.sum(points, axis=0)         # compute centroid

    A = np.zeros((3, 3))
    for i in range(n):                              # compute covariance matrix
        p = points[i] - centroid                    
        A += np.outer(p, p)                         
    A /= n                                          # normalize by number of points

    _, _, Vt = np.linalg.svd(A)                     # Compute the normal vector of the plane
    normal = Vt[-1]

    D = -np.dot(normal, centroid)                   # Compute the D parameter

    return (normal[0], normal[1], normal[2], D)


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
    und_points = cv2.undistortPoints(cameraMatrix=camera_matrix,
                                     distCoeffs=dist_coeffs,
                                     src=p2ds)
    und_points_squeezed = und_points.squeeze()      # remove extra dimension (n, 1, 2) -> (n, 2)
    return np.hstack((und_points_squeezed, np.ones((und_points_squeezed.shape[0], 1))))


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
    A, B, C, D = plane                                      # unpack plane parameters
    n = np.array([A, B, C])                                 # normal vector of the plane       
    points_3D = []
    for i in range(rays_origin.shape[0]):                   # iterate over rays
        p0 = rays_origin[i]
        d = rays_direction[i]

        denom = np.dot(n, d)                    
        if abs(denom) < epsilon:                            # ray is parallel to the plane   
            points_3D.append([np.Inf, np.Inf, np.Inf])
            continue

        t = -(np.dot(n, p0) + D) / denom
        if t < 0:                                           # intersection is behind the ray origin
            points_3D.append([np.Inf, np.Inf, np.Inf])
            continue

        intersection = p0 + t * d                           # compute intersection point
        points_3D.append(intersection)

    return np.array(points_3D)
