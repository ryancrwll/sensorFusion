import open3d as o3d
import numpy as np

def create_plane_mesh(plane, scale=1):
    """Creates a plane as two triangles, as an open3d mesh object"""
    A = plane[0]
    B = plane[1]
    C = plane[2]
    D = plane[3]

    plane_pts = np.array([[-0.5, -0.5, 0.5,  0.5],
                         [-0.5,  0.5, 0.5, -0.5],
                         [0, 0, 0, 0]])*scale 
    plane_pts = np.append(plane_pts, np.array([[1, 1, 1, 1]]), axis=0) # homogeneous coordinates
    n = np.array(np.array([A, B, C])) # The plane normal
    n = n / np.linalg.norm(n) # Make sure it is normalized
    u, v = compute_orthonormal_basis(n) # Compute a valid orthonormal basis for this normal

    # Compose a 3D tranformation matrix out of this basis and the projection of the origin into the plane
    origin_in_plane = -n*D
    w_Tf_p = np.eye(4,4)
    w_Tf_p[0:3, 0] = u
    w_Tf_p[0:3, 1] = v
    w_Tf_p[0:3, 2] = n
    w_Tf_p[0:3, 3] = origin_in_plane

    world_pts = w_Tf_p@plane_pts
    world_pts = np.delete(world_pts, 3, 0)

    # Create two triangles out of this points
    triangles = np.array([[0, 1, 2],
                          [2, 3, 0]]).astype(np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertices = o3d.utility.Vector3dVector(world_pts.T)

    return mesh
    

def compute_orthonormal_basis(n):
    """Computes an orthonormal basis for the input normal"""
    # Special cases ( axis-aligned normal )
    if np.all(np.absolute(n) == np.array([1.0, 0.0, 0.0])):
        x = np.array([0.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0])
        if n[0 < 0]:
            x = -1*x
            y = -1*y
        return x, y
    elif np.all(np.absolute(n) == np.array([0.0, 1.0, 0.0])):        
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 0.0, 1.0])
        if n[1] < 0:
            x = -1*x
            y = -1*y
        return x, y
    elif np.all(np.absolute(n) == np.array([0.0, 0.0, 1.0])):    
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        if n[2] < 0:
            x = -1*x
            y = -1*y
        return x, y
    else:
        # Compute a good basis
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([0.0, 0.0, 0.0])
        if abs(n[0]) < abs(n[1]):
            x[0] = n[2]
            x[1] = 0
            x[2] = -n[0]

            y[0] = n[0]*n[1]
            y[1] = -(n[0]*n[0] + n[2]*n[2])
            y[2] = n[1]*n[2]
        else:
            x[0] = n[1]
            x[1] = -n[0]
            x[2] = 0

            y[0] = n[0]*n[2]
            y[1] = n[1]*n[2]
            y[2] = -(n[0]*n[0] + n[1]*n[1])
   
    # Normalize x and y
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)

    return x, y


def draw_axis(img, rotation_vec, t, K, scale=0.1, dist=None):
    """
    Draw a 6dof axis (XYZ -> RGB) in the given rotation and translation
    :param img - rgb numpy array
    :rotation_vec - euler rotations, numpy array of length 3,
                    use cv2.Rodrigues(R)[0] to convert from rotation matrix
    :t - 3d translation vector, in meters (dtype must be float)
    :K - intrinsic calibration matrix , 3x3
    :scale - factor to control the axis lengths
    :dist - optional distortion coefficients, numpy array of length 4. If None distortion is ignored.
    """
    
    dist = np.zeros(4, dtype=float) if dist is None else dist
    points = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
    axis_points, _ = cv2.projectPoints(points, rotation_vec, t, K, dist)
    img = cv2.line(img, tuple(axis_points[3].ravel().astype(int)), tuple(axis_points[0].ravel().astype(int)), (0, 0, 255), 3)
    img = cv2.line(img, tuple(axis_points[3].ravel().astype(int)), tuple(axis_points[1].ravel().astype(int)), (0, 255, 0), 3)
    img = cv2.line(img, tuple(axis_points[3].ravel().astype(int)), tuple(axis_points[2].ravel().astype(int)), (255, 0, 0), 3)
    return img