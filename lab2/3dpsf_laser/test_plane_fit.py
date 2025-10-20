import argparse
import numpy as np
import drawing
import open3d as o3d
import my_library as lib


def test_plane_fit():
    # Handle input parameters
    parser = argparse.ArgumentParser(description="Test the plane fitting method")
    parser.add_argument('--num_pts', '-p', dest='num_pts', action='store', type=int, default=100,
                        help="Number of random points to generate in the test plane")
    parser.add_argument('--plane_a', '-a', dest='A', action='store', type=float, default=10,
                        help="A term of the test plane's equation")
    parser.add_argument('--plane_b', '-b', dest='B', action='store', type=float, default=10,
                        help="B term of the test plane's equation")
    parser.add_argument('--plane_c', '-c', dest='C', action='store', type=float, default=10,
                        help="C term of the test plane's equation")
    parser.add_argument('--plane_d', '-d', dest='D', action='store', type=float, default=10,
                        help="D term of the test plane's equation")
    param = parser.parse_args()

    # Create the plane
    normal = np.array([param.A, param.B, param.C])
    normal = normal/np.linalg.norm(normal)
    D = param.D
    point_in_plane =  normal * -D/np.linalg.norm(normal)

    # Compute an orthonormal basis for the normal of the plane
    vx, vy = drawing.compute_orthonormal_basis(normal)

    # Use the orthonormal basis to generate points in the plane
    num_pts = param.num_pts
    pts_in_plane = np.zeros((num_pts, 3))
    for i in range(num_pts):
        pts_in_plane[i, :] = point_in_plane + vx*np.random.uniform(-1.0, 1.0, 1) + vy*np.random.uniform(-1.0, 1.0, 1)

    # Fit a plane to the points
    fitted_plane = lib.fit_plane(pts_in_plane)

    # Compare the coefficients of the original and the fitted plane
    print("- Original plane: ({:f}, {:f}, {:f}, {:f})".format(normal[0], normal[1], normal[2], param.D))
    print("- Fitted plane: ({:f}, {:f}, {:f}, {:f})".format(fitted_plane[0], fitted_plane[1], fitted_plane[2], fitted_plane[3]))
    print("  (note that there might be an ambiguity on the sign!)")

    # Plot the points and fitted plane
    plane_pts_pcd = o3d.geometry.PointCloud()
    plane_pts_pcd.points = o3d.utility.Vector3dVector(pts_in_plane)
    plane_mesh = drawing.create_plane_mesh(fitted_plane, scale=3)
    print("- Showing the fitted plane and the sample points. Press 'q' or 'ESC' to finish...")
    o3d.visualization.draw_geometries([plane_pts_pcd, plane_mesh], mesh_show_back_face=True)


if __name__ == '__main__':
    test_plane_fit()