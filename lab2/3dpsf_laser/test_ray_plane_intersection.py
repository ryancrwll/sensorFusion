import argparse
import numpy as np
import drawing
import open3d as o3d
import my_library as lib


def test_ray_plane_intersection():
    # Handle input parameters
    parser = argparse.ArgumentParser(description="Check the ray-plane intersection method")
    parser.add_argument('--num_rays', '-r', dest='num_rays', action='store', type=int, default=5,
                        help="Number of random rays to test")
    param = parser.parse_args()

    # Random rays
    rays_origin = np.random.uniform(-1.0, 1.0, (param.num_rays, 3))
    rays_dir = np.random.uniform(-1.0, 1.0, (param.num_rays, 3))
    rays_dir_sum = np.sqrt((rays_dir*rays_dir).sum(axis=1))
    rays_dir = rays_dir/rays_dir_sum[:, np.newaxis]
    
    # Random plane
    plane_normal = np.random.uniform(-1.0, 1.0, 3)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    plane_point = np.random.uniform(-1.0, 1.0, 3)
    A = plane_normal[0]
    B = plane_normal[1]
    C = plane_normal[2]
    D = -np.dot(plane_normal, plane_point)

    # Compute the intersections
    intersections = lib.rays_plane_intersection((A, B, C, D), rays_origin, rays_dir)        

    # Remove infinite points (i.e., no intersections)
    intersections = intersections[~np.isinf(intersections).any(axis=1)]
    
    # Draw the problem
    #   - The plane
    plane_mesh = drawing.create_plane_mesh((A, B, C, D), scale=15)
    #   - The segments
    segment_length = 10
    rays_ends = rays_origin + rays_dir*segment_length # Draw the rays as segments, spanning 1 from the origin following the ray direction
    segments_pts = np.append(rays_origin, rays_ends, axis=0)
    lines = []
    for i in range(param.num_rays):
        lines.append([i, i+param.num_rays])
    rays_lineset = o3d.geometry.LineSet()
    rays_lineset.points = o3d.utility.Vector3dVector(segments_pts)
    rays_lineset.lines = o3d.utility.Vector2iVector(lines)
    rays_colors = [[0.0, 0.0, 1.0] for i in range(len(lines))]
    rays_lineset.colors = o3d.utility.Vector3dVector(rays_colors)
    #   - Print the origin of the rays as lines
    rays_origins_pcd = o3d.geometry.PointCloud()
    rays_origins_pcd.points = o3d.utility.Vector3dVector(rays_origin)
    rays_origins_colors = [[0.0, 1.0, 0.0] for i in range(param.num_rays)]
    rays_origins_pcd.colors = o3d.utility.Vector3dVector(rays_origins_colors)
    #   - Intersection points
    ints_pcd = o3d.geometry.PointCloud()
    ints_pcd.points = o3d.utility.Vector3dVector(intersections)
    ints_colors = [[1.0, 0.0, 0.0] for i in range(intersections.shape[0])]
    ints_pcd.colors = o3d.utility.Vector3dVector(ints_colors)
    #   - Visualize all together
    print("- Showing the rays (origin in green, direction in blue) and the intersections with the plane (in red). Press 'q' or 'ESC' to finish...")
    o3d.visualization.draw_geometries([plane_mesh, rays_lineset, ints_pcd, rays_origins_pcd], mesh_show_back_face=True)


if __name__ == '__main__':
    test_ray_plane_intersection()


