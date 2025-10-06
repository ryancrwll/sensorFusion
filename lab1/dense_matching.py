#!/usr/bin/env python
import numpy as np
import argparse
import cv2
import open3d as o3d
import open3d.visualization as vis

# Main function
def main():

    parser = argparse.ArgumentParser(description="Stereo dense matching")
    parser.add_argument('--left_image', '-l', dest='left_img_file', action='store', type=str, required=True,
                        help='The left image in the stereo pair')
    parser.add_argument('--right_image', '-r', dest='right_img_file', action='store', type=str, required=True,
                        help='The right image in the stereo pair')
    parser.add_argument('--rect_stereo_calib', '-c', dest='rect_stereo_calib_file', action='store', type=str, required=True,
                        help='The rectified stereo calibration file, in OpenCV''s YAML format (optional if you just want to see the matches, but needed to compute the 3D point set)')
    parser.add_argument('--out_pts_file', '-o', dest='out_ply', action='store', type=str, default="out.ply",
                        help='Output ply file with the 3D point set')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Debug flag. Shows intermediate results on screen (Note: waits for user input at each step).')
    param = parser.parse_args()
    
    # Load the images
    print('- Loading images...')
    imgL = cv2.imread(param.left_img_file)
    imgR = cv2.imread(param.right_img_file)

    # Convert to greyscale (the stereo matching does not use color)
    imgLG = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgRG = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Load the stereo camera calibration
    if param.rect_stereo_calib_file:
        print('- Loading the rectified stereo calibration...')    
        cv_file = cv2.FileStorage(param.rect_stereo_calib_file, cv2.FILE_STORAGE_READ)
        Q = cv_file.getNode("Q").mat()

    print('- Computing the disparity map...')
    ### Ex. 7: Play with the stereo matching parameters in order to obtain the best dense reconstruction for each of the stereo pairs provided
    # Stereo Matching parameters
    #  --- Disparity SSD ---
    block_size =   # SSD window size. Must be an odd number
    num_disparities =    # Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
    min_disparity =     # Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
    # --- Disparity pre-filtering ---
    prefilter_type = cv2.STEREO_BM_PREFILTER_XSOBEL # Prefilter to enhance results. Possibilities: cv2.STEREO_BM_PREFILTER_XSOBEL or cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE
    prefilter_size =      # Prefilter window size
    prefilter_cap =      # Truncation value for the prefiltered image pixels.
    # --- Disparity post-filtering ---
    disp12_max_diff =    # Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
    texture_threshold =   # Filters disparity readings based on the amount of texture in the SSD window
    uniqueness_ratio =    # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
    speckle_size =      # Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckle_range =       # Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
    ###

    ### Ex. 8: Change the Block matching algorithm below by the Semi-Global Block Matching algorithm and find the suitable parameters for the different pairs
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    stereo.setMinDisparity(min_disparity)
    stereo.setPreFilterType(prefilter_type)
    stereo.setPreFilterSize(prefilter_size)
    stereo.setPreFilterCap(prefilter_cap)
    stereo.setDisp12MaxDiff(disp12_max_diff)
    stereo.setTextureThreshold(texture_threshold)
    stereo.setUniquenessRatio(uniqueness_ratio)
    stereo.setSpeckleWindowSize(speckle_size)
    stereo.setSpeckleRange(speckle_range)
    ###

    disp = stereo.compute(imgLG, imgRG)
    # NOTE: StereoBM::compute returns a 16bit signed single channel image (CV_16S) containing a disparity map scaled by 16. 
    # We need to convert it to CV_32F and divide it by 16 to get the real disparity values
    disp = disp.astype(np.float32) / 16.0

    if param.debug:
        print('- Displaying the disparity map, press any key to continue...')
        cv2.namedWindow('Left image', cv2.WINDOW_NORMAL)
        cv2.imshow('Left image', imgL)
        cv2.namedWindow('Right image', cv2.WINDOW_NORMAL)
        cv2.imshow('Right image', imgR)
        cv2.namedWindow('Disparity map', cv2.WINDOW_NORMAL)
        cv2.imshow('Disparity map', (disp - min_disparity) / num_disparities)
        cv2.waitKey()

    if param.rect_stereo_calib_file:
        print('- Computing the 3d point cloud...')
        h, w = imgL.shape[:2]
        points = cv2.reprojectImageTo3D(disp, Q)
        colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

        # Filter invalid values
        mask = disp > disp.min()
        mask = mask & ~np.isnan(points[:, :, 0])
        mask = mask & ~np.isinf(points[:, :, 0])
        p3d = points[mask]
        p3d_colors = colors[mask]

        # Remove obvious outliers:
        #  - points behind the camera        
        p3d_colors = p3d_colors[np.where(p3d[:,2] > 0)]
        p3d = p3d[np.where(p3d[:,2] > 0)]
        #  - points too far from the camera (>10 meters, assuming the calibration was done in meters)
        p3d_colors = p3d_colors[np.where(p3d[:,2] < 10)]
        p3d = p3d[np.where(p3d[:,2] < 10)]        

        # Convert the points to Open3D for visualization/saving
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p3d)
        pcd.colors = o3d.utility.Vector3dVector(p3d_colors/255)

        if param.debug:            
            # Visualize the points
            print('- Showing the reconstructed 3D points, press ''q'' or close the window to continue...')
            o3d.visualization.draw_geometries([pcd])

        # Write the result to a PLY file
        print('- Saving the 3d point cloud...')
        o3d.io.write_point_cloud(param.out_ply, pcd)


if __name__ == '__main__':
   main()
