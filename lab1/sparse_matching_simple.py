#!/usr/bin/env python
import numpy as np
import cv2
import argparse
import open3d as o3d
import open3d.visualization as vis


# Main function
def main():
    parser = argparse.ArgumentParser(description="Stereo sparse matching")
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

    # Convert to greyscale (detectors do not use color)
    imgLG = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgRG = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Load the rectified stereo camera calibration
    if param.rect_stereo_calib_file:
        print('- Loading the rectified stereo calibration...')    
        cv_file = cv2.FileStorage(param.rect_stereo_calib_file, cv2.FILE_STORAGE_READ)
        left_P = cv_file.getNode("left_P").mat()
        right_P = cv_file.getNode("right_P").mat()

    # Compute the features for both images
    print('- Detecting/Describing keypoints in the stereo pair...')
    ### Ex. 6: Change the following three lines by other feature detectors/descriptors and check their performance
    ### see https://docs.opencv.org/4.6.0/db/d27/tutorial_py_table_of_contents_feature2d.html
    sift = cv2.SIFT_create()
    kpL, descL = sift.detectAndCompute(imgLG, None)
    kpR, descR = sift.detectAndCompute(imgRG, None)
    ###

    # Show the features in both images
    if param.debug:
        imgLKp = cv2.drawKeypoints(imgL, kpL, None)
        imgRKp = cv2.drawKeypoints(imgR, kpR, None)
        print('- Showing keypoints detected in left/right images, press any key to continue...')
        cv2.namedWindow('Left image keypoints', cv2.WINDOW_NORMAL)
        cv2.imshow('Left image keypoints', imgLKp)
        cv2.namedWindow('Right image keypoints', cv2.WINDOW_NORMAL)
        cv2.imshow('Right image keypoints', imgRKp)
        cv2.waitKey()

    # Match the descriptors
    print('- Matching the features...')
    bf = cv2.BFMatcher() # Create the matcher
    matches = bf.match(descL, descR)
    
    ### Ex. 5: Use the epipolar constraint on rectified images to further constrain the matches and reduce the outliers
    
    if param.rect_stereo_calib_file:
        
        good_matches = []
        vertical_threshold = 6.0  # pixels

        for m in matches:
            ptL = kpL[m.queryIdx].pt
            ptR = kpR[m.trainIdx].pt
            
            if abs(ptL[1] - ptR[1]) < vertical_threshold:
                good_matches.append(m)

        matches = good_matches
    ###

    if param.debug:
        # Show all the matches
        imgAllMatches = cv2.drawMatches(imgL, kpL, imgR, kpR, matches, None, flags=2)
        print('- Showing detected keypoints in left/right images, press any key to continue...')
        cv2.namedWindow('All matches', cv2.WINDOW_NORMAL)
        cv2.imshow('All matches', imgAllMatches)
        cv2.waitKey()

    # Reconstruct the 3D points if the camera calibration is available
    if param.rect_stereo_calib_file:
        print('- Computing the sparse 3d point cloud...')
        # Collect the 2D projections on each image
        p2dL = np.array([kpL[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        p2dR = np.array([kpR[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Triangulate the points
        p3dh = cv2.triangulatePoints(left_P, right_P, p2dL, p2dR)
        p3d = cv2.convertPointsFromHomogeneous(p3dh.T) # De-homogenize
        # De-homogenize
        p3dh = p3dh / p3dh[3] 
        p3d = p3dh[:3].T # And transpose, for convenience

        # Remove obvious outliers:
        #  - points behind the camera
        p3d = p3d[np.where(p3d[:,2] > 0)]        
        #  - points too far from the camera (>10 meters, assuming the calibration was done in meters)
        p3d = p3d[np.where(p3d[:,2] < 10)]

        # Convert the points to Open3D for visualization/saving
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p3d)
            
        if param.debug:            
            # Visualize the points
            print('- Showing the reconstructed 3D points, press ''q'' or close the window to continue...')
            o3d.visualization.draw_geometries([pcd])

        # Write the result to a PLY file
        print('- Saving the 3D point cloud...')
        o3d.io.write_point_cloud(param.out_ply, pcd)


if __name__ == '__main__':
   main()
