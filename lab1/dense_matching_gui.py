#!/usr/bin/env python
import numpy as np
import argparse
import cv2
import open3d as o3d
import open3d.visualization as vis

# Main function
def main():

    parser = argparse.ArgumentParser(description="Stereo dense matching GUI")
    parser.add_argument('--left_image', '-l', dest='left_img_file', action='store', type=str, required=True,
                        help='The left image in the stereo pair')
    parser.add_argument('--right_image', '-r', dest='right_img_file', action='store', type=str, required=True,
                        help='The right image in the stereo pair')
    parser.add_argument('--rect_stereo_calib', '-c', dest='rect_stereo_calib_file', action='store', type=str, required=True,
                        help='The rectified stereo calibration file, in OpenCV''s YAML format (optional if you just want to see the matches, but needed to compute the 3D point set)')
    param = parser.parse_args()
    
    # Load the images
    print('- Loading images...')
    imgL = cv2.imread(param.left_img_file)
    imgR = cv2.imread(param.right_img_file)

    # Convert to greyscale (the stereo matching does not use color)
    imgLG = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgRG = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    def nothing(x):
        pass

    cv2.namedWindow('Disparity Map', cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('Disparity Map', 600, 600)    
    cv2.createTrackbar('block_size (value*2+5)', 'Disparity Map', 5, 50, nothing)
    cv2.createTrackbar('num_disparities (value*16)', 'Disparity Map', 1, 25, nothing)
    cv2.createTrackbar('min_disparity', 'Disparity Map', 0, 500, nothing)
    cv2.createTrackbar('prefilter_type', 'Disparity Map', 1, 1, nothing)
    cv2.createTrackbar('prefilter_size (value*2+5)', 'Disparity Map', 2, 25, nothing)
    cv2.createTrackbar('prefilter_cap', 'Disparity Map', 5, 62, nothing)
    cv2.createTrackbar('disp12_max_diff', 'Disparity Map', 5, 25, nothing)
    cv2.createTrackbar('texture_threshold', 'Disparity Map', 10, 100, nothing)
    cv2.createTrackbar('uniqueness_ratio', 'Disparity Map', 15, 100, nothing)    
    cv2.createTrackbar('speckle_size', 'Disparity Map', 3, 25, nothing)
    cv2.createTrackbar('speckle_range', 'Disparity Map', 0, 100, nothing)

    # Creating an object of StereoBM algorithm
    stereo = cv2.StereoBM_create()

    while True:
        # Update the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('num_disparities (value*16)', 'Disparity Map')*16
        if numDisparities < 16:
            numDisparities = 16
        blockSize = cv2.getTrackbarPos('block_size (value*2+5)', 'Disparity Map')*2 + 5
        preFilterType = cv2.getTrackbarPos('prefilter_type', 'Disparity Map')
        preFilterSize = cv2.getTrackbarPos('prefilter_size (value*2+5)', 'Disparity Map')*2 + 5
        preFilterCap = cv2.getTrackbarPos('prefilter_cap', 'Disparity Map')
        if preFilterCap < 1:
            preFilterCap = 1
        textureThreshold = cv2.getTrackbarPos('texture_threshold', 'Disparity Map')
        uniquenessRatio = cv2.getTrackbarPos('uniqueness_ratio', 'Disparity Map')
        speckleRange = cv2.getTrackbarPos('speckle_range', 'Disparity Map')
        speckleWindowSize = cv2.getTrackbarPos('speckle_size', 'Disparity Map')*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12_max_diff', 'Disparity Map')
        minDisparity = cv2.getTrackbarPos('min_disparity', 'Disparity Map')

        # Set the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        # stereo.setBlockSize(blockSize)
        stereo.setBlockSize(11) # Block size must be 5, 7, 9 ... 255 (odd numbers). Using a fixed value here as suggested in OpenCV docs
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        # stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setUniquenessRatio(15)
        # stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleRange(100)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Compute the disparity map
        disparity = stereo.compute(imgLG, imgRG)
        
        # NOTE: StereoBM::compute returns a 16bit signed single channel image (CV_16S) containing a disparity map scaled by 16. 
        # We need to convert it to CV_32F and divide it by 16 to get the real disparity values
        disparity = disparity.astype(np.float32)

        # Scale and normalize the values for visualization
        disparity = (disparity/16.0 - stereo.getMinDisparity())/stereo.getNumDisparities()

        # Show the disparity map (greyscale)
        # cv2.imshow('Disparity Map', disparity)

        # Show the disparity map (color map)
        disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disp_normalized = np.uint8(disp_normalized)        
        disp_colormap = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
        cv2.imshow('Disparity Map', disp_colormap)

        # Close window using 'ESC' key
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
   main()
