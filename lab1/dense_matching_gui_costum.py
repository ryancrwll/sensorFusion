#!/usr/bin/env python
import numpy as np
import argparse
import cv2
import open3d as o3d
import open3d.visualization as vis

def main():
    parser = argparse.ArgumentParser(description="Stereo dense matching GUI")
    parser.add_argument('--left_image', '-l', dest='left_img_file', type=str, required=True,
                        help='The left image in the stereo pair')
    parser.add_argument('--right_image', '-r', dest='right_img_file', type=str, required=True,
                        help='The right image in the stereo pair')
    parser.add_argument('--rect_stereo_calib', '-c', dest='rect_stereo_calib_file', type=str, required=True,
                        help="The rectified stereo calibration file, in OpenCV's YAML format")
    param = parser.parse_args()

    # Load images
    print('- Loading images...')
    imgL = cv2.imread(param.left_img_file)
    imgR = cv2.imread(param.right_img_file)
    if imgL is None or imgR is None:
        raise RuntimeError('Could not load left or right image')

    # Convert to grayscale
    imgLG = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgRG = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Helper for trackbars
    def nothing(x):
        pass

    # Window 1 - visualization
    cv2.namedWindow('Disparity Map', cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('Disparity Map', 800, 600)

    # Window 2 - parameters
    PARAM_WIN = 'StereoBM Params'
    cv2.namedWindow(PARAM_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(PARAM_WIN, 420, 480)

    # Trackbars for parameter tuning
    cv2.createTrackbar('block_size (2*k+5)',        PARAM_WIN, 5, 50,  nothing)
    cv2.createTrackbar('num_disparities (16*k)',    PARAM_WIN, 1,  25,  nothing)
    cv2.createTrackbar('min_disparity',             PARAM_WIN, 0,  500, nothing)
    cv2.createTrackbar('prefilter_type',            PARAM_WIN, 1,  1,   nothing)
    cv2.createTrackbar('prefilter_size (2*k+5)',    PARAM_WIN, 2,  25,  nothing)
    cv2.createTrackbar('prefilter_cap',             PARAM_WIN, 5,  62,  nothing)
    cv2.createTrackbar('disp12_max_diff',           PARAM_WIN, 5,  25,  nothing)
    cv2.createTrackbar('texture_threshold',         PARAM_WIN, 10, 100, nothing)
    cv2.createTrackbar('uniqueness_ratio',          PARAM_WIN, 15, 100, nothing)
    cv2.createTrackbar('speckle_size (2*k)',        PARAM_WIN, 3,  50,  nothing)
    cv2.createTrackbar('speckle_range',             PARAM_WIN, 0,  100, nothing)

    # StereoBM object
    stereo = cv2.StereoBM_create()

    k = 0

    # Main loop
    while True:
        # Read params from the PARAM_WIN trackbars
        numDisparities = cv2.getTrackbarPos('num_disparities (16*k)', PARAM_WIN) * 16
        if numDisparities < 16:
            numDisparities = 16

        blockSize = cv2.getTrackbarPos('block_size (2*k+5)', PARAM_WIN) * 2 + 5
        if blockSize % 2 == 0:
            blockSize += 1
        if blockSize < 5:
            blockSize = 5

        preFilterType = cv2.getTrackbarPos('prefilter_type', PARAM_WIN)
        preFilterSize = cv2.getTrackbarPos('prefilter_size (2*k+5)', PARAM_WIN) * 2 + 5
        if preFilterSize % 2 == 0:
            preFilterSize += 1
        preFilterSize = max(5, min(preFilterSize, 255))

        preFilterCap = cv2.getTrackbarPos('prefilter_cap', PARAM_WIN)
        preFilterCap = max(1, min(preFilterCap, 63))

        textureThreshold = cv2.getTrackbarPos('texture_threshold', PARAM_WIN)
        uniquenessRatio = cv2.getTrackbarPos('uniqueness_ratio', PARAM_WIN)
        speckleRange = cv2.getTrackbarPos('speckle_range', PARAM_WIN)
        speckleWindowSize = cv2.getTrackbarPos('speckle_size (2*k)', PARAM_WIN) * 2
        disp12MaxDiff = cv2.getTrackbarPos('disp12_max_diff', PARAM_WIN)
        minDisparity = cv2.getTrackbarPos('min_disparity', PARAM_WIN)

        # Apply parameters
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Compute disparity
        disparity_raw = stereo.compute(imgLG, imgRG).astype(np.float32)

        # Normalize and colorize
        disp = (disparity_raw / 16.0 - stereo.getMinDisparity()) / float(stereo.getNumDisparities())
        disp_norm = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disp_norm = np.uint8(disp_norm)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

        # Create color legend (blue→red)
        legend = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET)
        legend = cv2.resize(legend, (40, disp_color.shape[0]))
        cv2.putText(legend, 'Near', (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(legend, 'Far',  (5, legend.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Combine disparity + legend
        disp_with_legend = np.hstack((disp_color, legend))

        # Resize to fit screen
        scale = 0.6  # adjust if still too big
        display = cv2.resize(disp_with_legend, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Show window
        cv2.imshow('Disparity Map', display)

        # ESC to quit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        k += 1
        if k % 100 == 0:
            print(f'Current parameters: blockSize={blockSize}, numDisparities={numDisparities}, minDisparity={minDisparity}, preFilterType={preFilterType}, preFilterSize={preFilterSize}, preFilterCap={preFilterCap}, textureThreshold={textureThreshold}, uniquenessRatio={uniquenessRatio}, speckleWindowSize={speckleWindowSize}, speckleRange={speckleRange}, disp12MaxDiff={disp12MaxDiff}')

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
