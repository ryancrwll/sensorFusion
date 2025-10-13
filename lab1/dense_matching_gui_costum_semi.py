#!/usr/bin/env python
import numpy as np
import argparse
import cv2
import open3d as o3d
import open3d.visualization as vis

def main():
    parser = argparse.ArgumentParser(description="Stereo dense matching GUI (SGBM)")
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

    # Visualization window
    cv2.namedWindow('Disparity Map', cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('Disparity Map', 800, 600)

    # Parameter window
    PARAM_WIN = 'StereoSGBM Params'
    cv2.namedWindow(PARAM_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(PARAM_WIN, 420, 480)

    # Trackbars for main parameters
    cv2.createTrackbar('block_size (odd)', PARAM_WIN, 5, 15, nothing)
    cv2.createTrackbar('num_disparities (16*k)', PARAM_WIN, 2, 25, nothing)
    cv2.createTrackbar('min_disparity', PARAM_WIN, 0, 200, nothing)
    cv2.createTrackbar('uniqueness_ratio', PARAM_WIN, 10, 50, nothing)
    cv2.createTrackbar('speckle_size', PARAM_WIN, 100, 200, nothing)
    cv2.createTrackbar('speckle_range', PARAM_WIN, 32, 64, nothing)
    cv2.createTrackbar('disp12_max_diff', PARAM_WIN, 1, 25, nothing)

    k = 0

    while True:
        # Read params
        numDisparities = cv2.getTrackbarPos('num_disparities (16*k)', PARAM_WIN) * 16
        if numDisparities < 16:
            numDisparities = 16

        blockSize = cv2.getTrackbarPos('block_size (odd)', PARAM_WIN)
        if blockSize % 2 == 0:
            blockSize += 1
        if blockSize < 3:
            blockSize = 3

        minDisparity = cv2.getTrackbarPos('min_disparity', PARAM_WIN)
        uniquenessRatio = cv2.getTrackbarPos('uniqueness_ratio', PARAM_WIN)
        speckleWindowSize = cv2.getTrackbarPos('speckle_size', PARAM_WIN)
        speckleRange = cv2.getTrackbarPos('speckle_range', PARAM_WIN)
        disp12MaxDiff = cv2.getTrackbarPos('disp12_max_diff', PARAM_WIN)

        # Compute penalties for smoothness
        P1 = 8 * 3 * blockSize ** 2
        P2 = 32 * 3 * blockSize ** 2

        # Create the SGBM matcher with current parameters
        stereo = cv2.StereoSGBM_create(
            minDisparity=minDisparity,
            numDisparities=numDisparities,
            blockSize=blockSize,
            P1=P1,
            P2=P2,
            disp12MaxDiff=disp12MaxDiff,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute disparity
        disparity_raw = stereo.compute(imgLG, imgRG).astype(np.float32) / 16.0

        # Normalize for visualization
        disp_norm = cv2.normalize(disparity_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disp_norm = np.uint8(disp_norm)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

        # Add color legend (Near→Far)
        legend = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET)
        legend = cv2.resize(legend, (40, disp_color.shape[0]))
        cv2.putText(legend, 'Near', (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(legend, 'Far', (5, legend.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        disp_with_legend = np.hstack((disp_color, legend))
        display = cv2.resize(disp_with_legend, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        cv2.imshow('Disparity Map', display)

        # Exit with ESC
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        k += 1
        if k % 60 == 0:
            print(f'[SGBM] block={blockSize}, numDisp={numDisparities}, minDisp={minDisparity}, uniq={uniquenessRatio}, speckleWin={speckleWindowSize}, range={speckleRange}')

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
