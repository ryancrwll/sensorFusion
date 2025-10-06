
import argparse
import cv2
import numpy
import os
import time

def capture_stereo_images(port, out_dir, continous):

    # Create the output directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    left_dir = os.path.join(out_dir, 'left')
    if not os.path.exists(left_dir):
        os.makedirs(left_dir)
    right_dir = os.path.join(out_dir, 'right')
    if not os.path.exists(right_dir):
        os.makedirs(right_dir)    

    # Open the ZED camera
    cap = cv2.VideoCapture(port)
    if cap.isOpened() == 0:
        print("[ERROR] Cannot open camera at port " + str(port))
        exit(-1)

    # Set the video resolution to HD1080 (3840*1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("*** Press 'c' to capture the current stereo pair, ESC to exit ***")

    image_id = 0
    while True :
        # Get a new frame from camera
        retval, frame = cap.read()
        # Extract left and right images from side-by-side
        left_right_image = numpy.split(frame, 2, axis=1)
        # Display images
        cv2.namedWindow("right", cv2.WINDOW_GUI_EXPANDED)
        cv2.namedWindow("left", cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("left", left_right_image[0])
        cv2.imshow("right", left_right_image[1])
        if continous < 0:
            k = cv2.waitKey(30)
        else:
            k = cv2.waitKey(round(continous*1000))
        if k == 27: # ESC key, exit
            break
        elif continous > 0 or k == ord('c'):
            left_img_name = f'left_{image_id:05d}.png'
            right_img_name = f'right_{image_id:05d}.png'
            cv2.imwrite(os.path.join(left_dir, left_img_name), left_right_image[0])
            cv2.imwrite(os.path.join(right_dir, right_img_name), left_right_image[1])
            image_id = image_id+1
            print("- Captured stereo pair " + str(image_id))           
    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Captures stereo images from a Zed camera")
    parser.add_argument('-p', dest='port', action='store', type=int, default=0,
                        help='The port corresponding to the Zed camera (see list_ports.py)')
    parser.add_argument('-c', dest='continous', action='store', type=float, default=-1.0, help="Continously capture images every this number of seconds")              
    parser.add_argument('out_dir', action='store', type=str,
                        help='The output directory where images will be stored')                        
    params = parser.parse_args()

    capture_stereo_images(params.port, params.out_dir, params.continous)