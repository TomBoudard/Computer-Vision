#!/usr/bin/env python3
""" computer vision AR demo"""

# ------- imports ------------------------------------------------------------
import sys
import logging
import argparse as ap
import cv2
import numpy as np


# ------- argument parser ----------------------------------------------------
parser = ap.ArgumentParser(description='Simple augmented reality application.')
parser.add_argument('file', metavar='video_file', type=str, nargs='?',
                    help='a video file to apply AR on')
parser.add_argument('-o', '--output', type=str, nargs='?',
                    help='save live input stream as a video file')
parser.add_argument('-g', '--grid-size', type=int, nargs=2, default=(6, 5),
                    help='calibration grid size')
parser.add_argument('-l', '--log', choices=['NOTSET', 'DEBUG', 'INFO',
                                            'WARNING', 'ERROR', 'CRITICAL'],
                    help='Log level used to display logs', default='INFO')

args = parser.parse_args()

# logging
fmt = '[%(asctime)s.%(msecs)-3.2s%(levelname)s] %(message)s'
logging.basicConfig(level=getattr(logging, args.log),
                    format=fmt, datefmt='%H:%M:%S')
logging.info('Python version ' + sys.version)
logging.info('Arguments: ' + str(args))


# ------- draw augmentation object (cube) -------------------------------------
# function to draw our augmented scene / cube
def draw(image, points2d):
    # TODO 3: draw virtual object on real frame
    return image


# ------- main initialization and loop ----------------------------------------
def main():

    # get source video capture object
    cam = cv2.VideoCapture(0 if args.file is None else args.file)

    # define the codec & create VideoWriter object in case we write video output
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        dim = int(cam.get(3)), int(cam.get(4))  # float width and height
        out = cv2.VideoWriter(args.output, fourcc, 20.0, dim)

    # open window where we will show video stream
    cv2.namedWindow("Calibration and augmented reality demo")

    # construct array of checkerboard coordinates in object coordinate frame
    grid = args.grid_size
    object_points = [[i, j, 0] for j in range(grid[1]) for i in range(grid[0])]
    object_points = np.array(object_points, np.float32)
    print(f'{object_points=}')

    # Arrays to store object points and image points from all the images.
    points3d = []    # 3d points of calibration in its coordinate frame
    points2d = []    # matching 2d points in image plane.

    # variables that will contain calibration result
    intrinsics, distortion = None, None

    # initialize coordinates that will be used to draw the cube
    cube = np.float32([[0, 0, 0],  [0, 3, 0],  [3, 3, 0],  [3, 0, 0],
                       [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

    print('\n\nWave the calibration checkboard to the camera for a few frames, '
          'then press C to calibrate and see AR cube on checkerboard.\n')
    print('Press ESC to close.')

    # corner refinement termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image_counter = 0
    scale = 0.3

    # main loop ---------------------------------------------------------------
    while cam.isOpened():
        # read an image and get return code
        ret, image = cam.read()

        # exit loop if we reach the end of stream
        if not ret:
            break

        # read key
        k = cv2.waitKey(1)

        # gray image conversion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # examine key events
        if k & 0xFF == 27:
            # exit if ESC pressed
            logging.info("Escape hit, closing...")
            break
        elif k & 0xFF == ord('c') and intrinsics is None:
            # when user presses 'c', perform calibration
            # TODO 2.1: use the point3d <-> point2d matches to calibrate
            # _, intrinsics, distortion, _, _ = ...

            logging.info('Calibration done')
            logging.info(f'Estimated distortion\n{distortion}')
            logging.info(f'Estimated intrinsics matrix\n{intrinsics}')

        if out:
            out.write(image)

        # TODO 1.1: chessboard detection on scaled down image for improved speed
        ret, corners = cv2.findChessboardCorners(image, grid)

        if ret:
            # TODO 1.3: if corners found, get sub-pixel refinement of corners
            corners = cv2.cornerSubPix(image, corners, )

            image_counter += 1
            if intrinsics is None:    # calibration ongoing

                # only take one in 30 frames for calibration to avoid overload
                if image_counter % 30 == 0:
                    # TODO 2.1: append each new point2d detection, and append
                    # the corresponding object point coordinates each time

                    # show frames used for calibration as inverted
                    image = cv2.bitwise_not(image)

                # TODO 1.2: draw and display the corners
                image = cv2.drawChessboardCorners(image, grid, corners, ret)

            else:
                # now that camera is calibrated, we can estimate extrinsics
                # TODO 2.2  find rotation and translation vectors.
                ret, rotation_vectors, translation_vectors, _ = ...

                # TODO 2.3 project 3D points to image using estimated parameters
                projected_points, jacobian = ...

                image = draw(image, projected_points)

        cv2.imshow('img', image)

    # finalize and release stream objects, input and output when applicable
    cam.release()
    if out:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
