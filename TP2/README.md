Computer Vision and Pattern Recognition
=========================================

Practical 2: Augmented reality with OpenCV
=========================================

### Goals
- discover a widely used computer vision software library (OpenCV)
- perform chessboard-based calibration with OpenCV
- write a basic augmented reality pipeline


### Overview

In the context of augmented reality and interaction with videos,
we often use specific known and easily detectable objects. 
Those objects, once detected, can serve as an 
anchor for interaction and allow the computer to make the link between 
the real world and a synthetic world that it manages.
For this practical we will be using a checkerboard pattern.

To insert virtual objects in the real image, we first need to calibrate
the camera,  i.e. infer its *intrinsics* matrix (see the lecture), 
then for each interaction frame
estimate the camera position or *camera extrinsics* given a fixed solution to 
the intrinsics. The latter is called the *Perspective-n-Point (PnP)* problem. 

The purpose of this TD is to achieve both tasks using OpenCV tools.
The checkerboard will be detected and positioned in 3D as part of the practical:
we can then draw 3D objects *in the coordinate system of the checkerboard object* 
and thus give the illusion that the virtual 3D object is in the real world 
(augmented reality). A source code skeleton is supplied, to be completed. 
 
Videos are made available to do the manipulations offline 
(on Ensimag machines in folder `/matieres/5MMVORF/02-dataset/`).
You can copy videos on the local disk (`/tmp`) to avoid 
unnecessarily overloading of the NFS.

### OpenCV functions of interest 

**findChessboardCorners** allows you to analyze an image to extract a rough 
estimate of the location of the projected image points of the checkerboard. 

**cornerSubPix** allows you to refine the first estimation of the location of the projected
points of the checkerboard in the image, starting with the previous estimate.

**drawChessboardCorners** allows you to draw the checkerboard detected in
a color image.

**calibrateCamera** allows, from a series of detected points of the
checkerboard at different times, to find the intrinsic camera settings,
as well as all extrinsic parameters associated with the successive positions 
of the checkerboard.

**solvePnPRansac** allows, once the calibration carried out,
to find the pose (extrinsic params) associated to the checkerboard, 
from the checkerboard image points and intrinsic settings already calculated
in the previous step.

**projectPoints** allows, from the known extrinsic and intrinsic parameters, 
to project 3D points of the object space in the image: this will allow in 
particular to draw a 3D virtual object that appears to be related to the
checkerboard.

Learn more about these functions by consulting the online 
[OpenCV documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
and see how they fit together in the code.


### Part 1: detection

We look here at detecting, refining, and drawing the projected image points of
the checkerboard from a captured image. 

A point data structure is initialized in size, for storing points.

1. Implement simple checkerboard spot detection with
   **findChessboardCorners** on grayscale version of the input 
   image.
2. Draw the checkerboard points on the color input image with 
   **drawChessboardCorners**.
3. The points from question 1 can be improved in precision by 
   refining based on a local window around the initial detections. 
   To this goal, before drawing the checkerboard points, add the refinement step
   using **cornerSubPix**.

### Part 2: calibration, positioning

1. Calibrate the camera using **calibrateCamera**. For this purpose, you need 
   to stack 3D to 2D matches at every detection that will be given
   to this function. 
2. Find the camera extrinsics using **solvePnPRansac**.
3. Use **projectPoints** to find the 2d reprojections of the 3D points using
   the estimated intrinsics and extrinsics to check that the result is correct.

### Part 3: drawing using reprojected points

You now have all the ingredients to draw a virtual object on the input image.
The coordinates of a 3D cube are provided which you can use as initial 
augmentation object, but you can draw whatever you wish. Use **circle**, 
**line**, **drawContours** for this purpose.

### Going further (optional)

The whole application relies on a known checkerboard grid, but the same thing 
could be done with an arbitrary printed image using arbitrary point matching.
Keypoints can be detected with SIFT, SURF or ORB (a patent-free alternative 
to the previous two) on the base image as initialization, then in the
interaction images with the user waving the printed image on-screen, instead
of *findChessboardCorners*. 
2D-3D matches can be obtained with keypoint matching and fed to the same 
functions *calibrateCamera*, *calibrateCamera*, *projectPoints*,
as previously explored. 

   
