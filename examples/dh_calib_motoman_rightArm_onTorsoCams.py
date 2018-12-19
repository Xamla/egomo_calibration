#!/usr/bin/env python

import numpy as np
import sys
import cv2 as cv
import os
import math
import torchfile
import xacro
import xml
from matplotlib import pyplot as plt

from numpy.linalg import inv
from numpy.linalg import norm

from copy import copy, deepcopy

import python.pattern_localisation as pattern_localisation
import python.Calibration_rightArm_onTorsoCams as calib
from python.xamla3d import Xamla3d

M_PI = 3.14159265359
helpers = Xamla3d()

imWidth = 1936
imHeight = 1216
intrinsic = np.zeros(shape=(3,3), dtype=np.float64)
distCoeffs = np.zeros(shape=(5,1), dtype=np.float64)

calibration_path = None
hand_pattern_path = None
number_of_images = None
left_images_path = None
right_images_path = None
joints_path = None
all_joints_path = None
all_joints_path2 = None
cut1 = None
all_joints_path3 = None
cut2 = None

print('Number of arguments:')
print(len(sys.argv))
print('Argument List:')
print(str(sys.argv))
if len(sys.argv) > 1:
  calibration_path = sys.argv[1]
if len(sys.argv) > 2:
  hand_pattern_path = sys.argv[2]
if len(sys.argv) > 3:
  number_of_images = int(sys.argv[3])
if len(sys.argv) > 4:
  left_images_path = sys.argv[4]
if len(sys.argv) > 5:
  right_images_path = sys.argv[5]
if len(sys.argv) > 6:
  joints_path = sys.argv[6]
if len(sys.argv) > 7:
  all_joints_path = sys.argv[7]
if len(sys.argv) > 8:
  all_joints_path2 = sys.argv[8]
if len(sys.argv) > 9:
  cut1 = int(sys.argv[9])
if len(sys.argv) > 10:
  all_joints_path3 = sys.argv[10]
if len(sys.argv) > 11:
  cut2 = int(sys.argv[11])

stereoCalib = np.load(calibration_path).item()
intrinsic = stereoCalib['camLeftMatrix']
distCoeffs = stereoCalib['camLeftDistCoeffs']
print("intrinsic:")
print(intrinsic)
print(intrinsic.dtype)
print("distCoeffs:")
print(distCoeffs)

hand_pattern = np.load(hand_pattern_path)
print("hand_pattern:")
print(hand_pattern)
print(hand_pattern.dtype)

hand_pattern_original = deepcopy(hand_pattern)


def createMotomanRobotModel(theta, d, a, alpha):

  dh = np.zeros((4,8))
  dh[0] = theta
  dh[1] = d
  dh[2] = a
  dh[3] = alpha

  joint_direction = np.ones(8) # { 1, -reflect, -reflect, -reflect, reflect, reflect, reflect, reflect }
  # left:  reflect =  1  => {  1, -1, -1, -1,  1,  1,  1,  1 }
  # right: reflect = -1  => {  1,  1,  1,  1, -1, -1, -1, -1 }
  joint_direction[0] =  1.0
  joint_direction[1] =  1.0
  joint_direction[2] =  1.0
  joint_direction[3] =  1.0
  joint_direction[4] = -1.0
  joint_direction[5] = -1.0
  joint_direction[6] = -1.0
  joint_direction[7] = -1.0

  result_table = { 'dh': dh, 'joint_direction': joint_direction }
  return result_table


theta = np.zeros(8, dtype=np.float64)
alpha = np.zeros(8, dtype=np.float64)
d = np.zeros(8, dtype=np.float64)
a = np.zeros(8, dtype=np.float64)
#a[0] = 0.09996 # left arm
a[0] = -0.09996 # right arm
d[0] = 0.3; d[1] = -0.2645; d[3] = -0.36; d[5] = -0.36; d[7] = -0.175
#theta[0] = 0.0 # left arm
theta[0] = -M_PI # right arm
alpha[0] = M_PI/2.0; alpha[1] = -M_PI/2.0; alpha[2] = M_PI/2.0; alpha[3] = -M_PI/2.0
alpha[4] = M_PI/2.0; alpha[5] = -M_PI/2.0; alpha[6] = M_PI/2.0; alpha[7] = M_PI

# optimization result:
#theta[1] = -1.19834558e-03 #-0.00119835
#theta[2] = -8.00673093e-04 #-0.000800673 
#theta[3] =  1.56595301e-03 #0.00156595 
#theta[4] =  3.45036065e-04 #0.000345036 
#theta[5] =  2.26587275e-03 #0.00226587 
#theta[6] = -1.10274794e-03 #-0.00110275 
#theta[7] = -4.52378065e-03 #-0.00452378 

robot_model = createMotomanRobotModel(theta, d, a, alpha)
print("robot_model:")
print(robot_model)

stereo = True
pattern = { "width": 8, "height": 21, "circleSpacing": 0.005 }
robotCalibration = calib(pattern, imWidth, imHeight, hand_pattern, robot_model, stereo)
robotCalibration.intrinsics = intrinsic
robotCalibration.distCoeffs = distCoeffs
robotCalibration.stereoCalib = stereoCalib

# load images (etc.) to robotCalibration
imagesLeft = []
imagesRight = []
for i in range(0, number_of_images) :
  image_left_fn = left_images_path + "_{:03d}.png".format(i+1)
  image_right_fn = right_images_path + "_{:03d}.png".format(i+1)
  image_left = cv.imread(image_left_fn)
  image_right = cv.imread(image_right_fn)
  imagesLeft.append(image_left)
  imagesRight.append(image_right)

jsposes = np.load(joints_path).item()
all_vals_tensors = np.load(all_joints_path)
print("all_vals_tensors[0]:")
print(all_vals_tensors[0])
if all_joints_path2 is not None :
  all_vals_tensors2 = np.load(all_joints_path2)
  print("all_vals_tensors2[0]:")
  print(all_vals_tensors2[0])
if all_joints_path3 is not None :
  all_vals_tensors3 = np.load(all_joints_path3)
  print("all_vals_tensors3[0]:")
  print(all_vals_tensors3[0])

robotPoses = []
jointValues = []
for i in range(0, len(imagesLeft)):
  robotPose = jsposes['recorded_poses'][i]
  robotPoses.append(robotPose)
  jointVals = np.zeros(8)
  jointVals[0] = all_vals_tensors[0]
  if all_joints_path2 is not None and i in range(cut1, len(imagesLeft)) :
    jointVals[0] = all_vals_tensors2[0]
  if all_joints_path3 is not None and i in range(cut2, len(imagesLeft)) :
    jointVals[0] = all_vals_tensors2[0]
  jointVals[1:8] = jsposes['recorded_joint_values'][i]
  jointValues.append(jointVals)

patternId = 22

points = []
pointsRight = []
not_found = []
patternSize = (8, 21)
for i in range(0, len(imagesLeft)):
  found1, point = helpers.findPattern(imagesLeft[i], cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, patternSize)
  if found1 :
    points.append(point)
  else :
    print("Pattern points could not be found for image {:03d}!!!".format(i))
    points.append("not found")
  found2, point_right = helpers.findPattern(imagesRight[i], cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, patternSize)
  if found2 :
    pointsRight.append(point_right)
  else :
    print("Pattern points could not be found for right image {:03d}!!!".format(i))
    pointsRight.append("not found")
  
  if not (found1 and found2) :
    not_found.append(i)

print("len(points):")
print(len(points))
print("Indices of images, in which the pattern could not be found:")
print(not_found)

# TODO: add scatter here!!!
# for runs in range (0, 9):
  # Scatter initial hand_eye and dh-Parameters:

for i in range(0, len(imagesLeft)):
  flag = 0
  for j in range(0, len(not_found)):
    if i == not_found[j] :
      print("Skip image {:d}.".format(i))
      flag = 1
  if flag == 0 :
    ok = False
    ok = robotCalibration.addStereoImage(imagesLeft[i], imagesRight[i], robotPoses[i], jointValues[i], patternId, points[i], pointsRight[i])
    #ok = robotCalibration.addImage(imagesLeft[i], robotPoses[i], jointValues[i], patternId, points[i])
    if not ok :
      print("addImage failed for image {:d}!!!".format(i))

robotCalibration.DHCrossValidate(1.0, 1)
