#!/usr/bin/env python

import numpy as np
import sys
import cv2 as cv
import os
import math
import torchfile
from matplotlib import pyplot as plt

from numpy.linalg import inv
from numpy.linalg import norm

from copy import copy, deepcopy

import python.pattern_localisation as pattern_localisation
import python.Calibration_rightArm_onTorsoCams as calib
from python.xamla3d import Xamla3d

M_PI = 3.14159265359
helpers = Xamla3d()

imWidth = 1920
imHeight = 1200
intrinsic = np.zeros(shape=(3,3), dtype=np.float64)
distCoeffs = np.zeros(shape=(5,1), dtype=np.float64) # torch.zeros(5,1)

calibration_path = "../../right_arm_data/calib_42_20_50/stereo_cams_CAMAU1639042_CAMAU1710001.npy"
#calibration_path = "../../right_arm_data/calib_after_27_21/stereo_cams_CAMAU1639042_CAMAU1710001.npy"
#calibration_path = "../../right_arm_data/calib_before_42/stereo_cams_CAMAU1639042_CAMAU1710001.npy"
stereoCalib = np.load(calibration_path).item()
intrinsic = stereoCalib['camLeftMatrix']
distCoeffs = stereoCalib['camLeftDistCoeffs']
print("intrinsic:")
print(intrinsic)
print(intrinsic.dtype)
print("distCoeffs:")
print(distCoeffs)

hand_pattern_path = "../../right_arm_data/calib_42_20_50/HandPattern.npy"
#hand_pattern_path = "../../right_arm_data/calib_42_20_50/HandPattern_optimized_rightArm_42_20_50.npy"
#hand_pattern_path = "../../right_arm_data/calib_after_27_21/HandPattern_optimized_rightArm_42_20_50.npy"
#hand_pattern_path = "../../right_arm_data/calib_before_42/HandPattern.npy"
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

  # TODO: Adapt reflect = 1 for left arm to reflect = -1 for right arm!!!  
  joint_direction = np.ones(8) # { 1, -reflect, -reflect, -reflect, reflect, reflect, reflect, reflect }
  # left:  reflect =  1  => {  1, -1, -1, -1,  1,  1,  1,  1 }
  # right: reflect = -1  => {  1,  1,  1,  1, -1, -1, -1, -1 }
  # left:
  #joint_direction[0] =  1.0
  #joint_direction[1] = -1.0
  #joint_direction[2] = -1.0
  #joint_direction[3] = -1.0
  #joint_direction[4] =  1.0
  #joint_direction[5] =  1.0
  #joint_direction[6] =  1.0
  #joint_direction[7] =  1.0
  # right:
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
alpha[4] = M_PI/2.0; alpha[5] = -M_PI/2.0; alpha[6] = M_PI/2.0

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
#print(robot_model["dh"].dtype)
#print(robot_model["joint_direction"].dtype)

stereo = True
pattern = { "width": 8, "height": 21, "circleSpacing": 0.005 }
robotCalibration = calib(pattern, imWidth, imHeight, hand_pattern, robot_model, stereo)
robotCalibration.intrinsics = intrinsic
robotCalibration.distCoeffs = distCoeffs # np.zeros(5,1)
robotCalibration.stereoCalib = stereoCalib

# load images (etc.) to robotCalibration
images = []
imagesRight = []
for i in range(0, 110):
#for i in range(0, 42) : #48) : #27) : #110) : #42) : #61) :
  image_fn = "../../right_arm_data/calib_42_20_50/capture_all/cam_CAMAU1639042_{:03d}.png".format(i+1)
  image_right_fn = "../../right_arm_data/calib_42_20_50/capture_all/cam_CAMAU1710001_{:03d}.png".format(i+1)
  #image_fn = "../../right_arm_data/calib_after_27/capture_all/cam_CAMAU1639042_{:03d}.png".format(i+1)
  #image_right_fn = "../../right_arm_data/calib_after_27/capture_all/cam_CAMAU1710001_{:03d}.png".format(i+1)
  #image_fn = "../../right_arm_data/calib_after_27_21/capture_all/cam_CAMAU1639042_{:03d}.png".format(i+1)
  #image_right_fn = "../../right_arm_data/calib_after_27_21/capture_all/cam_CAMAU1710001_{:03d}.png".format(i+1)
  #image_fn = "../../right_arm_data/calib_before_42/capture_all/cam_CAMAU1639042_{:03d}.png".format(i+1)
  #image_right_fn = "../../right_arm_data/calib_before_42/capture_all/cam_CAMAU1710001_{:03d}.png".format(i+1)
  image = cv.imread(image_fn)
  image_right = cv.imread(image_right_fn)
  images.append(image)
  imagesRight.append(image_right)

jsposes_fn = "../../right_arm_data/calib_42_20_50/jsposes_tensors.npy"
#jsposes_fn = "../../right_arm_data/calib_42_20_50/js_new.npy"
#jsposes_fn = "../../right_arm_data/calib_after_27/jsposes_tensors.npy"
#jsposes_fn = "../../right_arm_data/calib_after_27_21/jsposes_tensors.npy"
#jsposes_fn = "../../right_arm_data/calib_before_42/jsposes_tensors.npy"
jsposes = np.load(jsposes_fn).item()
#all_vals_tensors_fn = "../../right_arm_data/calib_after_27/all_vals_tensors.npy"
#all_vals_tensors_fn = "../../right_arm_data/calib_before_42/all_vals_tensors.npy"
#all_vals_tensors = np.load(all_vals_tensors_fn)
#print("all_vals_tensors[0]:")
#print(all_vals_tensors[0])
#all_vals_tensors_27_fn = "../../right_arm_data/calib_after_27_21/all_vals_tensors_27.npy"
#all_vals_tensors_21_fn = "../../right_arm_data/calib_after_27_21/all_vals_tensors_21.npy"
#all_vals_tensors_27 = np.load(all_vals_tensors_27_fn)
#all_vals_tensors_21 = np.load(all_vals_tensors_21_fn)
#print("all_vals_tensors_27[0]:")
#print(all_vals_tensors_27[0])
#print("all_vals_tensors_21[0]:")
#print(all_vals_tensors_21[0])
all_vals_tensors_42_fn = "../../right_arm_data/calib_42_20_50/all_vals_tensors_42.npy"
all_vals_tensors_20_fn = "../../right_arm_data/calib_42_20_50/all_vals_tensors_20.npy"
all_vals_tensors_50_fn = "../../right_arm_data/calib_42_20_50/all_vals_tensors_50.npy"
all_vals_tensors_42 = np.load(all_vals_tensors_42_fn)
all_vals_tensors_20 = np.load(all_vals_tensors_20_fn)
all_vals_tensors_50 = np.load(all_vals_tensors_50_fn)
print("all_vals_tensors_42[0]:")
print(all_vals_tensors_42[0]) # 1.5752789974213
print("all_vals_tensors_20[0]:")
print(all_vals_tensors_20[0]) # 1.5753470659256
print("all_vals_tensors_50[0]:")
print(all_vals_tensors_50[0]) # 1.5752619504929
robotPoses = []
jointValues = []
for i in range(0, len(images)):
  #robotPose = jsposes[b'recorded_poses'][i]
  robotPose = jsposes['recorded_poses'][i]
  robotPoses.append(robotPose)
  jointVals = np.zeros(8)
  #jointVals[0] = all_vals_tensors[0]
  #if i in range(0, 27):
  #  jointVals[0] = all_vals_tensors_27[0]
  #elif i in range(27, len(images)):
  #  jointVals[0] = all_vals_tensors_21[0]
  if i in range(0, 42):
    #print("First torso joint val i:")
    #print("========================")
    #print(i)
    jointVals[0] = all_vals_tensors_42[0]
  elif i in range(42, 61): # only 19, because the 1st is not taken/redundant! (42+19=61)
    #print("Second torso joint val i:")
    #print("=========================")
    #print(i)
    jointVals[0] = all_vals_tensors_20[0]
  elif i in range(61, len(images)): # only 49, because the 1st is not taken/redundant! (61+49=110)
    #print("Third torso joint val i:")
    #print("========================")
    #print(i)
    jointVals[0] = all_vals_tensors_50[0]
  jointVals[1:8] = jsposes['recorded_joint_values'][i]
  jointValues.append(jointVals)

patternId = 22

points = []
pointsRight = []
not_found = []
patternSize = (8, 21)
for i in range(0, len(images)):
  found1, point = helpers.findPattern(images[i], cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, patternSize)
  if found1 :
    points.append(point)
  else :
    print("Pattern points could not be found for image {:03d}!!!".format(i))
    points.append("not found")
  found2, point_right = helpers.findPattern(imagesRight[i], cv.CALIB_CB_ASYMMETRIC_GRID, patternSize)
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
  # Scatter hand_eye:

for i in range(0, len(images)):
  flag = 0
  for j in range(0, len(not_found)):
    if i == not_found[j] :
      print("Skip image {:d}.".format(i))
      flag = 1
  if flag == 0 :
    ok = False
    ok = robotCalibration.addStereoImage(images[i], imagesRight[i], robotPoses[i], jointValues[i], patternId, points[i], pointsRight[i])
    #ok = robotCalibration.addImage(images[i], robotPoses[i], jointValues[i], patternId, points[i])
    if not ok :
      print("addImage failed for image {:d}!!!".format(i))

robotCalibration.DHCrossValidate(1.0, 1)
