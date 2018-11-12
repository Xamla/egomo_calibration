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
import python.Calibration as calib
from python.xamla3d import Xamla3d

M_PI = 3.14159265359
helpers = Xamla3d()

imWidth = 1920
imHeight = 1200
intrinsic = np.zeros(shape=(3,3), dtype=np.float64)
intrinsic[0][0] = 4435.5590
intrinsic[1][1] = 4435.8500
intrinsic[0][2] = 986.0854 # ~ imWidth / 2 = 960
intrinsic[1][2] = 599.7553 # ~ imHeight / 2 = 600
intrinsic[2][2] = 1.0
distCoeffs = np.zeros(shape=(5,1), dtype=np.float64) # torch.zeros(5,1)
distCoeffs[0] = -0.0214
distCoeffs[1] = 0.6362

calibration_path = "../../calibration_rand50/stereo_cams_4103130811_4103189394.npy"
stereoCalib = np.load(calibration_path).item()

intrinsic = stereoCalib['camLeftMatrix']
distCoeffs = stereoCalib['camLeftDistCoeffs']
print("intrinsic:")
print(intrinsic)
print(intrinsic.dtype)

heye_path = "../../calibration_rand50/HandEye.npy"
hand_eye = np.load(heye_path)
#hand_eye = np.identity(4, dtype=np.float64)
#hand_eye[0:2,0:2] = np.identity(3)
#hand_eye[0][0] = -1.0
#hand_eye[1][1] = -1.0
#hand_eye[0][3] = -0.04
#hand_eye[1][3] = -0.06
#hand_eye[2][3] =  0.10
print("hand_eye:")
print(hand_eye)
print(hand_eye.dtype)

hand_eye_original = deepcopy(hand_eye)

#torch_Hc_path = "/home/inga/Rosvita/projects/SDA10D_left_arm_onboardCams/calibration/current_20_without_12/Hc_patternToCam.t7"
#Hc = torchfile.load(torch_Hc_path).astype(np.float64)

def createMotomanRobotModel(theta, d, a, alpha):

  dh = np.zeros((4,8))
  dh[0] = theta
  dh[1] = d
  dh[2] = a
  dh[3] = alpha

  # TODO: Adapt reflect = 1 for left arm to reflect = -1 for right arm!!!
  joint_direction = np.ones(8) # { 1, -1, -1, -1, 1, 1, 1, 1 }
  joint_direction[1] = -1.0
  joint_direction[2] = -1.0
  joint_direction[3] = -1.0
  
  result_table = { 'dh': dh, 'joint_direction': joint_direction }
  return result_table


theta = np.zeros(8, dtype=np.float64)
alpha = np.zeros(8, dtype=np.float64)
d = np.zeros(8, dtype=np.float64)
a = np.zeros(8, dtype=np.float64)
d[0] = 0.3; d[1] = -0.2645; d[3] = -0.36; d[5] = -0.36; d[7] = -0.175
a[0] = 0.09996
alpha[0] = M_PI/2.0; alpha[1] = -M_PI/2.0; alpha[2] = M_PI/2.0; alpha[3] = -M_PI/2.0
alpha[4] = M_PI/2.0; alpha[5] = -M_PI/2.0; alpha[6] = M_PI/2.0

robot_model = createMotomanRobotModel(theta, d, a, alpha)
print("robot_model:")
print(robot_model)
#print(robot_model["dh"].dtype)
#print(robot_model["joint_direction"].dtype)

stereo = True
pattern = { "width": 8, "height": 21, "circleSpacing": 0.005 }
robotCalibration = calib(pattern, imWidth, imHeight, hand_eye, robot_model, stereo)
robotCalibration.intrinsics = intrinsic
robotCalibration.distCoeffs = distCoeffs # np.zeros(5,1)
robotCalibration.stereoCalib = stereoCalib

# load images (etc.) to robotCalibration
images = []
imagesRight = []
for i in range(0, 50):
#for i in range(0, 25): # 42):
  image_fn = "../../calibration_rand50/capture_all/cam_4103130811_{:03d}.png".format(i+1)
  image_right_fn = "../../calibration_rand50/capture_all/cam_4103189394_{:03d}.png".format(i+1)
  image = cv.imread(image_fn)
  image_right = cv.imread(image_right_fn)
  images.append(image)
  imagesRight.append(image_right)

jsposes_fn = "../../calibration_rand50/jsposes_tensors.npy"
jsposes = np.load(jsposes_fn).item()
all_vals_tensors_fn = "../../calibration_rand50/all_vals_tensors.npy"
all_vals_tensors = np.load(all_vals_tensors_fn)
robotPoses = []
jointValues = []
for i in range(0, len(images)):
  #robotPose = jsposes[b'recorded_poses'][i]
  robotPose = jsposes['recorded_poses'][i]
  robotPoses.append(robotPose)
  jointVals = np.zeros(8)
  #jointVals[0] = -0.8168959021514733 # see world view
  jointVals[0] = all_vals_tensors[0]
  jointVals[1:8] = jsposes['recorded_joint_values'][i]
  jointValues.append(jointVals)

patternId = 21

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
not_found.append(47)
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
