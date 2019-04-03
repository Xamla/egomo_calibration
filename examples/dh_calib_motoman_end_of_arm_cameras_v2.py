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
import random

import python.pattern_localisation as pattern_localisation
import python.CalibrationV2 as calib
from python.xamla3d import Xamla3d

M_PI = 3.14159265359
helpers = Xamla3d()

imWidth = 1920
imHeight = 1200
intrinsic = np.zeros(shape=(3,3), dtype=np.float64)
distCoeffs = np.zeros(shape=(5,1), dtype=np.float64)

calibration_path = None
heye = None
number_of_images = None
left_images_path = None
right_images_path = None
joints_path = None
all_joints_path = None
patternId = 21
output_folder = None
output_robotModel_filename = None
output_handEye_filename = None
which_arm = None
alternate = False
runs = 1
train_test_split_percentage = 1.0
with_torso_movement_in_data = False
with_torso_optimization = False # makes only sense, if with_torso_movement_in_data is True
evaluate_only = False
with_jitter = False

print('Number of arguments:')
print(len(sys.argv))
print('Argument List:')
print(str(sys.argv))
if len(sys.argv) > 1:
  calibration_path = sys.argv[1]
if len(sys.argv) > 2:
  heye_path = sys.argv[2]
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
  patternId = int(sys.argv[8])
if len(sys.argv) > 9:
  output_folder = sys.argv[9]
if len(sys.argv) > 10:
  output_robotModel_filename = sys.argv[10]
if len(sys.argv) > 11:
  output_handEye_filename = sys.argv[11]
if len(sys.argv) > 12:
  which_arm = sys.argv[12]
if len(sys.argv) > 13:
  alternate = eval(sys.argv[13])
if len(sys.argv) > 14:
  runs = int(sys.argv[14])
if len(sys.argv) > 15:
  train_test_split_percentage = float(sys.argv[15])
if len(sys.argv) > 16:
  with_torso_movement_in_data = eval(sys.argv[16])
if len(sys.argv) > 17:
  with_torso_optimization = eval(sys.argv[17])
if len(sys.argv) > 18:
  evaluate_only = eval(sys.argv[18])
if len(sys.argv) > 19:
  with_jitter = eval(sys.argv[19])

stereoCalib = np.load(calibration_path).item()
intrinsic = stereoCalib['camLeftMatrix']
distCoeffs = stereoCalib['camLeftDistCoeffs']
print("intrinsic:")
print(intrinsic)
print(intrinsic.dtype)
hand_eye = np.load(heye_path)
print("hand_eye:")
print(hand_eye)
print(hand_eye.dtype)
hand_eye_original = deepcopy(hand_eye)
print("which_arm:")
print(which_arm)



def createMotomanRobotModel(theta, d, a, alpha, which_arm):
  dh = np.zeros((4,8))
  dh[0] = theta
  dh[1] = d
  dh[2] = a
  dh[3] = alpha
  joint_direction = np.ones(8) # { 1, -reflect, -reflect, -reflect, reflect, reflect, reflect, reflect }
  # left:  reflect =  1  => {  1, -1, -1, -1,  1,  1,  1,  1 }
  # right: reflect = -1  => {  1,  1,  1,  1, -1, -1, -1, -1 }
  if which_arm == "left" :
    joint_direction[1] = -1.0
    joint_direction[2] = -1.0
    joint_direction[3] = -1.0
  elif which_arm == "right" :
    joint_direction[0] =  1.0
    joint_direction[1] =  1.0
    joint_direction[2] =  1.0
    joint_direction[3] =  1.0
    joint_direction[4] = -1.0
    joint_direction[5] = -1.0
    joint_direction[6] = -1.0
    joint_direction[7] = -1.0
  else :
    print("Please choose arm \"left\" or \"right\" in run file.")
  result_table = { 'dh': dh, 'joint_direction': joint_direction }
  return result_table


theta = np.zeros(8, dtype=np.float64)
alpha = np.zeros(8, dtype=np.float64)
d = np.zeros(8, dtype=np.float64)
a = np.zeros(8, dtype=np.float64)

print("Please choose initial dh-parameter setting.")
print("1: Default values for SDA10D")
print("2: Default values for SDA10F")
print("3: Read from URDF")
choice = input("")
if choice == "1" :
  d[0] = 0.3; d[1] = -0.2645; d[3] = -0.36; d[5] = -0.36; d[7] = -0.175
  if which_arm == "left" :
    a[0] = 0.09996
    theta[0] = 0.0
  elif which_arm == "right" :
    a[0] = -0.09996
    theta[0] = -M_PI
  else :
    print("Please choose arm \"left\" or \"right\" in run file.")
  alpha[0] = M_PI/2.0; alpha[1] = -M_PI/2.0; alpha[2] = M_PI/2.0; alpha[3] = -M_PI/2.0
  alpha[4] = M_PI/2.0; alpha[5] = -M_PI/2.0; alpha[6] = M_PI/2.0; alpha[7] = M_PI
elif choice == "2" :
  print("2")
  d[0] = 0.32214; d[1] = -0.253; d[2] = -0.1031; d[3] = -0.35; d[4] = 0.024
  d[5] = -0.3486; d[6] = -0.023; d[7] = -0.168
  if which_arm == "left" :
    a[0] = 0.09996
    theta[0] = 0.0
  elif which_arm == "right" :
    a[0] = -0.09996
    theta[0] = -M_PI
  else :
    print("Please choose arm \"left\" or \"right\" in run file.")
  alpha[0] = M_PI/2.0; alpha[1] = -M_PI/2.0; alpha[2] = M_PI/2.0; alpha[3] = -M_PI/2.0
  alpha[4] = M_PI/2.0; alpha[5] = -M_PI/2.0; alpha[6] = M_PI/2.0; alpha[7] = M_PI
elif choice == "3" :
  print("Please type in file name of sda10d_macro.xacro or sda10f_macro.xacro")
  input_var = input("(or press \'enter\' to use robotModel/part_motoman_sda10d/sda10d_macro.xacro): ")
  if input_var == "" :
    sda_fn = "robotModel/part_motoman_sda10d/sda10d_macro.xacro"
  else :
    sda_fn = str(input_var)
  #print("You chose " + sda_fn)
  print("Please type in file name of arm_macro.xacro")
  input_var = input("(or press \'enter\' to use robotModel/part_motoman_sda10d/arm_macro.xacro): ")
  if input_var == "" :
    arm_fn = "robotModel/part_motoman_sda10d/arm_macro.xacro"
  else :
    arm_fn = str(input_var)
  #print("You chose " + arm_fn)

  # Read file sda10d_macro.xacro or sda10f_marco.xacro
  # ==================================================
  #fn_1 = "/home/inga/Rosvita/projects/laundrometer/robotModel/part_motoman_sda10d/sda10d_macro.xacro" # sda_fn
  #fn_1 = "/home/inga/Rosvita/projects/sda10f/robotModel/part_motoman_sda10f/sda10f_macro.xacro" # sda_fn
  fn_1 = sda_fn
  print("Read {:s}:".format(fn_1))
  print("================================================================================================")
  xml1 = open(fn_1)
  doc1 = xacro.parse(xml1)
  arms = doc1.getElementsByTagName("xacro:motoman_arm")
  for i in range(0, len(arms)) :
    if arms[i].hasAttribute("prefix"):
      if ((which_arm == "left") and ("left" in arms[i].getAttribute("prefix"))) or ((which_arm == "right") and ("right" in arms[i].getAttribute("prefix"))) :
        print("arm_{:d}:".format(i))
        print("======")
        print(arms[i].getAttribute("prefix"))
        reflect = arms[i].getAttribute("reflect")
        print("reflect = ", reflect)
        print(int(reflect))
        for node in arms[i].childNodes :
          if node.nodeType == node.ELEMENT_NODE:
            if node.tagName == "origin" :
              if node.hasAttribute("prefix"):
                print("prefix = ", node.getAttribute("prefix"))
              if node.hasAttribute("xyz") and node.hasAttribute("rpy"):
                print("origin xyz = ", node.getAttribute("xyz"), "  rpy = ", node.getAttribute("rpy"))
                first_space = node.getAttribute("xyz").find(" ")
                second_space = first_space+1 + node.getAttribute("xyz")[first_space+1:].find(" ")
                x = float(node.getAttribute("xyz")[:first_space])
                y = float(node.getAttribute("xyz")[first_space+1:second_space])
                z = float(node.getAttribute("xyz")[second_space+1:])
                a[0] = int(reflect) * x
                d[0] = z
                d[1] = -1.0 * int(reflect) * y
                if which_arm == "left" :
                  theta[0] = 0.0
                elif which_arm == "right" :
                  theta[0] = -M_PI
                else :
                  print("Please choose arm \"left\" or \"right\" in run file.")
                first_space = node.getAttribute("rpy").find(" ")
                roll = node.getAttribute("rpy")[:first_space]
                if roll[:1] == "-" :
                  r = "-" + roll[roll.find("{")+1:roll.find("}")]
                else :
                  r = roll[roll.find("{")+1:roll.find("}")]
                if r == "-pi" :
                  alpha[0] = -M_PI
                elif r == "pi" :
                  alpha[0] = M_PI
                elif r == "-pi/2" :
                  alpha[0] = -M_PI/2.0
                elif r == "pi/2" :
                  alpha[0] = M_PI/2.0
                else :
                  alpha[0] = 0.0
    print("\n")

  # Read file arm_macro.xacro
  # =========================
  #fn_2 = "/home/inga/Rosvita/projects/laundrometer/robotModel/part_motoman_sda10d/arm_macro.xacro" # arm_fn
  #fn_2 = "/home/inga/Rosvita/projects/sda10f/robotModel/part_motoman_sda10f/arm_macro.xacro" # arm_fn
  fn_2 = arm_fn
  print("Read {:s}:".format(fn_2))
  print("=============================================================================================")
  xml = open(fn_2)
  doc = xacro.parse(xml)
  joints = doc.getElementsByTagName("joint")
  for i in range(0, len(joints)) :
    print("joints{:d} = {:s}:".format(i, joints[i].getAttribute("name")))
    print("================================")
    for node in joints[i].childNodes :
      if node.nodeType == node.ELEMENT_NODE:
        if node.tagName == "origin" :
          if node.hasAttribute("xyz") and node.hasAttribute("rpy"):
            print("origin xyz = ", node.getAttribute("xyz"), "  rpy = ", node.getAttribute("rpy"))
            if i > 0 and i < len(joints)-1 :
              first_space = node.getAttribute("xyz").find(" ")
              second_space = first_space+1 + node.getAttribute("xyz")[first_space+1:].find(" ")
              x = float(node.getAttribute("xyz")[:first_space])
              y = float(node.getAttribute("xyz")[first_space+1:second_space])
              z = float(node.getAttribute("xyz")[second_space+1:])
              d[i] += z
              d[i+1] = -1.0 * y
            if i > 0 :
              first_space = node.getAttribute("rpy").find(" ")
              roll = node.getAttribute("rpy")[:first_space]
              if roll[:1] == "-" :
                r = "-" + roll[roll.find("{")+1:roll.find("}")]
              else :
                r = roll[roll.find("{")+1:roll.find("}")]
              if r == "-pi" :
                alpha[i] = -M_PI
              elif r == "pi" :
                alpha[i] = M_PI
              elif r == "-pi/2" :
                alpha[i] = -M_PI/2.0
              elif r == "pi/2" :
                alpha[i] = M_PI/2.0
              else :
                alpha[i] = 0.0
else :
  print("Improper choice! Please type 1, 2, or 3")
  sys.exit()

print("\n")
print("d = ", d)
print("a = ", a)
print("alpha = ", alpha)
print("theta = ", theta)

# Addition of random jitter to the theta start values:
if with_jitter :
  for i in range(0, 8) :
    r = random.uniform(-1,1)  # random number in ]-1;1[
    jitter = r * (math.pi/36.0)  # jitter in ]-5°;+5°[ = ]-M_PI/36;+M_PI/36[
    theta[i] += jitter
  print("theta with jitter = ", theta)

robot_model = createMotomanRobotModel(theta, d, a, alpha, which_arm)
print("robot_model:")
print(robot_model)

stereo = True
pattern = { "width": 8, "height": 21, "circleSpacing": 0.005 }
robotCalibration = calib(pattern, imWidth, imHeight, hand_eye, robot_model, stereo, output_folder, output_robotModel_filename, output_handEye_filename, which_arm)

robotCalibration.intrinsics = intrinsic
robotCalibration.distCoeffs = distCoeffs
robotCalibration.stereoCalib = stereoCalib

# load images (etc.) to robotCalibration
imagesLeft = []
imagesRight = []
for i in range(0, number_of_images):
  image_left_fn = left_images_path + "_{:03d}.png".format(i+1)
  image_right_fn = right_images_path + "_{:03d}.png".format(i+1)
  image_left = cv.imread(image_left_fn)
  image_right = cv.imread(image_right_fn)
  imagesLeft.append(image_left)
  imagesRight.append(image_right)

jsposes = np.load(joints_path).item()
all_vals_tensors = np.load(all_joints_path)
robotPoses = []
jointValues = []
for i in range(0, len(imagesLeft)):
  robotPose = jsposes['recorded_poses'][i]
  robotPoses.append(robotPose)
  jointVals = np.zeros(8)
  if with_torso_movement_in_data :
    jointVals[0:8] = jsposes['recorded_joint_values'][i]
  else :
    jointVals[0] = all_vals_tensors[0]
    jointVals[1:8] = jsposes['recorded_joint_values'][i]
  jointValues.append(jointVals)

pointsLeft = [None]*len(imagesLeft)
pointsRight = [None]*len(imagesRight)
not_found = []
patternSize = (8, 21)
for i in range(0, len(imagesLeft)):
  # Check with rectified images:
  imgLeftRectUndist, imgRightRectUndist = helpers.rectifyImages(imagesLeft[i], imagesRight[i], stereoCalib, patternSize)
  found1, point_left_rectified = helpers.findPattern(imgLeftRectUndist, cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, patternSize)
  if not found1 :
    print("Pattern points could not be found for rectified left camera image {:03d}!!!".format(i))
  found2, point_right_rectified = helpers.findPattern(imgRightRectUndist, cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, patternSize)
  if not found2 :
    print("Pattern points could not be found for rectified right camera image {:03d}!!!".format(i))
  if (found1 and found2) :
    pointsLeft[i] = point_left_rectified
    pointsRight[i] = point_right_rectified
  if not (found1 and found2) :
    not_found.append(i)
print("len(pointsLeft):")
print(len(pointsLeft))
print("len(pointsRight):")
print(len(pointsRight))
print("Indices of images, in which the pattern could not be found:")
print(not_found)

for i in range(0, len(imagesLeft)):
  flag = 0
  for j in range(0, len(not_found)):
    if i == not_found[j] :
      print("Skip image {:d}.".format(i))
      flag = 1
  if flag == 0 :
    ok = False
    ok = robotCalibration.addStereoImage(imagesLeft[i], imagesRight[i], robotPoses[i], jointValues[i], patternId, pointsLeft[i], pointsRight[i])
    #ok = robotCalibration.addImage(imagesLeft[i], robotPoses[i], jointValues[i], patternId, points[i])
    if not ok :
      print("addImage failed for image {:d}!!!".format(i))

robotCalibration.DHCrossValidate(train_test_split_percentage, runs, alternate, with_torso_optimization, evaluate_only)
