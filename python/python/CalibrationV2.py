#!/usr/bin/env python

import numpy as np
import sys
import cv2 as cv
import os
import math
from matplotlib import pyplot as plt

from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import eig

from copy import copy, deepcopy

from .pattern_localisation import PatternLocalisation
from .xamla3d import Xamla3d
from .env import clib
from cffi import FFI
import ctypes
ffi = FFI()

M_PI = 3.14159265359


class CalibrationV2:

  # generate matrix form denavit-hartenberg parameters
  def dh(self, theta, d, a, alpha) :
    sin = math.sin 
    cos = math.cos
    result = np.identity(4)
    result[0,:] = [ cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta) ]
    result[1,:] = [ sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta) ]
    result[2,:] = [          0,             sin(alpha),             cos(alpha),            d ]
    return result


  # generate inverse matrix form denavit-hartenberg parameters
  def dh_inverse(self, theta, d, a, alpha) :
    sin = math.sin 
    cos = math.cos
    result = np.identity(4)
    result[0,:] = [             cos(theta),             sin(theta),          0,            -a ]
    result[1,:] = [ -sin(theta)*cos(alpha),  cos(theta)*cos(alpha), sin(alpha), -d*sin(alpha) ]
    result[2,:] = [  sin(alpha)*sin(theta), -cos(theta)*sin(alpha), cos(alpha), -d*cos(alpha) ]
    return result


  # return final transform and individual frames
  def forward_kinematic(self, jointState, robotModel, join_dir) :
    poses = []

    T = np.identity(4)
    T[0][3] = 0.0925 # x-translation of torso-joint
    T[2][3] = 1.06 #0.9   # z-translation of torso-joint (cell_body->cell_tool0 = 0.16 + robot_base->torso_joint_b1 = 0.9)

    # Note: Here, base is the floor ground under the robot!!!
    
    for i in range (0, 8) : # 8 joints of motoman SDA1
      # "+" is the correct sign here!!!
      # ("-" will make this forward kinematic differ from the MoveIt-forward kinematic!!!)
      t = self.dh( join_dir[i] * jointState[i] + robotModel[0][i], # theta for joint i
                   robotModel[1][i], # d for joint i
                   robotModel[2][i], # a for joint i
                   robotModel[3][i]  # alpha for joint i
                 )
      poses.append(t)
      T = np.matmul(T, t)  

    return T #, poses


  def inv_forward_kinematic(self, jointState, robotModel, join_dir) :
    poses = []

    T = np.identity(4)
    inv_T_init = np.identity(4)
    inv_T_init[0][3] = -0.0925
    inv_T_init[2][3] = -1.06 #-0.9

    # Note: Here, base is the floor ground under the robot!!!
    
    for i in range(7, -1, -1) : # calculation of inverse tcp-pose (T^-1)
      t = self.dh_inverse( join_dir[i] * jointState[i] + robotModel[0][i], # theta for joint i
                           robotModel[1][i], # d for joint i
                           robotModel[2][i], # a for joint i
                           robotModel[3][i]  # alpha for joint i
                         )
      poses.append(t)
      # T^-1 = identity * t8^-1 * t7^-1 * ... * t1^-1 * T_init^-1 
      T = np.matmul(T, t)  # T^-1 = T^-1 * t
    
    T = np.matmul(T, inv_T_init) # T^-1 = T^-1 * T_init^-1

    return T #, poses


  def createUR5RobotModel(self, shoulder_height=0.089159, upper_arm_length=0.425, forearm_length=0.39225, 
                          wrist_1_length=0.10915, wrist_2_length=0.09465, wrist_3_length=0.0823) :
    dh = np.zeros((4, 6))

    # theta
    dh[0][0] = 0
    dh[0][1] = 0
    dh[0][2] = 0
    dh[0][3] = 0
    dh[0][4] = 0
    dh[0][5] = M_PI

    # d
    dh[1][0] = shoulder_height
    dh[1][1] = 0
    dh[1][2] = 0
    dh[1][3] = -wrist_1_length
    dh[1][4] = wrist_2_length
    dh[1][5] = -wrist_3_length

    # a
    dh[2][0] = 0
    dh[2][1] = upper_arm_length
    dh[2][2] = forearm_length
    dh[2][3] = 0
    dh[2][4] = 0
    dh[2][5] = 0

    # alpha
    dh[3][0] = M_PI/2
    dh[3][1] = 0
    dh[3][2] = 0
    dh[3][3] = M_PI/2
    dh[3][4] = -M_PI/2
    dh[3][5] = M_PI

    joint_direction = np.ones(6) # { 1, -1, -1, -1, 1, -1 }
    joint_direction[1] = -1.0
    joint_direction[2] = -1.0
    joint_direction[3] = -1.0
    joint_direction[5] = -1.0
  
    result_table = { 'dh': dh, 'joint_direction': joint_direction }
    return result_table


  def __init__(self, pattern, im_width=960, im_height=720, hand_eye=None, robot_model=None, stereo=False, output_folder=None, output_robotModel_filename=None, output_handEye_filename=None, which_arm="left") :
    self.pattern = pattern
    if self.pattern is None :
      self.pattern = { "width": 8, "height": 21, "circleSpacing": 0.005 }

    self.robotModel = robot_model
    if self.robotModel is None :
      self.robotModel = self.createUR5RobotModel()

    self.stereo = stereo
    if stereo is None :
      self.stereo = False

    self.im_width = im_width
    self.im_height = im_height

    # intrinsic parameters of camera
    self.intrinsics = np.zeros((3,3))

    # distortion parameters of camera
    self.distCoeffs = np.zeros((5,1))

    # stereo calibration (in case of stereo camera setup)
    self.stereoCalib = None

    self.handEye = hand_eye
    if hand_eye is None :
      self.handEye = np.identity(4) 
      self.handEye[0,:] = [  0.002505,  0.764157,  0.645025, 15.239477 / 1000.0 ]  # translation in meters (convert for mm)
      self.handEye[1,:] = [ -0.000719, -0.645026,  0.764161, 69.903526 / 1000.0 ]
      self.handEye[2,:] = [  0.999997, -0.002379, -0.001066, 55.941492 / 1000.0 ]
      
    self.images = []
    self.patternIDs = []
    self.output_folder = output_folder
    self.output_robotModel_filename = output_robotModel_filename
    self.output_handEye_filename = output_handEye_filename
    self.which_arm = which_arm


  def addStereoImage(self, imageLeft, imageRight, robotPose, jointState, patternId=21, pointsLeft=None, pointsRight=None, leftcam_pattern=None) :
    cam_pattern = leftcam_pattern

    if (imageLeft is None and pointsLeft is None) or (imageRight is None and pointsRight is None) or robotPose is None or jointState is None :
      print('Invalid arguments')
      return False

    pattern = patternId
    self.patternIDs.append(pattern)

    image_item = {
      'image': imageLeft,
      'imageRight': imageRight,
      'robotPose': robotPose,
      'poseToTarget': np.identity(4),
      'jointStates': jointState,
      'patternID': pattern
    }

    image_item['cam_pattern'] = cam_pattern
    
    self.images.append(image_item)
    return True


  # delete all images that have been added before
  def resetImages(self) :
    self.images = []


  def runHandEyeCalibration(self) :
    print('not implemented')
    return True, np.identity(4)


  def calcCamPoseAnd3dPoints(self) :
    # Create pattern localizer
    pattern_localizer = PatternLocalisation()
    pattern_localizer.circleFinderParams.minArea = 300
    pattern_localizer.circleFinderParams.maxArea = 4000
    pattern_localizer.setPatternIDdictionary(np.load("../python/python/patDictData.npy"))
    pattern_localizer.setPatternData(8, 21, 0.005)
    pattern_localizer.setStereoCalibration(self.stereoCalib)
    # Calculate cam->pattern poses with plane fit
    Hc = []
    points2dLeft = []
    points2dRight = []
    points3dInLeftCamCoord = []
    for i in range(0, len(self.images)) :
      # Here, the raw left and right images are needed (not undistorted)!
      result1, result2, result3, result4, imgLeftRectUndist, imgRightRectUndist, leftP, leftR = pattern_localizer.calcCamPoseViaPlaneFit(self.images[i]["image"], self.images[i]["imageRight"], "left", False)
      self.images[i]["imgLeftRectUndist"] = deepcopy(imgLeftRectUndist)
      #self.images[i]["imgRightRectUndist"] = deepcopy(imgRightRectUndist)
      Hc.append(result1)
      points2dLeft.append(result2)
      points2dRight.append(result3)
      self.images[i]["points2dLeft"] = deepcopy(result2)
      #self.images[i]["points2dRight"] = deepcopy(result3)
      points3dInLeftCamCoord.append(result4)
      self.images[i]["points3dLeft"] = deepcopy(result4)
      self.images[i]["newLeftCamMat"] = deepcopy(leftP)
      self.images[i]["leftR"] = deepcopy(leftR)
    return points3dInLeftCamCoord, Hc


  def prepareBAStructureWithImageIDsStereo(self, indices, points3dInLeftCamCoord) :
    print("Preparing problem structure (i.e. joint states and points3d in camera coordinates).")
    print("Using ", len(indices), " stereo cameras for preparing problem structure")
    nPts = self.pattern["height"] * self.pattern["width"]

    jointStates = np.zeros(shape=(len(indices), 8), dtype=np.float64)
    for i in range(0, len(indices)) :
      jointStates[i, 0:8] = deepcopy(self.images[indices[i]]["jointStates"])

    points3dInCamCoord = np.zeros(shape=(len(indices), nPts, 3), dtype=np.float64)
    for i in range(0, len(indices)) :
      points3dInCamCoord_oneImg = np.zeros(shape=(nPts, 3), dtype=np.float64)
      for j in range(0, nPts) :
        points3dInCamCoord_oneImg[j] = points3dInLeftCamCoord[indices[i]][j]
      points3dInCamCoord[i] = points3dInCamCoord_oneImg

    return jointStates, points3dInCamCoord


  def getImageIndicesForPattern(self, patternID) :
    indices = []
    for i in range(0,len(self.images)) :
      if self.images[i]["patternID"] == patternID :
        indices.append(i)
    return indices
  

  def calc_avg_leftCamBase(self, H, Hg, Hc) :
    # Rotation averaging: 
    # https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions
    Q = np.zeros(shape=(len(Hg), 4), dtype=np.float64)
    avg_pos = np.zeros(shape=3, dtype=np.float64)
    helpers = Xamla3d()
    for i in range(0, len(Hg)) : 
      leftCamPoseInBaseCoords = np.matmul( np.matmul(Hg[i], H), Hc[i] ) # Hg[i] * H * Hc[i]
      avg_pos = avg_pos + leftCamPoseInBaseCoords[0:3,3]
      q = helpers.transformMatrixToQuaternion(leftCamPoseInBaseCoords[0:3,0:3])
      Q[i] = q
    avg_pos = avg_pos / len(Hg)
    QtQ = np.matmul(np.transpose(Q), Q)
    e, V = eig(QtQ)
    maxEig_index = np.argmax(e)
    avg_q = np.transpose(V)[maxEig_index]
    avg_rot = helpers.transformQuaternionToMatrix(avg_q)   
    avg_LeftCamPose = np.zeros(shape=(4,4), dtype=np.float64)
    avg_LeftCamPose[0:3,0:3] = avg_rot
    avg_LeftCamPose[0:3,3] = avg_pos
    avg_LeftCamPose[3,3] = 1.0
    return avg_LeftCamPose
  
  
  def calcStandardDeviation(self, robotModel, handEye, points3dInLeftCamCoord, Hc) :
    # Calculate robot TCP poses with forward kinematic
    Hg = []
    for i in range(0, len(self.images)) :
      result = self.forward_kinematic(self.images[i]["jointStates"], robotModel, self.robotModel["joint_direction"])
      Hg.append(result)

    points3dInBaseCoord = []
    nPoints = 168 # pattern_geometry[1] * pattern_geometry[2]
    for i in range(0, len(self.images)) :
      pointsInLeftCamCoords = np.zeros(shape=4, dtype=np.float64)
      pointsInBaseCoords = np.zeros(shape=(nPoints, 4), dtype=np.float64)
      for j in range(0, nPoints) :
        pointsInLeftCamCoords[0] = points3dInLeftCamCoord[i][j][0]
        pointsInLeftCamCoords[1] = points3dInLeftCamCoord[i][j][1]
        pointsInLeftCamCoords[2] = points3dInLeftCamCoord[i][j][2]
        pointsInLeftCamCoords[3] = 1.0
        pointsInBaseCoords[j] = np.matmul( np.matmul(Hg[i], handEye), pointsInLeftCamCoords )
      points3dInBaseCoord.append(pointsInBaseCoords)

    # For j = 1,nPoints: Calculation of the standard deviation of pattern point j (in all images):
    variance = []
    standard_deviation = []
    for j in range(0, nPoints) :
      mean = np.zeros(shape=3, dtype=np.float64)
      for i in range(0, len(self.images)) :
        mean = mean + points3dInBaseCoord[i][j][0:3]
      mean = mean * (1.0 / len(self.images))

      var = np.zeros(shape=3, dtype=np.float64)
      for i in range(0, len(self.images)) :
        var[0] = var[0] + (points3dInBaseCoord[i][j][0] - mean[0]) * (points3dInBaseCoord[i][j][0] - mean[0])
        var[1] = var[1] + (points3dInBaseCoord[i][j][1] - mean[1]) * (points3dInBaseCoord[i][j][1] - mean[1])
        var[2] = var[2] + (points3dInBaseCoord[i][j][2] - mean[2]) * (points3dInBaseCoord[i][j][2] - mean[2])
      var = var * (1.0 / len(self.images))

      stdev = np.zeros(shape=3, dtype=np.float64)
      stdev[0] = math.sqrt(var[0])
      stdev[1] = math.sqrt(var[1])
      stdev[2] = math.sqrt(var[2])
      variance.append(var)
      standard_deviation.append(stdev)
    
    variance_all = np.zeros(shape=3, dtype=np.float64)
    stdev_all = np.zeros(shape=3, dtype=np.float64)
    for j in range(0, nPoints) :
      variance_all = variance_all + variance[j]
      stdev_all = stdev_all + standard_deviation[j]
    variance_all = variance_all * (1.0/nPoints)
    stdev_all = stdev_all * (1.0/nPoints)

    print("average standard deviation of all pattern points [in mm]:")
    stdev_all_mm = stdev_all * 1000.0
    print(stdev_all_mm)
    print("Length of average standard deviation [in mm]:")
    print(norm(stdev_all_mm))

    base_to_target_avg = self.calc_avg_leftCamBase(handEye, Hg, Hc)
    print("base-pattern average:")
    print(base_to_target_avg)

    return variance, standard_deviation, variance_all, stdev_all, base_to_target_avg


  # show the reprojection error for all pattern points in all given images
  def showReprojectionError(self, idx, robotModel, handEye, base_to_target, pointsX, pointsY, target_points, intrinsics, number, my_frames=None, my_frames_rectified=None) :
    print("********************************")
    print("* Show the reprojection error: *")
    print("********************************")

    frames = []
    frames_rectified = []
    for j in range(0, number) : #len(idx)) :
    #for j in range(0, 5) :
      tcp_pose = self.forward_kinematic(self.images[idx[j]]["jointStates"], robotModel, self.robotModel["joint_direction"])     
      camera_to_target = np.matmul(inv(np.matmul(tcp_pose, handEye)), base_to_target)
      reprojections = []
      reprojections_rectified = []
      points3d = deepcopy(self.images[idx[j]]["points3dLeft"]) # points 3d in cam coord
      points3d_rectified = deepcopy(points3d)
      points3d_reprojected = []
      points3d_reprojected_rectified = []
      leftP = deepcopy(self.images[idx[j]]["newLeftCamMat"]) # new intrinsics after rectification
      leftR = deepcopy(self.images[idx[j]]["leftR"]) # rectification transformation
      for k in range(0, pointsX*pointsY) :
         # The 3d points have been unrectified after "triangulatePoints".
         # => Thus, to display them on the rectified images they have to be rectified again!
        points3d_rectified[k] = leftR.dot(points3d[k])
        in_camera = np.matmul(camera_to_target, target_points[k][0])
        in_camera_rectified = leftR.dot(in_camera[0:3])
        # Scale into the image plane by distance away from camera
        xp, yp = in_camera[0], in_camera[1]
        xpr, ypr = in_camera_rectified[0], in_camera_rectified[1]
        p3d_x, p3d_y = points3d[k,0], points3d[k,1]
        p3dr_x, p3dr_y = points3d_rectified[k,0], points3d_rectified[k,1]
        if (abs(in_camera[2]) > 1e-5) : # avoid divide by zero
          xp /= in_camera[2]
          yp /= in_camera[2]
        if (abs(in_camera_rectified[2]) > 1e-5) : # avoid divide by zero
          xpr /= in_camera_rectified[2]
          ypr /= in_camera_rectified[2]
        if (abs(points3d[k,2]) > 1e-5) :
          p3d_x /= points3d[k,2]
          p3d_y /= points3d[k,2]
        if (abs(points3d_rectified[k,2]) > 1e-5) :
          p3dr_x /= points3d_rectified[k,2]
          p3dr_y /= points3d_rectified[k,2]
        # Perform projection using focal length and camera optical center into image plane
        xp_img = intrinsics[0][0] * xp + intrinsics[0][2]
        yp_img = intrinsics[1][1] * yp + intrinsics[1][2]
        xpr_img = leftP[0][0] * xpr + leftP[0][2]
        ypr_img = leftP[1][1] * ypr + leftP[1][2]
        reprojections.append((xp_img, yp_img))
        reprojections_rectified.append((xpr_img, ypr_img))
        p3d_x_img = intrinsics[0][0] * p3d_x + intrinsics[0][2]
        p3d_y_img = intrinsics[1][1] * p3d_y + intrinsics[1][2]
        p3dr_x_img = leftP[0][0] * p3dr_x + leftP[0][2]
        p3dr_y_img = leftP[1][1] * p3dr_y + leftP[1][2]
        points3d_reprojected.append((p3d_x_img, p3d_y_img))
        points3d_reprojected_rectified.append((p3dr_x_img, p3dr_y_img))
      frame = deepcopy(self.images[idx[j]]["image"])
      frame_rectified = deepcopy(self.images[idx[j]]["imgLeftRectUndist"])
      points2d = deepcopy(self.images[idx[j]]["points2dLeft"])
      if my_frames is not None :
        frame = deepcopy(my_frames[j])
        frame_rectified = deepcopy(my_frames_rectified[j])
      for k in range(0, len(reprojections)) :
        pt_x = int(round(reprojections[k][0]))
        pt_y = int(round(reprojections[k][1]))
        ptr_x = int(round(reprojections_rectified[k][0]))
        ptr_y = int(round(reprojections_rectified[k][1]))
        p3d_pt_x = int(round(points3d_reprojected[k][0]))
        p3d_pt_y = int(round(points3d_reprojected[k][1]))
        p3dr_pt_x = int(round(points3d_reprojected_rectified[k][0]))
        p3dr_pt_y = int(round(points3d_reprojected_rectified[k][1]))
        if my_frames is None :
          cv.circle(img=frame, center=(pt_x, pt_y), radius=4, color=(0, 0, 255))
          cv.circle(img=frame_rectified, center=(ptr_x, ptr_y), radius=4, color=(0, 0, 255))
        else :
          cv.circle(img=frame, center=(pt_x, pt_y), radius=4, color=(0, 255, 0))
          cv.circle(img=frame, center=(p3d_pt_x, p3d_pt_y), radius=4, color=(255, 0, 0))
          #cv.circle(img=frame_rectified, center=(points2d[k,0], points2d[k,1]), radius=4, color=(255, 0, 0))
          cv.circle(img=frame_rectified, center=(ptr_x, ptr_y), radius=4, color=(0, 255, 0))
          cv.circle(img=frame_rectified, center=(p3dr_pt_x, p3dr_pt_y), radius=4, color=(255, 0, 0))
      frames.append(frame)
      frames_rectified.append(frame_rectified)
      if my_frames is not None :
        cv.imshow("reprojection error for image {:d}".format(idx[j]), frame)
        cv.waitKey(500)
        cv.imshow("reprojection error on rectified image {:d}".format(idx[j]), frame_rectified)
        cv.waitKey(500)
        cv.imwrite("/home/inga/code/my_python_branch/egomo_calibration/result_imgs/reprojection_error_img_{:d}.png".format(idx[j]), frame)
        cv.imwrite("/home/inga/code/my_python_branch/egomo_calibration/result_imgs/reprojection_error_rectified_img_{:d}.png".format(idx[j]), frame_rectified)
    return frames, frames_rectified


  def evaluate(self, idxValidation, points3dInLeftCamCoord, Hc, pointsX, pointsY, target_points) :
    print("idxValidation:")
    print(idxValidation)
    if (len(idxValidation) > 0) :
      jointStates, points3d = self.prepareBAStructureWithImageIDsStereo(idxValidation, points3dInLeftCamCoord)

      intrinsics = deepcopy(self.intrinsics)
      handEye = deepcopy(self.handEye)
      robotModel = self.robotModel["dh"]
      jointStatesPred = deepcopy(jointStates)
      jointStatesObs = deepcopy(jointStates)
      points3dPred = deepcopy(points3d)
      points3dObs = deepcopy(points3d)
      num_joint_states = len(jointStates)
      num_points = len(points3d[0])
      validation_error = None

      handEye_cdata = ffi.cast("double *", ffi.from_buffer(handEye))
      jointStatesPred_cdata = ffi.cast("double *", ffi.from_buffer(jointStatesPred))
      jointStatesObs_cdata = ffi.cast("double *", ffi.from_buffer(jointStatesObs))
      robotModel_cdata = ffi.cast("double *", ffi.from_buffer(robotModel))
      points3dPred_cdata = ffi.cast("double *", ffi.from_buffer(points3dPred))
      points3dObs_cdata = ffi.cast("double *", ffi.from_buffer(points3dObs))

      arm = 0
      if self.which_arm == "right" :
        arm = 1

      validation_error = clib.evaluateDHV2(
        handEye_cdata,
        jointStatesPred_cdata,
        jointStatesObs_cdata,
        robotModel_cdata,
        points3dPred_cdata,
        points3dObs_cdata,
        num_joint_states,
        num_points,
        arm)
      print("*********************************************")
      print("Validation error:  ", validation_error)
      print("*********************************************")
      print("\n")
      variance, standard_deviation, variance_all, stdev_all, base_to_target = self.calcStandardDeviation(robotModel, handEye, points3dInLeftCamCoord, Hc)
      self.showReprojectionError(idxValidation, robotModel, handEye, base_to_target, pointsX, pointsY, target_points, intrinsics, len(idxValidation))


  def DHCrossValidate(self, trainTestSplitPercentage, iterations) :

    validationErrors = []
    helpers = Xamla3d()
    pattern_id = self.images[0]["patternID"]
    print("pattern_id:")
    print(pattern_id)

    print("===========================")
    print("= Simple comparison test: =")
    print("===========================")
    robotMod = deepcopy(self.robotModel["dh"])
    my_indices = self.getImageIndicesForPattern(pattern_id)
    print("self.images[my_indices[0]][\"robotPose\"]:")
    print(self.images[my_indices[0]]["robotPose"])
    #print("self.images[my_indices[0]][\"jointStates\"]:")
    #print(self.images[my_indices[0]]["jointStates"])   
    robotPoseWithJointStates = self.forward_kinematic(self.images[my_indices[0]]["jointStates"], robotMod, self.robotModel["joint_direction"])
    print("robot pose calculated with forward kinematic:")
    print(robotPoseWithJointStates)
    #inv_robotPoseWithJointStates = inv(robotPoseWithJointStates)
    #print("inverse robot pose:")
    #print(inv_robotPoseWithJointStates)
    #inv_pose = self.inv_forward_kinematic(self.images[my_indices[0]]["jointStates"], robotMod, self.robotModel["joint_direction"])
    #print("inverse robot pose calculated with inverse forward kinematic (with dh_inv):")
    #print(inv_pose)
    print("===========================")

    handEye_original = deepcopy(self.handEye)
    original_robotModel = deepcopy(self.robotModel["dh"])
    points3dInLeftCamCoord, Hc = self.calcCamPoseAnd3dPoints() # needed for calculation of standard deviation and for prepareBAStructure

    for i in range(0, iterations) :

      idxForValidationPerPattern = []

      idxPattern = self.getImageIndicesForPattern(pattern_id)
      nTraining = int(math.floor(len(idxPattern) * trainTestSplitPercentage))
      print("nTraining:", nTraining)
      helpers.shuffleTable(idxPattern)
      idxTraining = idxPattern[0:nTraining]
      idxValidation = idxPattern[nTraining+1:len(idxPattern)]
      idxForValidationPerPattern.append(idxValidation)
      print("idxTraining:")
      print(idxTraining)

      # Preparations to visualize the reprojection error:
      # =================================================
      # Generate ground truth circle center points of the calibration pattern.
      # Z is set to 0 for all points.
      pointsX = 8
      pointsY = 21
      pointSize = 0.005
      target_points = helpers.generatePatternPoints(pointsX, pointsY, pointSize)
      
      print("*************************************")
      print("* EVALUATION (before optimization): *")
      print("*************************************")
      #idxValidation = idxTraining # remove after evaluation!!!
      self.evaluate(idxValidation, points3dInLeftCamCoord, Hc, pointsX, pointsY, target_points)
      #sys.exit()

      jointStates, points3d = self.prepareBAStructureWithImageIDsStereo(idxTraining, points3dInLeftCamCoord)

      intrinsics = deepcopy(self.intrinsics)
      handEye = deepcopy(self.handEye)
      robotModel, robotJointDir = self.robotModel["dh"], self.robotModel["joint_direction"]

      print("************************************************************")
      print("* Standard deviation of pattern points before optimization *")
      print("************************************************************")
      before_variance, before_standard_deviation, before_variance_all, before_stdev_all, before_base_to_target = self.calcStandardDeviation(robotModel, handEye, points3dInLeftCamCoord, Hc)

      jointStatesPred = deepcopy(jointStates)
      jointStatesObs = deepcopy(jointStates)
      points3dPred = deepcopy(points3d)
      points3dObs = deepcopy(points3d)
      num_joint_states = len(jointStates)
      num_points = len(points3d[0])
      training_error = None

      print("******************************************************************************")
      print("* Show the reprojection error for all pattern points in all training images: *")
      print("******************************************************************************")
      frames_before, frames_rectified_before = self.showReprojectionError(idxTraining, robotModel, handEye, before_base_to_target, pointsX, pointsY, target_points, intrinsics, len(idxTraining))

      print("******************************************************")
      print("* All in one optimization (hand-eye and dh together) *")
      print("******************************************************")
      handEye_cdata = ffi.cast("double *", ffi.from_buffer(handEye))
      jointStatesPred_cdata = ffi.cast("double *", ffi.from_buffer(jointStatesPred))
      jointStatesObs_cdata = ffi.cast("double *", ffi.from_buffer(jointStatesObs))
      robotModel_cdata = ffi.cast("double *", ffi.from_buffer(robotModel))
      points3dPred_cdata = ffi.cast("double *", ffi.from_buffer(points3dPred))
      points3dObs_cdata = ffi.cast("double *", ffi.from_buffer(points3dObs))

      arm = 0
      if self.which_arm == "right" :
        arm = 1

      if i % 2 == 0 : # even (i=0,2,4,...)
        training_error = clib.optimizeDHV2(
          handEye_cdata,
          jointStatesPred_cdata,
          jointStatesObs_cdata,
          robotModel_cdata,
          points3dPred_cdata,
          points3dObs_cdata,
          num_joint_states,
          num_points,
          arm,
          False,     # optimize_hand_eye
          False,     # optimize_points
          True,      # optimize_robot_model_theta
          True,      # optimize_robot_model_d
          False,     # optimize_robot_model_a
          False,     # optimize_robot_model_alpha
          False      # optimize_joint_states
        )
        print("*********************************************")
        print("Error after (hand-eye and dh) optimization:  ", training_error)
        print("*********************************************")
        print("\n")
      else : # odd (i=1,3,5,...)
        training_error = clib.optimizeDHV2(
          handEye_cdata,
          jointStatesPred_cdata,
          jointStatesObs_cdata,
          robotModel_cdata,
          points3dPred_cdata,
          points3dObs_cdata,
          num_joint_states,
          num_points,
          arm,
          True,      # optimize_hand_eye
          False,     # optimize_points
          False,     # optimize_robot_model_theta
          False,     # optimize_robot_model_d
          False,     # optimize_robot_model_a
          False,     # optimize_robot_model_alpha
          False      # optimize_joint_states
        )
        print("*********************************************")
        print("Error after (hand-eye and dh) optimization:  ", training_error)
        print("*********************************************")
        print("\n")

      if (self.output_folder is not None) and (self.output_robotModel_filename is not None) :
        if not os.path.isdir(self.output_folder) :
          os.mkdir(self.output_folder)
        robotModel_fn = os.path.join(self.output_folder, (self.output_robotModel_filename + "_{:03d}".format(i)))
        np.save(robotModel_fn, robotModel)
      self.robotModel["dh"] = robotModel
        
      offset = np.zeros(shape=(4,8), dtype=np.float64)
      for j in range(0, 4) :
        for k in range(0, 8) :
          offset[j][k] = robotModel[j][k] - original_robotModel[j][k]
      
      offset_units = np.zeros(shape=(4,8), dtype=np.float64)
      for j in range(0, 8) :
        offset_units[0][j] = offset[0][j] * 180 / M_PI
        offset_units[1][j] = offset[1][j] * 1000.0
        offset_units[2][j] = offset[2][j] * 1000.0
        offset_units[3][j] = offset[3][j] * 180 / M_PI

      print("Offset (from original model) (i.e. robotModel = originalModel + Offset):")
      print("Joint 0 (theta, d, a, alpha): {:f} degree, {:f} mm, {:f} mm, {:f} degree".format(offset[0][0] * 180/M_PI, offset[1][0] * 1000, offset[2][0] * 1000, offset[3][0] * 180/M_PI))
      print("Joint 1 (theta, d, a, alpha): {:f} degree, {:f} mm, {:f} mm, {:f} degree".format(offset[0][1] * 180/M_PI, offset[1][1] * 1000, offset[2][1] * 1000, offset[3][1] * 180/M_PI))
      print("Joint 2 (theta, d, a, alpha): {:f} degree, {:f} mm, {:f} mm, {:f} degree".format(offset[0][2] * 180/M_PI, offset[1][2] * 1000, offset[2][2] * 1000, offset[3][2] * 180/M_PI))
      print("Joint 3 (theta, d, a, alpha): {:f} degree, {:f} mm, {:f} mm, {:f} degree".format(offset[0][3] * 180/M_PI, offset[1][3] * 1000, offset[2][3] * 1000, offset[3][3] * 180/M_PI))
      print("Joint 4 (theta, d, a, alpha): {:f} degree, {:f} mm, {:f} mm, {:f} degree".format(offset[0][4] * 180/M_PI, offset[1][4] * 1000, offset[2][4] * 1000, offset[3][4] * 180/M_PI))
      print("Joint 5 (theta, d, a, alpha): {:f} degree, {:f} mm, {:f} mm, {:f} degree".format(offset[0][5] * 180/M_PI, offset[1][5] * 1000, offset[2][5] * 1000, offset[3][5] * 180/M_PI))
      print("Joint 6 (theta, d, a, alpha): {:f} degree, {:f} mm, {:f} mm, {:f} degree".format(offset[0][6] * 180/M_PI, offset[1][6] * 1000, offset[2][6] * 1000, offset[3][6] * 180/M_PI))
      print("Joint 7 (theta, d, a, alpha): {:f} degree, {:f} mm, {:f} mm, {:f} degree".format(offset[0][7] * 180/M_PI, offset[1][7] * 1000, offset[2][7] * 1000, offset[3][7] * 180/M_PI))
      print("\n")

      print('Original hand-eye:')
      print(handEye_original)
      print('Optimized hand-eye:')
      print(handEye)
      if self.output_handEye_filename is not None :
        handEye_fn = os.path.join(self.output_folder, (self.output_handEye_filename + "_{:03d}".format(i)))
        np.save(handEye_fn, handEye)
      self.handEye = handEye
            
      print("***********************************************************")
      print("* Standard deviation of pattern points after optimization *")
      print("***********************************************************")
      variance, standard_deviation, variance_all, stdev_all, base_to_target = self.calcStandardDeviation(robotModel, handEye, points3dInLeftCamCoord, Hc)

      print("******************************************************************************")
      print("* Show the reprojection error for all pattern points in all training images: *")
      print("******************************************************************************")
      self.showReprojectionError(idxTraining, robotModel, handEye, base_to_target, pointsX, pointsY, target_points, intrinsics, 5, frames_before, frames_rectified_before) #len(idxTraining), frames_before)
      
    print("************************************")
    print("* EVALUATION (after optimization): *")
    print("************************************")
    self.evaluate(idxValidation, points3dInLeftCamCoord, Hc, pointsX, pointsY, target_points)
    