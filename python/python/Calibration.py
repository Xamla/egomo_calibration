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


class Calibration:

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
    T[0][3] = 0.09 # 0.0925 # x-translation of torso-joint
    T[2][3] = 1.06 # 0.9    # z-translation of torso-joint

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

    T[0][1] = (-1.0) * T[0][1]
    T[1][1] = (-1.0) * T[1][1]
    T[2][1] = (-1.0) * T[2][1]
    T[0][2] = (-1.0) * T[0][2]
    T[1][2] = (-1.0) * T[1][2]
    T[2][2] = (-1.0) * T[2][2]

    return T #, poses


  def inv_forward_kinematic(self, jointState, robotModel, join_dir) :
    poses = []

    T = np.identity(4)
    inv_T_init = np.identity(4)
    inv_T_init[0][3] = -0.09 # -0.0925
    inv_T_init[2][3] = -1.06 # -0.9

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

    T[1][0] = (-1.0) * T[1][0]
    T[1][1] = (-1.0) * T[1][1]
    T[1][2] = (-1.0) * T[1][2]
    T[1][3] = (-1.0) * T[1][3]
    T[2][0] = (-1.0) * T[2][0]
    T[2][1] = (-1.0) * T[2][1]
    T[2][2] = (-1.0) * T[2][2]
    T[2][3] = (-1.0) * T[2][3]

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


  def __init__(self, pattern, im_width=960, im_height=720, hand_eye=None, robot_model=None, stereo=False) :
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


  def addImage(self, image, robotPose, jointState, patternId=21, points=None) :
    if (image is None and points is None) or robotPose is None or jointState is None :
      print('Invalid arguments')
      return False

    pattern = patternId
    self.patternIDs.append(pattern)

    image_item = { 
      'image': image,
      'robotPose': robotPose,
      'poseToTarget': np.identity(4),
      'jointStates': jointState,
      'patternID': pattern
    }

    helpers = Xamla3d()
    if points is None :
      patternSize = (self.pattern["width"], self.pattern["height"])  # size of the pattern: x, y = (8, 21)
      found, points = helpers.findPattern(image, cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, patternSize)
      if not found :
        return False

    image_item['patternPoints2d'] = points
    self.images.append(image_item)
    return True


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

    helpers = Xamla3d()
    if pointsLeft is None :
      patternSize = (self.pattern["width"], self.pattern["height"])
      found, pointsLeft = helpers.findPattern(imageLeft, cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, patternSize)
      if not found :
        return False

    if pointsRight is None :
      patternSize = (self.pattern["width"], self.pattern["height"])
      found, pointsRight = helpers.findPattern(imageRight, cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, patternSize)
      if not found :
        return False

    image_item['patternPoints2d'] = pointsLeft
    image_item['patternPoints2dRight'] = pointsRight
    
    image_item['cam_pattern'] = cam_pattern
    
    self.images.append(image_item)
    return True


  # delete all images that have been added before
  def resetImages(self) :
    self.images = []


  def runHandEyeCalibration(self) :
    print('not implemented')
    return True, np.identity(4)


  # this function adds measurements, cameras and points3d for a given patternID to the last 3 variables.
  # if they are nil, we create new numpy.arrays
  def prepareBAStructureWithImageIDs(self, indices, measurementsE, jointPointIndicesE, jointStatesE, points3dE, calibrationData) :

    measOffset = 0
    jointStatesOffset = 0
    point3dOffset = 0

    intrinsics = deepcopy(self.intrinsics) # copy(self.intrinsics)
    distCoeffs = deepcopy(self.distCoeffs)
    handEye = deepcopy(self.handEye)

    robotModel, robotJointDir = self.robotModel["dh"], self.robotModel["joint_direction"]

    if calibrationData is not None :
      intrinsics = calibrationData.intrinsics
      distCoeffs = calibrationData.distCoeffs
      handEye = calibrationData.handEye
      robotModel = calibrationData.robotModel
      robotJointDir = calibrationData.joinDir
    else :
      print("Using GLOABAL robotParameters!")

    print('Using handEye:')
    print(handEye)
    print('Using robotModel:')
    print(robotModel)
    print('Using intrinsics:')
    print(intrinsics)

    if (measurementsE is not None) or (jointStatesE is not None) or (points3dE is not None) :
      assert(measurementsE)
      assert(jointStatesE)
      assert(points3dE)
      assert(jointPointIndicesE)
      measOffset = len(measurementsE)
      jointStatesOffset = len(jointStatesE)
      point3dOffset = len(points3dE)
      print("Offsets in prepareBA - Measurement: ", measOffset, " jointStates: ", jointStatesOffset, " points3d: ", point3dOffset)

    print("Using ", len(indices), " cameras for preparing BA Structure")
    nPts = self.pattern["height"] * self.pattern["width"]
    poses = []
    observations = []
    point3d = np.zeros(shape=(nPts,3), dtype=np.float64)

    for i in range(0, len(indices)) :
      imageEntry = self.images[indices[i]]
      poses.append(imageEntry['robotPose'])
      points = imageEntry['patternPoints2d']

      for m in range(0, len(points)) :
        meas = np.zeros(shape=(4,1), dtype=np.float64)
        meas[0] = len(poses)-1 + jointStatesOffset   # cameraID
        meas[1] = m + point3dOffset                  # pointID
        meas[2] = points[m][0][0]
        meas[3] = points[m][0][1]
        observations.append(meas)

    observationT = np.zeros(shape=(len(observations),2), dtype=np.float64)
    jointpointIndices = np.zeros(shape=(len(observations),2), dtype=np.int64)
    jointStates = np.zeros(shape=(len(indices),8), dtype=np.float64)

    for i in range(0, len(observations)) :
      observationT[i, 0:2] = observations[i][2:4, 0]
      jointpointIndices[i, 0:2] = observations[i][0:2, 0]

    nCnt = 0
    P = []
    for i in range(0, len(indices)) :    
      jointStates[nCnt, 0:8] = deepcopy(self.images[indices[i]]["jointStates"])
      robotPoseWithJointStates = self.forward_kinematic(self.images[indices[i]]["jointStates"], robotModel, robotJointDir)
      rh = np.matmul(robotPoseWithJointStates, handEye)
      inv_rh = inv(rh)
      c = np.matmul(intrinsics, inv_rh[0:3,:]) # only first 3 lines, because intrinsics is 3x3
      P.append(c)
      nCnt = nCnt + 1

    # make an initial guess for the 3d points by
    print("nPts = ", nPts)
    for i in range(0, nPts) :
      meas = []
      for j in range(0, len(indices)) :
        meas.append(deepcopy(self.images[indices[j]]['patternPoints2d'][i, 0:2]))

      if len(P) != len(meas) :
        print("ERROR: Measurements must be same size as camera poses (Poses {:d}, Measurements: {:d}).".format(len(P), len(meas)))
 
      helpers = Xamla3d()
      s, X = helpers.linearTriangulation(P, meas)
      if s != True :
        print("ERROR: Triangulation failed.")
      point3d[i, 0:3] = deepcopy(np.transpose(X))

    if measurementsE is not None :
      measurementsE = np.concatenate(measurementsE, observationT, axis=0)
      jointPointIndicesE = np.concatenate(jointPointIndicesE, jointpointIndices, axis=0)
      jointStatesE = np.concatenate(jointStatesE, jointStates, axis=0)
      points3dE = np.concatenate(points3dE, point3d, axis=0)
      return measurementsE, jointPointIndicesE, jointStatesE, points3dE
    else :
      return observationT, jointpointIndices, jointStates, point3d


  def getImageIndicesForPattern(self, patternID) :
    indices = []
    for i in range(0,len(self.images)) :
      if (self.images[i]["patternPoints2d"] is not None) and (self.images[i]["patternID"] == patternID) :
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
      result1, result2, result3, result4 = pattern_localizer.calcCamPoseViaPlaneFit(self.images[i]["image"], self.images[i]["imageRight"], "left", False)
      Hc.append(result1)
      points2dLeft.append(result2)
      points2dRight.append(result3)
      points3dInLeftCamCoord.append(result4)
    return points3dInLeftCamCoord, Hc
  
  
  def calcStandardDeviation(self, robotModel, handEye, points3dInLeftCamCoord, Hc) :
    # Calculate robot TCP poses with forward kinematic
    Hg = []
    for i in range(0, len(self.images)) :
      result = self.forward_kinematic(self.images[i]["jointStates"], robotModel, self.robotModel["joint_direction"])
      Hg.append(result)

    points3dInBaseCoord = []
    nPoints = 168 # pattern_geometry[1] * pattern_geometry[2]
    for i in range(0, len(self.images)) :
      pointsInLeftCamCoords = np.zeros(shape=4, dtype=np.float64) # torch.DoubleTensor(4)
      pointsInBaseCoords = np.zeros(shape=(nPoints, 4), dtype=np.float64) # torch.DoubleTensor(nPoints, 4)
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
    print(stdev_all * 1000.0)

    base_to_target_avg = self.calc_avg_leftCamBase(handEye, Hg, Hc)
    print("base-pattern average:")
    print(base_to_target_avg)

    return variance, standard_deviation, variance_all, stdev_all, base_to_target_avg
  

  def DHCrossValidate(self, trainTestSplitPercentage, iterations) :

    validationErrors = []
    helpers = Xamla3d()

    print("===========================")
    print("= Simple comparison test: =")
    print("===========================")
    robotMod = deepcopy(self.robotModel["dh"])
    my_indices = self.getImageIndicesForPattern(21)
    print("self.images[my_indices[0]][\"robotPose\"]:")
    print(self.images[my_indices[0]]["robotPose"])
    print("self.images[my_indices[0]][\"jointStates\"]:")
    print(self.images[my_indices[0]]["jointStates"])   
    robotPoseWithJointStates = self.forward_kinematic(self.images[my_indices[0]]["jointStates"], robotMod, self.robotModel["joint_direction"])
    print("robot pose calculated with forward kinematic:")
    print(robotPoseWithJointStates)
    inv_robotPoseWithJointStates = inv(robotPoseWithJointStates)
    print("inverse robot pose:")
    print(inv_robotPoseWithJointStates)
    inv_pose = self.inv_forward_kinematic(self.images[my_indices[0]]["jointStates"], robotMod, self.robotModel["joint_direction"])
    print("inverse robot pose calculated with inverse forward kinematic (with dh_inv):")
    print(inv_pose)
    print("===========================")

    handEye_original = deepcopy(self.handEye)
    original_robotModel = deepcopy(self.robotModel["dh"])
    points3dInLeftCamCoord, Hc = self.calcCamPoseAnd3dPoints() # needed for calculation of standard deviation

    for i in range(0, iterations) :

      observations = None
      jointPointIndices = None
      jointStates = None
      points3d = None
      calibrationData = None

      idxForValidationPerPattern = []

      #for k,v in ipairs(self.patternIDs) do
      k = 21 # TODO: Change this!
      idxPattern = self.getImageIndicesForPattern(k)
      nTraining = int(math.floor(len(idxPattern) * trainTestSplitPercentage))
      print("nTraining:", nTraining)
      helpers.shuffleTable(idxPattern)
      idxTraining = idxPattern[0:nTraining]
      idxValidation = idxPattern[nTraining+1:len(idxPattern)]
      idxForValidationPerPattern.append(idxValidation)
      print("idxTraining:")
      print(idxTraining)
      observations, jointPointIndices, jointStates, points3d = self.prepareBAStructureWithImageIDs(idxTraining, observations, jointPointIndices, jointStates, points3d, calibrationData)
      
      intrinsics = deepcopy(self.intrinsics)
      distCoeffs = deepcopy(self.distCoeffs)
      handEye = deepcopy(self.handEye)
      handEyeInv = inv(handEye)
      print('handEyeInv:')
      print(handEyeInv)

      robotModel, robotJointDir = self.robotModel["dh"], self.robotModel["joint_direction"]

      print("************************************************************")
      print("* Standard deviation of pattern points before optimization *")
      print("************************************************************")
      self.calcStandardDeviation(robotModel, handEye, points3dInLeftCamCoord, Hc)

      jointStatesOptimized = deepcopy(jointStates)
      num_joint_states = len(jointStates)
      num_points = len(points3d)
      training_error = None

      print("******************************************************")
      print("* All in one optimization (hand-eye and dh together) *")
      print("******************************************************")
      intrinsics_cdata = ffi.cast("double *", ffi.from_buffer(intrinsics))
      distCoeffs_cdata = ffi.cast("double *", ffi.from_buffer(distCoeffs))
      handEyeInv_cdata = ffi.cast("double *", ffi.from_buffer(handEyeInv))
      jointStatesOptimized_cdata = ffi.cast("double *", ffi.from_buffer(jointStatesOptimized))
      robotModel_cdata = ffi.cast("double *", ffi.from_buffer(robotModel))
      points3d_cdata = ffi.cast("double *", ffi.from_buffer(points3d))
      observations_cdata = ffi.cast("double *", ffi.from_buffer(observations))
      jointPointIndices_cdata = ffi.cast("long *", ffi.from_buffer(jointPointIndices))

      training_error = clib.optimizeDH(
        intrinsics_cdata,
        distCoeffs_cdata,
        handEyeInv_cdata,
        jointStatesOptimized_cdata,
        robotModel_cdata,
        points3d_cdata,
        observations_cdata,
        jointPointIndices_cdata,
        num_joint_states,
        num_points,
        True,      # optimize_hand_eye
        True,      # optimize_points
        True,      # optimize_robot_model_theta
        False,     # optimize_robot_model_d
        False,     # optimize_robot_model_a
        False,     # optimize_robot_model_alpha
        False,     # optimize_joint_states
        False,     # optimize_pp,
        False,     # optimize_focal_length,
        False      # optimize_distortion
      )
      print("*********************************************")
      print("Error after (hand-eye and dh) optimization:  ", training_error)
      print("*********************************************")
      print("\n")

      robotModel_fn = "robotModel_{:03d}".format(i)
      np.save(robotModel_fn, robotModel)
      self.robotModel["dh"] = robotModel

      offset = np.zeros(shape=(4,8), dtype=np.float64)
      for i in range(0, 4) :
        for j in range(0, 8) :
          offset[i][j] = robotModel[i][j] - original_robotModel[i][j]
      
      offset_units = np.zeros(shape=(4,8), dtype=np.float64)
      for i in range(0, 8) :
        offset_units[0][i] = offset[0][i] * 180 / M_PI
        offset_units[1][i] = offset[1][i] * 1000.0
        offset_units[2][i] = offset[2][i] * 1000.0
        offset_units[3][i] = offset[3][i] * 180 / M_PI

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
      handEye = inv(handEyeInv)
      print(handEye)
      #print('Original inverse hand-eye:')
      #print(inv(handEye_original))
      #print("Optimized inverse hand-eye:")
      #print(handEyeInv)
      handEye_fn = "handEye_{:03d}".format(i)
      np.save(handEye_fn, handEye)
      self.handEye = handEye
            
      print("***********************************************************")
      print("* Standard deviation of pattern points after optimization *")
      print("***********************************************************")
      variance, standard_deviation, variance_all, stdev_all, base_to_target = self.calcStandardDeviation(robotModel, handEye, points3dInLeftCamCoord, Hc)

      print("**********************************************************************")
      print("* Show the reprojection error for all pattern points in all images: *")
      print("**********************************************************************")

      # Preparations to visualize the reprojection error:
      # =================================================
      # Generate ground truth circle center points of the calibration pattern.
      # Z is set to 0 for all points.
      pointsX = 8
      pointsY = 21
      pointSize = 0.005
      # calculates the groundtruth x, y, z positions of the points of the asymmetric circle pattern
      target_points = np.zeros(shape=(pointsX*pointsY, 1, 4), dtype=np.float64)
      i = 0
      for y in range(0, pointsY) :
        for x in range(0, pointsX) :
          target_points[i][0][0] = (2 * x + y % 2) * pointSize
          target_points[i][0][1] = y * pointSize
          target_points[i][0][2] = 0
          target_points[i][0][3] = 1
          i = i + 1

      for i in range(0, len(self.images)) :
        tcp_pose = self.forward_kinematic(self.images[i]["jointStates"], robotModel, self.robotModel["joint_direction"])     
        camera_to_target = np.matmul(inv(np.matmul(tcp_pose, handEye)), base_to_target)
        reprojections = []
        for j in range(0, pointsX*pointsY) :
          in_camera = np.matmul(camera_to_target, target_points[j][0])
          xp1 = in_camera[0]
          yp1 = in_camera[1]
          zp1 = in_camera[2]
          # Scale into the image plane by distance away from camera
          xp = 0.0
          yp = 0.0
          if (zp1 == 0) : # avoid divide by zero
            xp = xp1
            yp = yp1
          else :
            xp = xp1 / zp1
            yp = yp1 / zp1
          # Perform projection using focal length and camera optical center into image plane
          x_image = intrinsics[0][0] * xp + intrinsics[0][2]
          y_image = intrinsics[1][1] * yp + intrinsics[1][2]
          reprojections.append((x_image, y_image))
        frame = deepcopy(self.images[i]["image"])
        for j in range(0, len(reprojections)) :
          pt_x = int(round(reprojections[j][0]))
          pt_y = int(round(reprojections[j][1]))
          cv.circle(img=frame, center=(pt_x, pt_y), radius=4, color=(0, 0, 255))
        cv.imshow("reprojection error for image {:d}".format(i), frame)
        cv.waitKey(500)     
      
