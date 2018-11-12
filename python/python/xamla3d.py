#!/usr/bin/env python

import numpy as np
import sys
import cv2 as cv
import os
import math
import random
from matplotlib import pyplot as plt

from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import norm

from copy import copy, deepcopy


class Xamla3d:

  def grayToRGB(self, inputImg) :
    if inputImg.shape[2] == 3 :
      return inputImg
    else :
      img = cv.cvtColor(src = inputImg, code = cv.COLOR_GRAY2RGB)
      return img
      

  # Open one image file, search for the circle pattern and extract the positions of the circle centers
  def findPattern (self, image, patternType, patternSize) :
    circleFinderParams = cv.SimpleBlobDetector_Params()
    circleFinderParams.thresholdStep = 5
    circleFinderParams.minThreshold = 60
    circleFinderParams.maxThreshold = 230
    circleFinderParams.minRepeatability = 3
    circleFinderParams.minDistBetweenBlobs = 1
    circleFinderParams.filterByColor = False
    circleFinderParams.blobColor = 0
    circleFinderParams.filterByArea = True  # area of the circle in pixels
    circleFinderParams.minArea = 200
    circleFinderParams.maxArea = 3000
    circleFinderParams.filterByCircularity = True
    circleFinderParams.minCircularity = 0.6
    circleFinderParams.maxCircularity = 10
    circleFinderParams.filterByInertia = False
    circleFinderParams.minInertiaRatio = 0.6
    circleFinderParams.maxInertiaRatio = 10
    circleFinderParams.filterByConvexity = True
    circleFinderParams.minConvexity = 0.8
    circleFinderParams.maxConvexity = 10
    ver = (cv.__version__).split('.')
    if int(ver[0]) < 3 :
      blobDetector = cv.SimpleBlobDetector(circleFinderParams)
    else : 
      blobDetector = cv.SimpleBlobDetector_create(circleFinderParams)

    resultsTmp = blobDetector.detect(image=image)
    results = []
    i = 0
    while i < len(resultsTmp) :
      results.append({"radius": resultsTmp[i].size/2.0,
                      "angle": resultsTmp[i].angle,
                      "pos": (resultsTmp[i].pt[0], resultsTmp[i].pt[1], 1) })
      i += 1

    circleScale = 16
    shiftBits = 4
      
    doDebug = False
    if doDebug == True :
      width = int(image.shape[1])
      height = int(image.shape[0])

      imgScale = cv.resize(image, (width, height))
      imgScale = self.grayToRGB(imgScale)
      #cv2.imshow("Scaled Image", imgScale)
      #cv2.waitKey(3000)
      #print(imgScale.shape)
      i = 0
      while i < len(results) : #for key,val in ipairs(results) do
        x = int(round(results[i]["pos"][0]*circleScale))
        y = int(round(results[i]["pos"][1]*circleScale))
        radius = int(round(results[i]["radius"]*circleScale))
        cv.circle(img = imgScale, center = (x, y), radius = radius, color = (0,255,255), 
                  thickness = 2, lineType = cv.LINE_AA, shift = shiftBits)
        i += 1
      cv.imshow("circleFinder", imgScale)
      cv.waitKey(500)
      cv.destroyWindow(winname = "circleFinder")

    pointFindSuccess, centers = cv.findCirclesGrid(image=image, patternSize=patternSize, flags=patternType, blobDetector = blobDetector)
    #print(pointFindSuccess)
    
    if pointFindSuccess : 
      return True, centers
    else :
      print("Calibration pattern not found!")
      cv.imshow("not found image", image)
      cv.waitKey(1000)
      return False, None


  # P is a table of 3x4 projection matrices
  # measurement is a table with 1x2 image measurements
  def linearTriangulation (self, P, measurements) :
    if len(P) != len(measurements) :
      return False, None

    m = len(measurements)

    A = np.zeros(shape=(2*m,3), dtype=np.float64)
    B = np.zeros(shape=(2*m,1), dtype=np.float64)

    for i in range(0,m) :
      p = P[i]
      x = measurements[i][0][0]
      y = measurements[i][0][1]

      A[i*2+0, 0] = p[2,0]*x - p[0,0]
      A[i*2+0, 1] = p[2,1]*x - p[0,1]
      A[i*2+0, 2] = p[2,2]*x - p[0,2]

      A[i*2+1, 0] = p[2,0]*y - p[1,0]
      A[i*2+1, 1] = p[2,1]*y - p[1,1]
      A[i*2+1, 2] = p[2,2]*y - p[1,2]

      B[i*2+0, 0] = p[0,3] - x*p[2,3]
      B[i*2+1, 0] = p[1,3] - y*p[2,3]
    
    AtA = np.zeros(shape=(3,3), dtype=np.float64)
    Atb = np.matmul(np.transpose(A), B)
    AtA = np.matmul(np.transpose(A), A)
    X = np.matmul(inv(AtA), Atb)
    return True, X


  def shuffleTable(self, t) :
    iterations = len(t)
    for i in range(iterations-1, -1, -1) :
      j = random.randint(0, i) #rand(i)
      t[i], t[j] = t[j], t[i]


  # Source: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
  def transformMatrixToQuaternion(self, rot) :
    sqrt = math.sqrt
    trace = rot[0][0] + rot[1][1] + rot[2][2]
    _next = [1, 2, 0]
    q = np.zeros(shape=4, dtype=np.float64)
    if trace > 0 :
      r = sqrt(trace + 1)
      s = 0.5 / r
      q[0] = 0.5 * r
      q[1] = (rot[2][1] - rot[1][2]) * s
      q[2] = (rot[0][2] - rot[2][0]) * s
      q[3] = (rot[1][0] - rot[0][1]) * s
    else :
      i = 0
      if rot[1][1] > rot[0][0] :
        i = 1
      if rot[2][2] > rot[i][i] :
        i = 2
      j = _next[i]
      k = _next[j]
      t = rot[i][i] - rot[j][j] - rot[k][k] + 1
      r = sqrt(t)
      s = 0.5 / sqrt(t)
      w = (rot[k][j] - rot[j][k]) * s
      q[0] = w
      q[i+1] = 0.5 * r
      q[j+1] = (rot[j][i] + rot[i][j]) * s
      q[k+1] = (rot[k][i] + rot[i][k]) * s

    return q/norm(q)


  def transformQuaternionToMatrix(self, q) :
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    result = np.zeros(shape=(3,3), dtype=np.float64)
    result[0][0] = 1 - 2*y*y - 2*z*z
    result[0][1] = 2*x*y - 2*w*z
    result[0][2] = 2*x*z + 2*w*y
    result[1][0] = 2*x*y + 2*w*z
    result[1][1] = 1 - 2*x*x - 2*z*z
    result[1][2] = 2*y*z - 2*w*x
    result[2][0] = 2*x*z - 2*w*y
    result[2][1] = 2*y*z + 2*w*x
    result[2][2] = 1 - 2*x*x - 2*y*y
    return result
  

  def __init__(self) :
    self.check = True
