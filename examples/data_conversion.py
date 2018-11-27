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


def main():

    # set default values
    torch_calibration_path = 'None'
    python_calibration_path = 'None'
    torch_heye_path = 'None'
    python_heye_path = 'None'
    torch_jsposes_fn = 'None'
    python_jsposes_fn = 'None'
    torch_all_vals_tensors_fn = 'None'
    python_all_vals_tensors_fn = 'None'

    print('Number of arguments:')
    print(len(sys.argv))
    print('Argument List:')
    print(str(sys.argv))
    if len(sys.argv) > 1:
      torch_calibration_path = sys.argv[1]
    if len(sys.argv) > 2:
      python_calibration_path = sys.argv[2]
    if len(sys.argv) > 3:
      torch_heye_path = sys.argv[3]
    if len(sys.argv) > 4:
      python_heye_path = sys.argv[4]
    if len(sys.argv) > 5:
      torch_jsposes_fn = sys.argv[5]
    if len(sys.argv) > 6:
      python_jsposes_fn = sys.argv[6]
    if len(sys.argv) > 7:
      torch_all_vals_tensors_fn = sys.argv[7]
    if len(sys.argv) > 8:
      python_all_vals_tensors_fn = sys.argv[8]

    if torch_calibration_path != 'None' :
      stereoCalib = torchfile.load(torch_calibration_path)
      stereo_calib = {}
      stereo_calib['camLeftMatrix'] = stereoCalib[b'camLeftMatrix'].astype(np.float64)
      stereo_calib['camRightMatrix'] = stereoCalib[b'camRightMatrix'].astype(np.float64)
      stereo_calib['camLeftDistCoeffs'] = stereoCalib[b'camLeftDistCoeffs'].astype(np.float64)
      stereo_calib['camRightDistCoeffs'] = stereoCalib[b'camRightDistCoeffs'].astype(np.float64)
      stereo_calib['trafoLeftToRightCam'] = stereoCalib[b'trafoLeftToRightCam'].astype(np.float64)
      stereo_calib['R'] = stereoCalib[b'R'].astype(np.float64)
      stereo_calib['F'] = stereoCalib[b'F'].astype(np.float64)
      stereo_calib['E'] = stereoCalib[b'E'].astype(np.float64)
      stereo_calib['T'] = stereoCalib[b'T'].astype(np.float64)
      stereo_calib['reprojError'] = stereoCalib[b'reprojError']
      stereo_calib['date'] = stereoCalib[b'date']
      stereo_calib['imHeight'] = stereoCalib[b'imHeight']
      stereo_calib['imWidth'] = stereoCalib[b'imWidth']
      patternGeometry = { 'width': stereoCalib[b'patternGeometry'][b'width'], 
                          'height': stereoCalib[b'patternGeometry'][b'height'], 
                          'pointDist': stereoCalib[b'patternGeometry'][b'pointDist'] }
      stereo_calib['patternGeometry'] = patternGeometry
      calibrationFlags = {}
      calibrationFlags['CALIB_RATIONAL_MODEL'] = stereoCalib[b'calibrationFlags'][b'CALIB_RATIONAL_MODEL']
      calibrationFlags['CALIB_FIX_PRINCIPAL_POINT'] = stereoCalib[b'calibrationFlags'][b'CALIB_FIX_PRINCIPAL_POINT']
      calibrationFlags['CALIB_USE_INTRINSIC_GUESS'] = stereoCalib[b'calibrationFlags'][b'CALIB_USE_INTRINSIC_GUESS']
      calibrationFlags['CALIB_FIX_ASPECT_RATIO'] = stereoCalib[b'calibrationFlags'][b'CALIB_FIX_ASPECT_RATIO']
      calibrationFlags['CALIB_FIX_K1'] = stereoCalib[b'calibrationFlags'][b'CALIB_FIX_K1']
      calibrationFlags['CALIB_FIX_K2'] = stereoCalib[b'calibrationFlags'][b'CALIB_FIX_K2']
      calibrationFlags['CALIB_FIX_K3'] = stereoCalib[b'calibrationFlags'][b'CALIB_FIX_K3']
      calibrationFlags['CALIB_FIX_K4'] = stereoCalib[b'calibrationFlags'][b'CALIB_FIX_K4']
      calibrationFlags['CALIB_FIX_K5'] = stereoCalib[b'calibrationFlags'][b'CALIB_FIX_K5']
      calibrationFlags['CALIB_FIX_K6'] = stereoCalib[b'calibrationFlags'][b'CALIB_FIX_K6']
      calibrationFlags['CALIB_ZERO_TANGENT_DIST'] = stereoCalib[b'calibrationFlags'][b'CALIB_ZERO_TANGENT_DIST']
      stereo_calib['calibrationFlags'] = calibrationFlags
      print("Stereo calibration:")
      print(stereo_calib)
      np.save(python_calibration_path, stereo_calib)
    
    if torch_heye_path != 'None' :
      hand_eye = torchfile.load(torch_heye_path).astype(np.float64)
      print("Hand-Eye matrix:")
      print(hand_eye)
      np.save(python_heye_path, hand_eye)

    if torch_jsposes_fn != 'None' :
      jsposes = torchfile.load(torch_jsposes_fn)
      jsposes_tensors = {}
      jsposes_tensors['recorded_poses'] = jsposes[b'recorded_poses']
      jsposes_tensors['recorded_joint_values'] = jsposes[b'recorded_joint_values']
      np.save(python_jsposes_fn, jsposes_tensors)

    if torch_all_vals_tensors_fn != 'None' :
      all_vals_tensors = torchfile.load(torch_all_vals_tensors_fn)
      print("torso_joint_b1 value:")
      print(all_vals_tensors[0])
      np.save(python_all_vals_tensors_fn, all_vals_tensors)

    print('finished')


if __name__ == "__main__":
    main()
