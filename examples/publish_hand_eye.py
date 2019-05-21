#!/usr/bin/env python3
from __future__ import print_function
from typing import Dict, List, Tuple
import cv2 as cv
import sys
import numpy as np

from xamla_motion.world_view_client import WorldViewClient
from xamla_motion.motion_client import MoveGroup
from xamla_motion.data_types import Pose


def addOrUpdatePose(world_view_client, display_name, folder, pose):
  try:
    get_value = world_view_client.get_pose(display_name, folder)
    world_view_client.update_pose(display_name, folder, pose)
  except:
    world_view_client.add_pose(display_name, folder, pose)


def main():
    world_view_client = WorldViewClient()

    print("Please type name (with path) of hand-eye-matrix, then press \'Enter\' (e.g. calibration/left_arm_cameras/HandEye_optimized.npy).")
    hand_eye_name = sys.stdin.readline()
    print("hand_eye_name:")
    print(hand_eye_name)

    flange_link_name = 'arm_left_link_tool0'
    print("For which arm is this hand-eye? Please type l or r, then press \'Enter\'.")
    print("l: left arm")
    print("r: right arm")
    which = sys.stdin.read(1)
    if which == "r" :
      flange_link_name = 'arm_right_link_tool0'
    print("flange_link_name:")
    print(flange_link_name)

    hand_eye_npy = np.load(hand_eye_name.rstrip('\n'))
    hand_eye_pose = Pose.from_transformation_matrix(matrix=hand_eye_npy, frame_id=flange_link_name, normalize_rotation=False)
    display_name = 'hand_eye_left_arm_left_cam'
    if which == "r" :
      display_name = 'hand_eye_right_arm_left_cam'
    addOrUpdatePose(world_view_client, display_name, '/Calibration', hand_eye_pose)
    print("Published " + display_name + " into World View folder Calibration.")


if __name__ == '__main__':
    main()
