#!/usr/bin/env python

import numpy as np
import sys
from copy import deepcopy

import rospy
import actionlib
import asyncio
import functools
import signal

from xamla_motion.data_types import *
from xamla_motion.motion_service import MotionService
from xamla_motion.motion_client import MoveGroup, EndEffector


def shutdown(loop, reason):
    print('shutdown asyncio due to : {}'.format(reason), flush=True)
    tasks = asyncio.gather(*asyncio.Task.all_tasks(loop=loop),
                           loop=loop, return_exceptions=True)
    tasks.add_done_callback(lambda t: loop.stop())
    tasks.cancel()

    # Keep the event loop running until it is either destroyed or all
    # tasks have really terminated
    while not tasks.done() and not loop.is_closed():
        loop.run_forever()


loop = asyncio.get_event_loop()
loop.add_signal_handler(signal.SIGTERM, functools.partial(shutdown, loop, signal.SIGTERM))
loop.add_signal_handler(signal.SIGINT, functools.partial(shutdown, loop, signal.SIGINT))


async def main():

    which_arm = "right" # "left"

    move_group_name = '/sda10d/sda10d_r1_controller'
    if (which_arm == "right") :
      move_group_name = '/sda10d/sda10d_r2_controller'    
    move_group = MoveGroup(move_group_name)

    robotModel = np.load("robotModel_optimized.npy")
    values = np.zeros(shape=7, dtype=np.float64)
    for i in range(0, 7) :
      values[i] = robotModel[0][i+1]
    
    # multiply values[i] by -joint_direction of joint[i] !!!
    
    if (which_arm == "left") :
      # Left arm: joint_direction = ((1), -1, -1, -1, 1, 1, 1, 1) # (first joint_dir is for torso)
      values[3] = -1.0 * values[3]
      values[4] = -1.0 * values[4]
      values[5] = -1.0 * values[5]
      values[6] = -1.0 * values[6]

    if (which_arm == "right") :
      # Right arm: joint_direction = ((1), 1, 1, 1, -1, -1, -1, -1) # (first joint_dir is for torso)
      values[0] = -1.0 * values[0]
      values[1] = -1.0 * values[1]
      values[2] = -1.0 * values[2]

    print("values:")
    print(values)

    #sys.exit()

    target = JointValues(move_group.joint_set, values)

    await move_group.move_joints(target, 0.05)

if __name__ == '__main__':
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()

