local calib = require 'egomo_calibration.env'
calib.xamla3d = require 'egomo_calibration.xamla3d'
require 'egomo_calibration.Calibration'

local optimizeDH = calib.lib.calib___optimizeDH
local evaluateDH = calib.lib.calib___evaluateDH

function calib.optimizeDH(
  intrinsics,           -- 3x3
  distortion,           -- 2 elments
  hand_eye,             -- 4x4
  joint_states,         -- Nx6
  robot_model,          -- 4x6
  points,               -- Nx3
  observations,         -- Nx2
  jointpoint_indices,   -- Nx2
  optimize_hand_eye,
  optimize_points,
  optimize_robot_model_theta,
  optimize_robot_model_d,
  optimize_robot_model_a,
  optimize_robot_model_alpha,
  optimize_joint_states)

  -- call ceres optimizer
  return optimizeDH(
    intrinsics:cdata(),
    distortion:cdata(),
    hand_eye:cdata(),
    joint_states:cdata(),
    robot_model:cdata(),
    points:cdata(),
    observations:cdata(),
    jointpoint_indices:cdata(),
    optimize_hand_eye or false,
    optimize_points or false,
    optimize_robot_model_theta or false,
    optimize_robot_model_d or false,
    optimize_robot_model_a or false,
    optimize_robot_model_alpha or false,
    optimize_joint_states or false
  )
end

function calib.evaluateDH(
  intrinsics,           -- 3x3
  distortion,           -- 2 elments
  hand_eye,             -- 4x4
  joint_states,         -- Nx6
  robot_model,          -- 4x6
  points,               -- Nx3
  observations,         -- Nx2
  jointpoint_indices    -- Nx2
)
  return evaluateDH(
    intrinsics:cdata(),
    distortion:cdata(),
    hand_eye:cdata(),
    joint_states:cdata(),
    robot_model:cdata(),
    points:cdata(),
    observations:cdata(),
    jointpoint_indices:cdata()
  )
end

return calib
