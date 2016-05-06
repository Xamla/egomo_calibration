local ffi = require 'ffi'

local calib = {}

local calib_cdef = [[

double calib___evaluateDH(
  THDoubleTensor *intrinsics,
  THDoubleTensor *distortion,
  THDoubleTensor *hand_eye,
  THDoubleTensor *joint_states,
  THDoubleTensor *robot_model,
  THDoubleTensor *points,
  THDoubleTensor *observations,
  THLongTensor *jointpoint_indices);

double calib___optimizeDH(
  THDoubleTensor *intrinsics,
  THDoubleTensor *distortion,
  THDoubleTensor *hand_eye,
  THDoubleTensor *joint_states,
  THDoubleTensor *robot_model,
  THDoubleTensor *points,
  THDoubleTensor *observations,
  THLongTensor *jointpoint_indices,
  bool optimize_hand_eye,
  bool optimize_points,
  bool optimize_robot_model_theta,
  bool optimize_robot_model_d,
  bool optimize_robot_model_a,
  bool optimize_robot_model_alpha,
  bool optimize_joint_states);
]]

ffi.cdef(calib_cdef)

calib.lib = ffi.load(package.searchpath('libegomo_calibration', package.cpath))

return calib
