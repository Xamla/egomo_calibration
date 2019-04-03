from cffi import FFI
import ctypes
import numpy as np

ffi = FFI()

ffi.cdef("""
         double optimizeDH(double *intrinsics,
                           double * distortion,
                           double *hand_eye,
                           double *joint_states,
                           double *robot_model,
                           double *points,
                           double *observations,
                           long *jointpoint_indices,
                           int num_joint_states,
                           int num_points,
                           int arm,
                           bool with_torso_optimization,
                           bool optimize_hand_eye,
                           bool optimize_points,
                           bool optimize_robot_model_theta,
                           bool optimize_robot_model_d,
                           bool optimize_robot_model_a,
                           bool optimize_robot_model_alpha,
                           bool optimize_joint_states,
                           bool optimize_pp,
                           bool optimize_focal_length,
                           bool optimize_distortion);
""")

ffi.cdef("""
         double evaluateDH(double *intrinsics,
                           double * distortion,
                           double *hand_eye,
                           double *joint_states,
                           double *robot_model,
                           double *points,
                           double *observations,
                           long *jointpoint_indices,
                           int num_joint_states,
                           int num_points,
                           int arm,
                           bool with_torso_optimization);
""")

ffi.cdef("""
         double optimizeDHV2(double *hand_eye,
                             double *joint_states,
                             double *joints_obs,
                             double *robot_model,
                             double *points,
                             double *points_obs,
                             int num_joint_states,
                             int num_points,
                             int arm,
                             bool optimize_hand_eye,
                             bool optimize_points,
                             bool optimize_robot_model_theta,
                             bool optimize_robot_model_d,
                             bool optimize_robot_model_a,
                             bool optimize_robot_model_alpha,
                             bool optimize_joint_states);
""")

ffi.cdef("""
         double evaluateDHV2(double *hand_eye,
                             double *joint_states,
                             double *joints_obs,
                             double *robot_model,
                             double *points,
                             double *points_obs,
                             int num_joint_states,
                             int num_points,
                             int arm);
""")

ffi.cdef("""
         double optimizeDH_rightArm(double *intrinsics,
                                    double * distortion,
                                    double *hand_pattern,
                                    double *inv_cam_pose,
                                    double *joint_states,
                                    double *robot_model,
                                    double *points,
                                    double *observations,
                                    long *jointpoint_indices,
                                    int num_joint_states,
                                    int num_points,
                                    bool optimize_hand_pattern,
                                    bool optimize_inv_cam_pose,
                                    bool optimize_points,
                                    bool optimize_robot_model_theta,
                                    bool optimize_robot_model_d,
                                    bool optimize_robot_model_a,
                                    bool optimize_robot_model_alpha,
                                    bool optimize_joint_states,
                                    bool optimize_pp,
                                    bool optimize_focal_length,
                                    bool optimize_distortion);
""")

ffi.cdef("""
         double evaluateDH_rightArm(double *intrinsics,
                                    double * distortion,
                                    double *hand_pattern,
                                    double *inv_cam_pose,
                                    double *joint_states,
                                    double *robot_model,
                                    double *points,
                                    double *observations,
                                    long *jointpoint_indices,
                                    int num_joint_states,
                                    int num_points);
""")

clib = ffi.dlopen("../build/libegomo_calibration_py.so")
