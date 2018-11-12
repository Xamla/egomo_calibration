#ifndef egomo_calib_h
#define egomo_calib_h


extern "C" {

  double evaluateDH(double *intrinsics, double *distortion, double *hand_eye, double *joint_states, 
                    double *robot_model, double *points, double *observations, long *jointpoint_indices,
                    int num_joint_states, int num_points);

  double optimizeDH(double *intrinsics, double *distortion, double *hand_eye, double *joint_states,
                    double *robot_model, double *points, double *observations, long *jointpoint_indices,
                    int num_joint_states, int num_points,
                    bool optimize_hand_eye, bool optimize_points, bool optimize_robot_model_theta,
                    bool optimize_robot_model_d, bool optimize_robot_model_a, bool optimize_robot_model_alpha,
                    bool optimize_joint_states, bool optimize_pp, bool optimize_focal_length,
                    bool optimize_distortion);

}

#include <stdexcept>

#endif  // egomo_calib_h