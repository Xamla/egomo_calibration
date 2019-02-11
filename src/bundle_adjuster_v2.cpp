// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A minimal, self-contained bundle adjuster using Ceres.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include "egomo_calibration.h"
#include <vector>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <math.h> // definition of M_PI

std::string arm_choice = "left";

template <typename T>
struct SDA10dKineInv {
  typedef ceres::MatrixAdapter<T,4,1> Mat44;

  // matrix math helpers...

  void mul(Mat44 &R, const Mat44 &A, Mat44 &B) {
    for (int i=0; i<4; ++i) {
      for (int j=0; j<4; ++j) {
        T sum(0.0);
        for (int k = 0; k < 4; ++k) {
          sum += A(i, k) * B(k, j);
        }
        R(i,j) = sum;
      }
    }
  }

  void copy(Mat44 &dst, const Mat44 &src) {
    for (int i=0; i<4; ++i) {
      for (int j=0; j<4; ++j) {
        dst(i,j) = src(i,j);
      }
    }
  }

  void identity(Mat44 &M) {
    for (int i=0; i<4; ++i) {
      for (int j=0; j<4; ++j) {
        M(i,j) = T(i==j ? 1.0 : 0.0);
      }
    }
  }

  // for calculation of inverse tcp-pose (pose_storage^-1):
  void dh_inverse(Mat44 &result, const T &theta, const T &d, const T &a, const T &alpha) {
    result(0,0) = cos(theta);
    result(0,1) = sin(theta);
    result(0,2) = T(0);
    result(0,3) = -a;

    result(1,0) = -sin(theta) * cos(alpha);
    result(1,1) = cos(theta) * cos(alpha);
    result(1,2) = sin(alpha);
    result(1,3) = -d * sin(alpha);

    result(2,0) = sin(alpha) * sin(theta);
    result(2,1) = -cos(theta) * sin(alpha);
    result(2,2) = cos(alpha);
    result(2,3) =  -d *cos(alpha);

    result(3,0) = T(0);
    result(3,1) = T(0);
    result(3,2) = T(0);
    result(3,3) = T(1);
  }

  void operator()(
    const T* const theta,
    const T* const d,
    const T* const a,
    const T* const alpha,
    const T* const joints,
    T* output_pose
  ) {
    T joint_dir[8] = { T(1), T(-1), T(-1), T(-1), T(1), T(1), T(1), T(1) }; // left arm
    if (arm_choice.compare("right") == 0) { // right arm
      joint_dir[0] = T(1);
      joint_dir[1] = T(1);
      joint_dir[2] = T(1);
      joint_dir[3] = T(1);
      joint_dir[4] = T(-1);
      joint_dir[5] = T(-1);
      joint_dir[6] = T(-1);
      joint_dir[7] = T(-1);
    }

    // compute forward kinematic
    T pose_storage[16];
    Mat44 pose(pose_storage);
    identity(pose);
    T inv_pose_storage_init[16];
    Mat44 inv_pose_init(inv_pose_storage_init);
    identity(inv_pose_init);
    inv_pose_init(0,3) = T(-0.0925); // negative x-translation of torso-joint
    inv_pose_init(2,3) = T(-1.06);   // negative z-translation of torso-joint

    // Note: Here, base is the floor ground under the robot!

    T tmp_storage[16];
    Mat44 tmp(tmp_storage);
    for (int i=7; i>=0; --i) {  // calculation of inverse tcp-pose (pose_storage^-1)
      T t_storage[16];
      Mat44 t(t_storage);
      dh_inverse(t, joint_dir[i] * joints[i] + theta[i], d[i], a[i], alpha[i]);
      
      // T^-1 = identity * t7^-1 * t6^-1 * ... * t0^-1 * T_init^-1 
      mul(tmp, pose, t); // T^-1 = T^-1 * t
      copy(pose, tmp);
    }
    mul(tmp, pose, inv_pose_init); // T^-1 = T^-1 * T_init^-1
    copy(pose, tmp);

    Mat44 output(output_pose);
    copy(output, pose);
  }
};


template <typename T>
struct SDA10dKine {
  typedef ceres::MatrixAdapter<T,4,1> Mat44;

  // matrix math helpers...

  void mul(Mat44 &R, const Mat44 &A, Mat44 &B) {
    for (int i=0; i<4; ++i) {
      for (int j=0; j<4; ++j) {
        T sum(0.0);
        for (int k = 0; k < 4; ++k) {
          sum += A(i, k) * B(k, j);
        }
        R(i,j) = sum;
      }
    }
  }

  void copy(Mat44 &dst, const Mat44 &src) {
    for (int i=0; i<4; ++i) {
      for (int j=0; j<4; ++j) {
        dst(i,j) = src(i,j);
      }
    }
  }

  void identity(Mat44 &M) {
    for (int i=0; i<4; ++i) {
      for (int j=0; j<4; ++j) {
        M(i,j) = T(i==j ? 1.0 : 0.0);
      }
    }
  }

  void dh(Mat44 &result, const T &theta, const T &d, const T &a, const T &alpha) {
    result(0,0) = cos(theta);
    result(0,1) = -sin(theta) * cos(alpha);
    result(0,2) = sin(theta) * sin(alpha);
    result(0,3) = a * cos(theta);

    result(1,0) = sin(theta);
    result(1,1) = cos(theta) * cos(alpha);
    result(1,2) = -cos(theta) * sin(alpha);
    result(1,3) = a * sin(theta);

    result(2,0) = T(0);
    result(2,1) = sin(alpha);
    result(2,2) = cos(alpha);
    result(2,3) = d;

    result(3,0) = T(0);
    result(3,1) = T(0);
    result(3,2) = T(0);
    result(3,3) = T(1);
  }

  void operator()(
    const T* const theta,
    const T* const d,
    const T* const a,
    const T* const alpha,
    const T* const joints,
    T* output_pose
  ) {
    T joint_dir[8] = { T(1), T(-1), T(-1), T(-1), T(1), T(1), T(1), T(1) }; // left arm
    if (arm_choice.compare("right") == 0) { // right arm
      joint_dir[0] = T(1);
      joint_dir[1] = T(1);
      joint_dir[2] = T(1);
      joint_dir[3] = T(1);
      joint_dir[4] = T(-1);
      joint_dir[5] = T(-1);
      joint_dir[6] = T(-1);
      joint_dir[7] = T(-1);
    }
    // compute forward kinematic
    T pose_storage[16];
    Mat44 pose(pose_storage);
    identity(pose);
    pose(0,3) = T(0.0925); // x-translation of torso-joint
    pose(2,3) = T(1.06);   // z-translation of torso-joint (base->torso_joint_b1 + cell-bottom-height)

    // Note: Here, base is the floor ground under the robot!

    T tmp_storage[16];
    Mat44 tmp(tmp_storage);
    for (int i=0; i<=7; ++i) {  // calculation of tcp-pose (pose_storage)
      T t_storage[16];
      Mat44 t(t_storage);
      dh(t, joint_dir[i] * joints[i] + theta[i], d[i], a[i], alpha[i]);
      
      // T = T_init * t0 * t1 * ... * t7 
      mul(tmp, pose, t); // T = T * t
      copy(pose, tmp);
    }

    Mat44 output(output_pose);
    copy(output, pose);
  }
};


// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojection {
  //SnavelyReprojection(double observed_x, double observed_y, double observed_z, double joint_0, double joint_1, double joint_2, double joint_3, double joint_4, double joint_5, double joint_6, double joint_7)
  //    : observed_x(observed_x), observed_y(observed_y), observed_z(observed_z), joint_0(joint_0), joint_1(joint_1), joint_2(joint_2), joint_3(joint_3), joint_4(joint_4), joint_5(joint_5), joint_6(joint_6), joint_7(joint_7) {}
  SnavelyReprojection(double observed_x, double observed_y, double observed_z)
      : observed_x(observed_x), observed_y(observed_y), observed_z(observed_z) {}

 template <typename T>
  bool operator()(
    const T* const robot_model_theta,
    const T* const robot_model_d,
    const T* const robot_model_a,
    const T* const robot_model_alpha,
    const T* const joint_state_pred,
    const T* const joint_state_obs,
    const T* const handEye,
    const T* const point_pred,
    T* residuals
  ) const {

    // compute forward-kinematic
    SDA10dKine<T> k;
    T pose_storage_pred[16];
    T pose_storage_obs[16];

    T point_obs[3];
    point_obs[0] = T(observed_x);
    point_obs[1] = T(observed_y);
    point_obs[2] = T(observed_z);
    
    T robot_model_d_[8];
    T robot_model_a_[8];
    T robot_model_theta_[8];
    T robot_model_alpha_[8];
    for (int i = 0; i < 8; ++i) {
      robot_model_d_[i] = robot_model_d[i];
      robot_model_a_[i] = robot_model_a[i];
      robot_model_theta_[i] = robot_model_theta[i];
      robot_model_alpha_[i] = robot_model_alpha[i];
    }

    // optimize only some of the dh-parameters 
    // and keep the other fix
    robot_model_theta_[0] = T(0.0); // left arm (theta for torso joint is not optimized, here.)
    if (arm_choice.compare("right") == 0) { 
      robot_model_theta_[0] = T(M_PI); // right arm (theta for torso joint is not optimized, here.)
    }

    /*
    std::cout << "robot_model_theta_ :" << std:: endl;
    std::cout << robot_model_theta_[0] << " " << robot_model_theta_[1] << " " << robot_model_theta_[2] << " " << robot_model_theta_[3] << std::endl;
    std::cout << robot_model_theta_[4] << " " << robot_model_theta_[5] << " " << robot_model_theta_[6] << " " << robot_model_theta_[7] << std::endl;
    std::cout << "robot_model_alpha_ :" << std:: endl;
    std::cout << robot_model_alpha_[0] << " " << robot_model_alpha_[1] << " " << robot_model_alpha_[2] << " " << robot_model_alpha_[3] << std::endl;
    std::cout << robot_model_alpha_[4] << " " << robot_model_alpha_[5] << " " << robot_model_alpha_[6] << " " << robot_model_alpha_[7] << std::endl;
    std::cout << "robot_model_d_ :" << std:: endl;
    std::cout << robot_model_d_[0] << " " << robot_model_d_[1] << " " << robot_model_d_[2] << " " << robot_model_d_[3] << std::endl;
    std::cout << robot_model_d_[4] << " " << robot_model_d_[5] << " " << robot_model_d_[6] << " " << robot_model_d_[7] << std::endl;
    std::cout << "robot_model_a_ :" << std:: endl;
    std::cout << robot_model_a_[0] << " " << robot_model_a_[1] << " " << robot_model_a_[2] << " " << robot_model_a_[3] << std::endl;
    std::cout << robot_model_a_[4] << " " << robot_model_a_[5] << " " << robot_model_a_[6] << " " << robot_model_a_[7] << std::endl;
    */
    
    //std::cout << "point_pred :" << std:: endl;
    //std::cout << point_pred[0] << " " << point_pred[1] << " " << point_pred[2] << std::endl;
    //std::cout << "point_obs :" << std:: endl;
    //std::cout << point_obs[0] << " " << point_obs[1] << " " << point_obs[2] << std::endl;

    k(robot_model_theta_, robot_model_d_, robot_model_a_, robot_model_alpha_, joint_state_pred, pose_storage_pred);
    k(robot_model_theta_, robot_model_d_, robot_model_a_, robot_model_alpha_, joint_state_obs, pose_storage_obs);

    /*
    std::cout << "pose_storage_pred (= Hg = TCP) :" << std:: endl;
    std::cout << pose_storage_pred[0] << " " << pose_storage_pred[1] << " " << pose_storage_pred[2] << " " << pose_storage_pred[3] << std::endl;
    std::cout << pose_storage_pred[4] << " " << pose_storage_pred[5] << " " << pose_storage_pred[6] << " " << pose_storage_pred[7] << std::endl;
    std::cout << pose_storage_pred[8] << " " << pose_storage_pred[9] << " " << pose_storage_pred[10] << " " << pose_storage_pred[11] << std::endl;
    std::cout << pose_storage_pred[12] << " " << pose_storage_pred[13] << " " << pose_storage_pred[14] << " " << pose_storage_pred[15] << std::endl;

    std::cout << "pose_storage_obs (= Hg = TCP) :" << std:: endl;
    std::cout << pose_storage_obs[0] << " " << pose_storage_obs[1] << " " << pose_storage_obs[2] << " " << pose_storage_obs[3] << std::endl;
    std::cout << pose_storage_obs[4] << " " << pose_storage_obs[5] << " " << pose_storage_obs[6] << " " << pose_storage_obs[7] << std::endl;
    std::cout << pose_storage_obs[8] << " " << pose_storage_obs[9] << " " << pose_storage_obs[10] << " " << pose_storage_obs[11] << std::endl;
    std::cout << pose_storage_obs[12] << " " << pose_storage_obs[13] << " " << pose_storage_obs[14] << " " << pose_storage_obs[15] << std::endl;
    */

    // transform point (= 3d pattern point in camera coordinates)
    // -> to a pattern point in base coordinates:
    // ------------------------------------------
    T p_pred[3];
    T p_obs[3];
    // p = heye * point_pred = heye * 3d-pattern-point_inCamCoord
    ceres::AngleAxisRotatePoint(handEye, point_pred, p_pred);
    p_pred[0] += handEye[3];
    p_pred[1] += handEye[4];
    p_pred[2] += handEye[5];
    ceres::AngleAxisRotatePoint(handEye, point_obs, p_obs);
    p_obs[0] += handEye[3];
    p_obs[1] += handEye[4];
    p_obs[2] += handEye[5];

    //std::cout << "p_pred :" << std:: endl;
    //std::cout << p_pred[0] << " " << p_pred[1] << " " << p_pred[2] << std::endl;
    //std::cout << "p_obs :" << std:: endl;
    //std::cout << p_obs[0] << " " << p_obs[1] << " " << p_obs[2] << std::endl;
    
    T p2_pred[3];
    T p2_obs[3];
    // p2 = pose_storage * heye * point
    p2_pred[0] = pose_storage_pred[0] * p_pred[0] + pose_storage_pred[1] * p_pred[1] + pose_storage_pred[2] * p_pred[2] + pose_storage_pred[3];
    p2_pred[1] = pose_storage_pred[4] * p_pred[0] + pose_storage_pred[5] * p_pred[1] + pose_storage_pred[6] * p_pred[2] + pose_storage_pred[7];
    p2_pred[2] = pose_storage_pred[8] * p_pred[0] + pose_storage_pred[9] * p_pred[1] + pose_storage_pred[10] * p_pred[2] + pose_storage_pred[11];
    p2_obs[0] = pose_storage_obs[0] * p_obs[0] + pose_storage_obs[1] * p_obs[1] + pose_storage_obs[2] * p_obs[2] + pose_storage_obs[3];
    p2_obs[1] = pose_storage_obs[4] * p_obs[0] + pose_storage_obs[5] * p_obs[1] + pose_storage_obs[6] * p_obs[2] + pose_storage_obs[7];
    p2_obs[2] = pose_storage_obs[8] * p_obs[0] + pose_storage_obs[9] * p_obs[1] + pose_storage_obs[10] * p_obs[2] + pose_storage_obs[11];

    //std::cout << "p2_pred :" << std:: endl;
    //std::cout << p2_pred[0] << " " << p2_pred[1] << " " << p2_pred[2] << std::endl;
    //std::cout << "p2_obs :" << std:: endl;
    //std::cout << p2_obs[0] << " " << p2_obs[1] << " " << p2_obs[2] << std::endl;

    // The error is the difference between the predicted and observed pattern point position (in base coordinates).
    residuals[0] = p2_pred[0] - p2_obs[0];
    residuals[1] = p2_pred[1] - p2_obs[1];
    residuals[2] = p2_pred[2] - p2_obs[2];

    //std::cout << "residuals[0] = " << residuals[0] << std::endl;
    //std::cout << "residuals[1] = " << residuals[1] << std::endl;
    //std::cout << "residuals[2] = " << residuals[2] << std::endl;
    //exit(1);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y,
                                     const double observed_z) {
    // observed point, robot_model_{theta,d,a,alpha}, joint_state of prediction, joint_state of observation, 
    // handeye, predicted point
    return (new ceres::AutoDiffCostFunction<SnavelyReprojection, 3, 8, 8, 8, 8, 8, 8, 6, 3>(
                new SnavelyReprojection(observed_x, observed_y, observed_z)));
  }

  double observed_x;
  double observed_y;
  double observed_z;
};


class DHCalibrationV2 {
public:
  DHCalibrationV2(
    double *hand_eye,
    double *joint_states_pred,
    double *joint_states_obs,
    double *robot_model,
    double *points_pred,
    double *points_obs,
    int num_joint_states, // number of images
    int num_points, // number of points per pattern (e.g. 168)
    int arm
  ) :
    hand_eye(hand_eye), joint_states_pred(joint_states_pred), joint_states_obs(joint_states_obs), 
    robot_model(robot_model), points_pred(points_pred), points_obs(points_obs), 
    num_joint_states(num_joint_states), num_points(num_points), arm(arm)
  {
    std::cout << "**************************" << std::endl;
    std::cout << "* Class DHCAlibrationV2: *" << std::endl;
    std::cout << "**************************" << std::endl;

    //arm_choice = "right";
    if (arm == 1) {
      arm_choice = "right";
    }
    std::cout << "arm_choice: " << arm_choice << std::endl;
    
    //num_observations = num_joint_states * num_points;
    std::cout << "# Stereo-Images: " << num_joint_states << std::endl;
    std::cout << "# Points per pattern: " << num_points << std::endl;
    //std::cout << "#Observations: " << num_observations << std::endl;
    
    robot_model_data = robot_model;
    robot_model_data_theta = robot_model_data;
    robot_model_data_d = robot_model_data + 8;
    robot_model_data_a = robot_model_data + 16;
    robot_model_data_alpha = robot_model_data + 24;
    
    std::cout << "robot_model_data_theta: " 
              << robot_model_data_theta[0] << " " 
              << robot_model_data_theta[1] << " "
              << robot_model_data_theta[2] << " "
              << robot_model_data_theta[3] << " "
              << robot_model_data_theta[4] << " "
              << robot_model_data_theta[5] << " "
              << robot_model_data_theta[6] << " "
              << robot_model_data_theta[7] << " " 
              << std::endl;
    std::cout << "robot_model_data_d: " 
              << robot_model_data_d[0] << " " 
              << robot_model_data_d[1] << " "
              << robot_model_data_d[2] << " "
              << robot_model_data_d[3] << " "
              << robot_model_data_d[4] << " "
              << robot_model_data_d[5] << " "
              << robot_model_data_d[6] << " "
              << robot_model_data_d[7] << " " 
              << std::endl;
    std::cout << "robot_model_data_a: " 
              << robot_model_data_a[0] << " " 
              << robot_model_data_a[1] << " "
              << robot_model_data_a[2] << " "
              << robot_model_data_a[3] << " "
              << robot_model_data_a[4] << " "
              << robot_model_data_a[5] << " "
              << robot_model_data_a[6] << " "
              << robot_model_data_a[7] << " " 
              << std::endl;
    std::cout << "robot_model_data_alpha: " 
              << robot_model_data_alpha[0] << " " 
              << robot_model_data_alpha[1] << " "
              << robot_model_data_alpha[2] << " "
              << robot_model_data_alpha[3] << " "
              << robot_model_data_alpha[4] << " "
              << robot_model_data_alpha[5] << " "
              << robot_model_data_alpha[6] << " "
              << robot_model_data_alpha[7] << " " 
              << std::endl;
    

    joint_states_pred_data = joint_states_pred;
    joint_states_obs_data = joint_states_obs;
    points_pred_data = points_pred;
    points_obs_data = points_obs;

    // read hand-eye matrix
    double angle_axis[3];
    double translation[3] = { hand_eye[3], hand_eye[7], hand_eye[11] };
    // convert to angle-axis representation
    {
      double rotation[9] = {
        hand_eye[0], hand_eye[1], hand_eye[2],
        hand_eye[4], hand_eye[5], hand_eye[6],
        hand_eye[8], hand_eye[9], hand_eye[10]
      };
      double rotation_cm[9] = { 0 };
      rotation_cm[0] = rotation[0];
      rotation_cm[1] = rotation[3];
      rotation_cm[2] = rotation[6];
      rotation_cm[3] = rotation[1];
      rotation_cm[4] = rotation[4];
      rotation_cm[5] = rotation[7];
      rotation_cm[6] = rotation[2];
      rotation_cm[7] = rotation[5];
      rotation_cm[8] = rotation[8];
      ceres::RotationMatrixToAngleAxis(rotation_cm, angle_axis);
    }
    hand_eye_data[0] = angle_axis[0];
    hand_eye_data[1] = angle_axis[1];
    hand_eye_data[2] = angle_axis[2];
    hand_eye_data[3] = translation[0];
    hand_eye_data[4] = translation[1];
    hand_eye_data[5] = translation[2];
    std::cout << "handeye rotAxis: (" << hand_eye_data[0] << ", " << hand_eye_data[1] << ", " << hand_eye_data[2] <<")" << std::endl;
    std::cout << "handeye translation: (" << hand_eye_data[3] << ", " << hand_eye_data[4] << ", " << hand_eye_data[5] <<")" << std::endl;

    prepareProblem();
  }

  ~DHCalibrationV2() {
    // free pointers to double arrays
  }

  void optimize(
    bool optimize_hand_eye,
    bool optimize_points,
    bool optimize_robot_model_theta,
    bool optimize_robot_model_d,
    bool optimize_robot_model_a,
    bool optimize_robot_model_alpha,
    bool optimize_joint_states
  ) {
    if (!optimize_hand_eye) {
      problem.SetParameterBlockConstant(hand_eye_data);
    }
  
    if (!optimize_points) {
      std::cout << "num_points = " << num_points << std::endl;
      for (int i = 0; i < num_points; i++) {
        problem.SetParameterBlockConstant(points_pred_data + (i * 3));
      }
    }

    if (!optimize_robot_model_theta) {
      problem.SetParameterBlockConstant(robot_model_data_theta);
    }
  
    if (!optimize_robot_model_d) {
      problem.SetParameterBlockConstant(robot_model_data_d);
    }

    if (!optimize_robot_model_a) {
      problem.SetParameterBlockConstant(robot_model_data_a);
    }

    if (!optimize_robot_model_alpha) {
      problem.SetParameterBlockConstant(robot_model_data_alpha);
    }

    //if (!optimize_joint_states) {
    std::cout << "num_joint_states = " << num_joint_states << std::endl;
    for (int i = 0; i < num_joint_states; i++) {
      problem.SetParameterBlockConstant(joint_states_pred_data + (i * 8));
    }

    for (int i = 0; i < num_joint_states; i++) {
      problem.SetParameterBlockConstant(joint_states_obs_data + (i * 8));
    }
    //}
    
    double *rm = robot_model_data;
    std::cout << "Robot Model (before)" << std::endl;
    for (int i = 0; i < 8; i++) {
      std::cout<< "Joint "<< i << "(theta, d, a, alpha)" << rm[0+i] << " " << rm[8 + i] << " " << rm[16 + i]  << " " << rm[24 + i] << std::endl;
    }

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    
    ceres::Solver::Options options;
    //options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.linear_solver_type = ceres::CGNR;
    //options.min_linear_solver_iterations = 5;
    //options.preconditioner_type = ceres::JACOBI;
    //options.linear_solver_ordering = ???;   
    //options.nonlinear_conjugate_gradient_type = ceres::FLETCHER_REEVES;

    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-16;  // default: 1e-6
    options.parameter_tolerance = 0.0; // default: 1e-8
    options.num_threads = 8;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::cout << "Robot Model (after)" << std::endl;
    for (int i = 0; i < 8; i++) {
      std::cout<< "Joint "<< i << "(theta, d, a, alpha)" << rm[0+i] << " " << rm[8+i] << " " << rm[16+i]  << " " << rm[24+i] << std::endl;
    }

    // convert hand-eye rotation back to matrix representations
    if (optimize_hand_eye)
    {
      double rotation_cm[9];
      ceres::AngleAxisToRotationMatrix(hand_eye_data, rotation_cm);

      double rotAxis[3] = {0};
      ceres::RotationMatrixToAngleAxis(rotation_cm, rotAxis);

      double rotation[9];
      rotation[0] = rotation_cm[0];
      rotation[3] = rotation_cm[1];
      rotation[6] = rotation_cm[2];
      rotation[1] = rotation_cm[3];
      rotation[4] = rotation_cm[4];
      rotation[7] = rotation_cm[5];
      rotation[2] = rotation_cm[6];
      rotation[5] = rotation_cm[7];
      rotation[8] = rotation_cm[8];

      // rotation part
      hand_eye[0] = rotation[0]; 
      hand_eye[1] = rotation[1];
      hand_eye[2] = rotation[2];
      hand_eye[4] = rotation[3];
      hand_eye[5] = rotation[4];
      hand_eye[6] = rotation[5];
      hand_eye[8] = rotation[6];
      hand_eye[9] = rotation[7];
      hand_eye[10] = rotation[8];

      // translation part
      hand_eye[3] = hand_eye_data[3];
      hand_eye[7] = hand_eye_data[4];
      hand_eye[11] = hand_eye_data[5];
    }
  }

  double calcAverageReproductionError() {
    double cost;
    std::vector<double> residuals;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, NULL, NULL);
    double sum = 0;
    for (size_t i = 0; i < residuals.size(); i = i+2) {
      sum += sqrt((residuals[i] * residuals[i]) + (residuals[i+1] * residuals[i+1]));
    }
    std::cout << "residuals.size() = " << residuals.size() << std::endl;
    double result = 0.0;
    if (residuals.size() > 0) {
      result = sum / (double(residuals.size()) / 2.0);
    }
    else {
      result = 0.0;
    }
    std::cout << "result: " << result << std::endl;
    return result;
  }

//private:
public:
  ceres::Problem problem;

  double *hand_eye, *hand_eye_;
  double *joint_states_pred, *joint_states_pred_;
  double *joint_states_obs, *joint_states_obs_;
  double *robot_model, *robot_model_;
  double *points_pred, *points_pred_;
  double *points_obs, *points_obs_;

  double hand_eye_data[6];

  int num_joint_states;
  int num_points;
  int arm;

  double *robot_model_data;
  double *robot_model_data_theta;
  double *robot_model_data_d;
  double *robot_model_data_a;
  double *robot_model_data_alpha;

  double *joint_states_pred_data;
  double *joint_states_obs_data;
  double *points_pred_data;
  double *points_obs_data;

  void prepareProblem() {

    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.

    for (int i = 0; i < num_joint_states; ++i) { // iterate over images
      for (int j = 0; j < num_points; ++j) { // iterate over pattern points

        // Each Residual block takes a pattern point in camera coordinates from one image 
        // and the same pattern point in camera coordinates from another image as input. 
        // Then by multiplication with the hand-eye-matrix and the robot pose (calculated
        // with forward kinematic) both points are transformed into base coordinates and
        // compared to each other.
        // Output is a 3 dimensional residual (calculated as the difference of these two points). 

        ceres::CostFunction *cost_function =
          SnavelyReprojection::Create(points_obs_data[i*3*num_points + 3 * j + 0],
                                      points_obs_data[i*3*num_points + 3 * j + 1],
                                      points_obs_data[i*3*num_points + 3 * j + 2]
                                      /*joint_states_data[8 * i + 0],
                                      joint_states_data[8 * i + 1],
                                      joint_states_data[8 * i + 2],
                                      joint_states_data[8 * i + 3],
                                      joint_states_data[8 * i + 4],
                                      joint_states_data[8 * i + 5],
                                      joint_states_data[8 * i + 6],
                                      joint_states_data[8 * i + 7]*/
                                      );

        double *joint_state_obs = joint_states_obs_data + 8 * i;

        for (int k = 0; k < num_joint_states; ++k) { // iterate over images
          if (k != i) { // take pattern point from another image for the prediction
            double *joint_state_pred = joint_states_pred_data + 8 * k;
            double *point_pred = points_pred_data + k*3*num_points + 3 * j;

            problem.AddResidualBlock(
              cost_function,
              NULL,   // squared loss
              robot_model_data_theta,
              robot_model_data_d,
              robot_model_data_a,
              robot_model_data_alpha,
              joint_state_pred,
              joint_state_obs,
              hand_eye_data,
              point_pred
            );

          }
        }
      }
    }

  }
};


// Function calls:
double evaluateDHV2(
  double *hand_eye,
  double *joint_states_pred,
  double *joint_states_obs,
  double *robot_model,
  double *points_pred,
  double *points_obs,
  int num_joint_states,
  int num_points,
  int arm
) {
  //google::InitGoogleLogging("/tmp");
  double evaluation_error = 0;
  DHCalibrationV2 calib(hand_eye, joint_states_pred, joint_states_obs, robot_model, points_pred, points_obs, num_joint_states, num_points, arm);
  std::cout << "=========================================================" << std::endl;
  std::cout << "Call of \"calcAverageReproductionError()\" in evaluation:" << std::endl;
  std::cout << "=========================================================" << std::endl;
  evaluation_error = calib.calcAverageReproductionError();
  std::cout << "evaluation_error = " << evaluation_error << std::endl;
  return evaluation_error;
}


double optimizeDHV2(
  double *hand_eye,
  double *joint_states_pred,
  double *joint_states_obs,
  double *robot_model,
  double *points_pred,
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
  bool optimize_joint_states
) {
  //google::InitGoogleLogging("/tmp");
  double err = 0;

  {
    DHCalibrationV2 calib(hand_eye, joint_states_pred, joint_states_obs, robot_model, points_pred, points_obs, num_joint_states, num_points, arm);
    std::cout << "===============================================================" << std::endl;
    std::cout << "Call of \"calcAverageReproductionError()\" before optimization:" << std::endl;
    std::cout << "===============================================================" << std::endl;
    calib.calcAverageReproductionError();
    std::cout << "=========================" << std::endl;
    std::cout << "= Call of \"optimize\" =:" << std::endl;
    std::cout << "=========================" << std::endl;
    calib.optimize(
      optimize_hand_eye,
      optimize_points,
      optimize_robot_model_theta,
      optimize_robot_model_d,
      optimize_robot_model_a,
      optimize_robot_model_alpha,
      optimize_joint_states
    );
    std::cout << "===============================================================" << std::endl;
    std::cout << "Call of \"calcAverageReproductionError()\" after optimization:" << std::endl;
    std::cout << "===============================================================" << std::endl;
    err = calib.calcAverageReproductionError();
    std::cout << "err = " << err << std::endl;
  }

  double crossCheck = 0;
  {
    std::cout << "======================================================" << std::endl;
    std::cout << "CrossCheck call of \"calcAverageReproductionError()\":" << std::endl;
    std::cout << "======================================================" << std::endl;
    DHCalibrationV2 calib2(hand_eye, joint_states_pred, joint_states_obs, robot_model, points_pred, points_obs, num_joint_states, num_points, arm);
    crossCheck = calib2.calcAverageReproductionError();
    printf("err: %f, crossCheck: %f\n", err, crossCheck);
  }

  return err;
}
