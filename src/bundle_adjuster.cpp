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

std::string which_arm = "left";
bool torso_opt = false;


template <typename T>
struct SDA10dKinematicInv {
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
    if (which_arm.compare("right") == 0) { // right arm
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
    inv_pose_init(2,3) = T(-1.06); //T(-0.9); //T(-1.06);   // negative z-translation of torso-joint

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
struct SDA10dKinematic {
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
    if (which_arm.compare("right") == 0) { // right arm
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
    pose(2,3) = T(1.06); //T(0.9); //T(1.06);   // z-translation of torso-joint (base->torso_joint_b1 + cell-bottom-height)

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
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

 template <typename T>
  bool operator()(
    const T* const robot_model_theta,
    const T* const robot_model_d,
    const T* const robot_model_a,
    const T* const robot_model_alpha,
    const T* const joint_state,
    const T* const handEye,
    const T* const point,
    const T* const focal,
    const T* const distortion,
    const T* const pp,
    T* residuals
  ) const {

    // compute forward-kinematic
    SDA10dKinematicInv<T> k;
    T pose_storage[16];
    
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
    // optimize only shoulder_height = d[0], upper_arm_length = a[1], 
    // forearm_length = a[2] and wrist{1,2,3}_length = d[3,4,5]:
    // -> Make d[1,2], a[0] and a[3,4,5] fix.
    //robot_model_d_[1] = T(0);
    //robot_model_d_[2] = T(0);
    //robot_model_a_[0] = T(0);
    //robot_model_a_[3] = T(0);
    //robot_model_a_[4] = T(0);
    //robot_model_a_[5] = T(0);

    // optimize only some of the dh-parameters and keep the other fix
    if (!torso_opt) {
      robot_model_theta_[0] = T(0.0); // left arm (theta for torso joint is not optimized, here.)
      if (which_arm.compare("right") == 0) {
        robot_model_theta_[0] = T(M_PI); // right arm (theta for torso joint is not optimized, here.)
      }
    }
    //robot_model_theta_[7] = T(0.0); // theta for t joint is not optimized, here.

    //---------------------------
    //robot_model_d_[2] = T(0);
    //robot_model_d_[4] = T(0);
    //robot_model_d_[6] = T(0);
    //---------------------------
    //robot_model_a_[1] = T(0);
    //robot_model_a_[2] = T(0);
    //robot_model_a_[3] = T(0);
    //robot_model_a_[4] = T(0);
    //robot_model_a_[5] = T(0);
    //robot_model_a_[6] = T(0);
    //robot_model_a_[7] = T(0);

    //std::cout << "robot_model_theta_ :" << std:: endl;
    //std::cout << robot_model_theta_[0] << " " << robot_model_theta_[1] << " " << robot_model_theta_[2] << " " << robot_model_theta_[3] << std::endl;
    //std::cout << robot_model_theta_[4] << " " << robot_model_theta_[5] << " " << robot_model_theta_[6] << " " << robot_model_theta_[7] << std::endl;
    //std::cout << "robot_model_alpha_ :" << std:: endl;
    //std::cout << robot_model_alpha_[0] << " " << robot_model_alpha_[1] << " " << robot_model_alpha_[2] << " " << robot_model_alpha_[3] << std::endl;
    //std::cout << robot_model_alpha_[4] << " " << robot_model_alpha_[5] << " " << robot_model_alpha_[6] << " " << robot_model_alpha_[7] << std::endl;
    //std::cout << "robot_model_d_ :" << std:: endl;
    //std::cout << robot_model_d_[0] << " " << robot_model_d_[1] << " " << robot_model_d_[2] << " " << robot_model_d_[3] << std::endl;
    //std::cout << robot_model_d_[4] << " " << robot_model_d_[5] << " " << robot_model_d_[6] << " " << robot_model_d_[7] << std::endl;
    //std::cout << "robot_model_a_ :" << std:: endl;
    //std::cout << robot_model_a_[0] << " " << robot_model_a_[1] << " " << robot_model_a_[2] << " " << robot_model_a_[3] << std::endl;
    //std::cout << robot_model_a_[4] << " " << robot_model_a_[5] << " " << robot_model_a_[6] << " " << robot_model_a_[7] << std::endl;

    k(robot_model_theta_, robot_model_d_, robot_model_a_, robot_model_alpha_, joint_state, pose_storage);
    
    //std::cout << "pose_storage^-1 (= Hg^-1 = TCP^-1) :" << std:: endl;
    //std::cout << pose_storage[0] << " " << pose_storage[1] << " " << pose_storage[2] << " " << pose_storage[3] << std::endl;
    //std::cout << pose_storage[4] << " " << pose_storage[5] << " " << pose_storage[6] << " " << pose_storage[7] << std::endl;
    //std::cout << pose_storage[8] << " " << pose_storage[9] << " " << pose_storage[10] << " " << pose_storage[11] << std::endl;
    //std::cout << pose_storage[12] << " " << pose_storage[13] << " " << pose_storage[14] << " " << pose_storage[15] << std::endl;

    //std::cout << "current point (= 3d pattern point in base coordinates):" << std:: endl;
    //std::cout << point[0] << " " << point[1] << " " << point[2] << std::endl;

    // transform point (= 3d pattern point in base coordinates)
    // -> to a 2d pattern point in camera coordinates:
    // -----------------------------------------------
    T p[3];
    // p = pose_storage^-1 * point = TCP^-1 * 3d-target-point_inBaseCoord
    p[0] = point[0] * pose_storage[0] + point[1] * pose_storage[1] + point[2] * pose_storage[2] + pose_storage[3];
    p[1] = point[0] * pose_storage[4] + point[1] * pose_storage[5] + point[2] * pose_storage[6] + pose_storage[7];
    p[2] = point[0] * pose_storage[8] + point[1] * pose_storage[9] + point[2] * pose_storage[10] + pose_storage[11];

    //std::cout << "TCP^-1 * 3d-pattern-point_inBaseCoord :" << std:: endl;
    //std::cout << p[0] << " " << p[1] << " " << p[2] << std::endl;

    // p2 = heye^-1 * p = heye^-1 * pose_storage^-1 * point
    T p2[3];
    // handEye rotation is in angle-axis-form,
    // such that we only have to optimize 3 rotation components instead of 9
    ceres::AngleAxisRotatePoint(handEye, p, p2); // handEye is handEyeInv here!
    p2[0] += handEye[3];
    p2[1] += handEye[4];
    p2[2] += handEye[5];

    //std::cout << "heye^-1 * TCP^-1 * 3d-pattern-point_inBaseCoord :" << std:: endl;
    //std::cout << p2[0] << " " << p2[1] << " " << p2[2] << std::endl;

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp =  p2[0] / p2[2];
    T yp =  p2[1] / p2[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = distortion[0];
    const T& l2 = distortion[1];
    const T& l3 = distortion[2];
    T r2 = xp*xp + yp*yp;
    T dist = T(1.0) + r2  * (l1 + l2 * r2 + l3*r2*r2);

    // Compute final projected point position.
    T predicted_x = (focal[0]  * xp * dist) + pp[0];
    T predicted_y = (focal[0]  * yp * dist) + pp[1];
    //T predicted_x = (focal[0]  * xp) + pp[0]; // without distortion
    //T predicted_y = (focal[0]  * yp) + pp[1]; // without distortion

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    //std::cout << "predicted_x = " << predicted_x << std::endl;
    //std::cout << "predicted_y = " << predicted_y << std::endl;
    //std::cout << "observed_x = " << observed_x << std::endl;
    //std::cout << "observed_y = " << observed_y << std::endl;
    //std::cout << "residuals[0] = " << residuals[0] << std::endl;
    //std::cout << "residuals[1] = " << residuals[1] << std::endl;
    //exit(1);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    // observed, robot_model_{theta,d,a,alpha}, joint_state, handeye, 3dpoint, focal, distortion, pp
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 8, 8, 8, 8, 8, 6, 3, 1, 3, 2>(
                new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};


class DHCalibration {
public:
  DHCalibration(
    double *intrinsics,   // camera intrinsics 3x3 matrix
    double *distortion,   // distortion 2 element vector
    double *hand_eye,
    double *joint_states,
    double *robot_model,
    double *points,
    double *observations,
    long *jointpoint_indices,
    int num_joint_states,
    int num_points,
    int arm,
    bool with_torso_optimization
  ) :
    intrinsics(intrinsics), distortion(distortion), hand_eye(hand_eye),
    joint_states(joint_states), robot_model(robot_model),
    points(points), observations(observations), jointpoint_indices(jointpoint_indices),
    num_joint_states(num_joint_states), num_points(num_points), arm(arm)
  {
    std::cout << "************************" << std::endl;
    std::cout << "* Class DHCAlibration: *" << std::endl;
    std::cout << "************************" << std::endl;

    //which_arm = "right";
    if (arm == 1) {
      which_arm = "right";
    }
    std::cout << "which_arm: " << which_arm << std::endl;
    if (with_torso_optimization) {
      torso_opt = true;
    }
    std::cout << "torso_opt: " << torso_opt << std::endl;

    num_observations = num_joint_states * num_points;
    std::cout << "#Joint states: " << num_joint_states << std::endl;
    std::cout << "#Points: " << num_points << std::endl;
    std::cout << "#Observations: " << num_observations << std::endl;

    focal[0] = (intrinsics[0] + intrinsics[4]) * 0.5;
    pp[0] = intrinsics[2];
    pp[1] = intrinsics[5];
    distortion_data[0] = distortion[0];
    distortion_data[1] = distortion[1];
    distortion_data[2] = distortion[4];
    
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
    

    joint_states_data = joint_states;
    points_data = points;
    observations_data = observations;
    jointpoint_indices_data = jointpoint_indices;

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
    std::cout << "handeyeInv rotAxis: (" << hand_eye_data[0] << ", " << hand_eye_data[1] << ", " << hand_eye_data[2] <<")" << std::endl;
    std::cout << "handeyeInv translation: (" << hand_eye_data[3] << ", " << hand_eye_data[4] << ", " << hand_eye_data[5] <<")" << std::endl;

    prepareProblem();
  }

  ~DHCalibration() {
    // free pointers to double arrays
    //delete [] intrinsics;
    //delete [] distortion;
    //delete [] hand_eye;
    //delete [] joint_states;
    //delete [] robot_model;
    //delete [] points;
    //delete [] observations;
    //delete [] jointpoint_indices;
    //free(intrinsics);
    //intrinsics = NULL;
    //free(distortion);
    //distortion = NULL;
    //free(hand_eye);
    //hand_eye = NULL;
  }

  void optimize(
    bool optimize_hand_eye,
    bool optimize_points,
    bool optimize_robot_model_theta,
    bool optimize_robot_model_d,
    bool optimize_robot_model_a,
    bool optimize_robot_model_alpha,
    bool optimize_joint_states,
    bool optimize_pp,
    bool optimize_focal_length,
    bool optimize_distortion
  ) {
    if (!optimize_hand_eye) {
      problem.SetParameterBlockConstant(hand_eye_data);
    }
  
    if (!optimize_points) {
      std::cout << "num_points = " << num_points << std::endl;
      for (int i = 0; i < num_points; i++) {
        problem.SetParameterBlockConstant(points_data + (i * 3));
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

    if (!optimize_joint_states) {
      std::cout << "num_joint_states = " << num_joint_states << std::endl;
      for (int i = 0; i < num_joint_states; i++) {
        problem.SetParameterBlockConstant(joint_states_data + (i * 8));
      }
    }

    if (!optimize_pp) {
      problem.SetParameterBlockConstant(pp);
    }

    if (!optimize_focal_length) {
      problem.SetParameterBlockConstant(focal);
    }

    if (!optimize_distortion) {
      problem.SetParameterBlockConstant(distortion_data);
    }
    
    double *rm = robot_model_data;
    std::cout << "Robot Model (before)" << std::endl;
    for (int i = 0; i < 8; i++) {
      std::cout<< "Joint "<< i << "(theta, d, a, alpha)" << rm[0+i] << " " << rm[8 + i] << " " << rm[16 + i]  << " " << rm[24 + i] << std::endl;
    }

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    
    ceres::Solver::Options options;

    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-16;  // default: 1e-6
    options.parameter_tolerance = 0.0; // default: 1e-8
    options.num_threads = 8;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::cout << "Focal: " << focal[0] << std::endl;
    std::cout << "Distortion: " << distortion_data[0] << " " << distortion_data[1] << " " << distortion_data[2] << std::endl;
    std::cout << "Principle Point: " << pp[0] << " " << pp[1] << std::endl;

    std::cout << "Robot Model (after)" << std::endl;
    for (int i = 0; i < 8; i++) {
      std::cout<< "Joint "<< i << "(theta, d, a, alpha)" << rm[0+i] << " " << rm[8+i] << " " << rm[16+i]  << " " << rm[24+i] << std::endl;
    }

    if (optimize_focal_length) {
      intrinsics[0] = focal[0];
      intrinsics[4] = focal[0];
    }

    if (optimize_pp) {
      intrinsics[2] = pp[0];
      intrinsics[5] = pp[1];
    }

    if (optimize_distortion) {
      distortion[0] = distortion_data[0];
      distortion[1] = distortion_data[1];
      distortion[4] = distortion_data[2];
    }

    // convert hand-eye rotation back to matrix representations
    if (optimize_hand_eye)
    {
      double rotation_cm[9];
      ceres::AngleAxisToRotationMatrix(hand_eye_data, rotation_cm);

      double rotAxis[3] = {0};
      ceres::RotationMatrixToAngleAxis(rotation_cm, rotAxis);

      /*printf("return cross check:\n");
      printf("heyedata: %f; %f; %f;    rotAxis: %f; %f; %f\n", hand_eye_data[0], hand_eye_data[1], hand_eye_data[2], rotAxis[0], rotAxis[1], rotAxis[2]);
      printf("translation: %f, %f, %f\n", hand_eye_data[3],hand_eye_data[4],hand_eye_data[5]);
      printf("rotation_cm output:");
      for (int i=0;i<9;++i) {
        printf("%d: %f\n", i, rotation_cm[i]);
      }*/

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

      //std::cout << "inverse hand_eye (after):" << std::endl;
      //std::cout << hand_eye[0] << " " << hand_eye[1] << " " << hand_eye[2] << " " << hand_eye[3] << std::endl;
      //std::cout << hand_eye[4] << " " << hand_eye[5] << " " << hand_eye[6] << " " << hand_eye[7] << std::endl;
      //std::cout << hand_eye[8] << " " << hand_eye[9] << " " << hand_eye[10] << " " << hand_eye[11] << std::endl;
      //std::cout << hand_eye[12] << " " << hand_eye[13] << " " << hand_eye[14] << " " << hand_eye[15] << std::endl;
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

  double *intrinsics, *intrinsics_;
  double *distortion, *distortion_;
  double *hand_eye, *hand_eye_;
  double *joint_states, *joint_states_;
  double *robot_model, *robot_model_;
  double *points, *points_;
  double *observations, *observations_;
  long *jointpoint_indices, *jointpoint_indices_;

  double focal[2];
  double pp[2];
  double distortion_data[3];
  double hand_eye_data[6];

  int num_joint_states;
  int num_points;
  int num_observations;
  int arm;

  double *robot_model_data;
  double *robot_model_data_theta;
  double *robot_model_data_d;
  double *robot_model_data_a;
  double *robot_model_data_alpha;

  double *joint_states_data;
  double *points_data;
  double *observations_data;
  long *jointpoint_indices_data;

  void prepareProblem() {

    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.

    //ceres::LossFunction* loss_function(new ceres::HuberLoss(1.0));
    
    for (int i = 0; i < num_observations; ++i) {
      // Each Residual block takes a point and a camera as input and outputs a 2
      // dimensional residual. Internally, the cost function stores the observed
      // image location and compares the reprojection against the observation.

      ceres::CostFunction *cost_function =
          SnavelyReprojectionError::Create(observations_data[2 * i + 0],
                                           observations_data[2 * i + 1]);

      long joint_index = jointpoint_indices_data[2 * i + 0];
      long point_index = jointpoint_indices_data[2 * i + 1];

      double *joint_state_for_observation = joint_states_data + 8 * joint_index;
      double *point_for_observation = points_data + 3 * point_index;

      problem.AddResidualBlock(
        cost_function,
        NULL,   // squared loss
        robot_model_data_theta,
        robot_model_data_d,
        robot_model_data_a,
        robot_model_data_alpha,
        joint_state_for_observation,
        hand_eye_data,
        point_for_observation,
        focal,
        distortion_data,
        pp
      );
    }
  }
};


// Function calls:
double evaluateDH(
  double *intrinsics,   // camera intrinsics 3x3 matrix
  double *distortion,   // distortion 2 element vector
  double *hand_eye,
  double *joint_states,
  double *robot_model,
  double *points,
  double *observations,
  long *jointpoint_indices,
  int num_joint_states,
  int num_points,
  int arm,
  bool with_torso_optimization
) {
  //google::InitGoogleLogging("/tmp");
  double evaluation_error = 0;
  DHCalibration calib(intrinsics, distortion, hand_eye, joint_states, robot_model, points, observations, jointpoint_indices, num_joint_states, num_points, arm, with_torso_optimization);
  std::cout << "=========================================================" << std::endl;
  std::cout << "Call of \"calcAverageReproductionError()\" in evaluation:" << std::endl;
  std::cout << "=========================================================" << std::endl;
  evaluation_error = calib.calcAverageReproductionError();
  std::cout << "evaluation_error = " << evaluation_error << std::endl;
  return evaluation_error;
}


double optimizeDH(
  double *intrinsics,   // camera intrinsics 3x3 matrix
  double *distortion,   // distortion 2 element vector
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
  bool optimize_distortion
) {
  //google::InitGoogleLogging("/tmp");
  double err = 0;

  {
    DHCalibration calib(intrinsics, distortion, hand_eye, joint_states, robot_model, points, observations, jointpoint_indices, num_joint_states, num_points, arm, with_torso_optimization);
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
      optimize_joint_states,
      optimize_pp,
      optimize_focal_length,
      optimize_distortion
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
    DHCalibration calib2(intrinsics, distortion, hand_eye, joint_states, robot_model, points, observations, jointpoint_indices, num_joint_states, num_points, arm, with_torso_optimization);
    crossCheck = calib2.calcAverageReproductionError();
    printf("err: %f, crossCheck: %f\n", err, crossCheck);
  }

  return err;
}
