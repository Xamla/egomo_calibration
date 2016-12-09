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
// A minimal, self-contained bundle adjuster using Ceres, that reads
// files from University of Washington' Bundle Adjustment in the Large dataset:
// http://grail.cs.washington.edu/projects/bal
//
// This does not use the best configuration for solving; see the more involved
// bundle_adjuster.cc file for details.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include "egomo_calibration.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

/*
void printPoints(std::string filename) {
  std::ofstream out(filename.c_str());
  for (int i = 0; i < num_pts(); i++) {
    out << mutable_points()[i*3+0] << " " << mutable_points()[i*3+1] << " " << mutable_points()[i*3+2] << std::endl;
  }
}


void printRobotModel() {
  std::cout << "Robot Model" << std::endl;
  double* rm = mutable_robot_model();
  for (int i = 0; i < 6; i++) {
    std::cout<< "Joint "<< i << "(theta, d, a, alpha)" <<   rm[i*4+0] << " " <<  rm[i*4+1] << " " <<  rm[i*4+2]  << " " <<  rm[i*4+3] << std::endl;
  }
}

void printHandEye() {
  std::cout << (parameters_ + hand_eye_offset_)[0] << " ";
  std::cout << (parameters_ + hand_eye_offset_)[1] << " ";
  std::cout << (parameters_ + hand_eye_offset_)[2] << " ";
  std::cout << (parameters_ + hand_eye_offset_)[3] << " ";
  std::cout << (parameters_ + hand_eye_offset_)[4] << " ";
  std::cout << (parameters_ + hand_eye_offset_)[5] << std::endl;
}
*/

template <typename T>
struct UR5Kinematic {
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
    result(2,3) = -d * cos(alpha);

    result(3,0) = T(0);
    result(3,1) = T(0);
    result(3,2) = T(0);
    result(3,3) = T(1);
  }

  void operator()(
    const T* const robot_model_theta,
    const T* const robot_model_d,
    const T* const robot_model_a,
    const T* const robot_model_alpha,
    const T* const joints,
    T* output_pose
  ) {
    T joint_dir[6] = { T(1), T(-1), T(-1), T(-1), T(1), T(-1) };

    // compute forward kinematic
    T pose_storage[16];
    Mat44 pose(pose_storage);
    identity(pose);

    T tmp_storage[16];
    Mat44 tmp(tmp_storage);
    //for (int i=5; i>=0; --i) {  // ##
    for (int i=4; i>=0; --i) {
      T t_storage[16];
      Mat44 t(t_storage);
      dh_inverse(t, joint_dir[i] * joints[i] + robot_model_theta[i], robot_model_d[i], robot_model_a[i], robot_model_alpha[i]);

      //T = T * t
      mul(tmp, pose, t);
      copy(pose, tmp);
    }

    // 3. extract robot rotation and translation from pose
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
    UR5Kinematic<T> k;
    T pose_storage[16];

    T robot_model_d_[6];
    T robot_model_a_[6];

    for (int i = 0; i < 6; ++i) {
      robot_model_d_[i] = robot_model_d[i];
      robot_model_a_[i] = robot_model_a[i];
    }

    // optimize only  shoulder_height, upper_arm_length, forearm_length, wrist{1,2,3}_length

    //robot_model_d_[0] = shoulder_height
    robot_model_d_[1] = T(0);
    robot_model_d_[2] = T(0);
    //robot_model_d_[3] = -wrist_1_length
    //robot_model_d_[4] = wrist_2_length
    //robot_model_d_[5] = -wrist_3_length

    robot_model_a_[0] = T(0);
    // robot_model_a_[1] = upper_arm_length
    // robot_model_a_[2] = forearm_length
    robot_model_a_[3] = T(0);
    robot_model_a_[4] = T(0);
    robot_model_a_[5] = T(0);

    k(robot_model_theta, robot_model_d_, robot_model_a_, robot_model_alpha, joint_state, pose_storage);


    //std::cout << "robot_model: " << robot_model[0] << " " <<  robot_model[1] << " " <<  robot_model[2] << " " << robot_model[3] << " " <<  robot_model[4] << " " <<  robot_model[5] <<std::endl;
    //std::cout << "joint_state: " << joint_state[0] << " " <<  joint_state[1] << " " <<  joint_state[2] << " " << joint_state[3] << " " <<  joint_state[4] << " " <<  joint_state[5] <<std::endl;
    //std::cout << "robot_translation: " << pose_storage[3] << " " <<  pose_storage[7] << " " <<  pose_storage[11] << std::endl;


    // transform point

    T p[3];

    //std::cout << "Pose:" << pose_storage[3] << " " << pose_storage[7] << " " << pose_storage[11] << std::endl;

    p[0] = point[0] * pose_storage[0] + point[1] * pose_storage[1] + point[2] * pose_storage[2] + pose_storage[3];
    p[1] = point[0] * pose_storage[4] + point[1] * pose_storage[5] + point[2] * pose_storage[6] + pose_storage[7];
    p[2] = point[0] * pose_storage[8] + point[1] * pose_storage[9] + point[2] * pose_storage[10] + pose_storage[11];

    // camera[0,1,2] are the angle-axis rotation.

    /*ceres::AngleAxisRotatePoint(robot_rotation, point, p);

    // camera[3,4,5] are the translation.
    p[0] += robot_translation[0];
    p[1] += robot_translation[1];
    p[2] += robot_translation[2];*/

    T p2[3];
    ceres::AngleAxisRotatePoint(handEye, p, p2);
    p2[0] += handEye[3];
    p2[1] += handEye[4];
    p2[2] += handEye[5];

    //std::cout << p2[0] << " " <<  p2[1] << " " <<  p2[2] << std::endl;

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
    // const T& focal = robot[6];
    T predicted_x = (focal[0]  * xp * dist) + pp[0];
    T predicted_y = (focal[0]  * yp * dist) + pp[1];

    //std::cout << "focal: " << focal[0] << std::endl;
    //exit(-1);

    //T predicted_x = focal[0] * (p2[0] / p2[2]);
    //T predicted_y = focal[0] * (p2[1] / p2[2]);

    //std::cout <<" PRED " <<  predicted_x << " " << predicted_y << std::endl;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6,6,6,6, 6,6, 3, 1,3,2>(    // observed, robot_model_{theta,d,a,alpha},joint_state, handeye,    3dpoint,   focal,distortion,pp
                new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};


class DHCalibration {
public:
  DHCalibration(
    THDoubleTensor *intrinsics,   // camera intrinsics 3x3 matrix
    THDoubleTensor *distortion,   // distortion 2 element vector
    THDoubleTensor *hand_eye,
    THDoubleTensor *joint_states,
    THDoubleTensor *robot_model,
    THDoubleTensor *points,
    THDoubleTensor *observations,
    THLongTensor *jointpoint_indices
  ) :
    intrinsics(intrinsics), distortion(distortion), hand_eye(hand_eye),
    joint_states(joint_states), robot_model(robot_model),
    points(points), observations(observations), jointpoint_indices(jointpoint_indices)
  {
     // ensure everything is continuous in memory
    intrinsics_ = THDoubleTensor_newContiguous(intrinsics);
    distortion_ = THDoubleTensor_newContiguous(distortion);
    hand_eye_ = THDoubleTensor_newContiguous(hand_eye);
    joint_states_ = THDoubleTensor_newContiguous(joint_states);
    robot_model_ = THDoubleTensor_newContiguous(robot_model);
    points_ = THDoubleTensor_newContiguous(points);
    observations_ = THDoubleTensor_newContiguous(observations);
    jointpoint_indices_ = THLongTensor_newContiguous(jointpoint_indices);

    focal[0] = (THDoubleTensor_get2d(intrinsics, 0, 0) + THDoubleTensor_get2d(intrinsics, 1, 1)) * 0.5;
    pp[0] = THDoubleTensor_get2d(intrinsics, 0, 2);
    pp[1] = THDoubleTensor_get2d(intrinsics, 1, 2);
    distortion_data[0] = THDoubleTensor_get2d(distortion, 0, 0);
    distortion_data[1] = THDoubleTensor_get2d(distortion, 0, 1);
    distortion_data[2] = THDoubleTensor_get2d(distortion, 0, 4);

    std::cout << "Distortion: " << distortion_data[0] << " " << distortion_data[1] << " " << distortion_data[2] << std::endl;

    num_joint_states = THDoubleTensor_size(joint_states_, 0);
    num_points = THDoubleTensor_size(points_, 0);
    num_observations = THDoubleTensor_size(observations_, 0);

    robot_model_data = THDoubleTensor_data(robot_model_);
    robot_model_data_theta = robot_model_data;
    robot_model_data_d = robot_model_data + 6;
    robot_model_data_a = robot_model_data + 12;
    robot_model_data_alpha = robot_model_data + 18;

    joint_states_data = THDoubleTensor_data(joint_states_);
    points_data = THDoubleTensor_data(points_);
    observations_data = THDoubleTensor_data(observations_);
    jointpoint_indices_data = THLongTensor_data(jointpoint_indices_);

/*
    // load debug robot model
    const double shoulder_height  = 0.089159;
    const double upper_arm_length = 0.425;
    const double forearm_length   = 0.39225;
    const double wrist_1_length   = 0.10915;
    const double wrist_2_length   = 0.09465;
    const double wrist_3_length   = 0.0823;

    // fill dh parameters from robot_model
    const double pi(3.14159265358979323846);

    double *rm = robot_model_data;
    std::cout << "Robot Model (A)" << std::endl;
    for (int i = 0; i < 6; i++) {
      std::cout<< "Joint "<< i << "(theta, d, a, alpha)" << rm[0+i] << " " << rm[6 + i] << " " << rm[12 + i]  << " " << rm[18 + i] << std::endl;
    }

    robot_model_data_theta[0] = 0;
    robot_model_data_theta[1] = 0;
    robot_model_data_theta[2] = 0;
    robot_model_data_theta[3] = 0;
    robot_model_data_theta[4] = 0;
    robot_model_data_theta[5] = pi;

    robot_model_data_d[0] = shoulder_height;
    robot_model_data_d[1] = 0;
    robot_model_data_d[2] = 0;
    robot_model_data_d[3] = -wrist_1_length;
    robot_model_data_d[4] = wrist_2_length;
    robot_model_data_d[5] = -wrist_3_length;

    robot_model_data_a[0] = 0;
    robot_model_data_a[1] = upper_arm_length;
    robot_model_data_a[2] = forearm_length;
    robot_model_data_a[3] = 0;
    robot_model_data_a[4] = 0;
    robot_model_data_a[5] = 0;

    robot_model_data_alpha[0] = pi*0.5;
    robot_model_data_alpha[1] = 0;
    robot_model_data_alpha[2] = 0;
    robot_model_data_alpha[3] = pi*0.5;
    robot_model_data_alpha[4] = -pi*0.5;
    robot_model_data_alpha[5] = pi;

    std::cout << "Robot Model (B)" << std::endl;
    for (int i = 0; i < 6; i++) {
      std::cout<< "Joint "<< i << "(theta, d, a, alpha)" << rm[0+i] << " " << rm[6 + i] << " " << rm[12 + i]  << " " << rm[18 + i] << std::endl;
    }
*/

      // read hand-eye matrix
    double angle_axis[3];
    double translation[3] = { THDoubleTensor_get2d(hand_eye, 0, 3), THDoubleTensor_get2d(hand_eye, 1, 3), THDoubleTensor_get2d(hand_eye, 2, 3) };

    // convert to angle-axis representation
    {
      double rotation[9] = {
        THDoubleTensor_get2d(hand_eye, 0, 0), THDoubleTensor_get2d(hand_eye, 0, 1), THDoubleTensor_get2d(hand_eye, 0, 2),
        THDoubleTensor_get2d(hand_eye, 1, 0), THDoubleTensor_get2d(hand_eye, 1, 1), THDoubleTensor_get2d(hand_eye, 1, 2),
        THDoubleTensor_get2d(hand_eye, 2, 0), THDoubleTensor_get2d(hand_eye, 2, 1), THDoubleTensor_get2d(hand_eye, 2, 2)
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

      /*std::cout << "handeye rotation:" << std::endl;
      for (int i=0; i <9; ++i) {
        printf("%d: %f\n", i, (double)rotation[i]);
      }

      printf("rotation_cm input:");
      for (int i=0;i<9;++i) {
        printf("%d: %f\n", i, rotation_cm[i]);
      }*/

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

    /*{
      double rotation_cm2[9] = {0};
      ceres::AngleAxisToRotationMatrix(hand_eye_data, rotation_cm2);

      double rotation[9];
      rotation[0] = rotation_cm2[0];
      rotation[3] = rotation_cm2[1];
      rotation[6] = rotation_cm2[2];
      rotation[1] = rotation_cm2[3];
      rotation[4] = rotation_cm2[4];
      rotation[7] = rotation_cm2[5];
      rotation[2] = rotation_cm2[6];
      rotation[5] = rotation_cm2[7];
      rotation[8] = rotation_cm2[8];

      std::cout << "handeye rotation decoded:" << std::endl;
      for (int i=0; i <9; ++i) {
        printf("%d: %f\n", i, (double)rotation[i]);
      }

    }*/

    prepareProblem();
  }

  ~DHCalibration() {
    // free tensors (normally only decrement ref count, copy if source tensor was not continuous)
    THDoubleTensor_freeCopyTo(intrinsics_, intrinsics);
    THDoubleTensor_freeCopyTo(distortion_, distortion);
    THDoubleTensor_freeCopyTo(hand_eye_, hand_eye);
    THDoubleTensor_freeCopyTo(joint_states_, joint_states);
    THDoubleTensor_freeCopyTo(robot_model_, robot_model);
    THDoubleTensor_freeCopyTo(points_, points);
    THDoubleTensor_freeCopyTo(observations_, observations);
    THLongTensor_freeCopyTo(jointpoint_indices_, jointpoint_indices);
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
      for (int i = 0; i < num_points; i++) {
        problem.SetParameterBlockConstant(points_data + (i*3));
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

    problem.SetParameterBlockConstant(joint_states_data);    // fix first joint configuration
    if (!optimize_joint_states) {
      for (int i = 0; i < num_joint_states; i++) {
        problem.SetParameterBlockConstant(joint_states_data + (i * 6));
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
    for (int i = 0; i < 6; i++) {
      std::cout<< "Joint "<< i << "(theta, d, a, alpha)" << rm[0+i] << " " << rm[6 + i] << " " << rm[12 + i]  << " " << rm[18 + i] << std::endl;
    }

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    //options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.num_threads = 8;

    //bal_problem.printToLua("input.lua", "input_");
    //bal_problem.printPoints("input_pts.csv");
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    //bal_problem.printPoints("optimized_pts.csv") ;

    std::cout << "Focal: " << focal[0] << std::endl;
    std::cout << "Distortion: " << distortion_data[0] << " " << distortion_data[1] << " " << distortion_data[2] << std::endl;
    std::cout << "Principle Point: " << pp[0] << " " << pp[1] << std::endl;

    std::cout << "Robot Model (after)" << std::endl;
    for (int i = 0; i < 6; i++) {
      std::cout<< "Joint "<< i << "(theta, d, a, alpha)" << rm[0+i] << " " << rm[6 + i] << " " << rm[12 + i]  << " " << rm[18 + i] << std::endl;
    }

    if (optimize_focal_length) {
      THDoubleTensor_set2d(intrinsics_, 0, 0, focal[0]);
      THDoubleTensor_set2d(intrinsics_, 1, 1, focal[0]);
    }

    if (optimize_pp) {
      THDoubleTensor_set2d(intrinsics_, 0, 2, pp[0]);
      THDoubleTensor_set2d(intrinsics_, 1, 2, pp[1]);
    }

    if (optimize_distortion) {
      THDoubleTensor_set2d(distortion_, 0, 0, distortion_data[0]);
      THDoubleTensor_set2d(distortion_, 0, 1, distortion_data[1]);
      THDoubleTensor_set2d(distortion_, 0, 4, distortion_data[2]);
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
      THDoubleTensor_set2d(hand_eye_, 0, 0, rotation[0]); THDoubleTensor_set2d(hand_eye_, 0, 1, rotation[1]); THDoubleTensor_set2d(hand_eye_, 0, 2, rotation[2]);
      THDoubleTensor_set2d(hand_eye_, 1, 0, rotation[3]); THDoubleTensor_set2d(hand_eye_, 1, 1, rotation[4]); THDoubleTensor_set2d(hand_eye_, 1, 2, rotation[5]);
      THDoubleTensor_set2d(hand_eye_, 2, 0, rotation[6]); THDoubleTensor_set2d(hand_eye_, 2, 1, rotation[7]); THDoubleTensor_set2d(hand_eye_, 2, 2, rotation[8]);

      // translation part
      THDoubleTensor_set2d(hand_eye_, 0, 3, hand_eye_data[3]);
      THDoubleTensor_set2d(hand_eye_, 1, 3, hand_eye_data[4]);
      THDoubleTensor_set2d(hand_eye_, 2, 3, hand_eye_data[5]);
    }
  }

  double calcAverageReproductionError() {
    double cost;
    std::vector<double> residuals;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, NULL, NULL);
    double sum = 0;
    for (size_t i = 0; i < residuals.size(); i = i+2) {
      sum += sqrt((residuals[i]* residuals[i]) + (residuals[i+1]* residuals[i+1]));
    }
    return sum / (double(residuals.size()) / 2.0);
  }

//private:
public:
  ceres::Problem problem;

  THDoubleTensor *intrinsics, *intrinsics_;
  THDoubleTensor *distortion, *distortion_;
  THDoubleTensor *hand_eye, *hand_eye_;
  THDoubleTensor *joint_states, *joint_states_;
  THDoubleTensor *robot_model, *robot_model_;
  THDoubleTensor *points, *points_;
  THDoubleTensor *observations, *observations_;
  THLongTensor *jointpoint_indices, *jointpoint_indices_;

  double focal[2];
  double pp[2];
  double distortion_data[3];
  double hand_eye_data[6];

  long num_joint_states;
  long num_points;
  long num_observations;

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
  /*
   params:
      const T* const robot_model,
      const T* const joint_state,
      const T* const handEye,
      const T* const point,
      const T* const focal,
      const T* const distortion,
      const T* const pp,
      ceres::LossFunction* loss_function(new ceres::HuberLoss(1.0));
  */

      long joint_index = jointpoint_indices_data[2 * i + 0];
      long point_index = jointpoint_indices_data[2 * i + 1];

      // std::cout << "joint_index: " << joint_index << "; point_index: " << point_index << std::endl;

      double *joint_state_for_observation = joint_states_data + 6 * joint_index;
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

CALIBIMP(double, _, evaluateDH)(
  THDoubleTensor *intrinsics,   // camera intrinsics 3x3 matrix
  THDoubleTensor *distortion,   // distortion 2 element vector
  THDoubleTensor *hand_eye,
  THDoubleTensor *joint_states,
  THDoubleTensor *robot_model,
  THDoubleTensor *points,
  THDoubleTensor *observations,
  THLongTensor *jointpoint_indices
) {
//  google::InitGoogleLogging("/tmp");
  DHCalibration calib(intrinsics, distortion, hand_eye, joint_states, robot_model, points, observations, jointpoint_indices);
  return calib.calcAverageReproductionError();
}

CALIBIMP(double, _, optimizeDH)(
  THDoubleTensor *intrinsics,   // camera intrinsics 3x3 matrix
  THDoubleTensor *distortion,   // distortion 2 element vector
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
  bool optimize_joint_states,
  bool optimize_pp,
  bool optimize_focal_length,
  bool optimize_distortion
) {
//  google::InitGoogleLogging("/tmp");
  double err = 0;

  {
    DHCalibration calib(intrinsics, distortion, hand_eye, joint_states, robot_model, points, observations, jointpoint_indices);
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
    err = calib.calcAverageReproductionError();
  }

  DHCalibration calib2(intrinsics, distortion, hand_eye, joint_states, robot_model, points, observations, jointpoint_indices);
  double crossCheck = calib2.calcAverageReproductionError();
  printf("err: %f, crossCheck: %f\n", err, crossCheck);

  return err;
}
