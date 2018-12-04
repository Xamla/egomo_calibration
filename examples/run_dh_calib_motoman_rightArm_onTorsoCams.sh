#/!bin/bash

# Arguments: 
# path of (stereo) calibration (e.g. "stereo_cams_CAMAU1639042_CAMAU1710001.npy")
# path of hand-pattern matrix (e.g. "HandPattern.npy")
# number of images for dh-parameter optimization (e.g. 50)
# path of left camera images without image number and without ".png" (e.g. "capture_all/cam_CAMAU1639042")
# path of right camera images without image number and without ".png" (e.g. "capture_all/cam_CAMAU1710001")
# path of joint configurations (e.g. "jsposes_tensors.npy")
# path of joint configuration containing the torso joint (e.g. "all_vals_tensors.npy")

python3 dh_calib_motoman_rightArm_onTorsoCams.py "../../right_arm_data/calib_42_20_50/stereo_cams_CAMAU1639042_CAMAU1710001.npy" "../../right_arm_data/calib_42_20_50/HandPattern.npy" 110 "../../right_arm_data/calib_42_20_50/capture_all/cam_CAMAU1639042" "../../right_arm_data/calib_42_20_50/capture_all/cam_CAMAU1710001" "../../right_arm_data/calib_42_20_50/jsposes_tensors.npy" "../../right_arm_data/calib_42_20_50/all_vals_tensors_42.npy" "../../right_arm_data/calib_42_20_50/all_vals_tensors_20.npy" 42 "../../right_arm_data/calib_42_20_50/all_vals_tensors_50.npy" 61

#python3 dh_calib_motoman_rightArm_onTorsoCams.py "../../right_arm_data/calib_42_20_50/stereo_cams_CAMAU1639042_CAMAU1710001.npy" "../../right_arm_data/calib_42_20_50/HandPattern_optimized_rightArm_42_20_50.npy" 110 "../../right_arm_data/calib_42_20_50/capture_all/cam_CAMAU1639042" "../../right_arm_data/calib_42_20_50/capture_all/cam_CAMAU1710001" "../../right_arm_data/calib_42_20_50/js_new.npy" "../../right_arm_data/calib_42_20_50/all_vals_tensors_42.npy" "../../right_arm_data/calib_42_20_50/all_vals_tensors_20.npy" 42 "../../right_arm_data/calib_42_20_50/all_vals_tensors_50.npy" 61

#python3 dh_calib_motoman_rightArm_onTorsoCams.py "../../right_arm_data/calib_after_27_21/stereo_cams_CAMAU1639042_CAMAU1710001.npy" "../../right_arm_data/calib_after_27_21/HandPattern_optimized_rightArm_42_20_50.npy" 48 "../../right_arm_data/calib_after_27_21/capture_all/cam_CAMAU1639042" "../../right_arm_data/calib_after_27_21/capture_all/cam_CAMAU1710001" "../../right_arm_data/calib_after_27_21/jsposes_tensors.npy" "../../right_arm_data/calib_after_27_21/all_vals_tensors_27.npy"

#python3 dh_calib_motoman_rightArm_onTorsoCams.py "../../right_arm_data/calib_before_42/stereo_cams_CAMAU1639042_CAMAU1710001.npy" "../../right_arm_data/calib_before_42/HandPattern.npy" 42 "../../right_arm_data/calib_before_42/capture_all/cam_CAMAU1639042" "../../right_arm_data/calib_before_42/capture_all/cam_CAMAU1710001" "../../right_arm_data/calib_before_42/jsposes_tensors.npy" "../../right_arm_data/calib_before_42/all_vals_tensors.npy"

