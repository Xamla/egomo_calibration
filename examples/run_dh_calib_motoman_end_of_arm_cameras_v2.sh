#/!bin/bash

# Arguments: 
# path of (stereo) calibration (e.g. "stereo_cams_CAMAU1639042_CAMAU1710001.npy")
# path of hand-pattern matrix (e.g. "HandPattern.npy")
# number of images for dh-parameter optimization (e.g. 50)
# path of left camera images without image number and without ".png" (e.g. "capture_all/cam_CAMAU1639042")
# path of right camera images without image number and without ".png" (e.g. "capture_all/cam_CAMAU1710001")
# path of joint configurations (e.g. "jsposes_tensors.npy")
# path of joint configuration containing the torso joint (e.g. "all_vals_tensors.npy")
# patternId
# output folder
# output filename for optimized robot model
# output filename for optimized hand-eye
# which arm ("left" or "right")

# Joint offset optimization for theta:
# ====================================
#python3 dh_calib_motoman_end_of_arm_cameras_v2.py \
#"../data/left_arm/2019-01-24/stereo_cams_4103130811_4103189394.npy" \
#"../data/left_arm/2019-01-24/HandEye.npy" \
#55 \
#"../data/left_arm/2019-01-24/capture_all/cam_4103130811" \
#"../data/left_arm/2019-01-24/capture_all/cam_4103189394" \
#"../data/left_arm/2019-01-24/jsposes_tensors.npy" \
#"../data/left_arm/2019-01-24/all_vals_tensors.npy" \
#22 \
#"results_left_arm_v2/" \
#"robotModel_leftArm_2019-01-24_v2" \
#"handEye_leftArm_2019-01-24_v2" \
#"left"

#python3 dh_calib_motoman_end_of_arm_cameras_v2.py \
#"../data/right_arm/2019-02-05/stereo_cams_4103217455_4103235743.npy" \
#"../data/right_arm/2019-02-05/HandEye.npy" \
#55 \
#"../data/right_arm/2019-02-05/capture_all/cam_4103217455" \
#"../data/right_arm/2019-02-05/capture_all/cam_4103235743" \
#"../data/right_arm/2019-02-05/jsposes_tensors.npy" \
#"../data/right_arm/2019-02-05/all_vals_tensors.npy" \
#22 \
#"results_right_arm_v2/" \
#"robotModel_rightArm_2019-02-05_v2" \
#"handEye_rightArm_2019-02-05_v2" \
#"right"


python3 dh_calib_motoman_end_of_arm_cameras_v2.py \
"../data/left_arm/2019-03-05/stereo_cams_4103217457_4103217454.npy" \
"../data/left_arm/2019-03-05/HandEye.npy" \
100 \
"../data/left_arm/2019-03-05/capture_all/cam_4103217457" \
"../data/left_arm/2019-03-05/capture_all/cam_4103217454" \
"../data/left_arm/2019-03-05/jsposes_tensors.npy" \
"../data/left_arm/2019-03-05/all_vals_tensors.npy" \
21 \
"results_left_arm/" \
"robotModel_leftArm_2019-03-05" \
"handEye_leftArm_2019-03-05" \
"left"
