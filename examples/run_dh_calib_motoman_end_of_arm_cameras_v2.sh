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
# which cam ("left" or "right")

# Joint offset optimization for theta:
# ====================================
#python3 dh_calib_motoman_end_of_arm_cameras_v2.py "../../left_arm_onboard_new/2019-01-24_55Images_forCalib/stereo_cams_4103130811_4103189394.npy" "../../left_arm_onboard_new/2019-01-24_55Images_forCalib/FlangeEye.npy" 55 "../../left_arm_onboard_new/2019-01-24_55Images_forCalib/capture_all/cam_4103130811" "../../left_arm_onboard_new/2019-01-24_55Images_forCalib/capture_all/cam_4103189394" "../../left_arm_onboard_new/2019-01-24_55Images_forCalib/jsposes_tensors.npy" "../../left_arm_onboard_new/2019-01-24_55Images_forCalib/all_vals_tensors.npy" 22 "results/" "robotModel_leftArm_2019-01-24" "handEye_leftArm_2019-01-24" "left"

python3 dh_calib_motoman_end_of_arm_cameras_v2.py "../../right_arm_onboard/2019-02-05_55Images_forCalib/stereo_cams_4103217455_4103235743_17Samples.npy" "../../right_arm_onboard/2019-02-05_55Images_forCalib/HandEye_initialGuess.npy" 55 "../../right_arm_onboard/2019-02-05_55Images_forCalib/capture_all/cam_4103217455" "../../right_arm_onboard/2019-02-05_55Images_forCalib/capture_all/cam_4103235743" "../../right_arm_onboard/2019-02-05_55Images_forCalib/jsposes_tensors.npy" "../../right_arm_onboard/2019-02-05_55Images_forCalib/all_vals_tensors.npy" 22 "results/" "robotModel_rightArm_2019-02-06" "handEye_rightArm_2019-02-06" "right"
