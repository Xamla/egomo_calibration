#/!bin/bash

# Arguments: 
# path of (stereo) calibration (e.g. "stereo_cams_CAMAU1639042_CAMAU1710001.npy")
# path of hand-eye matrix (e.g. "HandEye.npy")
# number of images for dh-parameter optimization (e.g. 50)
# path of left camera images without image number and without ".png" (e.g. "capture_all/cam_CAMAU1639042")
# path of right camera images without image number and without ".png" (e.g. "capture_all/cam_CAMAU1710001")
# path of joint configurations (e.g. "jsposes_tensors.npy")
# path of joint configuration containing the torso joint (e.g. "all_vals_tensors.npy")
# patternId

# Joint offset optimization for theta:
# ====================================
python3 dh_calib_motoman_rightArm_rightArmCams.py "../../right_arm_onboard/2019-01-16_50Images_forCalib/stereo_cams_4103217455_4103235743_2019-01-16_111536_sphere50.npy" "../../right_arm_onboard/2019-01-16_50Images_forCalib/FlangeEye_initialGues_2019-01-14_125243_sphere14.npy" 50 "../../right_arm_onboard/2019-01-16_50Images_forCalib/capture_all/cam_4103217455" "../../right_arm_onboard/2019-01-16_50Images_forCalib/capture_all/cam_4103235743" "../../right_arm_onboard/2019-01-16_50Images_forCalib/jsposes_tensors.npy" "../../right_arm_onboard/2019-01-16_50Images_forCalib/all_vals_tensors.npy" 22

