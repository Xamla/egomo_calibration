#/!bin/bash

# Arguments: 
# path of (stereo) calibration (e.g. "stereo_cams_CAMAU1639042_CAMAU1710001.npy")
# path of hand-pattern matrix (e.g. "HandPattern.npy")
# number of images for dh-parameter optimization (e.g. 50)
# path of left camera images without image number and without ".png" (e.g. "capture_all/cam_CAMAU1639042")
# path of right camera images without image number and without ".png" (e.g. "capture_all/cam_CAMAU1710001")
# path of joint configurations (e.g. "jsposes_tensors.npy")
# path of joint configuration containing the torso joint (e.g. "all_vals_tensors.npy")

python3 dh_calib_motoman.py "../../calibration_rand50/stereo_cams_4103130811_4103189394.npy" "../../calibration_rand50/HandEye.npy" 50 "../../calibration_rand50/capture_all/cam_4103130811" "../../calibration_rand50/capture_all/cam_4103189394" "../../calibration_rand50/jsposes_tensors.npy" "../../calibration_rand50/all_vals_tensors.npy"

#python3 dh_calib_motoman.py "../../calibration_rand50/stereo_cams_4103130811_4103189394.npy" "../../calibration_rand50/handEye_optimized.npy" 50 "../../calibration_rand50/capture_all/cam_4103130811" "../../calibration_rand50/capture_all/cam_4103189394" "../../calibration_rand50/jsposes_optimized_correct_sign.npy" "../../calibration_rand50/all_vals_tensors.npy"

#python3 dh_calib_motoman.py "../../calib_after_all/stereo_cams_4103130811_4103189394_new.npy" "../../calib_after_all/handEye_optimized.npy" 50 "../../calib_after_all/capture_all/cam_4103130811" "../../calib_after_all/capture_all/cam_4103189394" "../../calib_after_all/jsposes_tensors_correct_sign.npy" "../../calib_after_all/all_vals_tensors.npy"

#python3 dh_calib_motoman.py "../../calib_before_all/stereo_cams_4103130811_4103189394_new.npy" "../../calib_before_all/HandEye.npy" 50 "../../calib_before_all/capture_all/cam_4103130811" "../../calib_before_all/capture_all/cam_4103189394" "../../calib_before_all/jsposes_tensors.npy" "../../calib_before_all/all_vals_tensors.npy"

