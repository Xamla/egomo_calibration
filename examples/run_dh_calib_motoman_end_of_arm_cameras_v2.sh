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
# filename (with path) of robot.urdf
# which arm ("left" or "right")
# alternating optimization (hand-eye, DH, hand-eye, DH, ...)?
# number of runs
# train-test split percentage
# with torso movement in the data?
# with optimization of torso joint?
# only evaluation?
# with jitter on theta start value?
# optimize hand-eye?
# optimize pattern points?
# optimize DH-parameter theta?
# optimize DH-parameter d?
# optimize DH-parameter a?
# optimize DH-parameter alpha?


# Hand-Eye and DH-Parameter Optimization:
# =======================================

python3 dh_calib_motoman_end_of_arm_cameras_v2.py \
"../data/left_arm/2019-03-26_back/stereo_cams_4103189394_4103130811_kalibr_corrected_ordering.npy" \
"../data/left_arm/2019-03-26_back/HandEye_kalibr_corrected_ordering.npy" \
150 \
"../data/left_arm/2019-03-26_back/capture_all/cam_4103189394" \
"../data/left_arm/2019-03-26_back/capture_all/cam_4103130811" \
"../data/left_arm/2019-03-26_back/jsposes_tensors.npy" \
"../data/left_arm/2019-03-26_back/all_vals_tensors.npy" \
20 \
"results_left_arm/pattern_in_back_150/" \
"robotModel_v2_allDHOpt" \
"handEye_v2_allDHOpt" \
"../../../../Rosvita/projects/xamla_calibration/robot.urdf" \
"left" \
"False" \
1 \
1.0 \
"True" \
"True" \
"False" \
"False" \
"True" \
"False" \
"True" \
"True" \
"True" \
"True"

