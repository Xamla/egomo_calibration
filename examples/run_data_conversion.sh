#/!bin/bash

# Arguments: 
# calibration path of input torchfile
# calibration path of output pythonfile
# hand-eye path of input torchfile
# hand-eye path of output pythonfile
# jsposes path of input torchfile
# jsposes path of output pythonfile
# all_joint_values path of intput torchfile
# all_joint_values path of output pythonfile

python3 data_conversion.py "../../calibration_rand50/stereo_cams_4103130811_4103189394.t7" "../../calibration_rand50/stereo_cams_4103130811_4103189394.npy" "../../calibration_rand50/HandEye.t7" "../../calibration_rand50/HandEye.npy" "../../calibration_rand50/jsposes_tensors.t7" "../../calibration_rand50/jsposes_tensors.npy" "../../calibration_rand50/all_vals_tensors.t7" "../../calibration_rand50/all_vals_tensors.npy"

#python3 data_conversion.py "../../calib_after_all/stereo_cams_4103130811_4103189394_new.t7" "../../calib_after_all/stereo_cams_4103130811_4103189394_new.npy" 'None' 'None' "../../calib_after_all/jsposes_tensors.t7" "../../calib_after_all/jsposes_tensors.npy" "../../calib_after_all/all_vals_tensors.t7" "../../calib_after_all/all_vals_tensors.npy"

#python3 data_conversion.py "../../calib_before_all/stereo_cams_4103130811_4103189394_new.t7" "../../calib_before_all/stereo_cams_4103130811_4103189394_new.npy" 'None' 'None' "../../calib_before_all/jsposes_tensors.t7" "../../calib_before_all/jsposes_tensors.npy" "../../calib_before_all/all_vals_tensors.t7" "../../calib_before_all/all_vals_tensors.npy"




