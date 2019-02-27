#/!bin/bash

# Arguments: 
# calibration path of input torchfile
# calibration path of output pythonfile
# hand-eye path of input torchfile
# hand-eye path of output pythonfile
# jsposes_tensors path of input torchfile
# jsposes_tensors path of output pythonfile
# all_vals_tensors path of intput torchfile
# all_vals_tensors path of output pythonfile

# Don't forget to adapt stereo camera IDs!

mydir="$(dirname "$0")"
echo $mydir

python3 $mydir/data_conversion.py \
"stereo_cams_4103217455_4103235743.t7" \
"stereo_cams_4103217455_4103235743.npy" \
"HandEye.t7" \
"HandEye.npy" \
"jsposes_tensors.t7" \
"jsposes_tensors.npy" \
"all_vals_tensors.t7" \
"all_vals_tensors.npy"

