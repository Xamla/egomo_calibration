Install dependencies:
=====================

Ceres: see [ceres-sover.org](http://ceres-solver.org/installation.html)
------
* download Ceres (wget http://ceres-solver.org/ceres-solver-1.14.0.tar.gz)
* tar -xvzf ceres-solver-1.14.0.tar.gz
* cd ceres-solver-1.14.0
* sudo apt-get update
* sudo apt-get install cmake
* sudo apt-get install libgoogle-glog-dev
* sudo apt-get install libatlas-base-dev
* sudo apt-get install libeigen3-dev
* sudo apt-get install libsuitesparse-dev
* mkdir ceres-bin
* cd ceres-bin/
* cmake ..
* make -j3
* make test
* sudo make install
* bin/simple_bundle_adjuster ../data/problem-16-22106-pre.txt

Other dependencies:
-------------------
* sudo apt install python-pip python3-pip
* sudo python3 -m pip install torchfile   (pip3 install torchfile)
* sudo python3 -m pip install matplotlib  (pip3 install matplotlib)
* sudo python3 -m pip install cffi        (pip3 install cffi)
* sudo apt install python3-tk
* sudo apt install libboost-all-dev
* sudo python3 -m pip install opencv-python
* Installation of ROS (e.g. [ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu))
  (to be able to import "xacro" to read dh-parameters from urdf)

Calibration files:
------------------
* Run the sphere sampling in Rosvita: <br />
  ``th ../../lua/auto_calibration/configureCalibration.lua -cfg configuration.t7`` <br />
  -> "a Capture sphere sampling".
* Move the resulting folder (e.g. "/home/rosvita/Rosvita/projects/\<my-project\>/calibration/capture_shpere_sampling") into e.g. the folder "/home/rosvita/code/egomo_calibration/data/right_arm/".
* Additionally move an initial guess hand eye and stereocalibration into this folder (e.g. from "/home/rosvita/Rosvita/projects/\<my-project\>/calibration/right_arm_cameras/").
* Convert torchfiles (jsposes_tensors.t7, all_vals_tensors.t7, HandEye.t7, stereo_cams_\<id-left\>_\<id-right\>.t7) into numpy arrays: <br />
  ``cd /home/rosvita/code/egomo_calibration/data/right_arm/`` <br />
  ``./run_data_conversion.sh``

In egomo_calibration:
=====================
* cmake -E make_directory build
* cd build
* cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
* make
* cd ../python
* sudo python3 setup.py develop
* cd ../examples/
* ./run_dh_calib_motoman_end_of_arm_cameras (previously adapt paths within this file)
