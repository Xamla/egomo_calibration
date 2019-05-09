Install dependencies:
=====================

Ceres with dependencies: see [ceres-solver.org](http://ceres-solver.org)
------------------------
* sudo apt-get update
* sudo apt-get -y install libgoogle-glog-dev
* sudo apt-get -y install libatlas-base-dev
* sudo apt-get -y install libeigen3-dev
* sudo apt-get -y install libsuitesparse-dev
* sudo apt-get -y install libceres-dev

Python dependencies:
--------------------
* (sudo apt install python3-pip)
* sudo python3 -m pip install torchfile
* sudo python3 -m pip install matplotlib
* sudo python3 -m pip install cffi
* sudo python3 -m pip install opencv-python
* sudo python3 -m pip install defusedxml
* (sudo apt install python3-tk)
* (sudo apt install libboost-all-dev)

ROS:
----
Installation of ROS (e.g. [ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu))
(to be able to import "xacro" to read dh-parameters from urdf)

In egomo_calibration:
=====================
* cmake -E make_directory build
* cd build
* cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
* make
* cd ../python
* sudo python3 -m pip install ./python <br />
  (or: sudo python3 setup.py develop)
  
