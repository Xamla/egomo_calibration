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
* sudo python3 -m pip install torchfile   (pip3 install torchfile)
* sudo python3 -m pip install matplotlib  (pip3 install matplotlib)
* sudo apt-get install python3-tk
* sudo python3 -m pip install cffi        (pip3 install cffi)

Calibration files:
------------------
* From ... (TODO: choose a place for the data)
get "calibration_rand50.tgz" and extract folder.
* Move this folder (i.e. "calibration_rand50") into the parent directory of "egomo_calibration".

In egomo_calibration:
=====================
* cmake -E make_directory build
* cd build
* cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
* make
* cd ../python
* sudo python3 setup.py develop
* cd ../examples/
* python3 dh_calib_motoman.py (now: ./run_dh_calib_motoman.sh)
