# egomo_calibration

This package containts scripts for robot calibration and hand-eye optimization.
It is written in Python and C++ and uses [Ceres](http://ceres-solver.org/) to optimize the robot kinematic modelled by [Denavit-Hartenberg parameters](https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters).
Moreover, it is part of the [Rosvita](http://www.rosvita.com/) robot programming environment.


#### Major features are:

1. [Robot kinematic calibration (by optimizing DH-parameters)](#robot-kinematic-calibration)
2. [Hand-Eye optimization (starting from an initial guess)](#hand-eye-optimization)


#### Calibration pattern requirements:

For all robot kinematic and hand-eye calibrations one of our [circle patterns with ids](https://github.com/Xamla/auto_calibration/blob/master/Patterns_with_ID.pdf) (**Patterns_with_ID.pdf**) has to be used.
For a high-quality print of one of these patterns contact http://xamla.com/en/.


### Capturing calibration data via automatic sphere sampling

Automatic sphere sampling is performed via the [auto_calibration](https://github.com/Xamla/auto_calibration/) package of Rosvita (in Rosvita under ``/home/xamla/Rosvita.Control/lua/auto_calibration/``).

To use the automatic sphere sampling, you first have to define a good **starting pose** for the robot arm that will be calibrated. In the following, let us assume we have a stereo camera setup with the cameras mounted at the endeffector of the robot arm and looking in the direction of the z-axis of the endeffector. With this setup the calibration target has to be fixed onto the table and the robot has to be moved to a pose where the cameras look down at the target approximately straight from above and such that all target points are in the field of view (FOV) of the cameras. Save this starting pose of the robot (or better the joint values) to the Rosvita world view and move the robot to this posture before starting the sphere sampling.

Moveover, you need an initial guess hand-eye and stereo camera calibration, which you can receive e.g. by running the [auto_calibration](https://github.com/Xamla/auto_calibration/) scripts (see the corresponding [Readme](https://github.com/Xamla/auto_calibration/blob/master/README.md)). 

Now, to start the sphere sampling, with the Rosvita terminal go into your project folder and run the configuration script from the auto_calibration package:
```
cd /home/xamla/Rosvita.Control/projects/<your_project_folder>
th ../../lua/auto_calibration/configureCalibration.lua -cfg <name_of_your_configuration_file>.t7
```
Then press
```
a (Generate capture poses via sphere sampling)
```
Now, you have to choose the camera setup. Currently, two possibilities are implemented:
```
e end-of-arm camera setup
t torso camera setup
```
Here we choose option e (end-of-arm camera setup).

After that, we have to enter the number of capture poses we want to sample. 
In order to obtain good calibration results, you should choose a rather large number of about 100-200 poses.

Next you have to enter the paths to the previously generated intial guesses for the hand-eye calibration (e.g. ``/tmp/calibration/<data>_<time>/HandEye.t7``) and for the stereo calibration (e.g. ``/tmp/calibration/<data>_<time>/stereo_cams_<serial1>_<serial2>.t7``). Then after accepting the "identified target point" by pressing "enter" the sphere sampling will begin, i.e. the robot will start moveing and recording images and poses.

> **_NOTE:_**  Make sure, that all collision objects in your robot's work space are modelled carefully (with safety margin), before starting the sphere sampling. The robot will move relatively fast via moveit and collisition check, but only for collision objects, that are correctly modelled.


### Robot kinematic calibration




### Hand eye optimization
