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




### Robot kinematic calibration




### Hand eye optimization
