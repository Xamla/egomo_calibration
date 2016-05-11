local path = require 'pl.path'
local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d


local output_directory_path = '/data/ur5/ur5_calibration/2016-05-11'
local pictures_per_position = 30
local velocity_scaling = 0.5

local capture = calib.Capture(output_directory_path, 30, velocity_scaling)
capture:run()
