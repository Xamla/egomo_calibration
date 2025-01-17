local path = require 'pl.path'
local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d


local output_directory_path = '/data/ur5_calibration/' .. os.date('%Y-%m-%d') .. '/'
local pictures_per_position = 30
local velocity_scaling = 0.5

local capture = calib.Capture(output_directory_path, 30, velocity_scaling)
capture:run()
