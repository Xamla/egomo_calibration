local calib = require 'egomo_calibration'

local a = calib.ImageSaver('/tmp')
a:load()
local cap = calib.Capture()
cap:addGrabFunctions("IR", a.getNextImage, a)
local images = cap:doGrabbing()
print(images)
