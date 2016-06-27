local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
local path = require 'pl.path'
local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d


local imageSaver =  calib.ImageSaver('/home/hoppe/data/2016-06-23/intrinsics_new/')
imageSaver:load()

local camera_name = "WEBCAM"


local patternGeom = {}
patternGeom.width = 4
patternGeom.height = 11
patternGeom.circleSpacing = 0.023

local imWidth = 960
local imHeight = 720

local robotCalibration = calib.Calibration(patternGeom, imWidth, imHeight)

local nImages = imageSaver:getNumberOfImages()

for i = 1,nImages do
  local img, pose = imageSaver:loadImage(i)
  robotCalibration:addImage(img[camera_name], pose.MoveitPose, pose.JointPos, 1)
end

local ok, matrix = robotCalibration:runCameraCalibration()
xamla3d.utils.printMatrixAsTorchTensor(matrix, "intrinsics")
