local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
local path = require 'pl.path'
local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d


local imageSaver =  calib.ImageSaver('/home/hoppe/data/2016-06-27/ZCalib/')
imageSaver:load()

local camera_name = "DEPTH"


local patternGeom = {}
patternGeom.width = 4
patternGeom.height = 11
patternGeom.pointDistance = 0.023

local K = torch.eye(3,3)
K[1][1] = 570
K[2][2] = 570
K[1][3] = 640/2
K[2][3] = 480/2
local distortion = torch.zeros(5,1)

local depth_calibrator = calib.DepthCalibrator(K, distortion, patternGeom)

local nImages = imageSaver:getNumberOfImages()
for i = 1,nImages do
  local img, pose = imageSaver:loadImage(i)
  depth_calibrator:addImage(img["DEPTHCAM_NO_SPECKLE"], img["DEPTH"])  
end

depth_calibrator:doCalibration()
