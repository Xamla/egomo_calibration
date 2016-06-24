local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
local path = require 'pl.path'
local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d


local patterns = {
  { directory='/home/hoppe/data/2016-06-23/handeye_new'} 
}


local patternGeom = {}
patternGeom.width = 4
patternGeom.height = 11
patternGeom.circleSpacing = 23

local imWidth = 960
local imHeight = 720
local camera_name = "WEBCAM"
local intrinsic = torch.eye(3,3)
intrinsic[1][1] = 920
intrinsic[2][2] = 920
intrinsic[1][3] = imWidth / 2
intrinsic[2][3] = imHeight / 2

function dh_calibration(intrinsic, pattern_path, pattern_geom, imwidth, imheight, camera_name) 

  local robotCalibration = calib.Calibration(pattern_geom, imwidth, imheight)
  robotCalibration.intrinsics = intrinsic
  robotCalibration.distCoeffs = torch.zeros(5,1)
  local img_saver_group = calib.ImageSaverGroup()
  
  for i = 1,#patterns do
    local path = patterns[1].directory
    local img_saver = calib.ImageSaver(path)
    img_saver:load()
    img_saver_group:addImageSaver(img_saver)
  end
  
  
  
  for i = 1,#img_saver_group.image_saver do 
    local img_saver = img_saver_group.image_saver[i]
    local n = img_saver:getNumberOfImages()
    for j = 1,n do
      local img, pose = img_saver:loadImage(j)
       robotCalibration:addImage(img[camera_name], pose.MoveitPose, pose.JointPos, i)    
    end
  end
  
  
  local best, robotCalibrationData = robotCalibration:DHCrossValidate(0.8, 2)

end


dh_calibration(intrinsic,patterns,patternGeom,imWidth,imHeight,camera_name)