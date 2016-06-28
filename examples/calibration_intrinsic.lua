local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
local path = require 'pl.path'
local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d


function doCalibration(path, cam_name)
  local imageSaver =  calib.ImageSaver(path)
  imageSaver:load()
  
  local camera_name = cam_name
  
  
  local patternGeom = {}
  patternGeom.width = 4
  patternGeom.height = 11
  patternGeom.circleSpacing = 0.023
  
  local imWidth, imHeight = imageSaver:getImageSize(camera_name)
  
  local robotCalibration = calib.Calibration(patternGeom, imWidth, imHeight)
  
  local nImages = imageSaver:getNumberOfImages()
  for i = 1,nImages do
    local img, pose = imageSaver:loadImage(i)
    robotCalibration:addImage(img[camera_name], pose.MoveitPose, pose.JointPos, 1)
  end
  
  local ok, matrix, distcoeffs = robotCalibration:runCameraCalibration()
  xamla3d.utils.printMatrixAsTorchTensor(matrix, "intrinsics")
  return ok, matrix, distcoeffs, imWidth, imHeight 
  
end



local cmd=torch.CmdLine()
cmd:option('-path', "./noname/", 'Directory where calibration images are stored')
cmd:option('-calib', "./noname/calib.t7", 'File where calibration data is stored')
cmd:option('-camname', "WEBCAM", 'Name of the camera to be calibrated')


local params = cmd:parse(arg)


--local calib_data = torch.load(params.calib)
--cam:initializeFromCalibrationDataTable(calib_data)
local ok, K, dist, im_width, im_height = doCalibration(params.path, params.camname)

local cam = calib.Camera(K, dist, torch.eye(3,3), im_width, im_height)
local calib_data = cam:getCalibrationDataAsTable()
torch.save(params.calib, calib_data)