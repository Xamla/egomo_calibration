local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
local path = require 'pl.path'
local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d

local function doCalibration(path, cam_name_depth, cam_name_ir, camera)
  local imageSaver =  calib.ImageSaver(path)
  imageSaver:load()
 
  
  
  local patternGeom = {}
  patternGeom.width = 4
  patternGeom.height = 11
  patternGeom.pointDistance = 0.023
  
  local K = camera:getIntrinsic()
  local distortion = camera:getDistortion()
  
  local depth_calibrator = calib.DepthCalibrator(K, distortion, patternGeom)
  
  local nImages = imageSaver:getNumberOfImages()
  for i = 1,nImages do
    local img, pose = imageSaver:loadImage(i)
    depth_calibrator:addImage(img[cam_name_ir], img[cam_name_depth])  
  end
  
  return depth_calibrator:doCalibration()
end



local cmd=torch.CmdLine()
cmd:option('-path', "./noname/", 'Directory where calibration images are stored')
cmd:option('-calib', "./noname/calib.t7", 'File where calibration data is stored')
cmd:option('-camname_ir', "WEBCAM", 'Name of IR camera that has images without speckle pattern')
cmd:option('-camname_depth', "DEPTH", 'Name of depthcamera holding 2.5D views')


local params = cmd:parse(arg)

local calib_data = torch.load(params.calib)
local cam = calib.DepthCamera()

cam:initializeFromCalibrationDataTable(calib_data)
local a,b = doCalibration(params.path, params.camname_depth, params.camname_ir, cam)

cam:setZCalibration(b,a)

local calib_data = cam:getCalibrationDataAsTable()
torch.save(params.calib, calib_data)




