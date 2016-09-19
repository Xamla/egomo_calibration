local calib = require('egomo_calibration')
local xamla3d = calib.xamla3d
local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
require 'cv.features2d'

function printMatrixAsTorchTensor(m)
--if (m:nDimensions() ~= 2) then
-- return
--end

local write = io.write

write("a = torch.DoubleTensor({\n")
for i = 1,m:size()[1] do
  write("{")
  for j = 1,m:size()[2] do
  write(m[i][j].. ",")
  end
  write("},\n")
end

write("})\n")

end

local function doCalibration(image_path, camera, cam_name)
  
  local imageSaver =  calib.ImageSaver(image_path)
  imageSaver:load()
  
  
  
  local intrinsics = camera:getIntrinsic()
  local distortion = camera:getDistortion()
  print("Intrinsics")
  print(intrinsics)
  print(distortion)
  
  local circlesDistance = 0.023
  local patternSize = {}
  patternSize.width = 4
  patternSize.height = 11
  
  
  local min_error = 10000
  local bestHE = nil
    
  local Hg = {}
  local Hc = {} 
  
  
  
  local nImages = imageSaver:getNumberOfImages()
  for i = 1,nImages do
    local img, pose = imageSaver:loadImage(i)
    local found, pose_4x4 = xamla3d.calibration.getPoseFromTarget (img[cam_name], intrinsics, distortion, patternSize, circlesDistance)
    if found then     
      table.insert(Hg, pose.MoveitPose.full)
      table.insert(Hc, pose_4x4)
    end
  end
  
  local HE, error = calib.handEye.calibrateViaCrossValidation(Hg, Hc, 20, 1000)
  print("Max error: " ..torch.max(torch.abs(error)))
  print("Mean:      " ..torch.mean(torch.abs(error)))
  print("Variance:  " ..torch.var(torch.abs(error)))
  print(HE)
  return HE
  
end


local cmd=torch.CmdLine()
cmd:option('-path', "./noname/", 'Directory where calibration images are stored')
cmd:option('-calib', "./noname/calib.t7", 'File where calibration data of camera is stored')
cmd:option('-camname', "WEBCAM", 'Name of the camera to be calibrated')


local params = cmd:parse(arg)


local calib_data = torch.load(params.calib)
local cam = calib.Camera()
cam:initializeFromCalibrationDataTable(calib_data)
local HE = doCalibration(params.path, cam, params.camname)
cam:setHandEye(HE)
local calib_data = cam:getCalibrationDataAsTable()
torch.save(params.calib, calib_data)




