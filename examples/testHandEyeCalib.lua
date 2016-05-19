local calib = require('../lua/handEyeCalibration')
local xamla3d = require('../lua/xamla3d')
local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
require 'cv.features2d'


local dir= "/home/hoppe/data/2016-05-03.3/"
robotPoses = torch.load(path.join(dir, "spheresurface_000115.t7"))

local intrinsics = torch.load(dir..'intrinsics.t7')
local distortion = torch.load(dir..'distortion.t7')

local patternSize = {}
patternSize.width = 8
patternSize.height = 21


local posesPattern = {}

local min_error = 10000
local bestHE = nil
  
local Hg = {}
local Hc = {} 

 for i = 1, #robotPoses.MoveitPose do
    if robotPoses.MoveitPose[i] ~= nil then    
      local fn = dir.."/"..robotPoses.FileName[i].."_web.png"
      print(fn)
      local image = cv.imread{fn}    
      local found, pose_4x4 = xamla3d.calibration.getPoseFromTarget (image, intrinsics, distortion, patternSize, 0.005)       
    
      if found then     
        table.insert(Hg, robotPoses.MoveitPose[i].full)
        table.insert(Hc, pose_4x4)
    end
  end
end
  
local HE, error = calib.calibrateViaCrossValidation(Hg, Hc, 8, 300)
print("Max error: " ..torch.max(torch.abs(error)))
print("Mean:      " ..torch.mean(torch.abs(error)))
print("Variance:  " ..torch.var(torch.abs(error)))
print(HE)





