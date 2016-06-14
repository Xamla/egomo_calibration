local calib = require('egomo_calibration')
local xamla3d = calib.xamla3d
local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
require 'cv.features2d'


local dir= "/home/hoppe/data/cam_calib_ir/irCalibC/"
robotPoses = torch.load(path.join(dir, "irCalib_000059.t7"))

local intrinsics = torch.load('/home/hoppe/data/cam_calib_ir/irCalibB/irCalib_camIntrinsic_ir.t7')[6]
local distortion = torch.load('/home/hoppe/data/cam_calib_ir/irCalibB/irCalib_camIntrinsic_ir.t7')[7]

local patternSize = {}
patternSize.width = 4
patternSize.height = 11


local posesPattern = {}

local min_error = 10000
local bestHE = nil
  
local Hg = {}
local Hc = {} 

 for i = 1, #robotPoses.MoveitPose do
    if robotPoses.MoveitPose[i] ~= nil then    
      local fn = dir.."/"..robotPoses.FileName[i].."_ir.png"
      print(fn)
      local image = cv.imread{fn}    
      local found, pose_4x4 = xamla3d.calibration.getPoseFromTarget (image, intrinsics, distortion, patternSize, 0.023)       
    
      if found then     
        table.insert(Hg, robotPoses.MoveitPose[i].full)
        table.insert(Hc, pose_4x4)
    end
  end
end
  
local HE, error = calib.handEye.calibrateViaCrossValidation(Hg, Hc, 30, 1000) -- 30 has to be smaller then nImg
print("Max error: " ..torch.max(torch.abs(error)))
print("Mean:      " ..torch.mean(torch.abs(error)))
print("Variance:  " ..torch.var(torch.abs(error)))
print(HE)





