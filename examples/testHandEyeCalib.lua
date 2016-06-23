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



local dir= "/home/hoppe/data/2016-06-03/pose001/"
robotPoses = torch.load(path.join(dir, "pose001_.t7"))

local intrinsics = torch.load(dir..'/intrinsics.t7').intrinsics
local distortion = torch.load(dir..'/intrinsics.t7').distCoeffs

print("Intrinsics")
print(intrinsics)
print(distortion)

local patternSize = {}
patternSize.width = 8
patternSize.height = 21
circlesDistance = 0.008


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
      local found, pose_4x4 = xamla3d.calibration.getPoseFromTarget (image, intrinsics, distortion, patternSize, circlesDistance)       
    
      if found then     
        table.insert(Hg, robotPoses.MoveitPose[i].full)
        table.insert(Hc, pose_4x4)
    end
  end
end
  
local HE, error = calib.handEye.calibrateViaCrossValidation(Hg, Hc, 20, 1000)
print("Max error: " ..torch.max(torch.abs(error)))
print("Mean:      " ..torch.mean(torch.abs(error)))
print("Variance:  " ..torch.var(torch.abs(error)))
printMatrixAsTorchTensor(HE)





