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


local goodPoses = {}
local badPoses = {}

local min_error = 10000
local bestHE = nil

local nTrials = 1000
for nTrial = 1,nTrials do
  local idx = torch.randperm(#robotPoses.MoveitPose)
  local Hg = {}
  local Hc = {}
 
  local ids = {}
 
  local i = 1
  while #Hg < 8 do
    local id = idx[i] --i+nTrial
    i = i +1 

    if robotPoses.MoveitPose[id] ~= nil then
 
     table.insert(ids, id)
     local fn = dir.."/"..robotPoses.FileName[id].."_web.png"
     local image = cv.imread{fn}
     local found = false
     local pose_4x4 = nil
     if posesPattern[id] ~= nil then
      found = true
      pose_4x4 = posesPattern[id]
     end
     
     if not found then
       found, pose_4x4 = xamla3d.calibration.getPoseFromTarget (image, intrinsics, distortion, patternSize, 0.005)
       if found then
         posesPattern[id] = pose_4x4:clone()
       else
        pose_4x4 = torch.eye(4,4)
        posesPattern[id] = torch.eye(4,4) 
       end
     end
     if found and pose_4x4 ~= torch.eye(4,4) then     
      table.insert(Hg, robotPoses.MoveitPose[id].full)
      table.insert(Hc, pose_4x4)
     end
     end
  
  end
  
  for i = 1,#ids do
    io.write(ids[i].." ")
  end
  io.write("\n")
  
  local HE, res, res_angle = calib.calibrate(Hg,Hc)
  print("Max Alignment "..torch.max(res))
  print("Max Angle     "..torch.max(res_angle).." "..torch.max(res)) 
  if (torch.max(res) < min_error) then
    min_error =torch.max(res)   
    bestHE = HE:clone()     
  end
  print("MinError " ..min_error)  
   
end

print(bestHE)
  

  
  local HgValidate = {}
  local HcValidate = {}
  
  --[[
  for i = 6,#robotPoses.MoveitPose do
  
   local id = idx[i]   
   if robotPoses.FileName[id] ~= nil then     
     local fn = dir.."/"..robotPoses.FileName[id].."_web.png"
     
     local image = cv.imread{fn}
     
       local found = false
       local pose_4x4 = nil
       if posesPattern[id] ~= nil then
        found = true
        pose_4x4 = posesPattern[id]
       end
       
       if not found then
         found, pose_4x4 = xamla3d.calibration.getPoseFromTarget (image, intrinsics, distortion, patternSize, 0.005)
         if found then
           posesPattern[id] = pose_4x4:clone()
         else
           pose_4x4 = torch.eye(4,4)
           posesPattern[id] = torch.eye(4,4) 
         end
       end
        
     if found and pose_4x4 ~= torch.eye(4,4)  then
      table.insert(HgValidate, robotPoses.MoveitPose[id].full)
      table.insert(HcValidate, pose_4x4)
     end
    end   
  end
  
   local s, res =calib.getAlignError(HgValidate, HcValidate, HE)
   print(s)

end
]]




