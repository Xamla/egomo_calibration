local calib = require('egomo_calibration.handEyeCalibration')
local xamla3d = calib.xamla3d
local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
require 'cv.features2d'



local dir = '/home/hoppe/Xamla/prototyping/hoppe/shortCapA/'
local poses = torch.load(dir..'fivePoints_000005.t7')

local success, list = xamla3d.utils.readFileList(dir, "", "png")

local intrinsics = torch.load(dir..'intrinsics.t7')
local distortion = torch.load(dir..'distortion.t7')

local patternSize = {}
patternSize.width = 8
patternSize.height = 21


if success then

  
  local Hg = {}
  local Hc = {}

  for i = 1,#list do
    local image = cv.imread{list[i]}
    local found, pose_4x4 = xamla3d.calibration.getPoseFromTarget (image, intrinsics, distortion, patternSize, 0.005)
    
    if found then          
      table.insert(Hc, pose_4x4)          
      table.insert(Hg, poses[1][i].full)
    end
  end
  
  
  calib.calibrate(Hg, Hc)
  
  
end




--for i = 1, 5 do
--  table.insert(Hg, torch.rand(4,4))
--  table.insert(Hc, torch.rand(4,4))
--end


