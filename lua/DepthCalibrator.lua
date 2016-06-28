local calib = require 'egomo_calibration.env'
local torch = require 'torch'
local xamla3d = require 'egomo_calibration.xamla3d'

local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.imgcodecs'


 --calculates the pseudo-inverse of matrix m via SVD
local function pinv(m)
   local u, s, v = torch.svd(m, "A")

   -- transform s to a full matrix and calculate the pseudo inverse of that matrix
   local snew = torch.DoubleTensor(3, 4):zero()
   snew[1][1]=1.0/s[1]
   snew[2][2]=1.0/s[2]
   snew[3][3]=1.0/s[3]
   snew = snew:t()

   return v * snew * u:t()
end

-- Coordinates have to be provided as 3x1 tensors, not in homogene coordinates
local function PlaneLineIntersection(normalPlane, pointPlane, lineVector, camCenter)
   lineVector = lineVector / lineVector:norm()
   normalPlane = normalPlane / normalPlane:norm()

   local denom = torch.dot(normalPlane, lineVector)
   local intersect
   local length
   if torch.abs(denom) > 0.00001 then
      length = torch.dot( (pointPlane - camCenter), normalPlane) / denom
      intersect = camCenter + lineVector*length
   end

   return intersect, length
end


local DepthCalibrator = torch.class('egomo_calibration.DepthCalibrator', calib)

function DepthCalibrator:__init(intrinsic, distortion, pattern_geometry)
  self.intrinsic = intrinsic
  self.distortion = distortion
  self.pattern_geom = pattern_geometry
  
  self.depth_data = {}
  self.ir_data = {}
  
end


-- points: Tensor of size (nPoints, 2) with the second dimension representing the x and y coordinate of the point
local function fitData(points)
   local fitResult = cv.fitLine{points = points, distType = cv.DIST_L2, param = 0, reps = 0.01, aeps = 0.01}
   fitResult = fitResult:squeeze()
   -- form: a*x+b
   local a = fitResult[2]/fitResult[1]
   local b = fitResult[4] - a*fitResult[3]

   return a, b
end


function DepthCalibrator:addImage(ir_image, depthmap)
  table.insert(self.depth_data, depthmap)
  table.insert(self.ir_data, ir_image)
end

function DepthCalibrator:doCalibration()

  local pattern_normal = torch.Tensor({0, 0, 1})
  local pattern_origin = torch.Tensor({0, 0, 0})
  local cam_center = torch.Tensor({self.intrinsic[1][3], self.intrinsic[2][3], 1})
  
  local corr_depth = {}
  corr_depth.pattern = {}
  corr_depth.depth_img = {} 

  for i = 1,#self.depth_data do
    local ok, pose = xamla3d.calibration.getPoseFromTarget(self.ir_data[i], self.intrinsic, self.distortion, self.pattern_geom, self.pattern_geom.pointDistance)
    if ok then
      local P = self.intrinsic*pose[{{1,3}, {1,4}}]
      local Pinv = pinv(P)
      local center_line = Pinv * cam_center
      center_line = center_line / center_line[4]      
      center_line = center_line:view(4,1)[{{1,3},1}]     
      local intersection_pt, distance = PlaneLineIntersection(pattern_normal,pattern_origin, center_line:squeeze(),  pose[{{1,3}, 4}])     
      
      local depth_at_center = 0
      local depth_img = self.depth_data[i]:squeeze()
      local cnt = 0 
      
      for j = -1,1 do
        for k = -1, 1 do
           local x = self.intrinsic[1][3] + j
           local y = self.intrinsic[2][3] + k
           if depth_img[y][x] > 0.1 then
             depth_at_center = depth_at_center + depth_img[y][x]
             cnt = cnt + 1
           end
        end
      end 
      depth_at_center = depth_at_center / cnt
      
      if cnt > 4 then
        corr_depth.pattern[#corr_depth.pattern+1] = distance * -1
        corr_depth.depth_img[#corr_depth.depth_img + 1] = depth_at_center
      end
             
    end
  end

  local points_to_fit = torch.Tensor(#corr_depth.pattern, 2)
  for p = 1, #corr_depth.pattern do
    points_to_fit[p][1] = corr_depth.depth_img[p]
    points_to_fit[p][2] = corr_depth.pattern[p]
        
  end
  
  local a,b = fitData(points_to_fit)
  print(string.format("Scale factor %f", a))
  print(string.format("Offset %f", b))
  
  for p = 1, #corr_depth.pattern do
    local error = (a * points_to_fit[p][1] + b) - points_to_fit[p][2]          
    print(string.format("Pattern depth: %f Depth in image: %f (error: %fmm)",  points_to_fit[p][1],  points_to_fit[p][2], error*1000))
    
  end
  
  return a,b
  
end