local calib = require 'egomo_calibration.env'
local torch = require 'torch'

--- Class representing a depth camera
-- @classmod The class represents a depth camera
local DepthCamera = torch.class('egomo_calibration.DepthCamera', 'egomo_calibration.Camera', calib)


--- Depth camera constructor
-- Generates an instance of a depth camera
-- @param intrinsic 3x3 DoubleTensor - intrinsic camera parameters
-- @param distortion 5x1 DoubleTensor - distortion coefficients k1 k2 p1 p2 k3 (see opencv distortion model)
-- @param orientation 4x4 DoubleTensor - orientation of camera in space that rotates a point in space to camera coordinate system
-- @param width image width
-- @param height image height
-- @param z_offset offset in mm added to each z value
-- @param z_scaling each z value is scaled with this value
function DepthCamera:__init(intrinsic, distortion, heye, width, height, z_offset, z_scaling)
  calib.Camera.__init(self, intrinsic, distortion, heye, width, height)
  self.z_offset = z_offset or 0
  self.z_scaling = z_scaling or 1
end

function DepthCamera:__toString()
  print(string.format("Image width and height: %f %f", self.width, self.height))
  print("Intrinsics:")
  print(self.intrinsic)
  print("Distortion:")
  print(self.distortion)
  print("Orientation:")
  print(self.heye)  
  print(string.format("Z-Offset: %f Z-Scaling: %f", self.z_offset, self.z_scaling))
end


---
-- Returns the internal parameter of this camera as table
-- @return table - with fields intrinsic, distortion, width and height, z-offset and z-scaling
function DepthCamera:getCalibrationDataAsTable() 
  local data = {}
  data.intrinsic = self.intrinsic
  data.distortion = self.distortion
  data.width = self.width
  data.height = self.height  
  data.z_offset = self.z_offset
  data.z_scaling = self.z_scaling
  data.heye = self.heye  
  return data
end


function DepthCamera:toWorldCoordinates(robot_pose, point_3d)
  return (robot_pose * self.heye * point_3d:t()):t()  
end


---
-- @return x,y,3 
 function DepthCamera:getPointCloud(depth_img_orig)
  local depth_img = (depth_img_orig * self.z_scaling) + self.z_offset  

  local camIntrinsicsIRinverse = torch.inverse(self.intrinsic:type('torch.FloatTensor'))
  local xResolution = self.width
  local yResolution = self.height

  local z=torch.FloatTensor(yResolution, xResolution, 3)
  z[{{}, {}, 3}]=1
  z[{{}, {}, 1}]=torch.linspace(0.5, xResolution-0.5, xResolution):view(1,xResolution,1):expand(yResolution, xResolution,1)
  z[{{}, {}, 2}]=torch.linspace(0.5, yResolution-0.5, yResolution):view(yResolution,1,1):expand(yResolution, xResolution,1)

  local result= z:view(xResolution*yResolution, 3) * camIntrinsicsIRinverse:t()
  --result=result:t():clone()

  local mask = result[{{}, 3}]:le(0):view(result:size(1),1):expand(result:size())
  --result[mask]=0/0

  local depthExpanded=depth_img:view(result:size(1),1):expand(result:size())
  result:cmul(depthExpanded)

  local newCloud = torch.FloatTensor(yResolution, xResolution, 3)  
  newCloud[{{}, {}, {1,3}}] = result:view(yResolution,  xResolution, 3)

  return newCloud
  
end

function DepthCamera:setZCalibration(offset, scaling)
  self.z_offset = offset
  self.z_scaling = scaling  
end

---
-- Initializes the depth camera parameters given a table containing the parameters
-- @param caib_data - table contains (intrinsic, distortion, width, height, z_offset, z_scaling)
function DepthCamera:initializeFromCalibrationDataTable(calib_data)
  self.intrinsic = calib_data.intrinsic
  self.distortion = calib_data.distortion
  self.width = calib_data.width
  self.height = calib_data.height    
  self.heye = calib_data.heye    
  
  if calib_data.z_offset ~= nil then
    self.z_offset = calib_data.z_offset
    self.z_scaling = calib_data.z_scaling
  else
    self.z_offset = 0
    self.z_scaling = 1
  end
  
  
end
