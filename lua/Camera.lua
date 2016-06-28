local calib = require 'egomo_calibration.env'
local xamla3d = require 'egomo_calibration.xamla3d'
local torch = require 'torch'

--- Class representing a standard pinhole camera model
-- @classmod The class represents a standard pinhole camera model including intrinsics, extrinsics and radial distortion
local Camera = torch.class('egomo_calibration.Camera', calib)


--- Constructor
-- Creates a pinhole camera
-- @param intrinsic 3x3 DoubleTensor - intrinsic camera parameters
-- @param distortion 5x1 DoubleTensor - distortion coefficients k1 k2 p1 p2 k3 (see opencv distortion model)
-- @param orientation 4x4 DoubleTensor - orientation of camera in space that rotates a point in space to camera coordinate system
-- @param width image width
-- @param height image height
function Camera:__init(intrinsic, distortion, heye, width, height)
  self.intrinsic = intrinsic or torch.eye(3,3)
  self.distortion = distortion or torch.zeros(5,1)
  self.heye = heye or torch.eye(4,4)
  self.width = width or 0
  self.height = height or 0
  self.grab_function = nil  
end


function Camera:__toString()
  print(string.format("Image width and height: %f %f", self.width, self.height))
  print("Intrinsics:")
  print(self.intrinsic)
  print("Distortion:")
  print(self.distortion)
  print("Orientation:")
  print(self.heye)
end


function Camera:setGrabFunction(function_handle, instance)
  self.grab_function = {}
  self.grab_function.handle = function_handle
  self.grab_function.instance = instance
end


function Camera:grab()
  if self.grab_function ~= nil then
    return self.grab_function.handle(self.grab_function.instance)
  else
    error("Grab function in camera is nil")
    return nil
  end  
end

--- Projection
-- Projects a point from world coordinates into camera (no check if point is in image!)
-- @param point 3x1 DoubleTensor - point in space
-- @return point 2x1 DoubleTensor - point in image coordinates
function Camera:projectPoint(point, robot_pose)  
  return xamla3d.projectPoint(self.intrinsic, torch.inverse(robot_pose * self.heye), point)
end


function Camera:getDistortion()
  return self.distortion
end


function Camera:setDistortion(distortion)
  self.distortion = distortion:clone()
end


---
-- Sets intrinsic parameters of camera
-- @param 3x3 DoubleTensor - intrinsic matrix
function Camera:setIntrinsic(intrinsic)
  self.intrinsic = intrinsic:clone()
end


---
-- Get intrinsic matrix
-- @return 3x3 DoubleTensor - intrinsic matrix
function Camera:getIntrinsic()
  return self.intrinsic
end


---
-- Gets image width
-- @return image width in pixel
function Camera:getWidth()
  return self.width
end


---
-- Get image height
-- @return image height in pixel
function Camera:getHeight()
  return self.height
end

--- Sets the orientation of the camera 
-- Sets the orientation of the camera
-- @param orientation - 4x4 homogenous matrix [R|t] that rotates a point in world coordinates in camera space 
function Camera:setHandEye(hand_eye)
  self.heye = hand_eye:clone()
end


---
-- Returns camera orientation as 4x4 DoubleTensor
-- @return camera orientation as 4x4 DoubleTensor (homogenous matrix)
function Camera:getHandEye()
  return self.heye 
end


---
-- Sets the image size of the camera
-- @param width image width in pixel
-- @param height image height in pixel
function Camera:setImageSize(width, height)
  self.width = width
  self.height = height
end


---
-- Returns the calibration data of the camera, i.e. intrinsic, distortion, width and height
-- @return intrinsic 3x3 DoubleTensor
-- @return distortion 5x1 DoubleTensor (k1, k2, p1, p2, k3)
-- @return image width
-- @return image height
function Camera:getCalibrationData()
  return self.intrinsic, self.distortion, self.width, self.height
end


---
-- Returns the internal parameter of this camera as table
-- @return table - with fields intrinsic, distortion, width and height
function Camera:getCalibrationDataAsTable()
  local data = {}
  data.intrinsic = self.intrinsic
  data.distortion = self.distortion
  data.width = self.width
  data.height = self.height  
  data.heye = self.heye
  return data
end


---
-- Initializes the camera parameters given a table containing the parameters
-- @param caib_data - table contains (intrinsic, distortion, width, height)
function Camera:initializeFromCalibrationDataTable(calib_data)
  self.intrinsic = calib_data.intrinsic
  self.distortion = calib_data.distortion
  self.width = calib_data.width
  self.height = calib_data.height  
  self.heye = calib_data.heye
end
