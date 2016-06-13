local calib = require 'egomo_calibration.env'
require 'egomo_calibration.DepthCamera'
require 'egomo_calibration.Camera' 

---
-- @classmod Class that represents the egomo sensor
local EgomoSensor = torch.class('egomo_calibration.EgomoSensor', calib)

---
-- Constructor of Egomo sensor. The Egomo sensor head consists of
-- 1 or 2 RGB cameras and one depth camera  
function EgomoSensor:__init()
  self.side_cam_RGB = calib.Camera()
  self.side_cam_depth = calib.DepthCamera()
  self.front_cam = calib.Camera()
  
  self.side_cam_RGB_hand_eye = torch.eye(4,4)
  self.side_cam_depth_hand_eye = torch.eye(4,4)
  self.front_cam_hand_eye = torch.eye(4,4)
  
  -- Pose of the sensor head in robot coordinates
  self.pose = torch.eye(4,4)
  
end


---
-- Sets the pose of the Egomo sensor (in robot base coordinates, i.e. the TCP).
-- Everytime the pose of the robot updates, this function has to be called. This function
-- updates the positions of the camera 
-- @param pose - 4x4 DoubleTensor position of the robots TCP 

function EgomoSensor:setPose(pose)
  self.pose = pose:clone()
  
  -- update the cameras mounted on the egomo sensor  
  self.side_cam_RGB.setOrientation(torch.inverse(pose * self.side_cam_RGB_handy_eye))
  self.side_cam_depth.setOrientation(torch.inverse(pose * self.side_cam_depth_handy_eye))
  self.front_cam.setOrientation(torch.inverse(pose * self.front_cam_handy_eye))      
end


---
-- Returns the current position of the Egomo sensor in robot base coordinates
-- @return 4x4 DoubleTensor - pose of the Egomo
function EgomoSensor:getPose()
  return self.pose
end


---
-- Loads a calibration for the Egomo sensorhead and sets these parameters to the associated cameras
-- @param filepath - string where file is located 
function EgomoSensor:loadCalibration(filepath)
  
  if not xamla3d.utils.isDir(filepath) then
    return false
  end
 
  local calib_data = torch.load(filepath)
  self.side_cam_RGB:initializeFromCalibrationDataTable(calib_data.calib_side_cam_RGB)
  self.front_cam:initializeFromCalibrationDataTable(calib_data.calib_front_cam)
  
  
  return true  
end


---
-- Save parameters of Egomo sensor
-- @param filepath - 
function EgomoSensor:saveCalibration(filepath)

  -- test if the file can be created
  local file = io.open(filepath, "w")
  if file == nil then
    return false
  end
  file:close()
  
  local calib_data = {}
  calib_data.calib_side_cam_RGB = self.side_cam_RGB:getCalibrationDataAsTable()
  calib_data.calib_side_cam_depth = self.side_cam_depth:getCalibrationDataAsTable()
  calib_data.calib_front_cam = self.front_cam:getCalibrationDataAsTable()
  calib_data.side_cam_RGB_hand_eye = self.side_cam_RGB_hand_eye
  calib_data.side_cam_depth_hand_eye = self.side_cam_depth_hand_eye
  calib_data.front_cam_hand_eye = self.front_cam_hand_eye
  
  torch.save(filepath, calib_data)
  
  return true
  
end

