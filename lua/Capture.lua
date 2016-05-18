local calib = require 'egomo_calibration.env'
local egomoTools = require 'egomo-tools'
local torch = require 'torch'
local path = require 'pl.path'
local ros = require 'ros'
local pcl = require 'pcl'

local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
require 'cv.features2d'

local up = torch.DoubleTensor({0,0, 1})

-- private member functions
local function initializeRobot(self, velocity_scaling)
  self.roboControl = egomoTools.robot:new("capture", 0.4)
  print("Robot initialization finished.")

  self.ps = self.roboControl:GetPlanningSceneInterface()
  self.roboControl.rosMoveGroup:setPlanningTime(2.0)   -- we will ignore poses for which we do not find a plan within 2s

  local camIntrinsicsIR=torch.FloatTensor({
     {563, 0, 322},
     {0, 562, 240},
     {0, 0, 1}})
  local depthcam = egomoTools.structureio:new(camIntrinsicsIR)
  depthcam:Connect()
  depthcam:SetProjectorStatus(false)
  depthcam:SetIRresolution(640, 480)
  self.depthcam = depthcam
  print("Depthcam initialisation finished.")

  local webcam = egomoTools.webcam:new()
  webcam:ConnectDefault()
  self.webcam = webcam
  print("Webcam initialisation finished.")
end


local function showImage(img, winName, delay)
  winName = winName or 'Capture Output'
  local key = 0
  cv.imshow{winName, img}
  while not key%256 == string.byte("q") or key%256 == 27 do -- 27: esc key
    if delay ~= nil then
      cv.waitKey{delay}
      return
    else
      key=cv.waitKey{-1}
    end
  end
end


--[[
  Calculate the "true" 3D position (x,y,z) of the circle centers of the circle pattern.
  z position is set to 0 for all points

  Input params:
    arg.pointsX  -- number of points in horizontal direction
    arg.pointsY  -- number of points in vertical direction
    arg.pointDistance -- distance between two points of the pattern in meter
  Return value:
    Position of the circle centers
]]
local function calcPointPositions(arg)

  local corners = torch.FloatTensor(arg.pointsX*arg.pointsY, 1, 3):zero()
  local i=1
  for y=1, arg.pointsY do
    for x=1, arg.pointsX do
      corners[i][1][1] = (2*(x-1) + (y-1)%2) * arg.pointDistance
      corners[i][1][2] = (y-1)*arg.pointDistance
      corners[i][1][3] = 0
      i = i+1
    end
  end
  return corners
end

--[[
  transform a rotation vector as e.g. provided by solvePnP to a 3x3 rotation matrix using the Rodrigues' rotation formula
  see e.g. http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#void%20Rodrigues%28InputArray%20src,%20OutputArray%20dst,%20OutputArray%20jacobian%29

  Input parameters:
    vec = vector to transform
  Return value:
    3x3 rotation matrix
]]
local function rotVectorToRotMatrix(vec)
  local theta = torch.norm(vec)
  local r = vec/theta
  r = torch.squeeze(r)
  local mat = torch.Tensor({{0, -1*r[3], r[2]}, {r[3], 0, -1*r[1]}, {-1*r[2], r[1], 0}})
  r = r:resize(3,1)
  local result = torch.eye(3)*math.cos(theta) + (r*r:t())*(1-math.cos(theta)) + mat*math.sin(theta)
  return result
end


local function saveImages(path, prefix, count, imgIR, imgWeb)
  local fileName=string.format("%s_%06i", prefix, count)
  if imgIR then
   local saveSuccess = cv.imwrite{filename=path.."/"..fileName.."_ir.png", img=imgIR}
    if not saveSuccess then
      print("Could not save "..path.."/"..fileName.."_ir.png")
      return false
    end
  end
  if imgWeb then
    local saveSuccess = cv.imwrite{filename=path.."/"..fileName.."_web.png", img=imgWeb}
    if not saveSuccess then
      print("Could not save "..path.."/"..fileName.."_web.png")
      return false
    end
  end
  return fileName
end


local function savePoses(path, prefix, count, poseData)
  local fileName
  if not count==nil then
    fileName=string.format("%s_%06i.t7", prefix, count)
  else
    fileName=string.format("%s.t7", prefix)
  end
  torch.save(path.."/"..fileName, poseData, ascii)

  return fileName
end


local function mkdir_recursive(dir_path)
  dir_path = path.abspath(dir_path)
  local dir_names = string.split(dir_path, "/")
  print("----------------------------------------------------")
  print(dir_names)
  local current_path = '/'
  for i,fn in ipairs(dir_names) do
    current_path = path.join(current_path, fn)

    if not path.exists(current_path) then
      path.mkdir(current_path)
    elseif path.isfile(current_path) then
      error("Cannot create directory. File is in the way: '" .. current_path .. "'.'")
    end
  end
end


local Capture = torch.class('egomo_calibration.Capture', calib)


function Capture:__init(output_path, pictures_per_position, velocity_scaling)
  self.output_path = output_path
  self.pictures_per_position = pictures_per_position or 30

  -- initial guess for hand-eye matrix and camera parameters
  self.heye = torch.DoubleTensor({
    {  0.0025,   0.7642,   0.6450,  0.0152395 },
    { -0.0007,  -0.6450,   0.7642,  0.0699035 },
    {  1.0000,  -0.0024,  -0.0011,  0.0559415 },
    {  0.0000,   0.0000,   0.0000,  1.0000    }
  })
  self.intrinsics = torch.Tensor({
    {  918.3122,    0.0000,  481.8074 },
    {    0.0000,  917.5487,  359.0547 },
    {    0.0000,    0.0000,    1.0000 }
  })
  self.distortion = torch.Tensor({0.1448, -0.5273, -0.0007, 0.0028, 0.9005})
  self.pattern = { width = 8, height = 21, pointDistance = 0.005 }

  initializeRobot(self, velocity_scaling or 0.5)
end


function Capture:grabImage()
  return self.webcam:GrabGrayscaleImgROS()
end


function Capture:searchPatternCircular(center, radius, height)
  local overview_focus = 10

  self.webcam:SetFocusValue(overview_focus)      -- focus for overview pose
  print(string.format("Set focus for overview pose to %d.", overview_focus))

  local angle = 0

  while ros.ok() do

      -- move to search pose and look for calibration pattern
    local p = torch.mv(pcl.affine.rotateEuler(0,0,angle):double(), torch.Tensor({radius,0,0,0})):add(center)
    local robot_pose = self.roboControl:WebCamLookAt(p[{{1,3}}], height, math.rad(-30), math.rad(0.5), self.heye)
    if self.roboControl:MoveRobotTo(robot_pose) then
      sys.sleep(0.1)  -- wait for controller position convergence
      local img = self:grabImage()
      local ok,pattern_points = cv.findCirclesGrid{image=img, patternSize={height=self.pattern.height, width=self.pattern.width}, flags=cv.CALIB_CB_ASYMMETRIC_GRID}
      if ok then
        local circlePositions = calcPointPositions{pointsX=self.pattern.width, pointsY=self.pattern.height, pointDistance=self.pattern.pointDistance}
        local pose_found, pose_cam_rot_vector, pose_cam_trans=cv.solvePnP{objectPoints=circlePositions, imagePoints=pattern_points, cameraMatrix=self.intrinsics, distCoeffs=self.distortion}
        if not pose_found then
          error('could not calculate pose from calibration pattern')
        end

        local pose_cam_rot_matrix = rotVectorToRotMatrix(pose_cam_rot_vector)

        -- assemble the 4x4 transformation matrix
        local pattern_pose = torch.eye(4)
        pattern_pose[{{1,3}, {1,3}}] = pose_cam_rot_matrix
        pattern_pose[{{1,3}, {4}}] = pose_cam_trans

        local pattern_center_offset = torch.mv(pattern_pose, torch.Tensor({self.pattern.pointDistance * self.pattern.width, 0.5 * self.pattern.pointDistance * self.pattern.height, 0,0}))
        pattern_pose[{{},4}]:add(pattern_center_offset)

        return true, pattern_pose, robot_pose
      end

      print('Calibration pattern not in view.')
    else
      print('Move to capture pose failed.')
    end

    angle = angle + math.pi / 5
    if angle > 2*math.pi then
      return false, nil
    end
  end

  return false, nil
end


function Capture:searchPattern()
  local radius = 0.15
  local height = 0.40
  local patter_search_center = torch.Tensor({0.18, 0.48, 0, 1})

  while ros.ok() do
    local ok, pattern_pose, robot_pose = self:searchPatternCircular(patter_search_center, radius, height)
    if ok then
      return pattern_pose, robot_pose
    end

    if radius < 0.5 then
      radius = radius + 0.1
    else
      radius = 0.1  --restart search with 10cm distance
    end
  end

  error('Search for calibration pattern aborted.')
end


local function captureSphereSampling(self, path, filePrefix, robot_pose, transfer, count, min_radius, max_radius, focus, target_jitter)
    -- default values
  min_radius = min_radius or 0.13   -- min and max distance from target
  max_radius = max_radius or 0.15
  focus = focus or 30
  target_jitter = target_jitter or 0.015

  local t = robot_pose * self.heye * transfer
  local targetPoint = t[{{1,3},4}]

  print('identified target point:')
  print(targetPoint)

  local poseData = {}
  poseData["MoveitPose"] = {}
  poseData["UR5Pose"] = {}
  poseData["JointPos"] = {}
  poseData["FileName"] = {}

  self.webcam:SetFocusValue(focus)

  for i=1,count do

    -- generate random point in positive half shere
    local origin
    while true do
      origin = torch.randn(3)
      origin[3] = math.max(0.01, math.abs(origin[3]))
      origin:div(origin:norm())
      if origin[3] > 0.95 then
        break
      end
    end

    origin:mul(torch.lerp(min_radius, max_radius, math.random()))
    origin:add(targetPoint)

    local target = targetPoint + math.random() * target_jitter - 0.5 * target_jitter

    local up_ = up

    up_ = t[{1,{1,3}}] -- use pattern x axis in world

    if math.random(2) == 1 then
      up_ = -up_
    end

    local movePose = self.roboControl:PointAtPose(origin, target, up_, self.heye)

    if self.roboControl:MoveRobotTo(movePose) then
      sys.sleep(0.5)    -- wait for controller position convergence
      local imgWeb = self:grabImage()
      poseData["MoveitPose"][i] = self.roboControl:ReadRobotPose(true)
      local ur5state = self.roboControl:ReadUR5data()
      poseData["UR5Pose"][i] = self.roboControl:DecodeUR5TcpPose(ur5state, true)
      poseData["JointPos"][i] = self.roboControl:DecodeUR5actualJointState(ur5state)
      poseData["FileName"][i] = saveImages(path, filePrefix, i, nil, imgWeb)

      savePoses(path, filePrefix, i, poseData)
      i=i+1
    end
  end

  return savePoses(path, filePrefix, nil, poseData)
end


function Capture:run()
  print('Storing output in: '.. self.output_path)
  local capture_data_files = {}
  local i = 1
  while true do
    print(string.format('Please place pattern at position %d.', i))
    print('Ready? Please press enter.')
    io.stdin:read()

    local capture_output_path = path.join(self.output_path, string.format('pose%03d', i))
    mkdir_recursive(capture_output_path)    -- ensure output directory exists

    local pattern_pose, robot_pose = self:searchPattern()



    local file_prefix = string.format('pose%03d_', i)
    local pose_data_filename = captureSphereSampling(self, capture_output_path, file_prefix, robot_pose, pattern_pose, self.pictures_per_position)
    table.insert(capture_data_files, pose_data_filename)
    i = i + 1
  end
  return capture_data_files
end
