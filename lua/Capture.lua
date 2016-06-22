local calib = require 'egomo_calibration.env'
local xamla3d = require 'egomo_calibration.xamla3d'
local egomoTools = require 'egomo-tools'
local torch = require 'torch'
local path = require 'pl.path'
local ros = require 'ros'
local pcl = require 'pcl'
tf = ros.tf

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

  local webcam = egomoTools.webcam:new("egomo_webcam")
  webcam:ConnectDefault()
  webcam:ConnectToJpgStream()
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
]]



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


function Capture:calcCamPoseFromDesired2dPatternPoints(borderWidth, radius, pattern_in_robot)
  local intrinsics = self.intrinsics
  local w = self.imwidth
  local h = self.imheight

  local ul_3d = {x = 0, y = 0}
  local ur_3d = {x = (self.pattern.height-1) * self.pattern.pointDistance , y = 0}
  local lr_3d = {x = (self.pattern.height-1) * self.pattern.pointDistance , y = (self.pattern.width*2-2) * self.pattern.pointDistance}
  local ll_3d = {x = 0, y =  (self.pattern.width*2-2) * self.pattern.pointDistance}

  --local ul_3d = {x = 0, y = 0}
  --local ur_3d = {x = 0, y = 0.230}
  --local ll_3d = {x = 0.138, y = 0}
  --local lr_3d = {x = 0.138, y = 0.230}


  local p3d = torch.zeros(4,1,3);
  p3d[1][1][1] = ul_3d.y
  p3d[1][1][2] = ul_3d.x

  p3d[2][1][1] = ur_3d.y
  p3d[2][1][2] = ur_3d.x

  p3d[3][1][1] = lr_3d.y
  p3d[3][1][2] = lr_3d.x

  p3d[4][1][1] = ll_3d.y
  p3d[4][1][2] = ll_3d.x

    --Three corner points of our image
    local ul = {x = 0 + borderWidth + radius, y = 0 + borderWidth + radius}
    local ur = {x = w - borderWidth - radius, y = 0 + borderWidth + radius}
    local lr = {x = w - borderWidth - radius, y = h - borderWidth - radius}
    local ll = {x = 0 + borderWidth, y = h - borderWidth - radius}


    --Add some noise
    ul.x = ul.x + (math.random() - 0.5) * radius
    ul.y = ul.y + (math.random() - 0.5) * radius

    ur.x = ur.x + (math.random() - 0.5) * radius
    ur.y = ur.y + (math.random() - 0.5) * radius

    lr.x = lr.x + (math.random() - 0.5) * radius
    lr.y = lr.y + (math.random() - 0.5) * radius

    local img = torch.ByteTensor(h, w, 3):zero()

    cv.circle{img = img, center = ul, radius = 3, color = {80,80,255,1}, thickness = 5, lineType = cv.LINE_AA}
    cv.circle{img = img, center = ur, radius = 3, color = {80,80,255,1}, thickness = 5, lineType = cv.LINE_AA}
    cv.circle{img = img, center = lr, radius = 3, color = {80,80,255,1}, thickness = 5, lineType = cv.LINE_AA}
    cv.circle{img = img, center = ll, radius = 3, color = {80,80,255,1}, thickness = 5, lineType = cv.LINE_AA}


    local p2d = torch.zeros(4,1,2);
    p2d[1][1][1] = ul.x
    p2d[1][1][2] = ul.y

    p2d[2][1][1] = ur.x
    p2d[2][1][2] = ur.y

    p2d[3][1][1] = lr.x
    p2d[3][1][2] = lr.y

    p2d[4][1][1] = ll.x
    p2d[4][1][2] = ll.y

    local pose_found, pose_cam_rot_vector, pose_cam_trans=cv.solvePnP{objectPoints=p3d, imagePoints=p2d, cameraMatrix=self.intrinsics, distCoeffs=torch.zeros(5,1), flags = cv.CALIB_EPNP}



    local H = torch.eye(4,4)
    H[{{1,3},{1,3}}] = xamla3d.calibration.RotVectorToRotMatrix(pose_cam_rot_vector)
    H[{{1,3},4}] = pose_cam_trans


    local pp_3d = xamla3d.calibration.calcPatternPointPositions(self.pattern.width, self.pattern.height, self.pattern.pointDistance)
    for i = 1,pp_3d:size()[1] do
      local projection = xamla3d.projectPoint(self.intrinsics, H, pp_3d[{i,1,{}}])
       cv.circle{img = img, center = {x = projection[1], y = projection[2]}, radius = 10, color = {255,255,255,1}, thickness = 1, lineType = cv.LINE_AA}
    end

    print("----H----")
    print(H)
    print(pattern_in_robot * torch.inverse(H) * torch.inverse(self.heye))
    cv.imshow{"Pattern projected", img}
    cv.waitKey{-1}

     return pattern_in_robot * H
end

function Capture:__init(output_path, pictures_per_position, velocity_scaling)
  self.output_path = output_path
  self.pictures_per_position = pictures_per_position or 5

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
  self.pattern = { width = 8, height = 21, pointDistance = 0.008 }
  self.imwidth = 960
  self.imheight = 720
end

function Capture:setDefaultCameraValues(heye, pattern)
  --self.heye = heye
  --self.pattern = pattern
  initializeRobot(self, velocity_scaling or 0.5)
end


function Capture:grabImageGray()
  return self.webcam:GrabGrayscaleImgROS()
end


function Capture:grabImage()
  local imgIR = self.depthcam:GrabIRNoSpeckleViaROS()
  return self.webcam:GrabGrayscaleImgROS(), imgIR
end


function Capture:isPatternInImg(img)
   local ok,pattern_points = cv.findCirclesGrid{image=img, patternSize={height=self.pattern.height, width=self.pattern.width}, flags=cv.CALIB_CB_ASYMMETRIC_GRID}
   return ok
end

function Capture:findPattern()
  local img = self:grabImageGray()
  if img == nil then
    return false
  end

  local robot_pose = self.roboControl:ReadRobotPose(true).full

  local ok,pattern_points = cv.findCirclesGrid{image=img, patternSize={height=self.pattern.height, width=self.pattern.width}, flags=cv.CALIB_CB_ASYMMETRIC_GRID}
  if ok then
    local circlePositions = xamla3d.calibration.calcPatternPointPositions(self.pattern.width, self.pattern.height, self.pattern.pointDistance)
    local pose_found, pose_cam_rot_vector, pose_cam_trans=cv.solvePnP{objectPoints=circlePositions, imagePoints=pattern_points, cameraMatrix=self.intrinsics, distCoeffs=self.distortion}
    if not pose_found then
      error('could not calculate pose from calibration pattern')
    end

    local pose_cam_rot_matrix = rotVectorToRotMatrix(pose_cam_rot_vector)

    -- assemble the 4x4 transformation matrix
    local pattern_pose = torch.eye(4)
    pattern_pose[{{1,3}, {1,3}}] = pose_cam_rot_matrix
    pattern_pose[{{1,3}, {4}}] = pose_cam_trans

    local pattern_pose_original = pattern_pose:clone()

    local offset = torch.Tensor({self.pattern.pointDistance * self.pattern.width, 0.5 * self.pattern.pointDistance * self.pattern.height, 0, 1})

    local pattern_center_world = robot_pose * self.heye * pattern_pose * offset



    local pattern_points_in_base = {}

    for i =1,circlePositions:size()[1] do
      local X = torch.DoubleTensor(1,4)
      X[{1,{1,3}}] =  circlePositions:type('torch.DoubleTensor')[{i, 1, {1,3}}]
      X[1][4] = 1
      local base = robot_pose * self.heye * pattern_pose_original * X:t()
      table.insert(pattern_points_in_base, base[{{1,3},1}])
    end
    return true, pattern_pose, robot_pose, pattern_points_in_base, pattern_center_world
  end
  return false
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
      cv.imshow{"Image", img}
      cv.waitKey{10}
      local ok,pattern_points = cv.findCirclesGrid{image=img, patternSize={height=self.pattern.height, width=self.pattern.width}, flags=cv.CALIB_CB_ASYMMETRIC_GRID}
      if ok then
        return true, pattern_pose, robot_pose, pattern_points_in_base
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
    local ok, pattern_pose, robot_pose, pattern_points_base = self:searchPatternCircular(patter_search_center, radius, height)
    if ok then
      return pattern_pose, robot_pose, pattern_points_base
    end

    if radius < 0.5 then
      radius = radius + 0.1
    else
      radius = 0.1  --restart search with 10cm distance
    end
  end

  error('Search for calibration pattern aborted.')
end

local function checkPatternInImage(self, robot_pose, pattern_points_base)

  local cam_pos = torch.inverse(robot_pose * self.heye)
  local P = self.intrinsics * cam_pos[{{1,3}, {1,4}}]

  for i = 1,#pattern_points_base do
    local X = torch.DoubleTensor(4,1)
    X[{{1,3},1}] = pattern_points_base[i]:view(3,1)
    X[4][1] = 1
    local x = (P * X):squeeze()
    x = x  / x[3]
    if x[1] < 50 or x[1] > self.imwidth-50 or x[2] < 50 or x[2] > self.imheight-50 then
      return false
    end
  end

  return true

end


local function captureSphereSampling(self, path, filePrefix, robot_pose, transfer, count, capForHandEye, pattern_points_base, pattern_center_world, min_radius, max_radius, focus, target_jitter)

  min_radius = min_radius or 0.17   -- min and max distance from target
  max_radius = max_radius or 0.19
  focus = focus or 20
  target_jitter = target_jitter or 0.015
  capForHandEye = capForHandEye or false

  local targetPoint = pattern_center_world:view(4,1)[{{1,3},1}]
  -- pattern in world coordinates
  local t = robot_pose * self.heye * transfer

  print('identified target point:')
  print(targetPoint)

  local poseData = {}
  poseData["MoveitPose"] = {}
  poseData["UR5Pose"] = {}
  poseData["JointPos"] = {}
  poseData["FileName"] = {}

  self.webcam:SetFocusValue(focus)
  local i = 1
  while i < count do

    -- generate random point in positive half shere
    local sphereTh = 0.96
    if capForHandEye then
      sphereTh = 0.93
    end

    local origin
    while true do
      origin = torch.randn(4)
      origin[3] = math.max(0.01, math.abs(origin[3]))
      origin:div(origin:norm())
      if origin[3] > sphereTh then
        break
      end
    end
    --Lets express the position we want to look to relative to our pattern
    -- The targets z-axis goes into the table so we have a negative z-value w.r.t. pattern
    -- scale this vector to the desired length
    origin:mul(torch.lerp(min_radius, max_radius, math.random()))
    origin[3] = origin[3] * -1 --z is going into the table
    offset = torch.Tensor({self.pattern.pointDistance * self.pattern.width, 0.5 * self.pattern.pointDistance * self.pattern.height, 0, 1})
    origin:add(offset)
    origin[4] = 1 --make homogenoous vector

    -- bring the vector that is given relative to target to robot coordinates
    origin = robot_pose * self.heye * transfer * origin

    origin = origin:view(4,1)[{{1,3},1}]
    print("----------Origin..........")
    print(origin)

    local target = targetPoint + math.random() * target_jitter - 0.5 * target_jitter

    local up_ = up

    up_ = t[{1,{1,3}}] -- use pattern x axis in world

    if math.random(2) == 1 then
      up_ = -up_
    end

    local movePose = self.roboControl:PointAtPose(origin, target, up_, self.heye)


    if capForHandEye then
      print("Adapt parameters for hand - eye")
      local polarAngle = math.random()*180 - 90
      local azimuthalAngle = math.random()*60 - 30
      local radius = min_radius +0.20 +(max_radius - min_radius)*math.random()
      movePose = self.roboControl:WebCamLookAt(target, radius, math.rad(polarAngle), math.rad(azimuthalAngle), self.heye, math.random(1)-1)
      self.webcam:SetFocusValue(5)
    end

    --print("MovePose")
    --print(movePose)

    if checkPatternInImage(self, movePose, pattern_points_base) and  self.roboControl:MoveRobotTo(movePose) then
      sys.sleep(0.2)    -- wait for controller position convergence
      local imgWeb, imgIR = self:grabImage()
      local ok,pattern_points = cv.findCirclesGrid{image=imgWeb, patternSize={height=self.pattern.height, width=self.pattern.width}, flags=cv.CALIB_CB_ASYMMETRIC_GRID}
      if (ok) then
        poseData["MoveitPose"][i] = self.roboControl:ReadRobotPose(true)
        local ur5state = self.roboControl:ReadUR5data()
        poseData["UR5Pose"][i] = self.roboControl:DecodeUR5TcpPose(ur5state, true)
        poseData["JointPos"][i] = self.roboControl:DecodeUR5actualJointState(ur5state)
        poseData["FileName"][i] = saveImages(path, filePrefix, i, imgIR, imgWeb)

        savePoses(path, filePrefix, i, poseData)
        i=i+1
        print("Pattern found! Remaining images: ".. count - i)
      end
    end
  end

  return savePoses(path, filePrefix, nil, poseData)
end

function Capture:showLiveView()

  local cnt_nil = 0

  while true do
    local img = self.webcam:GrabJPEGstreamROS()
    if img ~= nil then
      cnt_nil = 0
      cv.imshow{"Live View", img}
      local key = cv.waitKey{30}
      if key == 1048689 then --q
        cv.destroyAllWindows{}
        return
      end
    else
      cnt_nil = cnt_nil + 1
    end
  end

end


function Capture:getBestFocusPoint()
  local image_stack = {}
  local brenner = {}

  local p = path.join(self.output_path, "focal_stack")
  mkdir_recursive(p)

  local best = {value = 0, focal = 0}

  for i = 0, 250/5 do
    local f = i*5
    self.webcam:SetFocusValue(f)
    local image_gray = nil
    for j = 1,2 do --ignore the first image because it is not affected by the focal settings
       local tmp = self.webcam:GrabGrayscaleImgROS()
       image_gray = tmp:type('torch.DoubleTensor')
    end

    -- calculate brenner
    local rows = image_gray:size()[1]
    local cols = image_gray:size()[2]

    local P = image_gray[{{1,rows-2}, {}}] - image_gray[{{3, rows},{}}]
    local b = torch.sum(torch.cmul(P,P))

    p = path.join(p, string.format('focal_%03d.png', i))
    cv.imwrite{p, image_gray:type('torch.ByteTensor')}

    if b > best.value then
      best.value = b
      best.focal = f
    else
      if torch.abs(best.focal - f) > 3 then -- Our focal function is monotonic and has a single peak,
        return best.focal                   -- So if we did not get a better value for a certain time
      end                                   -- we found already the maximum
    end
  end

  return image_stack

end

local function CreatePose(pos, rot)
  local pose = tf.Transform()
  pose:setOrigin(pos)
  pose:setRotation(rot)
  return pose:toTensor()
end

---
-- This function calculates the robot pose (TCP) that is required to rotate the camera around
-- its image axis x, y, z (where x represents the axis that is associated with the width of the
-- image and y the height of the image. Z is the vector that is associated with the viewing ray
-- passing the cameras center. The rotation order is x,y,z
-- @param robot_pose 4x4 torch.Tensor of current camera pose
-- @param rot_x_degree rotation around the images / cameras x axis in degree
-- @param rot_y_degree rotation around the images / cameras y axis in degree
-- @param rot_z_degree rotation around the images / cameras z axis in degree
-- @return the robots pose that is required to rotate the camera
function Capture:addRotationAroundCameraAxes(robot_pose, rot_x_degree, rot_y_degree, rot_z_degree)
  local pose_cam = robot_pose * self.heye
  local tfPose = tf.Transform()
  tfPose:fromTensor(pose_cam)
  local b=tfPose:getRotation()

  b = b:mul(tf.Quaternion({1,0,0}, math.rad(rot_x_degree) ))
  b = b:mul(tf.Quaternion({0,1,0}, math.rad(rot_y_degree) ))
  b = b:mul(tf.Quaternion({0,0,1}, math.rad(rot_z_degree) ))
  local c = tfPose:getOrigin()
  local next_pose = CreatePose(c, b)
  next_pose = next_pose * torch.inverse(self.heye)
  return next_pose
end


---
-- This function returns the cameras x, y, and z axis in robot base coordinates
-- @param robot_pose 4x4 torch.Tensor of robot pose the camera is attached to
-- @return x,y,z 3x1 torch.Tensor of the x, y, z axis in robot base coordinates
function Capture:getCameraAxesInRobotBase(robot_pose)
  local cam_pose = robot_pose * self.heye
  local x = cam_pose[{{1,3},{1,3}}] * torch.Tensor({1,0,0})
  local y = cam_pose[{{1,3},{1,3}}] * torch.Tensor({0,1,0})
  local z = x:cross(y)
  z = z / torch.norm(z) * -1
  return x,y,z
end


---
-- This function acquires a bunch of images to estimate the intrinsics of the camera.
-- Distortions are not estimated and therefore are set to zero.
-- The only assumption is that the hand eye matrix is given.
-- The idea is to move around the current camera position and acquire images. At the end
-- the intrinsic camera matrix is calculated. The movement pattern is the following:
-- First take a picture of the current image and then add random offsets to the current
-- position and rotate the camera around its axis. We start with small movements and then
-- increase the movement. This guarantees enough variation to estimate the intrinsic
-- camera parameters.
-- @param focus_setting the focus value the camera should be set to.
--
function Capture:acquireForApproxFocalLength(focus_setting)

  local images_pattern = {}
  local patterns = {}
  local objectPoints = {}

  self.webcam:SetFocusValue(focus_setting)
  local current_robot_pose = self.roboControl:ReadRobotPose(true).full:clone()

  local patternPoints3d = xamla3d.calibration.calcPatternPointPositions(self.pattern.width, self.pattern.height, self.pattern.pointDistance)

  local x_cam,y_cam,z_cam = self:getCameraAxesInRobotBase(current_robot_pose)

  while(#images_pattern < 8) do

    local robot_pose = current_robot_pose:clone()

	local scale_tensor = torch.Tensor({1,1,2})
    local scale_rot = 10
    local z_offset = 0.05
	if #images_pattern < 10 then
      --scale_tensor = torch.zeros(3)
      scale_rot = #images_pattern * 2
      z_offset = #images_pattern *0.01
	end

	local x_offset = x_cam * 0.04 * (math.random() - 0.5) * scale_tensor[1]
	local y_offset = y_cam * 0.04 * (math.random() - 0.5) * scale_tensor[2]
	local z_offset = (z_cam * z_offset) + (z_cam * 0.04 * (math.random() - 0.5) * scale_tensor[3])

	local jittered_cam_pose = (robot_pose * self.heye)
	jittered_cam_pose[{{1,3},{4}}] = jittered_cam_pose[{{1,3},{4}}] + x_offset + y_offset + z_offset
	robot_pose = jittered_cam_pose * torch.inverse(self.heye)

    local deg_x = (math.random()-0.5) * scale_rot
    local deg_y = (math.random()-0.5) * scale_rot
    local deg_z = (math.random()-0.5) * scale_rot



    robot_pose = self:addRotationAroundCameraAxes(robot_pose, deg_x, deg_y, deg_z)
    print("Going to pose:")
    print(robot_pose)

    if self.roboControl:MoveRobotTo(robot_pose) then
      sys.sleep(0.1)  -- wait for controller position convergence
      local img = self:grabImage()
      local ok,pattern_points = cv.findCirclesGrid{image=img, patternSize={height=self.pattern.height, width=self.pattern.width}, flags=cv.CALIB_CB_ASYMMETRIC_GRID}
      cv.imshow{"Grab!", img}
      cv.waitKey{-1}

      if ok then
        print("Image pattern found!")
        table.insert(images_pattern, img)
        table.insert(patterns, pattern_points)
        table.insert(objectPoints, patternPoints3d)
      else
        print("No Image found!")
      end
    end
  end

  local err_, camera_matrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera{objectPoints=objectPoints, imagePoints=patterns, imageSize={self.imwidth, self.imheight}, flag=cv.CALIB_ZERO_TANGENT_DIST+cv.CALIB_FIX_K3 + cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2}
  -- Move back to the initial position
  self.roboControl:MoveRobotTo(current_robot_pose)
 return camera_matrix, distCoeffs, err_
end

function Capture:captureForIntrinsics(pattern_pose, robot_pose, pattern_points_base, pattern_center_world)
  local capture_output_path = path.join(self.output_path, "intrinsic")
  mkdir_recursive(capture_output_path)    -- ensure output directory exists

  local file_prefix = string.format('intrinsic_')
  local pose_data_filename = captureSphereSampling(self, capture_output_path, file_prefix, robot_pose, pattern_pose, self.pictures_per_position, false, pattern_points_base, pattern_center_world)

  --table.insert(capture_data_files, pose_data_filename)
end


function Capture:captureForHandEye(pattern_pose, robot_pose, pattern_points_base, pattern_center_world, fname)
  fname = fname or "handeye"
  local capture_output_path = path.join(self.output_path, fname)
  mkdir_recursive(capture_output_path)    -- ensure output directory exists

  local file_prefix = string.format('handeye_')
  local pose_data_filename = captureSphereSampling(self, capture_output_path, file_prefix, robot_pose, pattern_pose, self.pictures_per_position, true, pattern_points_base, pattern_center_world)

  --table.insert(capture_data_files, pose_data_filename)
end



function Capture:run()
  print('Storing output in: '.. self.output_path)
  local capture_data_files = {}
  local i = 1

  local img = self:grabImage()
  self.imwidth = img:size()[2]
  self.imheight = img:size()[1]
  print(self.intrinsics)


  while true do
    print(string.format('Please place pattern at position %d.', i))
    print('Ready? Please press enter.')
    io.stdin:read()



    local capture_output_path = path.join(self.output_path, string.format('pose%03d', i))
    mkdir_recursive(capture_output_path)    -- ensure output directory exists

    local pattern_pose, robot_pose, pattern_points_base = self:searchPattern()


    local file_prefix = string.format('pose%03d_', i)
    local pose_data_filename = captureSphereSampling(self, capture_output_path, file_prefix, robot_pose, pattern_pose, self.pictures_per_position, true, pattern_points_base)
    table.insert(capture_data_files, pose_data_filename)
    i = i + 1
  end
  return capture_data_files
end
