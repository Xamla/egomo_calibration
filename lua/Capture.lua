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
  local depthcam = egomoTools.structureio:new(camIntrinsicsIR, "egomo_depthcam")
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


-- This functions calcuates the distance that is required to image a specific length in world units
-- in pixel. Example: Which distance the camera has to have if it wants to project a 2cm object to
-- 50 pixel.
local function calcDistanceToTarget(focal_length, width_in_px, width_in_world)
  return (focal_length / (0.5*width_in_px)) * (width_in_world*0.5)
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
  
   local pp_3d = xamla3d.calibration.calcPatternPointPositions(self.pattern.width, self.pattern.height, self.pattern.pointDistance)
    for i = 1,pp_3d:size()[1] do      
       cv.circle{img = test_img, center = {x = pp_3d[i][1][1]*2000, y = pp_3d[i][1][2]*2000}, radius = 2, color = {255,255,255,1}, thickness = 1, lineType = cv.LINE_AA}       
    end
    
    local wp = self.pattern.width
    local wh = self.pattern.height
    
    local npts = pp_3d:size()[1]
    
    cv.circle{img = test_img, center = {x = pp_3d[1][1][1]*2000, y = pp_3d[1][1][2]*2000}, radius = 2, color = {64,64,64,1}, thickness = 2, lineType = cv.LINE_AA}
    cv.circle{img = test_img, center = {x = pp_3d[wp][1][1]*2000, y = pp_3d[wp][1][2]*2000}, radius = 2, color = {64,64,64,1}, thickness = 2, lineType = cv.LINE_AA}
    cv.circle{img = test_img, center = {x = pp_3d[npts-wp][1][1]*2000, y = pp_3d[npts-wp][1][2]*2000}, radius = 2, color = {64,64,64,1}, thickness = 2, lineType = cv.LINE_AA}
    cv.circle{img = test_img, center = {x = pp_3d[npts-wp+1][1][1]*2000, y = pp_3d[npts-wp+1][1][2]*2000}, radius = 2, color = {64,64,64,1}, thickness = 2, lineType = cv.LINE_AA}
     
    cv.imshow{"Pattern", test_img}
    cv.waitKey{-1}
  

  print(p3d)



  for i = 1,100 do
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
  end

     return pattern_in_robot * H
end



---
-- removes all grab functions from the list
function Capture:removeAllGrabFunctions()
  self.grab_functions = {}
end

---
-- removes a grab function with the given name from the table of grab functions
-- @param string identfier of the function to be removed
function Capture:removeGrabFunction(name)
  if self.grab_functions[name] ~= nil then
     self.grab_functions[name] = nil
     return true
  end  
  return false
end


--- 
-- Add a grab function. Each grab function is identified by a name. Typically this is
-- the name of a camera like RGB_side or depth_no_speakle
-- This name is used later to identify which camera should be grabbed. e.g. the function showLiveView(cam_name)
-- indirectly calls the grab function with 'cam_name'
-- @param a string that uniquely identifies the grabbing function, i.e. the camera name or a specific stream
-- @param fct_handle the function handle
-- @param instance the instance that is given as 'self' to the function, can be null if it is not a class function 
function Capture:addGrabFunctions(identifier, fct_handle, instance)

  assert(type(fct_handle) == "function")
  assert(type(identifier) == "string")

  local fct_call = {}
  fct_call.name = identifier
  fct_call.instance = instance
  fct_call.fct_name = fct_handle
  table.insert(self.grab_functions, fct_call) 
end


---
-- Grabs images / a single image from the registers capturing functions.
-- you can grab either all cameras at the "same" time or you can select if you 
-- only want to capture the first camera, or a specific one by providing a name
-- or all by providing no argument
-- @param which_camera [optional] Selects which camera should be used for capturing.
-- If boolean = true then the first camera is captured, if string we capture the 
-- camera with the given name 
-- 
function Capture:doGrabbing(which_camera)
  local only_first = false
  local cam_name = nil
  
  if which_camera ~= nil then
    if type(which_camera) == "boolean" then
      only_first = which_camera
    elseif type(which_camera) == "string" then
      cam_name = which_camera
      -- Lets look if the camera is registered for grabbing
      local found = false
      for i = 1,#self.grab_functions do
        local fct_call = self.grab_functions[i]
        if fct_call.name == cam_name then
          found = true
          break 
        end
      end
      
      if found == false then
        error([[Trying to grab an image from a camera that does not exist! Please register
        a grab function with this name using addGrabFunctions() ]])
      end
      
    end
  end
  
  local pose_data = {}
  local images = {}  
  local first_image = nil
  
  --pose_data["MoveitPose"] = self.roboControl:ReadRobotPose(true)
  --local ur5state = self.roboControl:ReadUR5data()
  --pose_data["UR5Pose"] = self.roboControl:DecodeUR5TcpPose(ur5state, true)
  --pose_data["JointPos"]= self.roboControl:DecodeUR5actualJointState(ur5state) 
  
  function call(fct_call)
    if fct_call.instance == nil then
      return fct_call.fct_name()
    else
      return fct_call.fct_name(fct_call.instance)
    end      
  end
     
  
  for i = 1,#self.grab_functions do
    local fct_call = self.grab_functions[i]
    local img = nil
    
    if cam_name ~= nil and fct_call.name == cam_name then
      img = call(fct_call)
      return img, pose_data
    elseif only_first then
       img = call(fct_call)
      return img, pose_data
    elseif only_first == false and cam_name == nil then
      img = call(fct_call)
      images[fct_call.name] = img
    end
  end
  
  return images, pose_data
  
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
  self.grab_functions = {}  
  self.image_saver = calib.ImageSaver(output_path)
  
  
end

function Capture:setImageSaver(image_saver)
  self.image_saver = image_saver
end

function Capture:setDefaultCameraValues(heye, pattern)
  self.heye = heye
  self.pattern = pattern
  initializeRobot(self, velocity_scaling or 0.5)
end


function Capture:isPatternInImg(img)
   local ok,pattern_points = cv.findCirclesGrid{image=img, patternSize={height=self.pattern.height, width=self.pattern.width}, flags=cv.CALIB_CB_ASYMMETRIC_GRID}
   return ok
end

function Capture:findPattern(camera_name)
  assert(type(camera_name) == "string")

  local img = self:doGrabbing(camera_name)
  if img == nil then
    print("Grabbing failed")
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

    local pose_cam_rot_matrix =  xamla3d.calibration.RotVectorToRotMatrix(pose_cam_rot_vector)

    -- assemble the 4x4 transformation matrix
    local pattern_pose = torch.eye(4)
    pattern_pose[{{1,3}, {1,3}}] = pose_cam_rot_matrix
    pattern_pose[{{1,3}, {4}}] = pose_cam_trans

    local pattern_pose_original = pattern_pose:clone()

    local offset = torch.Tensor({self.pattern.pointDistance * self.pattern.width, 0.5 * self.pattern.pointDistance * self.pattern.height, 0, 1})

    local pattern_center_world = robot_pose * self.heye * pattern_pose * offset
    print("Target point in search!")
    print(pattern_center_world)



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


function Capture:searchPatternCircular(center, radius, height, camera_name)
  assert(type(camera_name) == "string")



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
      local img = self:doGrabbing(camera_name)
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


---
-- This function is thought for simulating if a pattern will be completely visible in the 
-- camera if the robot moves to a specific pose.
-- @param self the capturing instance
-- @param robot_pose the roboter pose that is used for evaluation if image is completely visible in image
-- @param pattern_points_base table of 3x1 torch.Tensor describing the patterns point 3d coordinates in the image
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


local function captureSphereSampling(self, robot_pose, transfer, count, capForHandEye, pattern_points_base, pattern_center_world, min_radius, max_radius, focus, target_jitter)
    
  local optimal_distance = calcDistanceToTarget(self.intrinsics[1][1], self.imwidth*0.8, (self.pattern.height - 1) * self.pattern.pointDistance)
  if optimal_distance < 0.1 then
    print(string.format("Optimal distance to target would be %fm. But we set it to 0.1 for safety reasons", optimal_distance) )
    optimal_distance = 0.1
  end

  min_radius = min_radius or optimal_distance - 0.01   -- min and max distance from target
  max_radius = max_radius or optimal_distance + 0.01
  focus = focus or 20
  target_jitter = target_jitter or 0.015
  capForHandEye = capForHandEye or false

  local targetPoint = pattern_center_world:view(4,1)[{{1,3},1}]
  -- pattern in world coordinates
  local t = robot_pose * self.heye * transfer

  print('identified target point:')
  print(targetPoint)
  
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
    local offset = torch.Tensor({self.pattern.pointDistance * self.pattern.width, 0.5 * self.pattern.pointDistance * self.pattern.height, 0, 1})
    origin:add(offset)
    origin[4] = 1 --make homogenoous vector    
    origin = robot_pose * self.heye * transfer * origin -- bring the vector that is given relative to target to robot coordinates
    origin = origin:view(4,1)[{{1,3},1}]

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
      local optimal_distance = calcDistanceToTarget(self.intrinsics[1][1], self.imwidth*0.5, (self.pattern.height - 1) * self.pattern.pointDistance)
      local radius = optimal_distance + (0.04)*math.random()
      movePose = self.roboControl:WebCamLookAt(target, radius, math.rad(polarAngle), math.rad(azimuthalAngle), self.heye, math.random(1)-1)
      self.webcam:SetFocusValue(5)
    end

    if checkPatternInImage(self, movePose, pattern_points_base) and  self.roboControl:MoveRobotTo(movePose) then
      sys.sleep(0.2)    -- wait for controller position convergence
      
      local images, poses = self:doGrabbing()
      local ok,pattern_points = cv.findCirclesGrid{image=images["WEBCAM"], patternSize={height=self.pattern.height, width=self.pattern.width}, flags=cv.CALIB_CB_ASYMMETRIC_GRID}
      if (ok) then
        self.image_saver:addCorrespondingImages(images, poses)           
        i=i+1
        print("Pattern found! Remaining images: ".. count - i)
      end
    end
  end
end

function Capture:showLiveView(cam_name)

  assert(type(cam_name) == "string")

  local cnt_nil = 0

  while true do    
    local img = self:doGrabbing(cam_name)
    if img ~= nil then
      cnt_nil = 0
      local img_gray = img
      
      if img:dim() == 3 then --only if it is a color image we have to convert it to grayscale
        img_gray = cv.cvtColor{src = img:type("torch.ByteTensor"), code = cv.COLOR_BGR2GRAY}
      end
      if img_gray:size()[2] < 640 then
        img_gray = cv.resize{src= img_gray, fx = 2, fy = 2}
      end
      local pattern_found = self:isPatternInImg(img_gray)
      if pattern_found then
         cv.circle{img = img, center = {x = 20, y = 20}, radius = 20, color = {0,255,0,1}, thickness = 5, lineType = cv.LINE_AA}
      else
         cv.circle{img = img, center = {x = 20, y = 20}, radius = 20, color = {0,0,255,1}, thickness = 5, lineType = cv.LINE_AA}
      end
      cv.imshow{"Live View", img}
      local key = cv.waitKey{30}
      if key == 1048689 then --q
        cv.destroyAllWindows{}
        cv.waitKey{20}
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

--- 
-- Helper function that creates a complete pose given the pose and rotation
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
function Capture:acquireForApproxFocalLength(focus_setting, cam_name)

  local found = false
  for i = 1,#self.grab_functions do
    if self.grab_functions[i].name == cam_name then
      found = true
    end
  end
  
  if found == false then
    error("No grabbing function with name ".. cam_name .. " registered!")
  end


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
  	local z_offset = (z_cam * z_offset) + (z_cam * 0.04 * (math.random()) * scale_tensor[3])
  
  	local jittered_cam_pose = (robot_pose * self.heye)
  	jittered_cam_pose[{{1,3},{4}}] = jittered_cam_pose[{{1,3},{4}}] + x_offset + y_offset + z_offset
  	robot_pose = jittered_cam_pose * torch.inverse(self.heye)
  
    local deg_x = (math.random()-0.5) * scale_rot
    local deg_y = (math.random()-0.5) * scale_rot
    local deg_z = (math.random()-0.5) * scale_rot
  
    robot_pose = self:addRotationAroundCameraAxes(robot_pose, deg_x, deg_y, deg_z)
    
    if self.roboControl:MoveRobotTo(robot_pose) then
      sys.sleep(0.1)  -- wait for controller position convergence
      local images, poses = self:doGrabbing()
      local img = images[cam_name]
      local ok,pattern_points = cv.findCirclesGrid{image=img, patternSize={height=self.pattern.height, width=self.pattern.width}, flags=cv.CALIB_CB_ASYMMETRIC_GRID}
  
      if ok then
        print("Image pattern found!")
        table.insert(images_pattern, img)
        table.insert(patterns, pattern_points)
        table.insert(objectPoints, patternPoints3d)
        
        if self.image_saver ~= nil then
          self.image_saver:addCorrespondingImages(images, poses)
        end
        
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


---
-- Capture a set of images that are thought for calculating the intrinsic parameters
-- @param pattern_pose 4x4 torch.Tensor pose of pattern with respect to camera. This is what we get from solvePnp for example
-- @param robot_pose the current robots pose
-- @param pattern_points_base table of 3x1 torch.Tensor describing the pattern points in robots base coordinates
-- @param pattern_center_world center of pattern in base coordinates
function Capture:captureForIntrinsics(pattern_pose, robot_pose, pattern_points_base, pattern_center_world)
  captureSphereSampling(self, robot_pose, pattern_pose, self.pictures_per_position, false, pattern_points_base, pattern_center_world)

end


---
-- Capture a set of images that are thought for calculating the handeye matrix
-- @param pattern_pose 4x4 torch.Tensor pose of pattern with respect to camera. This is what we get from solvePnp for example
-- @param robot_pose the current robots pose
-- @param pattern_points_base table of 3x1 torch.Tensor describing the pattern points in robots base coordinates
-- @param pattern_center_world center of pattern in base coordinates
function Capture:captureForHandEye(pattern_pose, robot_pose, pattern_points_base, pattern_center_world, fname)
  local file_prefix = string.format('handeye_')
   captureSphereSampling(self, robot_pose, pattern_pose, self.pictures_per_position, true, pattern_points_base, pattern_center_world)
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

    local pattern_pose, robot_pose, pattern_points_base = self:searchPattern()

    local pose_data_filename = captureSphereSampling(self, robot_pose, pattern_pose, self.pictures_per_position, true, pattern_points_base)
    table.insert(capture_data_files, pose_data_filename)
    i = i + 1
  end
  return capture_data_files
end
