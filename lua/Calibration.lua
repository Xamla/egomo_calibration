local calib = require 'egomo_calibration.env'

local cv  = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
require 'cv.features2d'

local path = require 'pl.path'
local pcl = require 'pcl'
local xamla3d = require 'egomo_calibration.xamla3d'
local M_PI = 3.14159265359


local Calibration = torch.class('egomo_calibration.Calibration', calib)


local function countKeys(x)
  local i = 0
  for k,v in pairs(x) do
    i = i + 1
  end
  return i
end


-- generate matrix form denavit-hartenberg parameters
local function dh(theta, d, a, alpha)
  local sin, cos = math.sin, math.cos
  return torch.Tensor({
    { cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta) },
    { sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta) },
    {          0,             sin(alpha),             cos(alpha),            d },
    {          0,                      0,                      0,            1 }
  })
end


-- return final transform and individual frames
local function forward_kinematic(jointState, robotModel, join_dir)
  local poses = {}

  local T = torch.eye(4)
  for i=1,6 do
    local t = dh(join_dir[i] * jointState[i] - robotModel[1][i], robotModel[2][i], robotModel[3][i], robotModel[4][i])
    table.insert(poses, t)
    T = T * t
  end
  return T, poses
end


function Calibration.createUR5RobotModel(shoulder_height, upper_arm_length, forearm_length, wrist_1_length, wrist_2_length, wrist_3_length)
  shoulder_height = shoulder_height or 0.089159
  upper_arm_length = upper_arm_length or 0.425
  forearm_length = forearm_length or 0.39225
  wrist_1_length = wrist_1_length or 0.10915
  wrist_2_length = wrist_2_length or 0.09465
  wrist_3_length = wrist_3_length or 0.0823

  local dh = torch.DoubleTensor(4, 6)

  -- theta
  dh[1][1] = 0
  dh[1][2] = 0
  dh[1][3] = 0
  dh[1][4] = 0
  dh[1][5] = 0
  dh[1][6] = M_PI

  -- d
  dh[2][1] = shoulder_height
  dh[2][2] = 0
  dh[2][3] = 0
  dh[2][4] = -wrist_1_length
  dh[2][5] = wrist_2_length
  dh[2][6] = -wrist_3_length

  -- a
  dh[3][1] = 0
  dh[3][2] = upper_arm_length
  dh[3][3] = forearm_length
  dh[3][4] = 0
  dh[3][5] = 0
  dh[3][6] = 0

  -- alpha
  dh[4][1] = M_PI/2
  dh[4][2] = 0
  dh[4][3] = 0
  dh[4][4] = M_PI/2
  dh[4][5] = -M_PI/2
  dh[4][6] = M_PI

  local joint_direction = { 1, -1, -1, -1, 1, -1 }

  return { dh = dh, joint_direction = joint_direction }
end


function Calibration:__init(pattern, im_width, im_height, hand_eye, robot_model)
  self.pattern = pattern
  if self.pattern == nil then
    self.pattern = {
      width = 8,
      height = 21,
      circleSpacing = 8
    }
  end

  self.robotModel = robot_model
  if self.robotModel == nil then
    self.robotModel = Calibration.createUR5RobotModel()
  end

  self.im_width = im_width or 960
  self.im_height = im_height or 720

  -- intrinsic parameters of camera
  self.intrinsics = torch.zeros(3,3)

  -- distortion parameters of camera
  self.distCoeffs = torch.zeros(5,1)

  self.handEye = hand_eye
  if handEye == nil then
    self.handEye = torch.DoubleTensor({
   {0.0019753199627884,0.7605059561815,0.64932795159573,0.015107794350183,},
   {0.0010427115144404,-0.64933043182179,0.76050568904041,0.074581531861269,},
   {0.99999750542876,-0.00082518033759893,-0.0020756236817346,0.0559444561459,},
   {0,0,0,1,},
})
    
  end

  self.images = {}
  self.patternIDs = {}
end


function Calibration:addImage(image, robotPose, jointState, patternId)
  if image == nil or robotPose == nil or jointState == nil then
    error('Invalid arguments')
  end
  
  local pattern = patternId or 1
  self.patternIDs[pattern] = true
 
  local image_item = {
    image = image,
    robotPose = robotPose,
    poseToTarget = torch.eye(4,4),
    jointStates = jointState,
    patternID = pattern,
  }

  local patternGeom = { height=self.pattern.height, width=self.pattern.width }    -- size of the pattern: x, y
  local found, points = xamla3d.calibration.findPattern(image:clone(), cv.CALIB_CB_ASYMMETRIC_GRID, patternGeom)
  if not found then
    return false
  end

  image_item.patternPoints2d = points:clone()
  table.insert(self.images, image_item)
  
--  print(self.images[#self.images])

  return true
end


-- delete all images that have been added before
function Calibration:resetImages()
  self.images = {}
end


function Calibration:runHandEyeCalibration()
  error('not implemented')
  --return true, torch.eye(4,4)
end


function Calibration:prepareBAStructureForPattern(patternID, measurementsE, jointPointIndicesE, jointStatesE, points3dE)
  local indices = {}
  for i = 1,#self.images do
    if self.images[i].patternPoints2d ~= nil and self.images[i].patternID == patternID then
      table.insert(indices, i)
    end
  end
  return self:prepareBAStructureWithImageIDs(indices, measurementsE, jointPointIndicesE, jointStatesE, points3dE)
end


-- this function adds measurements, cameras and points3d for a given patternID to the last 3 variables.
-- if they are nil, we create new torch.tensors
function Calibration:prepareBAStructureWithImageIDs(indices, measurementsE, jointPointIndicesE, jointStatesE, points3dE, calibrationData)

  local measOffset = 0
  local jointStatesOffset = 0
  local point3dOffset = 0 

  local intrinsics = self.intrinsics:clone()
  local distCoeffs = self.distCoeffs:clone()
  local handEye = self.handEye:clone()

  local robotModel, robotJointDir = self.robotModel.dh, self.robotModel.joint_direction
 
  if calibrationData ~= nil then
    intrinsics = calibrationData.intrinsics
    distCoeffs = calibrationData.distCoeffs 
    handEye = calibrationData.handEye
    robotModel = calibrationData.robotModel
    robotJointDir = calibrationData.joinDir  
  else
    print("Using GLOABAL robotParameters!") 
  end

  print('Using handEye:')
  print(handEye)
  print('Using robotModel:')
  print(robotModel)
  print('Using intrinsics:')
  print(intrinsics)

  if measurementsE ~= nil or jointStatesE ~= nil or points3dE ~= nil then
    assert(measurementsE)
    assert(jointStatesE)
    assert(points3dE)
    assert(jointPointIndicesE)
    
    measOffset = measurementsE:size()[1]
    jointStatesOffset = jointStatesE:size()[1]
    point3dOffset = points3dE:size()[1]
    print("Offsets in prepareBA - Measurement: " ..measOffset .. " jointStates: " .. jointStatesOffset .. " points3d: " .. point3dOffset)      
  end
  
  print("Using ".. #indices .. " cameras for preparing BA Structure")

  local patternGeom = {height=self.pattern.height, width=self.pattern.width}
  
  local nPts = self.pattern.height * self.pattern.width
  local poses = {}
  local observations = {}
  local point3d = torch.DoubleTensor(nPts,3):zero()

  for i = 1, #indices do
    local imageEntry = self.images[indices[i]]

    table.insert(poses, imageEntry.robotPose.full)
    local points = imageEntry.patternPoints2d

    for m = 1,points:size()[1] do
      local meas = torch.DoubleTensor(4,1)
      meas[1] = #poses -1 + jointStatesOffset --cameraID (zero - based for c++)
      meas[2] = m -1 + point3dOffset--pointID (make it zero based for c++)
      meas[3] = points[m][1][1]
      meas[4] = points[m][1][2]
      table.insert(observations, meas)
    end
  end
  
  local observationT = torch.DoubleTensor(#observations,2)
  local jointpointIndices = torch.LongTensor(#observations,2)
  local jointStates = torch.DoubleTensor(#indices, 6)
    
  for i = 1,#observations do
    observationT[{i,{1,2}}] = observations[i][{{3,4},1}]
    jointpointIndices[{i,{1,2}}] = observations[i][{{1,2},1}]
  end

  local nCnt = 0
  local P = {}
  for i = 1, #indices do
    nCnt = nCnt + 1
    jointStates[{nCnt, {1,6}}] = self.images[indices[i]].jointStates:view(1,6):clone()
    local robotPoseWithJointStates = forward_kinematic(self.images[indices[i]].jointStates,robotModel,robotJointDir)
    local c = intrinsics * torch.inverse(robotPoseWithJointStates * handEye)[{{1,3},{}}]
    table.insert(P, c)
  end
  
  --- make an initial guess for the 3d points by
  for i = 1, nPts do
    local meas = {}
    for j = 1,#indices do
      table.insert(meas, self.images[indices[j]].patternPoints2d[{i, 1, {1,2}}]:view(1,2):clone())
    end

    if #P ~= #meas then
      error(string.format("Measurements must be same size as camera poses (Poses %d, Measurement: s%d ).", #P, #meas))
    end

    --[[print('P')
    print(P[1])
    print("meas")
    print(meas[1])]]

    local s, X = xamla3d.linearTriangulation(P, meas)
    if s ~= true then
      error("Triangulation failed.")
    end
    point3d[{i,{1,3}}] = X:t():clone()
  end
  
  if measurementsE ~= nil then
    measurementsE = torch.cat(measurementsE, observationT, 1)
    jointPointIndicesE = torch.cat(jointPointIndicesE, jointpointIndices, 1)
    jointStatesE = torch.cat(jointStatesE, jointStates, 1)
    points3dE = torch.cat(points3dE, point3d, 1)
    return measurementsE, jointPointIndicesE, jointStatesE, points3dE
  else
    return observationT, jointpointIndices, jointStates, point3d
  end

end
 

function Calibration:getImageIndicesForPattern(patternID)
  local indices = {}
  for i = 1,#self.images do
    if self.images[i].patternPoints2d ~= nil and self.images[i].patternID == patternID then
      table.insert(indices, i)
    end
  end
  return indices
end


function Calibration:DHCrossValidate(trainTestSplitPercentage, iterations)
   
  local validationErrors = {}

  for i = 1,iterations do
     
    local observations, jointPointIndices, jointStates, points3d = nil

    local idxForValidationPerPattern = {}

    for k,v in ipairs(self.patternIDs) do
    
      local idxPattern = self:getImageIndicesForPattern(k)
      local nTraining = math.floor(#idxPattern * trainTestSplitPercentage)              
      xamla3d.shuffleTable(idxPattern)       
      local idxTraining = {unpack (idxPattern, 1, nTraining)}
      local idxValidation = {unpack (idxPattern, nTraining+1, #idxPattern)}
      
      table.insert(idxForValidationPerPattern, idxValidation)
                  
      observations, jointPointIndices, jointStates, points3d = self:prepareBAStructureWithImageIDs(idxTraining, observations, jointPointIndices, jointStates, points3d)

    end

    --local indices = torch.randperm(nImages):view(nImages,1):clone()    
    --local idxTraining = torch.totable(indices[{{1,nTraining},1}])
    --local idxValidation = torch.totable(indices[{{nTraining+1, indices:size()[1]},1}])

    local intrinsics = self.intrinsics:clone()
    local distCoeffs = self.distCoeffs:clone()
    local handEyeInv = torch.inverse(self.handEye)

    local robotModel = self.robotModel.dh:clone()
    local robotJointDir = self.robotModel.joint_direction

    local optimization_path = ""

    local jointStatesOptimized = jointStates:clone()
    --[[
    local init_error = calib.optimizeDH(intrinsics,
      distCoeffs,
      handEyeInv,
      jointStatesOptimized,
      robotModel,
      points3d,
      observations,
      jointPointIndices,
      true,      -- optimize_hand_eye
      true,      -- optimize_points
      false,     -- optimize_robot_model_theta
      false,     -- optimize_robot_model_d
      false,     -- optimize_robot_model_a
      false,     -- optimize_robot_model_alpha
      true       -- optimize_joint_states
    )
    print("Error after optimizing HandEye:                 "..init_error)     
]]

    local init_error = calib.optimizeDH(intrinsics,
      distCoeffs,
      handEyeInv,
      jointStates,
      robotModel,
      points3d,
      observations,
      jointPointIndices,
      false,     -- optimize_hand_eye
      true,      -- optimize_points
      false,     -- optimize_robot_model_theta
      false,     -- optimize_robot_model_d
      false,     -- optimize_robot_model_a
      false,     -- optimize_robot_model_alpha
      false      -- optimize_joint_states
    )
                   
    print("Error Initial:                                     " .. init_error)
    
    local training_error = calib.optimizeDH(intrinsics,
      distCoeffs,
      handEyeInv,
      jointStates,
      robotModel,
      points3d,
      observations,
      jointPointIndices,
      false,     -- optimize_hand_eye
      true,     -- optimize_points
      true,     -- optimize_robot_model_theta
      false,     -- optimize_robot_model_d
      false,     -- optimize_robot_model_a
      true,     -- optimize_robot_model_alpha
      false      -- optimize_joint_states
    )

    optimization_path  = optimization_path .. "(points + theta + alpha)"
      
    local training_error = calib.optimizeDH(intrinsics,
      distCoeffs,
      handEyeInv,
      jointStates,
      robotModel,
      points3d,
      observations,
      jointPointIndices,
      false,     -- optimize_hand_eye
      true,     -- optimize_points
      false,     -- optimize_robot_model_theta
      true,     -- optimize_robot_model_d
      true,     -- optimize_robot_model_a
      false,     -- optimize_robot_model_alpha
      false      -- optimize_joint_states
    )

    optimization_path  = optimization_path .. "(points + d + a)"

 --[[
      local training_error = calib.optimizeDH(intrinsics,
                 distCoeffs,
                 handEyeInv,   
                 jointStates,
                 robotModel,
                 points3d,
                 observations,
                 jointPointIndices,
                 true,     -- optimize_hand_eye
                 true,     -- optimize_points
                 false,     -- optimize_robot_model_theta
                 false,     -- optimize_robot_model_d
                 false,     -- optimize_robot_model_a
                 false,     -- optimize_robot_model_alpha
                 false      -- optimize_joint_states
                 )  
    optimization_path  = optimization_path .. "(handEye + points)"
]]
                   
    print("Error after optim of theta and alpha (Training):    " .. training_error)
  
    local calibData = {}
    calibData.intrinsics = intrinsics
    calibData.distCoeffs = distCoeffs
    calibData.handEye = torch.inverse(handEyeInv)
    calibData.robotModel = robotModel
    calibData.joinDir = { 1, -1, -1, -1, 1, -1 }     
  
    local observationsVal, jointPointIndicesVal, jointStatesVal, points3dVal = nil
    for k = 1,#idxForValidationPerPattern do
      local idxValidation = idxForValidationPerPattern[k]
       observationsVal, jointPointIndicesVal, jointStatesVal, points3dVal = self:prepareBAStructureWithImageIDs(idxValidation, observationsVal, jointPointIndicesVal, jointStatesVal, points3dVal, calibData)
    end
  
    local validation_error = calib.optimizeDH(intrinsics,
      distCoeffs,
      handEyeInv,
      jointStatesVal,
      robotModel,
      points3dVal,
      observationsVal,
      jointPointIndicesVal,
      false,     -- optimize_hand_eye
      true,     -- optimize_points
      false,     -- optimize_robot_model_theta
      false,     -- optimize_robot_model_d
      false,     -- optimize_robot_model_a
      false,     -- optimize_robot_model_alpha
      false      -- optimize_joint_states
    )
    print("Error after optim of theta and alpha (Validation): " .. validation_error)
    
    local tmp = {}
    tmp.validationError = validation_error
    tmp.trainingError = training_error
    tmp.calibData  = calibData
    tmp.trainingIndices = idxTraining
    table.insert(validationErrors, tmp)  
    tmp.optimizationPath = optimization_path
  end
  
  table.sort(validationErrors, function(a,b) return a.validationError < b.validationError end)

  local best = validationErrors[1]  
  return best, best.calibData
end


function Calibration:runBAMultiplePatterns()

  local observations, jointPointIndices, jointStates, points3d = nil
  
  for k,v in ipairs(self.patternIDs) do
    observations, jointPointIndices, jointStates, points3d = self:prepareBAStructureForPattern(k, observations, jointPointIndices, jointStates, points3d)
  end
  
  print("Intrinsics")
  print(self.intrinsics)
  print("DistCoeffs")
  print(self.distCoeffs)
  print("HandEye")
  print(self.handEye)

  --print("jointpointIndices")
  --print(jointpointIndices)

  print("jointStates size")
  print(jointStates:size())

  print("point3d size")
  print(points3d:size())

  print("observations size")
  print(observations:size())

  print("jointPointIndices size")
  print(jointPointIndices:size())


  --[[
  local error = calib.evaluateDH(self.intrinsics,
                   self.distCoeffs,
                   self.handEye,
                   jointStates,
                   self.robotModel.dh,
                   points3d,
                   observations,
                   jointPointIndices)
  print(error)
  os.exit()
  ]]

  local point3dBefore = points3d:clone()
  
  local final_error = calib.optimizeDH(
    self.intrinsics,
    self.distCoeffs,
    torch.inverse(self.handEye),
    jointStates,
    self.robotModel.dh:clone(),
    points3d,
    observations:clone(),
    jointPointIndices,
    false,     -- optimize_hand_eye
    true,      -- optimize_points
    false,     -- optimize_robot_model_theta
    false,     -- optimize_robot_model_d
    false,     -- optimize_robot_model_a
    false,     -- optimize_robot_model_alpha
    false      -- optimize_joint_states
  )

  print("Intrinsics2")
  print(self.intrinsics)
  print("DistCoeffs2")
  print(self.distCoeffs)
  print("HandEye2")
  print(self.handEye)

  print("Final error: " ..final_error)

  local v = pcl.PCLVisualizer('demo', true)
  v:addCoordinateSystem(0.5)
  ptsPcl = pcl.rand(points3d:size()[1])
  ptsPcl2 = pcl.rand(points3d:size()[1])
  
  for i = 1, points3d:size()[1] do
    ptsPcl:points()[{1,i,{1,3}}] = points3d[{i, {}}]
    ptsPcl2:points()[{1,i,{1,3}}] = point3dBefore[{i, {}}]  
  end
  v:addPointCloud(ptsPcl, 'ref')
  v:setPointCloudRenderingProperties3(pcl.RenderingProperties.PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, 'ref')
  v:setPointCloudRenderingProperties1(pcl.RenderingProperties.PCL_VISUALIZER_POINT_SIZE, 3, 'ref');
  
  v:addPointCloud(ptsPcl2, 'ref2')
  v:setPointCloudRenderingProperties3(pcl.RenderingProperties.PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, 'ref2')
  v:setPointCloudRenderingProperties1(pcl.RenderingProperties.PCL_VISUALIZER_POINT_SIZE, 3, 'ref2');

  print("HandEye before:")
  print(self.handEye)

  local handEye_inv = torch.inverse(self.handEye)
  local final_error = calib.optimizeDH(self.intrinsics,
    self.distCoeffs,
    handEye_inv,
    jointStates,
    self.robotModel.dh,
    points3d,
    observations,
    jointPointIndices,
    false,     -- optimize_hand_eye
    true,      -- optimize_points
    true,      -- optimize_robot_model_theta
    false,     -- optimize_robot_model_d
    false,     -- optimize_robot_model_a
    true,      -- optimize_robot_model_alpha
    false      -- optimize_joint_states
  )
  
  print("Last error: "..final_error)
  
  print("HandEye after:")
  self.handEye = torch.inverse(handEye_inv)
  print(self.handEye)

  print(final_error)

  local calibData = {}
  calibData.intrinsics = self.intrinsics
  calibData.distCoeffs = self.distCoeffs
  calibData.handEye = self.handEye
  calibData.robotModel = self.robotModel
  calibData.joinDir = { 1, -1, -1, -1, 1, -1 }
  
  torch.save("robotCalibration.t7", calibData)  

  v:spin()
end

-- runs a camera calibration on all images currently contained self.images
function Calibration:runCameraCalibration(load_existing_calibration)

  if load_existing_calibration then
    local c = torch.load('intrinsics.t7')
    self.intrinsics = c.intrinsics
    self.distCoeffs = c.distCoeffs
    return
  end

  local patternPoints3d = xamla3d.calibration.calcPatternPointPositions(self.pattern.width, self.pattern.height, self.pattern.circleSpacing)

  local objectPoints = {}
  local imgPoints = {}
  local detectedPatternsToCamera = {}
  local visualization = torch.DoubleTensor(self.im_height, self.im_width, 1):zero()

  for i = 1,#self.images do

    local patternGeom = {height=self.pattern.height, width=self.pattern.width}    -- size of the pattern: x, y

    cv.imshow{"img", self.images[i].image}
    cv.waitKey{30}

    local points = nil
    local found = false
    if self.images[i].patternPoints2d ~= nil then
      points = self.images[i].patternPoints2d
      found = true
    else
     found, points = xamla3d.calibration.findPattern(self.images[i].image:clone(), cv.CALIB_CB_ASYMMETRIC_GRID, patternGeom)
    end

    if found then
      for i = 1, (#points)[1] do
          local point = cv.Point(points[i][1][1], points[i][1][2])
          cv.circle{img = visualization, center = point, radius = 3, color = {80,80,255,1}, thickness = 1, lineType = cv.LINE_AA}
      end

      cv.imshow{"Measurement distribution in image plane", visualization}
      cv.waitKey{30}

      self.images[i].patternPoints2d = points
      table.insert(imgPoints, points)
      table.insert(objectPoints, patternPoints3d)
      table.insert(detectedPatternsToCamera, i)
    end
  end

  print("Performing camera calibration with ".. #imgPoints .. " images")

  if #imgPoints < 5 then
    error("Too few images for camera calibration (at least 5 required).")
  end

  local err_, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera{objectPoints=objectPoints, imagePoints=imgPoints, imageSize={self.im_width, self.im_height}, flag=cv.CALIB_ZERO_TANGENT_DIST+cv.CALIB_FIX_K3}

  print("Error: "..err_)
  print("Intrinsics")
  print(cameraMatrix)
  print("Distortions")
  print(distCoeffs)
  
  torch.save("intrinsics.t7", { intrinsics = cameraMatrix, distCoeffs = distCoeffs })

  self.intrinsics = cameraMatrix
  self.distCoeffs = distCoeffs

  for i = 1,#rvecs do
    local rotation = xamla3d.calibration.RotVectorToRotMatrix(rvecs[i])
    local transfer = torch.zeros(4,4)
    transfer[{{1,3}, {1,3}}] = rotation
    transfer[{{1,3}, {4}}] = tvecs[i]
    transfer[4][4]=1

    local imageID = detectedPatternsToCamera[i]
    self.images[imageID].poseToTarget = transfer

    --cv.undistortPoints{src= self.images[imageID].patternPoints2d, cameraMatrix = self.intrinsics, distCoeffs = self.distCoeffs}

    if self.images[imageID].patternPoints2d == nil then
      error("patternPoints2d is nil")
    end

  end

  return true

end
