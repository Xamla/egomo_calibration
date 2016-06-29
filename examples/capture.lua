local path = require 'pl.path'
local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d
local ros = require 'ros'
local egomoTools = require 'egomo-tools'


local function askForPattern(pattern)
  print("Current pattern parameter:")
  print(string.format("Width: %d Height: %d", pattern.width, pattern.height))
  print(string.format("Circle Spacing: %f [meter]", pattern.pointDistance))
  local answer
  repeat
    io.write("Are these parameters valid (y/n)? ")
    io.flush()
    answer=io.read()
  until answer=="y" or answer=="n"
  if answer == "y" then
    return true
  else
    return false
  end
end

local function askForDHParameter()
  print("Do you want to acquire images for DH Parameter optimization?")
  print("If yes, then move the calibration target within your roboter workspace")
  print("move the robot in freedrive to pattern position and answer this question")
  print("with \"y\" ")
   local answer
  repeat
    io.write("Do you want to acquire images for DH parameter (y/n)? ")
    io.flush()
    answer=io.read()
  until answer=="y" or answer=="n"
  if answer == "y" then
    return true
  else
    return false
  end

end


local function showLiveView(capture, cam_name)
  print("Please move robot in freedrive mode about 40cm above the pattern!")
  print("If you are done, please press \"q\" ")
  capture:showLiveView(cam_name)
end

local function doCaptureZCalib(capture, cam_name_depth, cam_name_ir, output_directory_path, pattern_geom, hand_eye)
  capture:setDefaultCameraValues(hand_eye, pattern_geom)
  showLiveView(capture, cam_name_ir)
  local pattern_found, pattern_pose, robot_pose, pattern_points_base, pattern_center_world = capture:findPattern(cam_name_ir)
  if pattern_found then
    local image_saver = calib.ImageSaver(path.join(output_directory_path, "ZCalib"))
    print("ZCalib path:" .. image_saver.path)
    capture.setImageSaver(image_saver)
    capture:capturePatternFrontoParallel(cam_name_ir, robot_pose, pattern_pose, pattern_center_world, 0.4, 0.6)
  else
    print("Pattern in Z Calibration not found! So we skip Z Calibration")
  end

end


local function calibrateIntrinsic(image_saver, cam_name, pattern_geom)
  image_saver:load()
  local nImages = image_saver:getNumberOfImages()
  local image_stack = image_saver:loadImage(1)
  local image = image_stack[cam_name]
  local im_width = image:size()[2]
  local im_height = image:size()[1]

  local robotCalibration = calib.Calibration(pattern_geom, im_width, im_height)

  for i = 1,nImages do
    local img, pose = image_saver:loadImage(i)
    robotCalibration:addImage(img[cam_name], pose.MoveitPose, pose.JointPos, 1)
  end
  local ok, matrix, coeffs = robotCalibration:runCameraCalibration()
  return matrix, coeffs
end


local function doCaptureProcess(capture, cam_name, output_directory_path, pattern_geom, hand_eye)
  capture:setDefaultCameraValues(hand_eye, pattern_geom)

  -- Visualize the livestream in a window
  showLiveView(capture, cam_name)
  -- Capture an image stack and find the best value
  --local bestFocus = capture:getBestFocusPoint()
  --print(string.format("Best focus setting is %d",bestFocus))


  local imgSaver = calib.ImageSaver(path.join(output_directory_path, "approxFocal"))
  capture:setImageSaver(imgSaver)
  local cam_intrinsics = capture:acquireForApproxFocalLength(5, cam_name)
  capture.intrinsics = cam_intrinsics
  print("Approx Camera intrinsics")
  print(cam_intrinsics)
  torch.save(path.join(imgSaver:getPath(), 'approxIntrinsics.t7'), cam_intrinsics)

  print("Now acquire for intrinsic calibration")
  local pattern_found = false
  while pattern_found == false do
    showLiveView(capture, cam_name)
    pattern_found, pattern_pose, robot_pose, pattern_points_base, pattern_center_world = capture:findPattern(cam_name)
  end

  local imgSaverIntrinsic = nil
  local imgSaverHandEye = nil
  local intrinsic = nil
  local distortion = nil

  if pattern_found then

    imgSaverIntrinsic = calib.ImageSaver(path.join(output_directory_path, "intrinsics"))
    capture:setImageSaver(imgSaverIntrinsic)
    capture:captureForIntrinsics(cam_name, pattern_pose, robot_pose, pattern_points_base, pattern_center_world)
    intrinsic, distortion = calibrateIntrinsic(imgSaverIntrinsic, cam_name, pattern_geom)
    capture.intrinsics = intrinsic

    imgSaverHandEye = calib.ImageSaver(path.join(output_directory_path, "handeye"))
    capture:setImageSaver(imgSaverHandEye)
    capture:captureForHandEye(cam_name, pattern_pose, robot_pose, pattern_points_base, pattern_center_world)
  end

  return imgSaverIntrinsic, imgSaverHandEye, intrinsic, distortion

end


local function main()
  ros.init('webtest')
  ros.Time.init()
  spinner = ros.AsyncSpinner()
  spinner:start()

  local webcam = egomoTools.webcam:new("egomo_webcam")
  webcam:ConnectDefault()
  webcam:ConnectToJpgStream()
  print("Webcam initialisation finished in main routine.")


 local camIntrinsicsIR=torch.FloatTensor({
     {563, 0, 322},
     {0, 562, 240},
     {0, 0, 1}})
   local depthcam = egomoTools.structureio:new(camIntrinsicsIR, "egomo_depthcam")
  depthcam:Connect()
  depthcam:SetProjectorStatus(false)
  depthcam:SetIRresolution(640, 480)


  local egomo = calib.EgomoSensor()
  egomo.side_cam_RGB = calib.Camera(torch.eye(3,3), torch.zeros(5,1), torch.eye(4,4), 960, 720)
  egomo.side_cam_hand_eye =  torch.DoubleTensor({
      {  0.0025,   0.7642,   0.6450,  0.0152395 },
      { -0.0007,  -0.6450,   0.7642,  0.0699035 },
      {  1.0000,  -0.0024,  -0.0011,  0.0559415 },
      {  0.0000,   0.0000,   0.0000,  1.0000    }
    })

  egomo.side_cam_depth = calib.DepthCamera(torch.eye(3,3), torch.zeros(5,1), torch.eye(4,4), 640, 480)
  egomo.side_cam_depth_hand_eye = torch.DoubleTensor({
        {-0.00998,  -0.78267,   0.62236,   0.04679},
        {0.00177,   0.62238,   0.78271,   0.05434},
        {-0.99995,   0.00891,  -0.00483,   0.08246},
        {0.00000,   0.00000,   0.00000,   1.00000}})


  local output_directory_path = '/data/ur5_calibration/' .. os.date('%Y-%m-%d') .. '/'
  local pictures_per_position = 26
  local velocity_scaling = 0.5

  local pattern = {}
  pattern.width = 4
  pattern.height = 11
  pattern.pointDistance = 0.023
  pattern.circleSpacing = pattern.pointDistance

  local capture = calib.Capture(output_directory_path, pictures_per_position, velocity_scaling)

  capture:addGrabFunctions("DEPTHCAM_NO_SPECKLE", depthcam.GrabIRNoSpeckleViaROS, depthcam)
  capture:addGrabFunctions("WEBCAM", webcam.GrabGrayscaleImgROS, webcam)

  capture.imwidth = 640
  capture.imheight = 480

   local camIntrinsicsIR=torch.DoubleTensor({
     {563, 0, 322},
     {0, 562, 240},
     {0, 0, 1}})

  --local saver_intrinsic, saver_heye, K, distortion = doCaptureProcess(capture, "DEPTHCAM_NO_SPECKLE", path.join(output_directory_path, "DEPTHCAM_NO_SPECKLE"), pattern, egomo.side_cam_depth_hand_eye)
  capture.intrinsics = camIntrinsicsIR
  capture.distortion = torch.zeros(5,1)--distortion


  --capture:addGrabFunctions("DEPTH", depthcam.GrabDepthImageViaROS, depthcam)
  --doCaptureZCalib(capture, "DEPTH", "DEPTHCAM_NO_SPECKLE", output_directory_path, pattern, egomo.side_cam_depth_hand_eye)




  capture.imwidth = 960
  capture.imheight = 720
  doCaptureProcess(capture, "WEBCAM", path.join(output_directory_path, "WEBCAM"), pattern, egomo.side_cam_hand_eye)


  local do_dh = askForDHParameter()
  if do_dh then
    for i = 1,3 do
      showLiveView(capture, "DEPTHCAM_NO_SPECKLE")
      local pattern_found, pattern_pose, robot_pose, pattern_points_base, pattern_center_world = capture:findPattern()
      if pattern_found then
        print("Pattern found")
        capture:captureForHandEye(pattern_pose, robot_pose, pattern_points_base, pattern_center_world, string.format("dh_%03d", i))
      end
    end
  end


end


main()
