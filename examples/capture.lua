local path = require 'pl.path'
local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d
local ros = require 'ros'


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


local function showLiveView(capture)
  print("Please move robot in freedrive mode about 40cm above the pattern!")
  print("If you are done, please press \"q\" ")
  capture:showLiveView()
end



local function main()
  --ros.init('webtest')
  --ros.Time.init()
  --spinner = ros.AsyncSpinner()
  --spinner:start()



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
  local pictures_per_position = 30
  local velocity_scaling = 0.5

  local pattern = {}
  pattern.width = 4
  pattern.height = 11
  pattern.pointDistance = 0.023

  --Ask user if parameters of pattern are correct
  if askForPattern(pattern) then
    local capture = calib.Capture(output_directory_path, pictures_per_position, velocity_scaling)
    capture:setDefaultCameraValues(egomo.side_cam_hand_eye, pattern)

    -- Visualize the livestream in a window
    showLiveView(capture)
    -- Capture an image stack and find the best value
    --local bestFocus = capture:getBestFocusPoint()
    --print(string.format("Best focus setting is %d",bestFocus))
    --local cam_intrinsics = capture:acquireForApproxFocalLength(bestFocus)
    --capture.intrinsics = cam_intrinsics
    --print("Approx Camera intrinsics")
    --print(cam_intrinsics)


  local pattern_found, pattern_pose, robot_pose, pattern_points_base, pattern_center_world = capture:findPattern()

   local pattern_pose_robot = robot_pose * capture.heye * pattern_pose
   print("---------Robot Pose-----")
   print(robot_pose)
   print("---------Pattern Pose-----")
   print(pattern_pose)

   --capture:calcCamPoseFromDesired2dPatternPoints(100, 50, pattern_pose_robot)

  if pattern_found then
    print("Pattern found")
    print(pattern_pose)
    capture:captureForIntrinsics(pattern_pose, robot_pose, pattern_points_base, pattern_center_world)
    capture:captureForHandEye(pattern_pose, robot_pose, pattern_points_base, pattern_center_world)
  end

  local do_dh = askForDHParameter()
  if do_dh then
    for i = 1,3 do
      showLiveView(capture)
      local pattern_found, pattern_pose, robot_pose, pattern_points_base, pattern_center_world = capture:findPattern()
      if pattern_found then
        print("Pattern found")
        capture:captureForHandEye(pattern_pose, robot_pose, pattern_points_base, pattern_center_world, string.format("dh_%03d", i))
      end
    end
  end



  else
    print("Calibration cancelled! Change your calibration board!")
  end


end


main()
