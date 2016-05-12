local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
local path = require 'pl.path'
local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d

local patterns = {
  { directory='/data/ur5_calibration/2016-05-11/pose001', robotPoses='pose001_.t7' },
  { directory='/data/ur5_calibration/2016-05-11/pose002', robotPoses='pose002_.t7' },
  { directory='/data/ur5_calibration/2016-05-11/pose004', robotPoses='pose004_.t7' },
  { directory='/data/ur5_calibration/2016-05-11/pose005', robotPoses='pose005_.t7' },
  --{ directory='/data/ur5_calibration/2016-05-11/pose006', robotPoses='pose006_.t7' },
  --{ directory='/data/ur5_calibration/2016-05-11/pose007', robotPoses='pose007_.t7' }
}

for i,x in ipairs(patterns) do
  local p = path.join(x.directory, x.robotPoses)
  x.robotPoses = torch.load(p)
end

--[[local patterns = {}
local p1 = {}
p1.directory = "/home/hoppe/data/2016-05-03.4/"
p1.robotPoses = torch.load(path.join(p1.directory, "spheresurface_000115.t7"))
table.insert(patterns, p1)

local p2 = {}
p2.directory = "/home/hoppe/data/2016-05-03.3/"
p2.robotPoses = torch.load(path.join(p2.directory, "spheresurface_000115.t7"))
table.insert(patterns, p2)
]]

local robotCalibration = calib.Calibration()
local calibrated = false
for p = 1,#patterns do
  local directory = patterns[p].directory
  local robotPoses = patterns[p].robotPoses

  local success, list = xamla3d.utils.readFileList(directory, "web", "png")
  print(#list.." Files found!")

  for i, fn in ipairs(robotPoses.FileName) do
    local fn = path.join(directory, robotPoses.FileName[i].."_web.png")
    local image = cv.imread{filename = fn}
    print(fn)
    if robotPoses.MoveitPose[i] == nil then
      print("NIL POSE!!!")
    end

    robotCalibration:addImage(image, robotPoses.MoveitPose[i], robotPoses.JointPos[i], p)
  end
  if not calibrated and (#robotCalibration.images > 75 or p == #patterns) then
    robotCalibration:runCameraCalibration()
    calibrated = true
  end
end

local best, robotCalibrationData = robotCalibration:DHCrossValidate(0.6, 1)

print("Best Result:")
print("Training   Error:"..best.trainingError)
print("Validation Error:"..best.validationError)
print("OptimPath:" .. best.optimizationPath)
print()
print('intrinsics')
print(best.calibData.intrinsics)
print('distCoeffs:')
print(best.calibData.distCoeffs)
print('handEye:')
print(best.calibData.handEye)
print('robotModel:')
print(best.calibData.robotModel)
print('jointDir:')
print(best.calibData.joinDir)

torch.save('calibration.t7', robotCalibrationData)
