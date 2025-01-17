local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
local path = require 'pl.path'
local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d


function printMatrixAsTorchTensor(m)
--if (m:nDimensions() ~= 2) then
-- return
--end

local write = io.write

write("a = torch.DoubleTensor({\n")
for i = 1,m:size()[1] do
  write("{")
  for j = 1,m:size()[2] do
  write(m[i][j].. ",")
  end
  write("},\n")
end

write("})\n")

end


local patterns = {
<<<<<<< Updated upstream
  { directory='/data/ur5_calibration/2016-06-13/pose001', robotPoses='pose001_.t7' },
--  { directory='/data/ur5_calibration/2016-06-11/pose002', robotPoses='pose002_.t7' },
--  { directory='/data/ur5_calibration/2016-06-11/pose004', robotPoses='pose004_.t7' },
--  { directory='/data/ur5_calibration/2016-06-11/pose005', robotPoses='pose005_.t7' },
=======
  { directory='/home/hoppe/data/2016-06-03/pose001/', robotPoses='pose001_.t7' },
  { directory='/home/hoppe/data/2016-06-03/pose002/', robotPoses='pose002_.t7' },
  { directory='/home/hoppe/data/2016-06-03/pose003/', robotPoses='pose003_.t7' },
  --{ directory='/data/ur5_calibration/2016-05-11/pose002', robotPoses='pose002_.t7' },
  --{ directory='/data/ur5_calibration/2016-05-11/pose004', robotPoses='pose004_.t7' },
  --{ directory='/data/ur5_calibration/2016-05-11/pose005', robotPoses='pose005_.t7' },
>>>>>>> Stashed changes
  --{ directory='/data/ur5_calibration/2016-05-11/pose006', robotPoses='pose006_.t7' },
  --{ directory='/data/ur5_calibration/2016-05-11/pose007', robotPoses='pose007_.t7' }
}

for i,x in ipairs(patterns) do
  local p = path.join(x.directory, x.robotPoses)
  x.robotPoses = torch.load(p)
end

--[[
local patterns = {}
local p1 = {}
p1.directory = "/home/hoppe/data/2016-05-03.4/"
p1.robotPoses = torch.load(path.join(p1.directory, "spheresurface_000115.t7"))
table.insert(patterns, p1)

local p2 = {}
p2.directory = "/home/hoppe/data/2016-05-03.3/"
p2.robotPoses = torch.load(path.join(p2.directory, "spheresurface_000115.t7"))
table.insert(patterns, p2)
]]

local patternGeom = {}
patternGeom.width = 8
patternGeom.height = 21
patternGeom.circleSpacing = 8

imWidth = 2304
imHeight = 1536


local robotCalibration = calib.Calibration(patternGeom, imWidth, imHeight)
local calibrated = false
for p = 1,1 do
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
<<<<<<< Updated upstream
  --if not calibrated and (#robotCalibration.images > 15 or p == #patterns) then
  --if not calibrated and (#robotCalibration.images > 20 or p == #patterns) then
  --if not calibrated and (#robotCalibration.images > 20 or p == #patterns) then
  if p == 1 then
    robotCalibration:runCameraCalibration()
=======
  if not calibrated and (#robotCalibration.images > 80 or p == #patterns) then
    robotCalibration:runCameraCalibration(true)
>>>>>>> Stashed changes
    calibrated = true
  end
end

<<<<<<< Updated upstream
os.exit()

local best, robotCalibrationData = robotCalibration:DHCrossValidate(0.6, 20)
=======

local best, robotCalibrationData = robotCalibration:DHCrossValidate(0.8, 20)
>>>>>>> Stashed changes

print("Best Result:")
print("Training   Error:"..best.trainingError)
print("Validation Error:"..best.validationError)
print("OptimPath:" .. best.optimizationPath)
print()
print('intrinsics')
printMatrixAsTorchTensor(best.calibData.intrinsics)
print('distCoeffs:')
printMatrixAsTorchTensor(best.calibData.distCoeffs)
print('handEye:')
printMatrixAsTorchTensor(best.calibData.handEye)
print('robotModel:')
printMatrixAsTorchTensor(best.calibData.robotModel)
print('jointDir:')
print(best.calibData.joinDir)

for i = 1,6 do
  local r = best.calibData.robotModel
  print("<property name=\"theta_"..i.."\" value=\""..r[1][i].."\" />")
  print("<property name=\"ur5_d"..i.."\" value=\""..r[2][i].."\" />")
  print("<property name=\"ur5_a"..i.."\" value=\""..r[3][i].."\" />")
  print("<property name=\"alpha_"..i.."\" value=\""..r[4][i].."\" />")  
end



torch.save('calibration.t7', robotCalibrationData)
