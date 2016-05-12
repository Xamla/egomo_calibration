local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
local path = require 'pl.path'
local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d

local patterns = {}

local p1 = {}
p1.directory = "/home/hoppe/data/2016-05-03.4/"
p1.robotPoses = torch.load(path.join(p1.directory, "spheresurface_000115.t7"))
table.insert(patterns, p1)

local p2 = {}
p2.directory = "/home/hoppe/data/2016-05-03.3/"
p2.robotPoses = torch.load(path.join(p2.directory, "spheresurface_000115.t7"))
table.insert(patterns, p2)

local robotCalibration = calib.Calibration()

for p = 1,#patterns do
  local directory = patterns[p].directory
  local robotPoses = patterns[p].robotPoses

  local success, list = xamla3d.utils.readFileList(directory, "web", "png")
  print(#list.." Files found!")

  for i=1, 100 do
    if robotPoses.FileName[i] ~= nil then
      local fn = path.join(directory, robotPoses.FileName[i].."_web.png")
      local image = cv.imread{filename = fn}
      print(fn)
      if robotPoses.MoveitPose[i] == nil then
        print("NIL POSE!!!")
      end

      robotCalibration:addImage(image, robotPoses.MoveitPose[i], robotPoses.JointPos[i], p)

      if p == 1 and i == 50 then
       robotCalibration:runCameraCalibration()
      end

    end
  end
end

local best, robotCalibrationData = robotCalibration:DHCrossValidate(0.6, 1)

print("Best Result:")
print("Training   Error:"..best.trainingError)
print("Validation Error:"..best.validationError)
print("Robot Model:     ")
print(best.calibData.robotModel)
print("OptimPath:" .. best.optimizationPath)
