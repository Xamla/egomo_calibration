local calib = require('../lua/handEyeCalibration')

local Hg = {}
local Hc = {}


directory = "/home/hoppe/data/2016-05-03.4/"
robotPoses = torch.load(path.join(directory, "spheresurface_000115.t7"))

for i = 1, #robotPoses.MoveitPose do
  if robotPoses.MoveitPose[i] ~= nil then
   
    print(robotPoses.MoveitPose[i].full)
    table.insert(Hg, robotPoses.MoveitPose[i].full)
    table.insert(Hc, torch.inverse(robotPoses.MoveitPose[i].full))
  end
end

calib.calibrate(Hg, Hc)


