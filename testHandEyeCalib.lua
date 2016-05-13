local calib = require('lua/handEyeCalibration')

local Hg = {}
local Hc = {}


for i = 1, 5 do
  table.insert(Hg, torch.rand(4,4))
  table.insert(Hc, torch.rand(4,4))
end

calib.calibrate(Hg, Hc)


