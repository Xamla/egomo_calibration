local calib = require 'egomo_calibration'
local a = calib.ImageSaver('/tmp')

local imgs = {}
local img1 = torch.ByteTensor(300,300)
local img2 = torch.ByteTensor(300,300)
imgs["IR"] = img1
imgs["RGB"] = img2


local pose = {}
pose.Moveit = torch.Tensor(4,4)
pose.UR5 = torch.Tensor(4,4)
pose.Joints = {}
pose.Joints.j1 = 234

a:addCorrespondingImages(imgs, pose)
a:addCorrespondingImages(imgs, pose)
a:addCorrespondingImages(imgs, pose)
a:addCorrespondingImages(imgs ,pose)

table.remove(imgs, 1)

a:addCorrespondingImages(imgs, pose)



a:addImage(img1, "IR",  pose)

a:addCorrespondingImages(imgs, pose)

local b = calib.ImageSaver('/tmp')
b:load()

local limage, lposes = b:loadImage(5)
print(limage)
print(lposes)

