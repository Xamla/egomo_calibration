local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
require 'cv.features2d'

local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d

local rgb_intrinsics = torch.eye(3,3)
local rgb_distortion = torch.zeros(5,1)

local ir_intrinsics = torch.eye(3,3)
local ir_distortion = torch.zeros(5,1)

ir_intrinsics[1][1] = 570
ir_intrinsics[2][2] = 569
ir_intrinsics[1][3] = 326
ir_intrinsics[2][3] = 241

rgb_intrinsics[1][1] = 1658.4245
rgb_intrinsics[2][2] = 1655.8632
rgb_intrinsics[1][3] = 1144.3745
rgb_intrinsics[2][3] = 772.1538


function printMatrixAsTorchTensor(m, variable )
--if (m:nDimensions() ~= 2) then
-- return
--end

local write = io.write

write(variable.. " = torch.DoubleTensor({\n")
for i = 1,m:size()[1] do
  write("{")
  for j = 1,m:size()[2] do
  write(m[i][j].. ",")
  end
  write("},\n")
end

write("})\n")

end


local patternGeom  = {}
patternGeom.width = 8
patternGeom.height = 21
patternGeom.circleSpacing = 0.008

local directory = "/home/hoppe/data/2016-06-03.1/pose001/"
local poses_file = "pose001_.t7"

local robot_poses = torch.load(path.join(directory, poses_file))

local ir_centers = {}
local rgb_centers = {}
local object_points = {} 
local ir_images = {}
local rgb_images = {}


for i, fn in ipairs(robot_poses.FileName) do
  local rgb_fn = path.join(directory, robot_poses.FileName[i].."_web.png")
  local ir_fn = path.join(directory, robot_poses.FileName[i].."_ir.png")
  local rgb_image = cv.imread{filename = rgb_fn}
  local ir_image = cv.imread{filename = ir_fn}
  
  local rgb_success, rgb_patterns = xamla3d.calibration.findPattern(rgb_image, cv.CALIB_CB_ASYMMETRIC_GRID, patternGeom)
  local ir_success, ir_patterns = xamla3d.calibration.findPattern(ir_image, cv.CALIB_CB_ASYMMETRIC_GRID, patternGeom)
  
  
  if rgb_success and ir_success then
    table.insert(ir_centers, ir_patterns)
    table.insert(rgb_centers, rgb_patterns)
    table.insert(object_points, xamla3d.calibration.calcPatternPointPositions(patternGeom.width, patternGeom.height, patternGeom.circleSpacing))    
    table.insert(rgb_images, rgb_image)
    table.insert(ir_images, ir_image)
  end
end

print(string.format("Found in %d %d images the calibration pattern", #rgb_centers, #ir_centers))


local reprojError, _MatrixCam1, _DistortCam1, _MatrixCam2, _DistortCam2, R, T, E, F = cv.stereoCalibrate{objectPoints=object_points,
                      imagePoints1=ir_centers, imagePoints2=rgb_centers,
          cameraMatrix1=ir_intrinsics, distCoeffs1=ir_distortion,
          cameraMatrix2=rgb_intrinsics, distCoeffs2=rgb_distortion,
          imageSize={640,480}, flags=cv.CALIB_FIX_INTRINSIC} 
print(string.format("Reprojection error: %f", reprojError))          
print(R)
T = T:squeeze()
print(string.format("%f %f %f", T[1], T[2], T[3]))

local H = torch.eye(4,4)
H[{{1,3},{1,3}}] = R
H[{{1,3},4}] = T

printMatrixAsTorchTensor(H, "rgb_offset")


for i = 1,#rgb_images do
  xamla3d.drawEpipolarLineWithF(F, rgb_images[i], ir_centers[i], ir_images[i])
  
  cv.imshow{"RGB", rgb_images[i]}
  cv.imshow{"IR", ir_images[i]}
  cv.waitKey{-1}
  
end





