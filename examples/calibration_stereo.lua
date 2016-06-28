local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'cv.calib3d'
require 'cv.imgcodecs'
require 'cv.features2d'

local calib = require 'egomo_calibration'
local xamla3d = calib.xamla3d


local function doCalibration(path, cam_name_ir, cam_name_rgb, cam_ir, cam_rgb)

  local patternGeom  = {}
  patternGeom.width = 4
  patternGeom.height = 11
  patternGeom.circleSpacing = 0.023
  
  local image_saver = calib.ImageSaver(path)
  image_saver:load()
  
  local ir_centers = {}
  local rgb_centers = {}
  local object_points = {} 
  local ir_images = {}
  local rgb_images = {}


  local nImages = image_saver:getNumberOfImages()
  for i = 1,nImages do
    local img, pose = image_saver:loadImage(i)
    local rgb_image = img[cam_name_rgb]
    local ir_image = img[cam_name_ir] 
    
    assert(rgb_image ~=nil)
    assert(ir_image ~=nil)
    
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

  local reprojError, _MatrixCam1, _DistortCam1, _MatrixCam2, _DistortCam2, R, T, E, F = cv.stereoCalibrate{objectPoints=object_points,
                      imagePoints1=ir_centers, imagePoints2=rgb_centers,
          cameraMatrix1=cam_ir:getIntrinsic(), distCoeffs1=cam_ir:getDistortion(),
          cameraMatrix2=cam_rgb:getIntrinsic(), distCoeffs2=cam_rgb:getDistortion(),
          imageSize={640,480}, flags=cv.CALIB_FIX_INTRINSIC} 
  print(string.format("Reprojection error: %f", reprojError))          
  print(R)
  T = T:squeeze()
  print(string.format("%f %f %f", T[1], T[2], T[3]))
  
  local H = torch.eye(4,4)
  H[{{1,3},{1,3}}] = R
  H[{{1,3},4}] = T
  
  return H
  
  --[[
  for i = 1,#rgb_images do
    xamla3d.drawEpipolarLineWithF(F, rgb_images[i], ir_centers[i], ir_images[i])
    
    cv.imshow{"RGB", rgb_images[i]}
    cv.imshow{"IR", ir_images[i]}
    cv.waitKey{-1}
    
  end
  print(string.format("Found in %d %d images the calibration pattern", #rgb_centers, #ir_centers))
  ]]
end





local cmd=torch.CmdLine()
cmd:option('-path', "./noname/", 'Directory where calibration images are stored')
cmd:option('-calib_ir', "./noname/calib.t7", 'File where calibration data is stored')
cmd:option('-calib_rgb', "./noname/calib.t7", 'File where calibration data for rgb is stored')
cmd:option('-camname_ir', "DEPTHCAM_NO_SPECKLE", 'Name of IR camera that has images without speckle pattern')
cmd:option('-camname_rgb', "WEBCAM", 'Name of rgbcamera')
cmd:option('-calib_out', "./noname/rgb_rig.t7", 'Name of rgbcamera')


local params = cmd:parse(arg)

local cam_ir = calib.DepthCamera()
local cam_rgb = calib.Camera()

cam_ir:initializeFromCalibrationDataTable(torch.load(params.calib_ir))
cam_rgb:initializeFromCalibrationDataTable(torch.load(params.calib_rgb))

local H = doCalibration(params.path, params.camname_ir, params.camname_rgb, cam_ir, cam_rgb)

local rgbd_rig = {}
rgbd_rig.depth = cam_ir:getCalibrationDataAsTable()
rgbd_rig.rgb = cam_rgb:getCalibrationDataAsTable()
rgbd_rig.depth_to_rgb = H
torch.save(params.calib_out, rgbd_rig)










