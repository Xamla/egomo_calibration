local calib = require 'egomo_calibration.env'
local torch = require 'torch'

local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.imgcodecs'


local function mkdirRecursive(dir_path)
  dir_path = path.abspath(dir_path)
  local dir_names = string.split(dir_path, "/")
  local current_path = '/'
  for i,fn in ipairs(dir_names) do
    current_path = path.join(current_path, fn)

    if not path.exists(current_path) then
      path.mkdir(current_path)
    elseif path.isfile(current_path) then
      error("Cannot create directory. File is in the way: '" .. current_path .. "'.'")
    end
  end
end


---
-- adds an image to  
local function writeImage(self, image, filename)
 
 local fn = path.join(self.path, string.format(filename, self.cnt)) 
 cv.imwrite{fn, image}
 
end

local ImageSaver = torch.class('egomo_calibration.ImageSaver', calib)

---
-- Initializes the image Saver
function ImageSaver:__init(path)
  
  self.path = path  
  self.img_data = {}
  self.img_data.filename = {}
  self.img_data.poses = {}
  self.cnt = 1
  self.filename = "imageData.t7"
  self.curr_image = 1
  
  mkdirRecursive(path)
end

function ImageSaver:getPath()
  return self.path
end


function ImageSaver:load()
  self.img_data = torch.load(path.join(self.path, self.filename))
  
  local max_poses = 0
  
  for k,v in pairs(self.img_data.poses) do
    local pose = v -- pose of a specific coordinate system
    if #pose > max_poses then
      max_poses = #pose
    end
  end
  
  self.cnt = max_poses
  
  print(string.format("%d images loaded", self.cnt)) 
  
end


function ImageSaver:getNextImage()
  local images =  self:loadImage(self.curr_image)
  self.curr_image = self.curr_image + 1
  return images
end

function ImageSaver:getNumberOfImages()
  return self.cnt
end

function ImageSaver:loadImage(cnt)
  if cnt < 1 or cnt > self.cnt then
    return nil
  end
  
  local images = {}
  local pose_info = {}
  
  for k,v in pairs(self.img_data.poses) do
    pose_info[k] = v[cnt]     
  end
  
  for k,v in pairs(self.img_data.filename) do
    local fn = v[cnt]
    if fn ~= nil then
      local img = cv.imread{path.join(self.path, fn)}
      images[k] = img
    end
  end
  
  return images, pose_info
     
end


--- poses - table with "NameOfPose" = Data
-- @param images table of images. Key is an identifier like IR, RGB DEPTH
-- @param pose_info table of poses (poses.MoveIt = 4x4 Tensor, poses.Joints = table) 
function ImageSaver:addCorrespondingImages(images,  pose_info)
  if images == nil then
    return false 
  end


  assert(type(images) == "table")
  

  for k,v in pairs(images) do
    local img = v
    local prefix = k
    local fn = string.format("%s_%06d.png", prefix, self.cnt)
    
    writeImage(self, img, fn)   
    
    if self.img_data.filename[prefix] == nil then    
      self.img_data.filename[prefix] = {}
    end
      
    self.img_data.filename[prefix][self.cnt] = fn
    
    for k,v in pairs(pose_info) do
      if self.img_data.poses[k] == nil then
         self.img_data.poses[k] = {}
      end
      self.img_data.poses[k][self.cnt] = v 
    end        
  end
    
  self.cnt = self.cnt + 1
  
  torch.save(path.join(self.path, self.filename), self.img_data)
  
  return true
   
end


function ImageSaver:addImage(image, file_prefix, pose_info)
  local i = {}
  i[file_prefix] = image  
  
  self:addCorrespondingImages(i, pose_info)
  
end
