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
function ImageSaver:__init(path, filename)

  self.filename = filename or "imageData.t7"
  self.path = path
  self.img_data = {}
  self.img_data.filename = {}
  self.img_data.poses = {}
  self.img_data.additional = {}
  self.cnt = 1
  self.curr_image = 1

  mkdirRecursive(path)
end

function ImageSaver:getPath()
  return self.path
end


function ImageSaver:load()
  self.img_data = torch.load(path.join(self.path, self.filename))

 -- store absolute paths
  for k,v in pairs(self.img_data.filename) do
    for cnt = 1, #v do
      local fn = v[cnt]
      if fn ~= nil then
        local pos = string.find(fn, "/")
        if pos ~= nil then
          fn = fn:sub(pos+1)
        end
        v[cnt] = fn
      end
    end
  end


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
  local images, poses =  self:loadImage(self.curr_image)
  self.curr_image = self.curr_image + 1
  return images, poses
end

function ImageSaver:getNumberOfImages()
  return self.cnt
end


function ImageSaver:getAdditionalInformation(key)
  if self.img_data.additional == nil then
     print("ImageSaver:getAdditionalInformation -- Old data format, no additional data available.")
     return nil
  end

  return self.img_data.additional[key]
end

function ImageSaver:addAdditionalInformation(key, value)
  self.img_data.additional[key] = value
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
      local img = nil
      if string.find(fn, ".png") then
        img = cv.imread{fn}
      else
        img = torch.load(path.join(self.path, fn))
      end
      images[k] = img
    end
  end

  return images, pose_info

end

function ImageSaver:getImageSize(cam_name)
  local img = (self:loadImage(1))[cam_name]
  local w = img:size()[2]
  local h = img:size()[1]
  return w, h
end


function ImageSaver:updateImage(index, images, pose_info)

end

--- poses - table with "NameOfPose" = Data
-- @param images table of images. Key is an identifier like IR, RGB DEPTH
-- @param pose_info table of poses (poses.MoveIt = 4x4 Tensor, poses.Joints = table)
-- @param index [optional] if this is set, then the existing information at index is overwritten (Use for update information)
function ImageSaver:addCorrespondingImages(images,  pose_info, index)
  if images == nil then
    return false
  end

  assert(type(images) == "table")

  if index == nil then
    index = self.cnt
    self.cnt = self.cnt + 1
  end

  for k,v in pairs(images) do
    local img = v
    local prefix = k
    local fn = nil
    if v.type == nil then
      print(v)
      print(string.format("%s is nil!", k))
    end

    if v:type() == "torch.ByteTensor" then
      fn = string.format("%s_%06d.png", prefix, index)
      writeImage(self, img, fn)
    else
      fn = string.format("%s_%06d.t7", prefix, index)
      local p = path.join(self.path, fn)
      torch.save(p, img)
    end

    if self.img_data.filename[prefix] == nil then
      self.img_data.filename[prefix] = {}
    end

    self.img_data.filename[prefix][index] = fn

    for k,v in pairs(pose_info) do
      if self.img_data.poses[k] == nil then
         self.img_data.poses[k] = {}
      end
      self.img_data.poses[k][index] = v
    end
  end

  torch.save(path.join(self.path, self.filename), self.img_data)

  return true

end


function ImageSaver:addImage(image, file_prefix, pose_info)
  local i = {}
  i[file_prefix] = image

  self:addCorrespondingImages(i, pose_info)

end


local ImageSaverGroup = torch.class('egomo_calibration.ImageSaverGroup', calib)

function ImageSaverGroup:__init()
  self.image_saver = {}
  self.current_image = 1
  self.cnt = 1
  self.im_number_to_saver = {}
  self.im_number_to_local_number = {}

end

function ImageSaverGroup:addImageSaver(image_saver)
  table.insert(self.image_saver, image_saver)
  local n = image_saver:getNumberOfImages()
  for i = 1,n do
    self.im_number_to_saver[self.cnt] = image_saver
    self.im_number_to_local_number[self.cnt] = n
    self.cnt = self.cnt + 1
  end
end

function ImageSaverGroup:loadImage(cnt)
  if cnt > self.cnt or cnt < 1 then
    return false
  end

  return self.im_number_to_saver[cnt].loadImage(self.im_number_to_local_number[cnt])
end

function ImageSaverGroup:getNextImage()
   self.current_image = self.current_image + 1
   return self:loadImage(self.current_image-1)
end
