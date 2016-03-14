package.path = package.path .. ';/home/nvesdapu/opencv/?.lua'

local CaltechProcessor = require 'caltech_processor'
local class = require 'class'
local M = class('CaltechHogLuvProcessor', 'CaltechProcessor')

function M:__init(opt)
  CaltechProcessor.__init(self, opt)
end

function M.preprocess(path)
  require 'features'

  local img = cv.imread{path, cv.IMREAD_COLOR}:float():transpose(3, 1, 2)
  local mean_pixel = torch.FloatTensor{103.939, 116.779, 123.68}:view(3, 1, 1):expandAs(img)
  return hog_luv(img - mean_pixel)
end

return M
