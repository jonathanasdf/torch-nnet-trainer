package.path = package.path .. ';/home/nvesdapu/opencv/?.lua'
local class = require 'class'

local CaltechProcessor = require 'caltech_processor'
local M = class('CaltechHogLuvProcessor', 'CaltechProcessor')

function M:__init(opt)
  CaltechProcessor.__init(self, opt)
end

function M.preprocess(path, opt, isTraining)
  local CaltechProcessor = require 'caltech_processor'
  require 'features'
  return hog_luv(CaltechProcessor.preprocess(path, opt, isTraining))
end

return M
