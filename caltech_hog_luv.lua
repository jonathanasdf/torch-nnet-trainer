local class = require 'class'

local CaltechProcessor = require 'caltech_processor'
local M = class('CaltechHogLuvProcessor', 'CaltechProcessor')

function M:__init(opt)
  CaltechProcessor.__init(self, opt)
end

function M.preprocess(path, isTraining, opt)
  local dir, name = path:match("(.*)base(.*)")
  local imgs = {}
  for i=1,10 do
    imgs[i] = image.load(dir .. tostring(i) .. name)
    if isTraining then
      imgs[i] = image.scale(imgs[i], opt.imageSize, opt.imageSize)
    end
  end
  return torch.cat(imgs, 1)
end

return M
