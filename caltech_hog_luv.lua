local class = require 'class'

local CaltechProcessor = require 'caltech_processor'
local M = class('CaltechHogLuvProcessor', 'CaltechProcessor')

function M:__init(opt)
  CaltechProcessor.__init(self, opt)
end

function M.preprocess(path, isTraining, opt)
  local imgs = {}
  local dir, name = path:match("(.*)base(.*)")
  for i=1,10 do
    imgs[#imgs+1] = image.load(dir .. tostring(i) .. name)
  end
  return torch.cat(imgs, 1)
end

return M
