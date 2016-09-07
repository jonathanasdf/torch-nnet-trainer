require 'torchzlib'

local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechCheckerboardsProcessor', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)
end

function M:preprocess(path, augmentations)
  local augs = {}
  local dir = paths.dirname(paths.dirname(path)) .. '/chkt7/'
  local name = paths.basename(path, '.png')
  local imgs = torch.load(dir .. name .. '.t7'):decompress()
  local sz = self.processorOpts.imageSize
  imgs = Transforms.Scale(sz, sz)[2](imgs)
  imgs = Transforms.Apply(augs, imgs)
  return imgs:cuda(), augs
end

return M
