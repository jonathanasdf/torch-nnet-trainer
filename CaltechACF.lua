require 'image'
require 'torchzlib'
local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechACF', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)
end

function M:loadInput(path, augmentations)
  local augs = {}
  if augmentations ~= nil then
    for i=1,#augmentations do
      local name = augmentations[i][1]
      if name == 'hflip' then
        augs[#augs+1] = augmentations[i]
      end
    end
  else
    if opts.phase == 'train' then
      if self.flip ~= 0 then
        augs[#augs+1] = Transforms.HorizontalFlip(self.flip)
      end
    end
  end
  local dir = paths.dirname(paths.dirname(path)) .. '/acft7/'
  local name = dir .. paths.basename(path, '.png') .. '.t7'
  local imgs = torch.load(name):decompress()

  local sz = self.imageSize
  imgs = Transforms.Scale(sz, sz)[2](imgs)
  imgs = Transforms.Apply(augs, imgs)
  return imgs:cuda(), augs
end

return M
