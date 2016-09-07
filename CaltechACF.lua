require 'torchzlib'

local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechACFProcessor', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)
end

function M:preprocess(path, augmentations)
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
      if self.processorOpts.flip ~= 0 then
        augs[#augs+1] = Transforms.HorizontalFlip(self.processorOpts.flip)
      end
    end
  end
  local dir = paths.dirname(paths.dirname(path)) .. '/acft7/'
  local name = paths.basename(path, '.png')
  local imgs = torch.load(dir .. name .. '.t7'):decompress()
  local sz = self.processorOpts.imageSize
  imgs = Transforms.Scale(sz, sz)[2](imgs)
  imgs = Transforms.Apply(augs, imgs)
  return imgs:cuda(), augs
end

return M
