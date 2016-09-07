package.path = package.path .. ';/home/nvesdapu/scripts/?.lua'
local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechFilterbankProcessor', 'CaltechProcessor')

require 'features'

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

  local img = image.load(path, 3)
  local sz = self.processorOpts.imageSize
  img = Transforms.Scale(sz, sz)[2](img)
  img = Transforms.Apply(augs, img)
  img = filterbank(img)
  return img:cuda(), augs
end

return M
