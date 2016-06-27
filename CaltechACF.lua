require 'torchzlib'

local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechACFProcessor', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)
end

function M:preprocess(path, pAugment)
  if pAugment == nil then
    pAugment = {}
    if opts.phase == 'train' then
      if self.processorOpts.flip ~= 0 then
        pAugment['hflip'] = self.processorOpts.flip
      end
    end
  end
  local dir = paths.dirname(paths.dirname(path)) .. '/acft7/'
  local name = paths.basename(path, '.png')
  local imgs = torch.load(dir .. name .. '.t7'):decompress()

  local sz = self.processorOpts.imageSize
  if imgs:size(2) ~= sz or imgs:size(3) ~= sz then
    imgs = image.scale(imgs, sz, sz)
  end

  local augmentations
  if opts.phase == 'train' then
    if self.processorOpts.flip ~= 0 then
      imgs, augmentations = Transforms.HorizontalFlip(pAugment['hflip'])(imgs)
    end
  end

  self:checkAugmentations(pAugment, augmentations)
  return imgs:cuda(), augmentations
end

return M
