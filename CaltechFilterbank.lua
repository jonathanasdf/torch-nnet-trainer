package.path = package.path .. ';/home/nvesdapu/scripts/?.lua'
local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechFilterbankProcessor', 'CaltechProcessor')

require 'features'

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
  local img = image.load(path, 3)
  local sz = self.processorOpts.imageSize
  if img:size(2) ~= sz or img:size(3) ~= sz then
    img = image.scale(img, sz, sz)
  end

  local augmentations
  if opts.phase == 'train' then
    if self.processorOpts.flip ~= 0 then
      img, augmentations = Transforms.HorizontalFlip(pAugment['hflip'])(img)
    end
  end
  img = filterbank(img)

  self:checkAugmentations(pAugment, augmentations)
  return img:cuda(), augmentations
end

return M
