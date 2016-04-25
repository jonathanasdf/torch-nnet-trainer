package.path = package.path .. ';/home/nvesdapu/scripts/?.lua'
local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechFilterbankProcessor', 'CaltechProcessor')

require 'features'

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)
end

function M.preprocess(path, isTraining, processorOpts)
  local img = image.load(path, 3)
  local sz = processorOpts.imageSize
  if img:size(2) ~= sz or img:size(3) ~= sz then
    img = image.scale(img, sz, sz)
  end
  if isTraining then
    img = Transforms.HorizontalFlip(processorOpts.flip)(img)
  end
  img = filterbank(img)
  return img
end

return M
