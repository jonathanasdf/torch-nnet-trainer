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
  if isTraining then
    local sz = processorOpts.imageSize
    if img:size(2) ~= sz or img:size(3) ~= sz then
      img = image.scale(img, sz, sz)
    end
    img = Transforms.HorizontalFlip(processorOpts.flip)(img)
    img = filterbank(img)
  end
  return img
end

function M.preprocessWindows(windows)
  local first = filterbank(windows[1])
  local output = first.new():resize(windows:size(1), first:size(1), first:size(2), first:size(3))
  output[1] = first
  for i=2,windows:size(1) do
    output[i] = filterbank(windows[i])
  end
  return output
end

return M
