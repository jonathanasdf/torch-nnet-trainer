local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechHogLuvProcessor', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)
end

function M.preprocess(path, isTraining, processorOpts)
  local dir, name = path:match("(.*)base(.*)")
  local w,h
  if isTraining then
    w = processorOpts.imageSize
    h = processorOpts.imageSize
  else
    w = processorOpts.testImageWidth
    h = processorOpts.testImageHeight
  end
  local imgs = torch.Tensor(10, h, w)
  for i=11,20 do
    local img = image.load(dir .. tostring(i) .. name, 1)
    if img:size(2) ~= h or img:size(3) ~= w then
      img = image.scale(img, w, h)
    end
    imgs[i-10] = img
  end
  if isTraining then
    imgs = Transforms.HorizontalFlip(processorOpts.flip)(imgs)
  end
  return imgs
end

return M
