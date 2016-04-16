local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechHogLuvProcessor', 'CaltechProcessor')

function M:__init()
  CaltechProcessor.__init(self)
end

function M.preprocess(path, isTraining, processorOpts)
  local dir, name = path:match("(.*)base(.*)")
  local imgs = {}
  for i=11,20 do
    imgs[i-10] = image.load(dir .. tostring(i) .. name, 1)
  end
  imgs = cat(imgs, 1)
  if isTraining then
    if imgs:size(2) ~= processorOpts.imageSize or imgs:size(3) ~= processorOpts.imageSize then
      imgs = image.scale(imgs, processorOpts.imageSize, processorOpts.imageSize)
    end
    imgs = Transforms.HorizontalFlip(processorOpts.flip)(imgs)
  end
  assert(imgs:size(1) == 10)
  return imgs
end

return M
