local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechHogLuvProcessor', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)
end

function M.preprocess(path, isTraining, processorOpts)
  local dir, name = path:match("(.*)base(.*)")
  local first = image.load(dir .. '11' .. name, 1)
  local imgs = torch.Tensor(10, first:size(2), first:size(3))
  imgs[1] = first
  for i=12,20 do
    imgs[i-10] = image.load(dir .. tostring(i) .. name, 1)
  end
  if isTraining then
    local sz = processorOpts.imageSize
    if imgs:size(2) ~= sz or imgs:size(3) ~= sz then
      imgs = image.scale(imgs, sz, sz)
    end
    imgs = Transforms.HorizontalFlip(processorOpts.flip)(imgs)
  end
  return imgs
end

return M
