require 'hdf5'

local Transforms = require 'Transforms'
local CaltechProcessor2 = require 'CaltechProcessor2'
local M = torch.class('CaltechACFProcessor', 'CaltechProcessor2')

function M:__init(model, processorOpts)
  CaltechProcessor2.__init(self, model, processorOpts)
end

function M.preprocess(path, isTraining, processorOpts)
  local dir = paths.dirname(paths.dirname(path)) .. '/acf/'
  local name = paths.basename(path, '.png')
  local f = hdf5.open(dir .. name .. '.h5', 'r')
  local imgs = f:read('/img'):all()
  f:close()

  local sz = processorOpts.imageSize
  if imgs:size(2) ~= sz or imgs:size(3) ~= sz then
    imgs = image.scale(imgs, sz, sz)
  end

  if isTraining then
    imgs = Transforms.HorizontalFlip(processorOpts.flip)(imgs)
  end

  return imgs
end

return M
