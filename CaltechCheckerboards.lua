require 'torchzlib'

local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechCheckerboardsProcessor', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)
end

function M.preprocess(path, isTraining, processorOpts)
  local dir = paths.dirname(paths.dirname(path)) .. '/chkt7/'
  local name = paths.basename(path, '.png')
  local imgs = torch.load(dir .. name .. '.t7'):decompress()

  local sz = processorOpts.imageSize
  if imgs:size(2) ~= sz or imgs:size(3) ~= sz then
    imgs = image.scale(imgs, sz, sz)
  end

  return imgs
end

return M
