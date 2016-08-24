require 'torchzlib'

local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechCheckerboardsProcessor', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)
end

function M:preprocess(path, augmentations)
  local augs = {}
  if augmentations ~= nil then
    for i=1,#augmentations do
      local name = augmentations[i][1]
      if name == 'scale' then
        augs[#augs+1] = augmentations[i]
      end
    end
  else
    local sz = self.processorOpts.imageSize
    augs[#augs+1] = Transforms.Scale(sz, sz)
  end

  local dir = paths.dirname(paths.dirname(path)) .. '/chkt7/'
  local name = paths.basename(path, '.png')
  local imgs = torch.load(dir .. name .. '.t7'):decompress()
  Transforms.Apply(augs, imgs)
  self:checkAugmentations(augmentations, augs)
  return imgs:cuda(), augs
end

return M
