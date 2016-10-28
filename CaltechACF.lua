require 'image'
local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechACFProcessor', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)
end

function M:preprocess(path, augmentations)
  local augs = {}
  if augmentations ~= nil then
    for i=1,#augmentations do
      local name = augmentations[i][1]
      if name == 'hflip' then
        augs[#augs+1] = augmentations[i]
      end
    end
  else
    if opts.phase == 'train' then
      if self.flip ~= 0 then
        augs[#augs+1] = Transforms.HorizontalFlip(self.flip)
      end
    end
  end
  local dir = paths.dirname(paths.dirname(path)) .. '/acf/'
  local name = paths.basename(path, '.png')
  local img = image.loadPNG(dir .. name .. 'a.png')
  local imgs = torch.Tensor(10, img:size(2), img:size(3));
  imgs[{{1,3}, {}, {}}] = img;
  imgs[{{4,6}, {}, {}}] = image.loadPNG(dir .. name .. 'b.png');
  imgs[{{7,10}, {}, {}}] = image.loadPNG(dir .. name .. 'c.png');

  local sz = self.imageSize
  imgs = Transforms.Scale(sz, sz)[2](imgs)
  imgs = Transforms.Apply(augs, imgs)
  return imgs:cuda(), augs
end

return M
