local CaltechProcessor = require 'caltech_processor'
local M = torch.class('CaltechHogLuvProcessor', 'CaltechProcessor')

function M:__init()
  CaltechProcessor.__init(self)
end

function M.preprocess(path, isTraining, processor_opts)
  local dir, name = path:match("(.*)base(.*)")
  local imgs = {}
  for i=1,10 do
    imgs[i] = image.load(dir .. tostring(i) .. name, 1)
  end
  imgs = cat(imgs, 1)
  if isTraining and (imgs:size(2) ~= processor_opts.imageSize or imgs:size(3) ~= processor_opts.imageSize) then
    imgs = image.scale(imgs, processor_opts.imageSize, processor_opts.imageSize)
  end
  assert(imgs:size(1) == 10)
  return imgs
end

return M
