local Processor = require 'Processor'
local M = torch.class('RPNProvider', 'Processor')

function M:__init(model, processorOpts)
  self.cmd:option('-boxesPerImage', 100, 'Number of boxes output per image.')
  Processor.__init(self, model, processorOpts)
end

function M:preprocess(path, augmentations)
  local img = image.load(path, 3)
  return img:cuda(), {}
end

function M:getLabels(pathNames, outputs)
  local n = self.processorOpts.boxesPerImage
  local boxes = torch.CudaTensor(#pathNames, n*4)
  local scores = torch.CudaTensor(#pathNames, n)
  local matio = require 'matio'
  matio.use_lua_strings = true
  for i=1,#pathNames do
    local d = matio.load(pathNames[i] .. '.mat')
    boxes[i] = d.bf_boxes:view(-1)
    scores[i] = d.bf_scores
  end
  return {boxes, scores:view(-1)}
end

function M:forward(pathNames, inputs, deterministic)
  self.model:get(1).output = getLabels(pathNames)
  return self.model:get(1).output
end

return M
