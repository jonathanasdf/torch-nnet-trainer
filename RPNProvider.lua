local Processor = require 'Processor'
local M = torch.class('RPNProvider', 'Processor')

function M:__init(model, processorOpts)
  self.cmd:option('-boxesPerImage', 100, 'Number of boxes output per image.')
  Processor.__init(self, model, processorOpts)

  self.gt = torch.load('/data/rpn/datasets/caltech/gt.t7')
end

-- Only called by TrainStudentModel.lua
function M:getStudentCriterion()
  local softCriterion = Processor.getStudentCriterion(self)
  local smoothL1Criterion = nn.SmoothL1Criterion()
  smoothL1Criterion.sizeAverage = false
  local criterion = nn.ParallelCriterion()
    :add(smoothL1Criterion)
    :add(softCriterion)
  criterion.sizeAverage = false
  return criterion:cuda()
end

local t = torch.CudaTensor(1)
function M:preprocess(path, augmentations)
  return t, {}
end

function M:getLabels(pathNames, outputs)
  local n = self.processorOpts.boxesPerImage
  local boxes = torch.CudaTensor(#pathNames, n, 4)
  local scores = torch.CudaTensor(#pathNames, n)
  for i=1,#pathNames do
    local name = paths.basename(pathNames[i])
    boxes[i] = self.gt.boxes[name];
    scores[i] = self.gt.scores[name];
  end
  return {boxes, scores:view(-1)}
end

function M:forward(pathNames, inputs, deterministic)
  return Processor.forward(self, pathNames, self:getLabels(pathNames), deterministic)
end

function M:train()
  error('Cannot train RPNProvider.')
end

return M
