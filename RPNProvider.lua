local Processor = require 'Processor'
local M = torch.class('RPNProvider', 'Processor')

function M:__init(model, processorOpts)
  Processor.__init(self, model, processorOpts)

  local boxesFile = '/file1/caltechrpn/boxes.txt'
  self.nBoxes = 0
  for _ in io.lines(boxesFile) do
    self.nBoxes = self.nBoxes + 1
  end

  self.data = torch.load('/file1/caltechrpn/ssd.t7')
end

function M:getLoss(outputs, labels)
  return 0, 0
end

-- Only called by TrainStudentModel.lua
function M:getStudentLoss(outputs, labels)
  local smoothL1Criterion = nn.SmoothL1Criterion()
  smoothL1Criterion.sizeAverage = false
  local criterion = nn.ParallelCriterion()
    :add(softCriterion)
    :add(smoothL1Criterion)
    :cuda()
  criterion.sizeAverage = false
  return criterion
end

local t = torch.CudaTensor(1)
function M:preprocess(path, augmentations)
  return t, {}
end

function M:getLabels(pathNames, outputs)
  local n = self.nBoxes
  local scores = torch.zeros(#pathNames, n):cuda()
  local offsets = torch.zeros(#pathNames, n, 4):cuda()
  for i=1,#pathNames do
    local name = paths.basename(pathNames[i], '.jpg')
    local box = self.data[name]
    if box then
      for j=1,box:size(1) do
        scores[i][box[j][1]+1] = box[j][6];
        offsets[i][box[j][1]+1] = box[j][{{2,5}}];
      end
    end
  end
  return {scores:view(-1), offsets}
end

function M:forward(pathNames, inputs, deterministic)
  return Processor.forward(self, pathNames, self:getLabels(pathNames), deterministic)
end

function M:train()
  error('Cannot train RPNProvider.')
end

return M
