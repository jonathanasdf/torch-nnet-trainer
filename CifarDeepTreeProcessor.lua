local CifarProcessor = require 'CifarProcessor'
local M = torch.class('CifarDeepTreeProcessor', 'CifarProcessor')

function M:__init(model, processorOpts)
  CifarProcessor.__init(self, model, processorOpts)

  self.criterion = nn.TrueNLLCriterion(nil, false):cuda()
end

function M:loadAndPreprocessInputs(pathNames, augmentations)
  local inputs = CifarProcessor.loadAndPreprocessInputs(self, pathNames, augmentations)
  return {inputs, torch.ones(inputs:size(1), 1):cuda()}
end

-- Output: {{batchSize x label_dist, batchSize x leaf_prob}, ...}
function M:updateStats(pathNames, outputs, labels)
  local avg = torch.zeros(#pathNames, 10)
  for i=1,#pathNames do
    for j=1,#outputs do
      avg[i]:add(outputs[j][1][i]:mul(outputs[j][2][i][1]):float())
    end
  end
  self.stats:batchAdd(avg, labels)
end

-- Performs a single forward and backward pass through the model
function M:train(pathNames)
  if self.criterion.sizeAverage ~= false then
    error('this function assumes criterion.sizeAverage == false because we divide through by batchCount.')
  end

  local inputs = self:loadAndPreprocessInputs(pathNames)
  local outputs = self:forward(pathNames, inputs)
  local labels = self:getLabels(pathNames, outputs)
  local loss = 0
  local gradOutputs = {}
  for i=1,#outputs do
    gradOutputs[i] = {torch.zeros(outputs[i][1]:size()):cuda(), torch.zeros(outputs[i][2]:size()):cuda()}
    for j=1,#pathNames do
      loss = loss + self.criterion:forward(outputs[i][1][j], labels[j]) * outputs[i][2][j][1] / opts.batchCount
      gradOutputs[i][1][j] = self.criterion:backward(outputs[i][1][j], labels[j]) * outputs[i][2][j][1] / opts.batchCount
    end
  end
  self:backward(inputs, gradOutputs)

  self:updateStats(pathNames, outputs, labels)
  return loss, #pathNames
end

-- return {aggregatedLoss, #instancesTested}
function M:test(pathNames)
  local inputs = self:loadAndPreprocessInputs(pathNames)
  local outputs = self:forward(pathNames, inputs, true)
  local labels = self:getLabels(pathNames, outputs)
  local loss = 0
  for i=1,#outputs do
    for j=1,#pathNames do
      loss = loss + self.criterion:forward(outputs[i][1][j], labels[j]) * outputs[i][2][j][1] / opts.batchCount
    end
  end

  self:updateStats(pathNames, outputs, labels)
  return loss, #pathNames
end

return M
