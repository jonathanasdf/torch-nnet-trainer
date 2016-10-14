local CifarProcessor = require 'CifarProcessor'
local M = torch.class('CifarDeepTreeProcessor', 'CifarProcessor')

function M:__init(model, processorOpts)
  CifarProcessor.__init(self, model, processorOpts)

  self.criterion = nn.TrueNLLCriterion(nil, false):cuda()
  self.pairwiseDistance = nn.PairwiseDistance(2):cuda()
  self.hingeCriterion = nn.HingeEmbeddingCriterion():cuda()
end

function M:loadAndPreprocessInputs(pathNames, augmentations)
  local inputs = CifarProcessor.loadAndPreprocessInputs(self, pathNames, augmentations)
  return {inputs, torch.ones(inputs:size(1), 1):cuda()}
end

function M:getLoss(outputs, labels)
  if self.criterion.sizeAverage ~= false then
    error('this function assumes criterion.sizeAverage == false because we divide through by batchCount.')
  end

  local loss = 0
  local gradOutputs = {}
  for i=1,#outputs do
    gradOutputs[i] = {torch.zeros(outputs[i][1]:size()):cuda(), torch.zeros(outputs[i][2]:size()):cuda()}
    for j=1,labels:size(1) do
      loss = loss + self.criterion:forward(outputs[i][1][j], labels[j]) * outputs[i][2][j][1] / opts.batchCount
      gradOutputs[i][1][j] = self.criterion:backward(outputs[i][1][j], labels[j]) * outputs[i][2][j][1] / opts.batchCount
    end
  end

  --for i=1,#outputs do
  --  for j=i+1,#outputs do
  --    for k=1,labels:size(1) do
  --      local I = {outputs[i][1][k], outputs[j][1][k]}
  --      local d = self.pairwiseDistance:forward(I)
  --      loss = loss + self.hingeCriterion:forward(d, -1)
  --      local g = self.hingeCriterion:backward(d, -1)
  --      local gradInputs = self.pairwiseDistance:backward(I, g)
  --      gradOutputs[i][1][k] = gradOutputs[i][1][k] + gradInputs[1] / opts.batchCount
  --      gradOutputs[j][1][k] = gradOutputs[j][1][k] + gradInputs[2] / opts.batchCount
  --    end
  --  end
  --end

  return loss, gradOutputs
end

function M:resetStats()
  CifarProcessor.resetStats(self)
  self.classCounts = torch.zeros(10)
  self.leafSum = {}
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

  for i=1,#pathNames do
    local c = labels[i]
    self.classCounts[c] = self.classCounts[c] + 1
    if self.leafSum[c] == nil then
      self.leafSum[c] = torch.zeros(#outputs)
    end
    for j=1,#outputs do
      self.leafSum[c][j] = self.leafSum[c][j] + outputs[j][2][i][1]
    end
  end
end

function M:getStats()
  local s = CifarProcessor.getStats(self)
  local r = 'Leaf probabilities per class:'
  for i=1,#self.leafSum do
    r = r .. '\n' .. tostring(i) .. ':'
    for j=1,self.leafSum[i]:size(1) do
      r = r .. ' ' .. tostring(math.floor((self.leafSum[i][j]/self.classCounts[i])*10000+0.5)/100)
    end
  end
  return r
end

return M
