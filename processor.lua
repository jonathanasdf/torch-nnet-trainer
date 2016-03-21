local class = require 'class'
require 'image'
require 'paths'
require 'utils'

local M = class('Processor')

M.cmd = torch.CmdLine()
function M:__init(opt)
  self.opt = {}
  for k,v in pairs(opt) do
    self.opt[k] = v
  end

  local new_opts = self.cmd:parse(opt.processor_opts:split(' ='))
  for k,v in pairs(new_opts) do
    self.opt[k] = v
  end

  self.preprocessFn = bind_post(self.preprocess, self.opt)
end

-- Takes a path as input and returns something that can be used as input to the model, such as an image
-- Note that this function is M.preprocess, not M:preprocess
-- This function is executed on multiple threads, so try not to pass anything very big to it
function M.preprocess(path, opt, isTraining)
  return image.load(path, 3)
end

-- return the labels associated with the list of paths
function M:getLabels(pathNames)
  error('getLabels is not defined.')
end

-- Feed gradients backwards through the model.
-- Note: don't call zeroGradParameters, that's already done automatically.
function M:trainBatch(pathNames, inputs)
  if not(self.criterion) then
    error('self.criterion is not defined. Either define a criterion or a custom trainBatch.')
  end

  local outputs = self.model:forward(inputs)
  local labels = self:getLabels(pathNames)
  if nGPU > 0 then
    labels = labels:cuda()
  end

  --Assumes self.criterion.sizeAverage = false
  self.criterion:forward(outputs, labels)
  local grad_outputs = self.criterion:backward(outputs, labels)
  self.model:backward(inputs, grad_outputs)
end

-- forward and accumulate stats for a test batch. Returns {#correct, #total, aggregated_loss}
function M:testBatch(pathNames, inputs)
  if not(self.criterion) then
    error('self.criterion is not defined. Either define a criterion or a custom testBatch.')
  end

  local outputs = self.model:forward(inputs, true)
  local labels = self:getLabels(pathNames)
  if nGPU > 0 then
    labels = labels:cuda()
  end

  --Assumes classification
  local _, pred = torch.max(outputs, 2)
  local correct = torch.eq(pred:squeeze(), labels):sum()
  local total = #pathNames

  --Assumes self.criterion.sizeAverage = false
  local loss = self.criterion:forward(outputs, labels)
  return correct, total, loss
end

-- Called before each validation run
function M:resetStats() end
-- Called at the end of forward.lua. Print whatever you want
function M:printStats() end

return M
