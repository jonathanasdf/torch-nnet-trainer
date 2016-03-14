local class = require 'class'
require 'image'
require 'paths'

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
end

function M.preprocess(path)
  return image.load(path, 3)
end

function M:getLabels(pathNames)
  error('getLabels is not defined.')
end

function M:evaluateBatch(pathNames, outputs)
  if not(self.criterion) then
    error('self.criterion is not defined. Either define a criterion or a custom evaluateBatch.')
  end

  local labels = self:getLabels(pathNames)
  if nGPU > 0 then
    labels = labels:cuda()
  end

  --Assumes classification
  local _, pred = torch.max(outputs, 2)
  local correct = torch.eq(pred:squeeze(), labels):sum()

  --Assumes self.criterion.sizeAverage = false
  local loss = self.criterion:forward(outputs, labels) / self.opt.batchCount
  local grad_outputs = self.criterion:backward(outputs, labels) / self.opt.batchCount
  return loss, grad_outputs, correct
end

function M:testBatch(pathNames, outputs) end
function M:printStats() end

return M
