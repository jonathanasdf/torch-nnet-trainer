require 'image'

require 'Utils'

local M = torch.class('Processor')

M.cmd = torch.CmdLine()
function M:__init(model, processorOpts)
  self.model = model
  self.processorOpts = self.cmd:parse(processorOpts:split(' ='))

  self.trainFn = bind(self.train, self)
  self.testFn = bind(self.test, self)
end

function M:checkAugmentations(a, b)
  if a == nil then
    return
  end

  b = b or {}
  for k,v in pairs(a) do
    if b[k] == nil then
      error('Augmentation ' .. k .. ' specified but not performed.')
    end
  end
  for k,v in pairs(b) do
    if a[k] == nil then
      error('Augmentation ' .. k .. ' not specified but performed.')
    end
  end
end

-- Takes path as input and returns a cuda tensor that can be used
-- as input to the model such as an image, as well as the augmentations used
function M:preprocess(path, pAugment)
  self:checkAugmentations(pAugment, nil)
  return image.load(path, 3):cuda(), nil
end

function M:loadAndPreprocessInputs(pathNames, pAugment)
  pAugment = pAugment or {}
  augmentations = {}
  local first
  first, augmentations[1] = self:preprocess(pathNames[1], pAugment[1])
  local size = torch.LongStorage(first:dim() + 1)
  size[1] = #pathNames
  for i=1,first:dim() do
    size[i+1] = first:size(i)
  end
  local inputs = first.new(size)
  inputs[1] = first
  for i=2,#pathNames do
    inputs[i], augmentations[i] = self:preprocess(pathNames[i], pAugment[i])
  end
  return inputs, augmentations
end

-- return the labels associated with the list of paths as cuda tensors
function M:getLabels(pathNames)
  error('getLabels is not defined.')
end

function M:forward(inputs, deterministic)
  return self.model:forward(inputs, deterministic)
end

function M:backward(inputs, gradOutputs, gradLayer)
  return self.model:backward(inputs, gradOutputs, gradLayer)
end

-- Called before each validation/test run
function M:resetStats() end
-- Calculate and update stats
function M.updateStats(pathNames, outputs, labels) end
-- Called after each epoch. Returns something printable
function M:getStats() end

-- Performs a single forward and backward pass through the model
function M:train(pathNames)
  if not(self.criterion) then
    error('processor criterion is not defined. Either define a criterion or a custom train function.')
  end
  if self.criterion.sizeAverage ~= false then
    error('this function assumes criterion.sizeAverage == false because we divide through by batchCount.')
  end

  local labels = self:getLabels(pathNames)
  local inputs = self:loadAndPreprocessInputs(pathNames)

  local outputs = self:forward(inputs)
  local loss = self.criterion:forward(outputs, labels)
  local gradOutputs = self.criterion:backward(outputs, labels)
  self:backward(inputs, gradOutputs / opts.batchCount)

  self:updateStats(pathNames, outputs, labels)

  return loss, labels:size(1)
end

-- return {aggregatedLoss, #instancesTested}
function M:test(pathNames)
  local labels = self:getLabels(pathNames)
  local inputs = self:loadAndPreprocessInputs(pathNames)

  local outputs = self:forward(inputs, true)
  local loss = self.criterion:forward(outputs, labels)

  self:updateStats(pathNames, outputs, labels)

  return loss, #pathNames
end

return M
