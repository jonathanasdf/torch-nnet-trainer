require 'image'

require 'Model'

local M = torch.class('Processor')

M.cmd = torch.CmdLine()
function M:__init(model, processorOpts)
  self.model = model
  for k,v in pairs(self.cmd:parse(processorOpts:split(' ='))) do
    self[k] = v
  end
end

-- nil means anything is fine. {} means no augmentations.
local function checkAugmentations(a, b)
  if a == nil or b == nil then
    return
  end

  local fail = false
  if #a ~= #b then
    fail = true
  else
    for i=1,#a do
      -- augmentations are {'name', fn}
      if b[i][1] ~= a[i][1] then
        fail = true
      end
    end
  end

  if fail then
    local msg = 'Augmentations do not match:\n{'
    for i=1,#a do
      msg = msg + a[i][1] + ','
    end
    msg = msg + '}\n{'
    for i=1,#b do
      msg = msg + b[i][1] + ','
    end
    msg = msg + '}'
    error(msg)
  end
end

-- Takes path as input and what augmentations should be performed, and
-- returns a cuda tensor as input to the model and the augmentations used
function M:preprocess(path, augmentations)
  return image.load(path, 3):cuda(), {}
end

function M:loadAndPreprocessInputs(pathNames, augmentations)
  augmentations = augmentations or {}
  augs = {}
  local first
  first, augs[1] = self:preprocess(pathNames[1], augmentations[1])
  local size = torch.LongStorage(first:dim() + 1)
  size[1] = #pathNames
  for i=1,first:dim() do
    size[i+1] = first:size(i)
  end
  local inputs = first.new(size)
  inputs[1] = first
  for i=2,#pathNames do
    inputs[i], augs[i] = self:preprocess(pathNames[i], augmentations[i])
  end
  for i=1,#pathNames do
    checkAugmentations(augmentations[i], augs[i])
  end
  return inputs, augs
end

-- return the labels associated with the list of paths as cuda tensors
function M:getLabels(pathNames, outputs)
  error('getLabels is not defined.')
end

function M:forward(pathNames, inputs, deterministic)
  return self.model:forward(inputs, deterministic)
end

function M:backward(inputs, gradOutputs, gradLayer)
  return self.model:backward(inputs, gradOutputs, gradLayer)
end

function M:getLoss(outputs, labels)
  if not(self.criterion) then
    error('processor criterion is not defined. Either define a criterion or a custom getLoss function.')
  end
  if self.criterion.sizeAverage ~= false then
    error('this function assumes criterion.sizeAverage == false because we divide through by batchCount.')
  end

  local loss = self.criterion:forward(outputs, labels)
  local gradOutputs = self.criterion:backward(outputs, labels)
  if type(gradOutputs) == 'table' then
    for i=1,#gradOutputs do
      gradOutputs[i] = gradOutputs[i] / opts.batchCount
    end
  else
    gradOutputs = gradOutputs / opts.batchCount
  end

  return loss, gradOutputs
end

-- Only called by TrainStudentModel.lua
function M:getStudentLoss(student, studentOutputs, teacherOutputs)
  if self.softCriterion == nil then
    if opts.dropoutBayes > 1 then
      self.softCriterion = nn.SquareMahalanobisCriterion(false)
    elseif opts.useMSE then
      self.softCriterion = nn.MSECriterion(false)
    else
      self.softCriterion = nn.SoftCrossEntropyCriterion(opts.T, false)
    end
    self.softCriterion = self.softCriterion:cuda()
  end

  local criterion = self.criterion
  self.criterion = self.softCriterion
  local loss, gradOutputs = M.getLoss(self, studentOutputs, teacherOutputs)
  self.criterion = criterion
  return loss, gradOutputs
end

-- Called before each validation/test run
function M:resetStats() end
-- Calculate and update stats
function M.updateStats(pathNames, outputs, labels) end
-- Called after each epoch. Returns something printable
function M:getStats() end

-- Performs a single forward and backward pass through the model
function M:train(pathNames)
  local inputs = self:loadAndPreprocessInputs(pathNames)
  local outputs = self:forward(pathNames, inputs)
  local labels = self:getLabels(pathNames, outputs)
  local loss, gradOutputs = self:getLoss(outputs, labels)
  self:backward(inputs, gradOutputs)

  self:updateStats(pathNames, outputs, labels)
  return loss, #pathNames
end

-- return {aggregatedLoss, #instancesTested}
function M:test(pathNames)
  local inputs = self:loadAndPreprocessInputs(pathNames)
  local outputs = self:forward(pathNames, inputs, true)
  local labels = self:getLabels(pathNames, outputs)
  local loss, _ = self:getLoss(outputs, labels)

  self:updateStats(pathNames, outputs, labels)
  return loss, #pathNames
end

return M
