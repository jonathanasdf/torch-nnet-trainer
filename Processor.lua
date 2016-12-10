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
-- returns a cuda tensor as input to the model, the label for it, and the augmentations used
function M:loadInput(path, augmentations)
  return image.load(path, 3):cuda(), {}
end

function M:getLabel(path)
  return torch.Tensor(1):cuda()
end

function M:preprocess(path, augmentations)
  local input, augs = self:loadInput(path, augmentations)
  local label = self:getLabel(path)
  return input, label, augs
end

function M:loadAndPreprocessInputs(pathNames, augmentations)
  augmentations = augmentations or {}
  local inputs = {}
  local labels = {}
  local augs = {}
  for i=1,#pathNames do
    inputs[i], labels[i], augs[i] = self:preprocess(pathNames[i], augmentations[i])
    checkAugmentations(augmentations[i], augs[i])
  end
  return batchConcat(inputs), batchConcat(labels), augs
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
  local inputs, labels = self:loadAndPreprocessInputs(pathNames)
  local outputs = self:forward(pathNames, inputs)
  local loss, gradOutputs = self:getLoss(outputs, labels)
  self:backward(inputs, gradOutputs)

  self:updateStats(pathNames, outputs, labels)
  return loss, #pathNames
end

-- return {aggregatedLoss, #instancesTested}
function M:test(pathNames)
  local inputs, labels = self:loadAndPreprocessInputs(pathNames)
  local outputs = self:forward(pathNames, inputs, true)
  local loss, _ = self:getLoss(outputs, labels)

  self:updateStats(pathNames, outputs, labels)
  return loss, #pathNames
end

return M
