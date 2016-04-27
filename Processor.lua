require 'image'
require 'optim'
require 'paths'

require 'Utils'

local M = torch.class('Processor')

M.cmd = torch.CmdLine()
function M:__init(model, processorOpts)
  assert(model)
  self.model = model

  self.processorOpts = self.cmd:parse(processorOpts:split(' ='))
  self.preprocessFn = bindPost(self.preprocess, self.processorOpts)
end

function M:initializeThreads()
  print("Copying models to threads...")
  self.model.needsSync = torch.ByteTensor{0}
  self.model:zeroGradParameters()
  gpu = 1
  _model = self.model
  _processor = self
  processorOpts = self.processorOpts
  if nThreads == 0 then
    mutex = {}
    mutex.lock = function() end
    mutex.unlock = function() end
    self.models = {_model}
    self.mutexes = {mutex}
    return
  end
  local specific = threads:specific()
  threads:specific(true)
  local this = self
  if nGPU > 0 then assert(cutorch.getDevice() == 1) end
  local nDevices = math.max(nGPU, 1)
  local localModel = self.model
  local localCriterion
  if self.criterion then localCriterion = self.criterion end
  local models = {}
  local mutexes = {}
  for device=1,nDevices do
    if device ~= 1 then
      cutorch.setDevice(device)
      localModel = localModel:clone()
      localModel:zeroGradParameters()
      localModel.params, localModel.gradParams = localModel:getParameters()
      if localCriterion then localCriterion = localCriterion:clone() end
    end
    models[device] = localModel
    -- Separate mutex for each GPU
    mutexes[device] = (require 'threads').Mutex()
    local mutexId = mutexes[device]:id()
    for i=device-1,nThreads,nDevices do if i > 0 then
      threads:addjob(i,
        function()
          gpu = device
          if nGPU > 0 then cutorch.setDevice(device) end
          _model = localModel
          _processor = this
          _processor.criterion = localCriterion
          processorOpts = this.processorOpts
          mutex = (require 'threads').Mutex(mutexId)
        end
      )
    end end
  end
  if nGPU > 0 then cutorch.setDevice(1) end
  threads:specific(specific)

  assert(#models == nDevices)
  assert(#mutexes == nDevices)
  self.models = models
  self.mutexes = mutexes
end

-- Takes path as input and returns something that can be used as input to the model, like an image
function M.preprocess(path, isTraining, processorOpts)
  return image.load(path, 3)
end

-- return the labels associated with the list of paths
function M.getLabels(pathNames)
  error('getLabels is not defined.')
end

function M.forward(inputs, deterministic)
  return _model:forward(inputs, deterministic)
end

-- accumulate gradients in model
function M.backward(inputs, gradOutputs, gradLayer)
  -- TODO: hack for https://github.com/torch/nn/issues/792
  local gradCopy = _model.gradParams:clone()
  _model:zeroGradParameters()
  -- TODO: hack for https://github.com/torch/nn/issues/792
  if gradLayer then
    -- feed gradients through a specific layer
    for i=gradLayer,2,-1 do
      gradOutputs = _model.model:get(i):backward(_model.model:get(i-1).output, gradOutputs)
    end
    _model.model:get(1):backward(inputs, gradOutputs)
  else
    _model:backward(inputs, gradOutputs)
  end
  -- TODO: hack for https://github.com/torch/nn/issues/792
  _model.gradParams:add(gradCopy)
  -- TODO: hack for https://github.com/torch/nn/issues/792
  _model.needsSync[1] = 1
end

-- Performs a forward and backward pass through the model
function M.train(pathNames, inputs)
  if not(_processor.criterion) then
    error('processor.criterion is not defined. Either define a criterion or a custom train function.')
  end
  if _processor.criterion.sizeAverage ~= false then
    error('this function assumes criterion.sizeAverage == false because we divide through by batchCount.')
  end
  local labels = _processor.getLabels(pathNames)
  if nGPU > 0 and not(inputs.getDevice) then inputs = inputs:cuda() end
  if nGPU > 0 and not(labels.getDevice) then labels = labels:cuda() end

  mutex:lock()
  local outputs = _processor.forward(inputs)
  local stats = _processor.calcStats(pathNames, outputs, labels)
  local loss = _processor.criterion:forward(outputs, labels)
  local gradOutputs = _processor.criterion:backward(outputs, labels)
  _processor.backward(inputs, gradOutputs / opts.batchCount)
  mutex:unlock()
  return loss, labels:size(1), stats
end

function M:updateModel()
  self.mutexes[1]:lock()
  for i=2,nGPU do
    if self.models[i].needsSync[1] == 1 then
      self.mutexes[i]:lock()
      self.model.gradParams:add(self.models[i].gradParams:clone())
      self.models[i]:zeroGradParams()
      self.models[i].needsSync[1] = 0
      self.mutexes[i]:unlock()
    end
  end
  self.mutexes[1]:unlock()

  if self.model.needsSync[1] == 1 then
    self.mutexes[1]:lock()
    optim.sgd(function() return 0, self.model.gradParams end, self.model.params, opts.optimState)
    self.model:zeroGradParameters()
    self.model.needsSync[1] = 0
    self.mutexes[1]:unlock()

    if nGPU > 1 then
      for i=2,nGPU do
        cutorch.setDevice(i)
        self.mutexes[i]:lock()
        self.models[i].params:copy(self.model.params:clone())
        self.mutexes[i]:unlock()
      end
      cutorch.setDevice(1)
    end
  end
end

-- Calculate and return stats, but don't accumulate them since this is likely on another thread
function M.calcStats(pathNames, outputs, labels) end
-- Called before each validation/test run
function M:resetStats() end
-- Accumulate stats from the result of calcStats
function M:accStats(new_stats) end
-- Called after each epoch
function M:processStats(phase) end

-- return {aggregatedLoss, #instancesTested, stats}
function M.testWithLabels(pathNames, inputs, labels)
  mutex:lock()
  local outputs = _processor.forward(inputs, true)

  local stats = _processor.calcStats(pathNames, outputs, labels)
  local loss = _processor.criterion:forward(outputs, labels)
  mutex:unlock()

  return loss, labels:size(1), stats
end

function M.test(pathNames, inputs)
  local labels = _processor.getLabels(pathNames)
  if nGPU > 0 and not(inputs.getDevice) then inputs = inputs:cuda() end
  if nGPU > 0 and not(labels.getDevice) then labels = labels:cuda() end
  return _processor.testWithLabels(pathNames, inputs, labels)
end

return M
