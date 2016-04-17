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
  local models = {}
  local mutexes = {}
  for device=1,nDevices do
    if device ~= 1 then
      cutorch.setDevice(device)
      localModel = localModel:clone()
      localModel.params, localModel.gradParams = localModel:getParameters()
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
          _processor = this
          if _processor.criterion then
            _processor.criterion = _processor.criterion:clone()
          end
          processorOpts = this.processorOpts
          if opts.replicateModel then
            if __threadid <= nDevices then
              _model = localModel
            else
              _model = localModel:clone('weight', 'bias')
              _model.params, _model.gradParams = _model:getParameters()
            end
            collectgarbage(); collectgarbage()
            isReplica = __threadid ~= nDevices
            if isReplica then
              mutex = {}
              mutex.lock = function() end
              mutex.unlock = function() end
            else
              mutex = (require 'threads').Mutex(mutexId)
            end
          else
            _model = localModel
            isReplica = device ~= 1
            mutex = (require 'threads').Mutex(mutexId)
          end
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
  local outputs = _model:forward(inputs, deterministic)
  outputs = outputs:view(inputs:size(1), -1)
  return outputs
end

-- Return gradParams if isReplica, otherwise accumulate gradients in model and return nil
function M.backward(inputs, gradOutputs, gradLayer)
  if isReplica then
    _model:zeroGradParameters()
  end

  if gradLayer then
    -- feed gradients through a specific layer
    for i=gradLayer,2,-1 do
      gradOutputs = _model.model:get(i):backward(_model.model:get(i-1).output, gradOutputs)
    end
    _model.model:get(1):backward(inputs, gradOutputs)
  else
    _model:backward(inputs, gradOutputs)
  end

  local gradParams
  if isReplica then
    gradParams = _model.gradParams:clone()
  end
  return gradParams
end

-- Performs a forward and backward pass through the model
function M.train(pathNames, inputs)
  if not(_processor.criterion) then
    error('processor.criterion is not defined. Either define a criterion or a custom trainBatch.')
  end
  if nGPU > 0 and not(inputs.getDevice) then inputs = inputs:cuda() end
  local labels = _processor.getLabels(pathNames)

  mutex:lock()
  local outputs = _processor.forward(inputs)
  --Assumes criterion.sizeAverage = false
  local loss = _processor.criterion:forward(outputs, labels)
  local stats = _processor.calcStats(pathNames, outputs, labels)
  local gradOutputs = _processor.criterion:backward(outputs, labels)
  local gradParams = _processor.backward(inputs, gradOutputs)
  mutex:unlock()
  return gradParams, loss, labels:size(1), stats
end

function M:updateModel()
  self.mutexes[1]:lock()
  local p = self.model.params:clone()
  optim.sgd(function() return 0, self.model.gradParams / opts.batchCount end, self.model.params, opts.optimState)
  self.model:zeroGradParameters()
  self.mutexes[1]:unlock()

  for i=2,nGPU do
    cutorch.setDevice(i)
    self.mutexes[i]:lock()
    self.models[i].params:copy(self.model.params:clone())
    self.mutexes[i]:unlock()
  end
  cutorch.setDevice(1)
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

  --Assumes criterion.sizeAverage = false
  local loss = _processor.criterion:forward(outputs, labels)
  local stats = _processor.calcStats(pathNames, outputs, labels)
  mutex:unlock()

  return loss, labels:size(1), stats
end

function M.test(pathNames, inputs)
  if nGPU > 0 and not(inputs.getDevice) then inputs = inputs:cuda() end
  local labels = _processor.getLabels(pathNames)
  return _processor.testWithLabels(pathNames, inputs, labels)
end

return M
