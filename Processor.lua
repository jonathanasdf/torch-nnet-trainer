require 'image'
require 'optim'
require 'paths'

require 'Utils'

local M = torch.class('Processor')

M.cmd = torch.CmdLine()
function M:__init()
  self.processorOpts = self.cmd:parse(opts.processorOpts:split(' ='))
  self.preprocessFn = bindPost(self.preprocess, self.processorOpts)
end

-- copy the processor and model to multiple threads/GPUs
function M:initializeThreads()
  assert(self.model)
  if nThreads == 0 then
    gpu = 1
    model = self.model
    processor = self
    mutex = {}
    mutex.lock = function() end
    mutex.unlock = function() end
    self.models = {model}
    self.mutexes = {mutex}
    return
  end
  print("Copying models to threads...")
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
          processor = this
          if opts.replicateModel then
            if __threadid <= nDevices then
              model = localModel
            else
              model = localModel:clone('weight', 'bias')
              model.params, model.gradParams = model:getParameters()
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
            model = localModel
            isReplica = device ~= 1
            mutex = (require 'threads').Mutex(mutexId)
          end

          if processor.criterion then
            processor.criterion = processor.criterion:clone()
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
  local outputs = model:forward(inputs, deterministic)
  outputs = outputs:view(inputs:size(1), -1)
  return outputs
end

-- Return gradParams if isReplica, otherwise accumulate gradients in model and return nil
function M.backward(inputs, gradOutputs)
  local gradParams
  if isReplica then
    model:zeroGradParameters()
    model:backward(inputs, gradOutputs)
    if nGPU > 0 and model.gradParams:getDevice() ~= 1 then
      cutorch.setDevice(1)
      gradParams = model.gradParams:clone()
      cutorch.setDevice(gpu)
    else
      gradParams = model.gradParams:clone()
    end
  else
    model:backward(inputs, gradOutputs)
  end
  return gradParams
end

-- Performs a forward and backward pass through the model
function M.train(pathNames, inputs)
  if not(processor.criterion) then
    error('processor.criterion is not defined. Either define a criterion or a custom trainBatch.')
  end
  if nGPU > 0 and not(inputs.getDevice) then inputs = inputs:cuda() end
  local labels = processor.getLabels(pathNames)

  mutex:lock()
  local outputs = processor.forward(inputs)
  --Assumes criterion.sizeAverage = false
  local loss = processor.criterion:forward(outputs, labels)
  local stats = processor.calcStats(pathNames, outputs, labels)
  local gradOutputs = processor.criterion:backward(outputs, labels)
  local gradParams = processor.backward(inputs, gradOutputs)
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
  local outputs = processor.forward(inputs, true)

  --Assumes criterion.sizeAverage = false
  local loss = processor.criterion:forward(outputs, labels)
  local stats = processor.calcStats(pathNames, outputs, labels)
  mutex:unlock()

  return loss, labels:size(1), stats
end

function M.test(pathNames, inputs)
  if nGPU > 0 and not(inputs.getDevice) then inputs = inputs:cuda() end
  local labels = processor.getLabels(pathNames)
  return processor.testWithLabels(pathNames, inputs, labels)
end

return M
