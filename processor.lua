require 'image'
require 'optim'
require 'paths'
require 'utils'

local M = torch.class('Processor')

M.cmd = torch.CmdLine()
function M:__init()
  self.processor_opts = self.cmd:parse(opts.processor_opts:split(' ='))
  self.preprocessFn = bind_post(self.preprocess, self.processor_opts)
end

-- copy the processor and model to multiple threads/GPUs
function M:initializeThreads()
  assert(self.model)
  print("Copying models to threads...")
  local specific = threads:specific()
  threads:specific(true)
  local this = self
  if nGPU > 0 then assert(cutorch.getDevice() == 1) end
  local nDevices = math.max(nGPU, 1)
  local local_model = self.model
  local models = {}
  local mutexes = {}
  for device=1,nDevices do
    if device ~= 1 then
      cutorch.setDevice(device)
      local_model = local_model:clone()
    end
    models[device] = local_model
    -- Separate mutex for each GPU
    mutexes[device] = (require 'threads').Mutex()
    local mutex_id = mutexes[device]:id()
    for i=device-1,nThreads,nDevices do if i > 0 then
      threads:addjob(i,
        function()
          gpu = device
          if nGPU > 0 then cutorch.setDevice(device) end
          processor = this
          if opts.replicateModel then
            if __threadid <= nDevices then
              model = local_model
            else
              model = local_model:clone('weight', 'bias')
            end
            isReplica = __threadid ~= nDevices
            if isReplica then
              mutex = {}
              mutex.lock = function() end
              mutex.unlock = function() end
            else
              mutex = (require 'threads').Mutex(mutex_id)
            end
          else
            model = local_model
            isReplica = device ~= 1
            mutex = (require 'threads').Mutex(mutex_id)
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
function M.preprocess(path, isTraining, processor_opts)
  return image.load(path, 3)
end

-- return the labels associated with the list of paths
function M.getLabels(pathNames)
  error('getLabels is not defined.')
end

function M.forward(inputs, deterministic)
  return model:forward(inputs, deterministic)
end

-- Return gradParameters if isReplica, otherwise accumulate gradients in model and return nil
function M.backward(inputs, grad_outputs)
  local gradParameters
  if isReplica then
    model:zeroGradParameters()
    model:backward(inputs, grad_outputs)
    if nGPU > 0 and model.gradParameters:getDevice() ~= 1 then
      cutorch.setDevice(1)
      gradParameters = model.gradParameters:clone()
      cutorch.setDevice(gpu)
    else
      gradParameters = model.gradParameters:clone()
    end
  else
    model:backward(inputs, grad_outputs)
  end
  return gradParameters
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
  processor.criterion:forward(outputs, labels)
  local grad_outputs = processor.criterion:backward(outputs, labels)
  local gradParameters = processor.backward(inputs, grad_outputs)
  mutex:unlock()
  return gradParameters
end

function M:updateModel()
  self.mutexes[1]:lock()
  optim.sgd(function() return 0, self.model.gradParameters end, self.model.parameters, opts.optimState)
  self.model:zeroGradParameters()
  self.mutexes[1]:unlock()

  for i=2,nGPU do
    cutorch.setDevice(i)
    self.mutexes[i]:lock()
    self.models[i].parameters:copy(self.model.parameters:clone())
    self.mutexes[i]:unlock()
  end
  cutorch.setDevice(1)
end

-- Calculate and return stats, but don't accumulate them since this is likely on another thread
function M.calcStats(pathNames, outputs, labels) end
-- Called before each validation/test run
function M:resetStats() end
-- Accumulate stats from the result of calcStats
function M:accStats(...) end
-- Called after each validation/test run
function M:printStats() end

-- return {aggregated_loss, #instances_tested, stats}
function M.testWithLabels(pathNames, inputs, labels)
  mutex:lock()
  local outputs = processor.forward(inputs, true)

  --Assumes criterion.sizeAverage = false
  local loss = processor.criterion:forward(outputs, labels)
  local stats = {processor.calcStats(pathNames, outputs, labels)}
  mutex:unlock()

  return loss, labels:size(1), unpack(stats)
end

function M.test(pathNames, inputs)
  if nGPU > 0 and not(inputs.getDevice) then inputs = inputs:cuda() end
  local labels = processor.getLabels(pathNames)
  return processor.testWithLabels(pathNames, inputs, labels)
end

return M
