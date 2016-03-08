--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'cunn'
require 'cudnn'
require 'optim'
require 'paths'

require 'dataLoader'
require 'utils'

local M = torch.class('Model')
local loadDataParallel

function M:__init(path)
  if nGPU == 0 then
    self.backend = 'nn'
  else
    cutorch.setDevice(1)
    self.backend = 'cudnn'
  end

  if path then
    self:load(path)
  end
end

function M:load(path)
  assert(paths.filep(path), 'File not found: ' .. path)
  if paths.extname(path) == 'lua' then
    print('Creating model from file: ' .. path)
    self.model = paths.dofile(path)
    if self.backend == 'cudnn' then
      cudnn.convert(self.model, cudnn)
    elseif self.backend ~= 'nn' then
      error('Unsupported backend')
    end
  else
    print('Loading model from file: ' .. path)
    self.model = loadDataParallel(path, self.backend)
  end
  if nGPU > 0 then
    self.model = self.model:cuda()
  end
  self.parameters, self.gradParameters = self.model:getParameters()

  print('=> Model')
  print(self.model)
  return self
end

local function trainBatch(model, updates, opt, pathNames, inputs)
  if nGPU > 0 then
    inputs = inputs:cuda()
  end

  opt.train_iter = opt.train_iter + 1
  local feval = updates(model, pathNames, inputs)
  if opt.train_iter % opt.update_every == 0 then
    optim.sgd(feval, model.parameters, opt.optimState)
    model:zeroGradParameters()
  end

  if model.model.needsSync then
    model.model:syncParameters()
  end
end

local function validBatch(model, processor, pathNames, inputs)
  model.valid_loss = model.valid_loss +
    processor:processBatch(pathNames, model:forward(inputs, true))
  model.valid_count = model.valid_count + #pathNames
end

function M:train(opt, updates)
  if not(opt.input) then
    error('Input must be defined for training.')
  end

  local train_loader = DataLoader{
    path = opt.input,
    preprocessor = opt.processor.preprocess
  }

  local valid_loader
  if opt.val ~= '' then
    valid_loader = DataLoader{
      path = opt.val,
      preprocessor = opt.processor.preprocess
    }
  end

  if opt.optimState ~= '' then
    opt.optimState = torch.load(opt.optimState)
  else
    opt.optimState = {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      dampening = 0.0,
      nesterov = true,
      weightDecay = opt.weightDecay
    }
  end

  self:zeroGradParameters()
  opt.train_iter = 0
  local trainFn = bind(trainBatch, self, updates, opt)
  local valFn = bind(validBatch, self, opt.processor)
  for epoch=1,opt.epochs do
    print('==> training epoch # ' .. epoch)

    train_loader:runAsync(opt.batchSize,
                          opt.epochSize,
                          true, --shuffle
                          trainFn)

    if opt.val ~= '' and epoch % opt.val_every == 0 then
      self.valid_count = 0
      self.valid_loss = 0
      valid_loader:runAsync(opt.batchSize,
                            opt.valSize,
                            false, --don't shuffle
                            valFn)
      self.valid_loss = self.valid_loss / self.valid_count
      print(string.format('  Validation loss: %.6f', self.valid_loss))
    end

    if opt.cache_every ~= -1 and epoch % opt.cache_every == 0 and
       opt.output and opt.output ~= '/dev/null' then
      self:save(opt.output .. '.cached')
      opt.optimState.dfdx = nil
      torch.save(opt.output .. '.optimState', opt.optimState)
    end
  end
end

function M:forward(inputs, deterministic)
  if nGPU > 0 then
    inputs = inputs:cuda()
  end
  if deterministic then
    self.model:evaluate()
  else
    self.model:training()
  end
  return self.model:forward(inputs)
end

function M:zeroGradParameters()
  self.model:zeroGradParameters()
end

function M:backward(inputs, gradOutputs)
  return self.model:backward(inputs, gradOutputs)
end



function makeDataParallel(model)
  if not(noUseDataParallelTable) and nGPU > 0 then
    print('converting model to nn.DataParallelTable')
    local gpu = torch.range(1, nGPU):totable()
    local model_single = model
    model = nn.DataParallelTable(1)
    model:add(model_single:clone(), gpu)
  end
  return model
end

function loadDataParallel(filename, backend)
  nn.DataParallelTable.deserializeNGPUs = nGPU

  local model
  if paths.extname(filename) == 'caffemodel' then
    require 'loadcaffe'
    model = loadcaffe.load(paths.dirname(filename) .. '/deploy.prototxt', filename, backend)
  else
    model = torch.load(filename)
  end

  if torch.type(model) == 'nn.DataParallelTable' then
    return makeDataParallel(model:get())
  elseif torch.type(model) == 'nn.Sequential' then
    for i,module in ipairs(model.modules) do
      if torch.type(module) == 'nn.DataParallelTable' then
        model.modules[i] = makeDataParallel(module:get())
      end
    end
    return model
  else
    error('The loaded model is not a Sequential or DataParallelTable module.')
  end
end

local function cleanDPT(module)
  -- This assumes this DPT was created by the function above: all the
  -- module.modules are clones of the same network on different GPUs
  -- hence we only need to keep one when saving the model to the disk.
  local newDPT = nn.DataParallelTable(1)
  newDPT:add(module:get(), 1)
  return newDPT
end

function M:save(filename)
  self.model:clearState()

  if torch.type(self.model) == 'nn.DataParallelTable' then
    torch.save(filename, cleanDPT(self.model))
  elseif torch.type(self.model) == 'nn.Sequential' then
    local temp_model = nn.Sequential()
    for i, module in ipairs(self.model.modules) do
      if torch.type(module) == 'nn.DataParallelTable' then
        temp_model:add(cleanDPT(module))
      else
        temp_model:add(module)
      end
    end
    torch.save(filename, temp_model)
  else
    error('This saving function only works with Sequential or DataParallelTable modules.')
  end
end

return M
