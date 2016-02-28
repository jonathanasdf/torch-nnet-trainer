--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'paths'
require 'utils'
local argcheck = require 'argcheck'

local M = torch.class('Model')

local initcheck = argcheck{
  pack=true,
  {name="backend",
  type="string",
  help="Options: cudnn | ccn2 | cunn | nn",
  default = "cudnn"},
  
  {name="gpu",
  type="string",
  help="Comma-separated list of GPUs to use",
  default = ""},

  {name="nGPU",
  type="number",
  help="Number of GPUs to use. Ignored if gpu is set above.",
  default = 4}
}

function M:__init(...)
  -- argcheck
  local args = initcheck(...)
  self.backend = args.backend
  self.gpu = args.gpu:split(',')
  if tablelength(self.gpu) == 0 then
    for i=1, args.nGPU do
      table.insert(self.gpu, i)
    end
  end
  self.nGPU = tablelength(self.gpu)
  cutorch.setDevice(self.gpu[1])

  if self.nGPU == 0 and self.backend ~= 'nn' then
    print('\27[31mWarning: Only the nn backend can be used with no GPUs. Switching backend to nn.\27[0m')
    self.backend = 'nn'
  end
end



function M:create(path)
  assert(paths.filep(path), 'File not found: ' .. path)
  print('Creating model from file: ' .. path)
  gpu = self.gpu --make variable visible for dofile below
  self.model = paths.dofile(path)
  if self.backend == 'cudnn' then
    cudnn.convert(self.model, cudnn)
  elseif self.backend ~= 'nn' then
    error('Unsupported backend')
  end
  print('=> Model')
  print(self.model)
end

function M:load(path)
  if paths.extname(path) == "lua" then
    self:create(path)
    return
  end
  assert(paths.filep(path), 'File not found: ' .. path)
  print('Loading model from file: ' .. path)
  self.model = self:loadDataParallel(path)
  print('=> Model')
  print(self.model)
end

function M:forward(input, deterministic)
  if deterministic then
    self.model:evaluate()
  else
    self.model:training()
  end
  if self.nGPU > 0 then
    input = input:cuda()
  end
  return self.model:forward(input)
end

function M:getParameters()
  return self.model:getParameters()
end
   
function M:zeroGradParameters()
  self.model:zeroGradParameters()
end

function M:backward(inputs, gradOutputs)
  return self.model:backward(inputs, gradOutputs)
end



function makeDataParallel(model, gpu)
  if tablelength(gpu) > 1 then
    local device = cutorch.getDevice()
    print('converting model to nn.DataParallelTable')
    local model_single = model
    model = nn.DataParallelTable(1)
    for i,g in ipairs(gpu) do
      cutorch.setDevice(g)
      model:add(model_single:clone():cuda(), g)
    end
    cutorch.setDevice(device)
  end
  return model
end

function M:loadDataParallel(filename)
  local model
  if paths.extname(filename) == 'caffemodel' then
    require 'loadcaffe'
    model = loadcaffe.load(paths.dirname(filename) .. '/deploy.prototxt', filename, self.backend)
  else
    model = torch.load(filename)
  end

  if torch.type(model) == 'nn.DataParallelTable' then
    return makeDataParallel(model:get(1):float(), self.gpu)
  elseif torch.type(model) == 'nn.Sequential' then
    for i,module in ipairs(model.modules) do
      if torch.type(module) == 'nn.DataParallelTable' then
        model.modules[i] = makeDataParallel(module:get(1):float(), self.gpu)
      end
    end
    if self.nGPU > 0 then
      model = model:cuda()
    end
    return model
  else
    error('The loaded model is not a Sequential or DataParallelTable module.')
  end
end

local function cleanDPT(module, gpu)
  -- This assumes this DPT was created by the function above: all the
  -- module.modules are clones of the same network on different GPUs
  -- hence we only need to keep one when saving the model to the disk.
  cutorch.setDevice(gpu)
  local newDPT = nn.DataParallelTable(1)
  newDPT:add(module:get(1), gpu)
  return newDPT
end

function M:saveDataParallel(filename)
  self.model:clearState()
  collectgarbage()
  if torch.type(self.model) == 'nn.DataParallelTable' then
    torch.save(filename, cleanDPT(self.model, self.gpu[1]))
  elseif torch.type(self.model) == 'nn.Sequential' then
    local temp_model = nn.Sequential()
    for i, module in ipairs(self.model.modules) do
      if torch.type(module) == 'nn.DataParallelTable' then
        temp_model:add(cleanDPT(module, self.gpu[1]))
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
