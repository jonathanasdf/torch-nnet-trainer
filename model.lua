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
  help="Options: cudnn | ccn2 | cunn",
  default = "cudnn"},
  
  {name="nGPU",
  type="number",
  help="Number of GPUs to use",
  default = 4},

  {name="gpu",
  type="number",
  help="Default GPU to use",
  default = 1}
}

function M:__init(...)
  -- argcheck
  local args = initcheck(...)
  for k,v in pairs(args) do self[k] = v end  
end



function M:create(path)
  assert(paths.filep(path), 'File not found: ' .. path)
  print('Creating model from file: ' .. path)
  nGPU = self.nGPU
  self.model = paths.dofile(path)
  if self.backend == 'cudnn' then
    cudnn.convert(self.model, cudnn)
  elseif self.backend ~= 'nn' then
    error'Unsupported backend'
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
  return self.model:forward(input:cuda())
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



function makeDataParallel(model, nGPU)
  device = cutorch.getDevice()
  if nGPU > 1 then
    print('converting model to nn.DataParallelTable')
    assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
    local model_single = model
    model = nn.DataParallelTable(1)
    for i=1, nGPU do
      cutorch.setDevice(i)
      model:add(model_single:clone():cuda(), i)
    end
  end
  cutorch.setDevice(device)
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
    return makeDataParallel(model:get(1):float(), self.nGPU)
  elseif torch.type(model) == 'nn.Sequential' then
    for i,module in ipairs(model.modules) do
      if torch.type(module) == 'nn.DataParallelTable' then
        model.modules[i] = makeDataParallel(module:get(1):float(), self.nGPU)
      end
    end
    return model:cuda()
  else
    error('The loaded model is not a Sequential or DataParallelTable module.')
  end
end

local function cleanDPT(module)
  -- This assumes this DPT was created by the function above: all the
  -- module.modules are clones of the same network on different GPUs
  -- hence we only need to keep one when saving the model to the disk.
  local newDPT = nn.DataParallelTable(1)
  cutorch.setDevice(module.gpu)
  newDPT:add(module:get(1), module.gpu)
  return newDPT
end

function saveDataParallel(filename, model)
  collectgarbage()
  if torch.type(model) == 'nn.DataParallelTable' then
    torch.save(filename, cleanDPT(model))
  elseif torch.type(model) == 'nn.Sequential' then
    local temp_model = nn.Sequential()
    for i, module in ipairs(model.modules) do
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
