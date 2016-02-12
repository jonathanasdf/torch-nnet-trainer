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
local argcheck = require 'argcheck'

local M = torch.class('model')

local initcheck = argcheck{
  {name="backend",
   type="string",
   help="Options: cudnn | ccn2 | cunn",
   default = "cudnn"},
}

function M:__init(...)
  -- argcheck
  local args = initcheck(...)
  for k,v in pairs(args) do self[k] = v end  

  self.GPU = 1
  self.nGPU = 4
end

local makeDataParallel, loadDataParallel

function M:create(path)
   assert(paths.filep(path), 'File not found: ' .. path)
   print('Creating model from file: ' .. path)
   self.model = paths.dofile(path)
   if self.opt.backend == 'cudnn' then
      cudnn.convert(model, cudnn)
   elseif self.opt.backend ~= 'nn' then
      error'Unsupported backend'
   end
   self.model = makeDataParallel(self.model, self.nGPU)
   print('=> Model')
   print(self.model)
end

function M:load(path)
   assert(paths.filep(path), 'File not found: ' .. path)
   print('Loading model from file: ' .. path)
   self.model = loadDataParallel(path, self.nGPU)
   print('=> Model')
   print(self.model)
end

function M:test(input, resultHandler, ...)
   self.model:evaluate()
   resultHandler(self.model:forward(input:cuda()), ...)
end



function makeDataParallel(model, nGPU)
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
   cutorch.setDevice(1)
   return model
end

local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTable(1)
   cutorch.setDevice(module.GPU)
   newDPT:add(module:get(1), module.GPU)
   return newDPT
end

local function saveDataParallel(filename, model)
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

function loadDataParallel(filename, nGPU)
   local model
   if paths.extname(filename) == 'caffemodel' then
     require 'loadcaffe'
     model = loadcaffe.load(paths.dirname(filename) .. '/deploy.prototxt', filename, opt.backend)
   else
     model = torch.load(filename)
   end

   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1):float(), nGPU)
   elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
         end
      end
      return model:cuda()
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

return M
