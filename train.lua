package.path = package.path .. ';/home/jshen/scripts/?.lua'
torch.setdefaulttensortype('torch.FloatTensor')

require 'cutorch'
require 'cudnn'
require 'model'
require 'paths'
require 'utils'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained model')
--defined in utils.lua
defineBaseOptions(cmd)
defineTrainingOptions(cmd)
-- Additional processor functions:
--   -updates(model, paths, inputs): custom updates function for optim

local opt = processArgs(cmd)
assert(paths.filep(opt.model), 'Cannot find model ' .. opt.model)

local model = Model(opt.model)

local function updates(model, paths, inputs)
  local loss, grad_outputs = opt.processor:processBatch(paths, model:forward(inputs))
  model:zeroGradParameters()
  model:backward(inputs, grad_outputs)
  return function(x)
    return loss, model.gradParameters
  end
end

model:train(opt, opt.processor.updates or updates)
model:save(opt.output)
