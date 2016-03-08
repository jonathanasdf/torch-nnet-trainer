package.path = package.path .. ';/home/jshen/scripts/?.lua'
torch.setdefaulttensortype('torch.FloatTensor')

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
--   -updates(model, pathNames, inputs): custom updates function for optim

local opt = processArgs(cmd)
assert(paths.filep(opt.model), 'Cannot find model ' .. opt.model)

local model = Model(opt.model)
opt.processor.model = model

local function updates(model, pathNames, inputs)
  local loss, grad_outputs = opt.processor:processBatch(pathNames, model:forward(inputs))
  model:backward(inputs, grad_outputs)
  return function(x)
    return loss, model.gradParameters
  end
end

model:train(opt, opt.processor.updates or updates)
model:save(opt.output)
