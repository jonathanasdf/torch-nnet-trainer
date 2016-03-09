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
--   -trainBatch(model, pathNames, inputs): custom function for training. Propagate gradients back through model and return loss

local opt = processArgs(cmd)
assert(paths.filep(opt.model), 'Cannot find model ' .. opt.model)

local model = Model(opt.model)
opt.processor.model = model

local function trainBatch(model, pathNames, inputs)
  local loss, grad_outputs = opt.processor:evaluateBatch(pathNames, model:forward(inputs))
  model:backward(inputs, grad_outputs)
  return loss
end

model:train(opt, opt.processor.trainBatch or trainBatch)
