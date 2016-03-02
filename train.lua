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

local model = Model{gpu=opt.gpu, nGPU=opt.nGPU}:load(opt.model)

local function updates(processor, model, paths, inputs)
  return function(x)
    local loss, grad_outputs = processor:processBatch(paths, model:forward(inputs))

    model:zeroGradParameters()
    model:backward(inputs, grad_outputs)
    return loss, model.gradParameters
  end
end

model:train(opt, opt.processor.updates or bind(updates, opt.processor))
model:saveDataParallel(opt.output)
