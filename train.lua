package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'cutorch'
require 'cudnn'
require 'model'
require 'paths'
require 'optim'
require 'dataLoader'
require 'utils'

torch.setdefaulttensortype('torch.FloatTensor')

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
assert(paths.filep(opt.model), "Cannot find model " .. opt.model)

local loader = DataLoader{
  path = opt.input,
  preprocessor = opt.processor.preprocess,
  verbose = true
}

local model = Model{gpu=opt.gpu, nGPU=opt.nGPU}:load(opt.model)

local function updates(processor, model, paths, inputs)
  local parameters, grad_parameters = model:getParameters()
  return function(x)
    grad_parameters:zero()
    local outputs = model:forward(inputs)
    local loss, grad_outputs = processor:processBatch(paths, outputs)
    model:backward(inputs, grad_outputs)
    return loss, grad_parameters
  end
end

model:train(loader, opt, opt.processor.updates or bind(updates, opt.processor))
model:saveDataParallel(opt.output)
