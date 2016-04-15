package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'paths'
require 'Model'
require 'Utils'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained model')
--defined in utils.lua
defineBaseOptions(cmd)
defineTrainingOptions(cmd)
processArgs(cmd)

assert(paths.filep(opts.model), 'Cannot find model ' .. opts.model)

local model = Model(opts.model)
opts.processor.model = model
opts.processor:initializeThreads()
model:train()
print("Done!\n")
