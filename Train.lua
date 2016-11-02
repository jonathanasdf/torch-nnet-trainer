package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'Model'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained model')
defineBaseOptions(cmd)      -- defined in Utils.lua
defineTrainingOptions(cmd)  -- defined in Utils.lua
processArgs(cmd)

local model = Model(opts.model)
model:train()
print("Done!\n")
