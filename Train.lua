package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'paths'
require 'Model'
require 'Utils'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained model')
cmd:argument(
    '-processor',
    'REQUIRED. lua file that does the heavy lifting. ' ..
    'See processor.lua for functions that can be defined.\n'
  )
defineBaseOptions(cmd)     --defined in utils.lua
cmd:option('-processorOpts', '', 'additional options for the processor')
defineTrainingOptions(cmd)
processArgs(cmd)

assert(paths.filep(opts.model), 'Cannot find model ' .. opts.model)
if opts.nThreads > 1 then
  error('There is currently a bug with nThreads > 1.')
end


local model = Model(opts.model)
local processor = requirePath(opts.processor).new(model, opts.processorOpts)
processor:initializeThreads()
model:train()
print("Done!\n")
