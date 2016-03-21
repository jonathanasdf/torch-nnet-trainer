package.path = package.path .. ';/home/jshen/scripts/?.lua'
torch.setdefaulttensortype('torch.FloatTensor')

require 'model'
require 'dataLoader'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to load')
cmd:argument('-input', 'input file or folder')
defineBaseOptions(cmd)     --defined in utils.lua

local opt = processArgs(cmd)
assert(paths.filep(opt.model), 'Cannot find model ' .. opt.model)

local model = Model(opt.model)
opt.processor.model = model

opt.processor:resetStats()
DataLoader{path = opt.input}:runAsync(
  opt.batchSize,
  opt.epochSize,
  true,          -- shuffle,
  bind_post(opt.processor.preprocessFn, false),
  bind(opt.processor.testBatch, opt.processor))
opt.processor:printStats()
