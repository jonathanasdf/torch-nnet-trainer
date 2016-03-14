package.path = package.path .. ';/home/jshen/scripts/?.lua'
torch.setdefaulttensortype('torch.FloatTensor')

require 'model'
require 'dataLoader'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to load')
cmd:argument('-input', 'input file or folder')
defineBaseOptions(cmd)     --defined in utils.lua
-- Additional processor functions:
--   -testBatch(pathNames, outputs): accumulate statistics
--   -printStats(): outputs statistics

local opt = processArgs(cmd)
assert(paths.filep(opt.model), 'Cannot find model ' .. opt.model)

local model = Model(opt.model)
opt.processor.model = model

local function testBatch(pathNames, inputs)
  opt.processor:testBatch(pathNames, model:forward(inputs, true))
end

DataLoader{path = opt.input}:runAsync(
  opt.batchSize,
  opt.epochSize,
  true,          -- shuffle,
  opt.processor.preprocess,
  testBatch)     -- resultHandler

opt.processor:printStats()
