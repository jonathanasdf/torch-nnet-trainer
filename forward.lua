package.path = package.path .. ';/home/jshen/scripts/?.lua'
torch.setdefaulttensortype('torch.FloatTensor')

require 'model'
require 'dataLoader'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to load')
cmd:argument('-input', 'input file or folder')
defineBaseOptions(cmd)     --defined in utils.lua
-- Additional processor functions:
--   -printStats(): outputs statistics

local opt = processArgs(cmd)
assert(paths.filep(opt.model), 'Cannot find model ' .. opt.model)

local model = Model(opt.model)
opt.processor.model = model

local function testBatch(pathNames, inputs)
  opt.processor:processBatch(pathNames, model:forward(inputs, true), true)
end

DataLoader{
  path = opt.input,
  preprocessor = opt.processor.preprocess
}:runAsync(opt.batchSize,
           opt.epochSize,
           true,          -- shuffle,
           testBatch)     -- resultHandler

opt.processor:printStats()
