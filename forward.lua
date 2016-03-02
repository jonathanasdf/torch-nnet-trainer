package.path = package.path .. ';/home/jshen/scripts/?.lua'
torch.setdefaulttensortype('torch.FloatTensor')

require 'cutorch'
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

local model = Model{gpu=opt.gpu, nGPU=opt.nGPU}:load(opt.model)

local function testBatch(paths, inputs)
  opt.processor:processBatch(paths, model:forward(inputs, true), true)
end

DataLoader{
  path = opt.input,
  preprocessor = opt.processor.preprocess,
  verbose = true
}:runAsync(opt.batchSize, 
           -1,          -- epochSize
           false,       -- don't shuffle,
           testBatch)   -- resultHandler

opt.processor:printStats()
