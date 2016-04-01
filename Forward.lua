package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'DataLoader'
require 'Model'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to load')
cmd:argument('-input', 'input file or folder')
defineBaseOptions(cmd)     --defined in utils.lua
cmd:option('-seed', '-1', 'random seed')
processArgs(cmd)

assert(paths.filep(opts.model), 'Cannot find model ' .. opts.model)

if opts.seed == -1 then
  opts.seed = torch.random()
end
torch.manualSeed(opts.seed)
cutorch.manualSeed(opts.seed)
augmentThreadState(function()
  cutorch.manualSeed(opts.seed + __threadid)
end)
print("Seed = " .. opts.seed)

local model = Model(opts.model)
opts.processor.model = model
opts.processor:initializeThreads()

local function accResults(loss, cnt, ...)
  opts.processor:accStats(...)
  jobDone()
end

opts.processor:resetStats()
DataLoader{path = opts.input}:runAsync(
  opts.batchSize,
  opts.epochSize,
  true,          -- shuffle,
  bindPost(opts.processor.preprocessFn, false),
  opts.processor.test,
  accResults)
opts.processor:printStats()
