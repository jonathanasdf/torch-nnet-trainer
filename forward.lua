package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'model'
require 'dataLoader'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to load')
cmd:argument('-input', 'input file or folder')
defineBaseOptions(cmd)     --defined in utils.lua
processArgs(cmd)

assert(paths.filep(opts.model), 'Cannot find model ' .. opts.model)

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
  bind_post(opts.processor.preprocessFn, false),
  opts.processor.test,
  accResults)
opts.processor:printStats()
