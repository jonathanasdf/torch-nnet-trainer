package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'DataLoader'
require 'Model'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to load')
cmd:argument('-input', 'input file or folder')
defineBaseOptions(cmd)     --defined in utils.lua
cmd:option('-cacheEvery', 20, 'save forwarding stats every n batches')
cmd:option('-resume', '', 'resume forwarding from saved state. Command must be identical')
processArgs(cmd)

assert(paths.filep(opts.model), 'Cannot find model ' .. opts.model)

local model = Model(opts.model)
opts.processor.model = model
opts.processor:initializeThreads()
opts.processor:resetStats()
local loader = DataLoader{inputs = opts.input, randomize = true}
local state = {shuffle = loader.shuffle, completed = 0}
opts.cacheFile = os.tmpname()
if opts.resume ~= '' then
  opts.cacheFile = opts.resume
  state = torch.load(opts.resume)
  loader.shuffle = state.shuffle
  opts.processor.stats = state.stats
end
print("Cache file: " .. opts.cacheFile)

local function accResults(loss, cnt, stats)
  opts.processor:accStats(stats)
  jobDone()

  state.completed = state.completed + 1
  if opts.cacheEvery ~= -1 and state.completed % opts.cacheEvery == 0 then
    state.stats = opts.processor.stats
    torch.save(opts.cacheFile, state)
  end
end

loader:runAsync(
  opts.batchSize,
  opts.epochSize,
  false,               -- randomSample,
  bindPost(opts.processor.preprocessFn, false),
  opts.processor.test,
  accResults,
  state.completed + 1) -- startBatch
print(opts.processor:processStats())
