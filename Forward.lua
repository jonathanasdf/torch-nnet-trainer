package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'DataLoader'
require 'Model'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to load')
cmd:argument('-input', 'input file or folder')
cmd:argument(
    '-processor',
    'REQUIRED. lua file that does the heavy lifting. ' ..
    'See processor.lua for functions that can be defined.\n'
  )
defineBaseOptions(cmd)     --defined in utils.lua
cmd:option('-processorOpts', '', 'additional options for the processor')
cmd:option('-noshuffle', false, 'if true then forward in order')
cmd:option('-cacheEvery', 10, 'save forwarding stats every n batches')
cmd:option('-resume', '', 'resume forwarding from saved state. Command must be identical')
processArgs(cmd)
opts.testing = true

assert(paths.filep(opts.model), 'Cannot find model ' .. opts.model)

local model = Model(opts.model)
local processor = requirePath(opts.processor).new(model, opts.processorOpts)
processor:initializeThreads()
processor:resetStats()

local loader = DataLoader{inputs = opts.input, randomize = not(opts.noshuffle)}
local state = {shuffle = loader.shuffle, completed = 0}
if opts.resume ~= '' then
  print("Resuming from cache file: " .. opts.resume)
  opts.cacheFile = opts.resume
  state = torch.load(opts.resume)
  loader.shuffle = state.shuffle
  processor.stats = state.stats
else
  opts.cacheFile = os.tmpname()
  print("Cache file: " .. opts.cacheFile)
end

local function accResults(loss, cnt, stats)
  processor:accStats(stats)
  jobDone()

  state.completed = state.completed + 1
  if opts.cacheEvery ~= -1 and state.completed % opts.cacheEvery == 0 then
    state.stats = processor.stats
    torch.save(opts.cacheFile, state)
  end
end

loader:runAsync(
  opts.batchSize,
  opts.epochSize,
  false,               -- randomSample,
  bindPost(processor.preprocessFn, false),
  processor.test,
  accResults,
  state.completed + 1) -- startBatch
if opts.cacheEvery ~= -1 then
  state.stats = processor.stats
  torch.save(opts.cacheFile, state)
end
print(processor:processStats('test'))
print()
