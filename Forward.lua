package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'Model'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to load')
cmd:argument('-input', 'input file or folder')
defineBaseOptions(cmd)  -- defined in Utils.lua
cmd:option('-noshuffle', false, 'if true then forward in order')
cmd:option('-cacheEvery', 10, 'save forwarding stats every n batches')
cmd:option('-resume', '', 'resume forwarding from saved state. Command must be identical')
processArgs(cmd)
setPhase('test')

local model = Model(opts.model)
model.processor:resetStats()

local dataloader = DataLoader{inputs = opts.input, randomize = not(opts.noshuffle)}
local state = {shuffle = dataloader.shuffle, completed = 0}
if opts.resume ~= '' then
  print("Resuming from cache file: " .. opts.resume)
  opts.cacheFile = opts.resume
  state = torch.load(opts.resume)
  dataloader.shuffle = state.shuffle
  model.processor.stats = state.stats
else
  opts.cacheFile = os.tmpname()
  print("Cache file: " .. opts.cacheFile)
end

local function accResults(loss, cnt)
  state.completed = state.completed + 1
  if opts.cacheEvery ~= -1 and state.completed % opts.cacheEvery == 0 then
    state.stats = model.processor.stats
    torch.save(opts.cacheFile, state)
  end
end

model:run(dataloader,
          opts.batchSize,
          opts.epochSize,
          false,               -- randomSample,
          bind(model.processor.test, model.processor),
          accResults,
          state.completed + 1) -- startBatch
if opts.cacheEvery ~= -1 then
  state.stats = model.processor.stats
  torch.save(opts.cacheFile, state)
end
print(model.processor:getStats())
print()
