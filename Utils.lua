function defineBaseOptions(cmd)
  cmd:argument(
    '-processor',
    'REQUIRED. lua file that does the heavy lifting. ' ..
    'See processor.lua for functions that can be defined.\n'
  )
  cmd:option('-processorOpts', '', 'additional options for the processor')
  cmd:option('-batchSize', 32, 'batch size')
  cmd:option('-epochSize', -1, 'num batches per epochs. -1 means run all available data once')
  cmd:option('-dropout', 0.5, 'dropout probability')
  cmd:option('-nThreads', 4, 'number of worker threads')
  cmd:option('-replicateModel', false, 'Replicate model across threads? Speeds up everything, but takes more memory')
  cmd:option('-nGPU', 1, 'number of GPU to use. Set to -1 to use CPU')
end

function defineTrainingOptions(cmd)
  cmd:option('-LR', 0.001, 'learning rate')
  cmd:option('-momentum', 0.9, 'momentum')
  cmd:option('-weightDecay', 0.0005, 'weight decay')
  cmd:option('-inputWeights', '', 'comma separated weights to balance input classes for each batch')
  cmd:option('-epochs', 50, 'num epochs')
  cmd:option('-updateEvery', 1, 'update model with sgd every n batches')
  cmd:option('-cacheEvery', 20, 'save model every n epochs. Set to -1 or a value >epochs to disable')
  cmd:option('-val', '', 'validation data')
  cmd:option('-valBatchSize', -1, 'batch size for validation')
  cmd:option('-valSize', -1, 'num batches to validate. -1 means run all available data once')
  cmd:option('-valEvery', 1, 'run validation every n epochs')
  cmd:option('-optimState', '', 'optimState to resume from')
end

function processArgs(cmd)
  torch.setdefaulttensortype('torch.FloatTensor')
  opts = cmd:parse(arg or {})
  if opts.nGPU == 0 then
    error('nGPU should not be 0. Please set nGPU to -1 if you want to use CPU.')
  end
  if opts.nGPU == -1 then opts.nGPU = 0 end
  if opts.nThreads < opts.nGPU-1 then
    if opts.nThreads == 0 then
      print('Not enough threads to use all gpus. Setting nGPU=1 since nThreads=0.')
      opts.nGPU = 1
    else
      opts.nThreads = opts.nGPU-1
      print('Not enough threads to use all gpus. Increasing nThreads to ' .. opts.nThreads)
    end
  end
  if opts.val and opts.val ~= '' then
    if opts.valBatchSize == -1 then
      opts.valBatchSize = opts.batchSize
    end
  end

  nGPU = opts.nGPU
  nThreads = opts.nThreads

  if not opts.updateEvery then opts.updateEvery = 1 end
  opts.batchCount = opts.batchSize * opts.updateEvery

  if opts.input then
    opts.input = opts.input:split(';')
  end
  if opts.val and opts.val ~= '' then
    opts.val = opts.val:split(';')
  end
  if opts.inputWeights then
    opts.inputWeights = opts.inputWeights:split(';')
    for i=1,#opts.inputWeights do
      opts.inputWeights[i] = tonumber(opts.inputWeights[i])
    end
  end

  if opts.output and opts.output ~= '' then
    opts.dirname = paths.dirname(opts.output) .. '/'
    opts.basename = paths.basename(opts.output, paths.extname(opts.output))
    opts.backupdir = opts.dirname .. 'backups/'
    paths.mkdir(opts.backupdir)
    opts.logdir = opts.dirname .. 'logs/' .. opts.basename .. os.date("_%Y%m%d_%H%M%S/")
    paths.mkdir(opts.logdir)
    cmd:log(opts.logdir .. 'log.txt', opts)
    cmd:addTime()

    opts.lossGraph = gnuplot.pngfigure(opts.logdir .. 'loss.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('loss')
    gnuplot.grid(true)
  end

  if opts.processor == '' then
    error('A processor must be supplied.')
  end
  local processorPath = opts.processor
  opts.processor = requirePath(opts.processor).new()

  if opts.nThreads > 0 then
    local opt = opts
    torch.setnumthreads(opts.nThreads)
    local Threads = require 'threads'
    Threads.serialization('threads.sharedserialize')
    threads = Threads(opts.nThreads,
      function()
        package.path = package.path .. ';/home/jshen/scripts/?.lua'
        package.path = package.path .. ';/home/nvesdapu/opencv/?.lua'
      end,
      function()
        torch.setdefaulttensortype('torch.FloatTensor')
        require 'cudnn'
        require 'cunn'
        cv = require 'cv'
        require 'cv.cudawarping'
        require 'cv.imgcodecs'
        require 'dpnn'
        require 'draw'
        require 'fbnn'
        require 'gnuplot'
        require 'image'
        require 'optim'
        require 'paths'

        require 'Model'
        require 'Utils'
        requirePath(processorPath)

        local min = math.min
        local max = math.max
        local floor = math.floor
        local ceil = math.ceil
      end,
      function()
        cudnn.benchmark = true
        opts = opt
        nGPU = opts.nGPU
        nThreads = opts.nThreads
      end
    )
  else
    threads = {}
    threads.addjob = function(self, f1, f2, ...)
      local r = {f1(...)}
      if f2 then f2(unpack(r)) end
    end
    threads.synchronize = function() end
  end
end

function augmentThreadState(...)
  if nThreads == 0 then
    return
  end
  local specific = threads:specific()
  threads:specific(true)
  local funcs = {...}
  for j=1,#funcs do
    for i=1,nThreads do
      threads:addjob(i, funcs[j])
    end
  end
  threads:specific(specific)
end

local jobsDone, jobSize
function startJobs(count)
  jobsDone = 0
  jobSize = count
  xlua.progress(jobsDone, jobSize)
end

function jobDone()
  jobsDone = jobsDone + 1
  xlua.progress(jobsDone, jobSize)
end

function requirePath(path)
  local oldPackagePath = package.path
  package.path = paths.dirname(path) .. '/?.lua;' .. package.path
  local M = require(paths.basename(path, 'lua'))
  package.path = oldPackagePath
  return M
end

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

function cat(T, dim)
  local out = torch.Tensor():typeAs(T[1])
  torch.cat(out, T, dim)
  return out
end

-- Concatenates a table of tensors **of the same size** along a new dimension at the front
function tableToBatchTensor(T)
  for i=1,#T do
    T[i] = nn.utils.addSingletonDimension(T[i])
  end
  return cat(T, 1)
end

local RGBBGR = torch.LongTensor{3,2,1}
function convertRGBBGR(T, dim)
  if not dim then dim = 1 end
  return T:index(dim, RGBBGR)
end

function tensorToCVImg(T)
  if T:dim() == 2 then
    return (T*255):byte()
  elseif T:dim() == 3 then
    return (T:index(1, RGBBGR):permute(2, 3, 1)*255):byte()
  else
    error('2d or 3d tensor expected')
  end
end

function cvImgToTensor(I)
  if I:dim() == 2 then
    return I:float() / 255.0
  elseif I:dim() == 3 then
    return I:index(3, RGBBGR):permute(3, 1, 2):float() / 255.0
  else
    error('2d or 3d image expected')
  end
end

function convertTensorToSVMLight(labels, tensor)
  assert(tensor:size(1) == labels:nElement())
  local data = {}
  for i=1,tensor:size(1) do
    local values = tensor[i]:view(-1):float()
    local indices = torch.range(1, values:nElement()):int()
    data[#data+1] = {labels[i], {indices, values}}
  end
  return data
end

function string:split(sep)
  sep = sep or ','
  local fields = {}
  local pattern = string.format('([^%s]+)', sep)
  self:gsub(pattern, function(c) fields[#fields+1] = c end)
  return fields
end

function findModuleByName(model, name)
  if model.modules then
    for i=1,#model.modules do
      local recur = findModuleByName(model.modules[i], name)
      if recur then return recur end
      if model.modules[i].name == name then
        return model.modules[i]
      end
    end
  end
  return nil
end

function setDropout(model, p)
  if torch.isTypeOf(model, 'nn.Dropout') then
    model:setp(p)
  elseif model.modules then
    for i=1,#model.modules do
      setDropout(model.modules[i], p)
    end
  end
end

function printOutputSizes(model)
  if torch.isTypeOf(model, 'Model') then
    printOutputSizes(model.model)
    return
  end
  if model.output then
    print(model, #model.output)
  end
  if not(model.modules) then return end
  for i=1,#model.modules do
    printOutputSizes(model.modules[i])
  end
end

local va = require 'vararg'
function bind(f, ...)
  local outer = va(...)
  local function closure(...)
    return f(va.concat(outer, va(...)))
  end
  return closure
end

function bindPost(f, arg)
  local function closure(...)
    local args = {...}
    args[#args+1] = arg
    return f(unpack(args))
  end
  return closure
end
