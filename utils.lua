function defineBaseOptions(cmd)
  cmd:argument(
    '-processor',
    'REQUIRED. lua file that does the heavy lifting. ' ..
    'See processor.lua for functions that can be defined.\n'
  )
  cmd:option('-processor_opts', '', 'additional options for the processor')
  cmd:option('-batchSize', 32, 'batch size')
  cmd:option('-epochSize', -1, 'num batches per epochs. -1 means run all available data once')
  cmd:option('-nThreads', 4, 'number of worker threads')
  cmd:option('-replicateModel', false, 'Replicate model across threads? Speeds up everything, but takes more memory')
  cmd:option('-nGPU', 1, 'number of GPU to use. Set to -1 to use CPU')
end

function defineTrainingOptions(cmd)
  cmd:option('-LR', 0.001, 'learning rate')
  cmd:option('-momentum', 0.9, 'momentum')
  cmd:option('-weightDecay', 0.0005, 'weight decay')
  cmd:option('-epochs', 50, 'num epochs')
  cmd:option('-update_every', 1, 'update model with sgd every n batches')
  cmd:option('-cache_every', 20, 'save model every n epochs. Set to -1 or a value >epochs to disable')
  cmd:option('-val', '', 'validation data')
  cmd:option('-valBatchSize', -1, 'batch size for validation')
  cmd:option('-valSize', -1, 'num batches to validate. -1 means run all available data once')
  cmd:option('-val_every', 20, 'run validation every n epochs')
  cmd:option('-optimState', '', 'optimState to resume from')
end

function processArgs(cmd)
  torch.setdefaulttensortype('torch.FloatTensor')
  opts = cmd:parse(arg or {})
  if opts.nGPU == 0 then
    error('nGPU should not be 0. Please set nGPU to -1 if you want to use CPU.')
  end
  if opts.nGPU == -1 then opts.nGPU = 0 end
  nGPU = opts.nGPU

  if opts.nThreads < math.max(1, opts.nGPU-1) then
    error('nThreads < max(1, nGPU-1).')
  end
  if opts.nThreads > math.max(1, opts.nGPU) then
    print('\27[31mThere is currently a bug when nThreads > nGPU. Setting nThreads to nGPU.\27[0m')
    opts.nThreads = math.max(1, opts.nGPU)
  end
  nThreads = opts.nThreads

  if opts.batchSize <= 1 then
    error('Sorry, this framework only supports batchSize > 1.')
  end
  if opts.valBatchSize then
    if opts.valBatchSize == -1 then
      opts.valBatchSize = opts.batchSize
    end
    if opts.valBatchSize <= 1 then
      error('Sorry, this framework only supports valBatchSize > 1.')
    end
  end

  if not opts.update_every then opts.update_every = 1 end
  opts.batchCount = opts.batchSize * opts.update_every
  if opts.LR then opts.LR = opts.LR / opts.batchCount end

  if opts.output and opts.output ~= '' then
    opts.basename = paths.dirname(opts.output) .. '/' .. paths.basename(opts.output, paths.extname(opts.output))
    opts.logdir = opts.basename .. os.date("_%Y%m%d_%H%M%S/")
    paths.mkdir(opts.logdir)
    cmd:log(opts.logdir .. 'log.txt', opts)
    cmd:addTime()
  end

  if opts.processor == '' then
    error('A processor must be supplied.')
  end
  local processor_path = opts.processor
  opts.processor = requirePath(opts.processor).new()

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
      require 'cunn'
      require 'cudnn'
      cv = require 'cv'
      require 'cv.cudawarping'
      require 'cv.imgcodecs'
      require 'dpnn'
      require 'fbnn'
      require 'image'
      require 'model'
      require 'paths'
      require 'utils'
      requirePath(processor_path)

      local min = math.min
      local max = math.max
      local floor = math.floor
      local ceil = math.ceil
    end,
    function()
      opts = opt
      nGPU = opts.nGPU
      nThreads = opts.nThreads
    end
  )
end

function augmentThreadState(...)
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
  package.path = paths.dirname(path) .. '/?.lua' .. ';' .. package.path
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

local RGB_BGR = torch.LongTensor{3,2,1}
function tensorToCVImg(T)
  if T:dim() == 2 then
    return (T*255):byte()
  elseif T:dim() == 3 then
    return (T:index(1, RGB_BGR):permute(2, 3, 1)*255):byte()
  else
    error('2d or 3d tensor expected')
  end
end

function cvImgToTensor(I)
  if I:dim() == 2 then
    return I:float() / 255.0
  elseif I:dim() == 3 then
    return I:index(3, RGB_BGR):permute(3, 1, 2):float() / 255.0
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
  local sep, fields = sep or ',', {}
  local pattern = string.format('([^%s]+)', sep)
  self:gsub(pattern, function(c) fields[#fields+1] = c end)
  return fields
end

function findModuleByName(model, name)
  if torch.isTypeOf(model, 'Model') then
    return findModuleByName(model.model, name)
  end
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
  local outer_args = va(...)
  local function closure(...)
    return f(va.concat(outer_args, va(...)))
  end
  return closure
end

function bind_post(f, arg)
  local function closure(...)
    local args = {...}
    args[#args+1] = arg
    return f(unpack(args))
  end
  return closure
end
