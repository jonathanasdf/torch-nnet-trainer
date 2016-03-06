function defineBaseOptions(cmd)
  cmd:argument(
    '-processor',
    'REQUIRED. lua file that preprocesses input and handles output. '
    .. 'Functions that can be defined:\n'
    .. '    -preprocess(img): takes a single img and prepares it for the network\n'
    .. '    -processBatch(paths, outputs, calculateStats): returns [loss, grad_outputs]\n'
  )
  cmd:option('-processor_opts', '', 'additional options for the processor')
  cmd:option('-batchSize', 32, 'batch size')
  cmd:option('-nThreads', 8, 'number of threads')
  cmd:option('-nGPU', 4, 'number of GPU to use. Set to 0 to use CPU')
  cmd:option('-gpu', '', 'comma separated list of GPUs to use. overrides nGPU if set')
end

function defineTrainingOptions(cmd)
  cmd:option('-LR', 0.001, 'learning rate')
  cmd:option('-momentum', 0.9, 'momentum')
  cmd:option('-epochs', 50, 'num epochs')
  cmd:option('-epochSize', -1, 'num batches per epochs')
  cmd:option('-cache_every', 20, 'save model every n epochs')
  cmd:option('-val', '', 'validation data')
  cmd:option('-valSize', -1, 'num batches to validate')
  cmd:option('-val_every', 20, 'run validation every n epochs')
  cmd:option('-optimState', '', 'optimState to resume from')
end

function processArgs(cmd)
  local opt = cmd:parse(arg or {})

  gpu = opt.gpu:split(',')
  if tablelength(gpu) == 0 then
    for i=1, opt.nGPU do
      table.insert(gpu, i)
    end
  end
  nGPU = tablelength(gpu)

  local Threads = require 'threads'
  Threads.serialization('threads.sharedserialize')
  threads = Threads(
    opt.nThreads,
    function()
      require 'torch'
      require 'cutorch'
      require 'cunn'
      require 'cudnn'
      require 'fbnn'
      require 'image'
      require 'paths'
      require 'utils'
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  )

  if opt.processor == '' then
    error('A processor must be supplied.')
  end
  opt.processor = dofile(opt.processor).new(opt)

  return opt
end

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

function tableToBatchTensor(T) -- Assumes 3 dimensions
  for k, v in pairs(T) do
    local sz = torch.totable(v:size())
    table.insert(sz, 1, 1)
    T[k] = v:view(torch.LongStorage(sz))
  end
  return torch.cat(T, 1)
end

function string:split(sep)
  local sep, fields = sep or ',', {}
  local pattern = string.format('([^%s]+)', sep)
  self:gsub(pattern, function(c) fields[#fields+1] = c end)
  return fields
end

function findModuleByName(model, name)
  for i=1,#model.modules do
    if model.modules[i].name == name then
      return model.modules[i]
    end
  end
  return nil
end

local va = require 'vararg'
function bind(f, ...)
  local outer_args = va(...)
  local function closure (...)
    return f(va.concat(outer_args, va(...)))
  end
  return closure
end
