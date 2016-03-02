function defineBaseOptions(cmd)
  cmd:argument(
    '-processor',
    'REQUIRED. lua file that preprocesses input and handles output. '
    .. 'Functions that can be defined:\n'
    .. '    -setOptions(opt): Passes command line options to the processor\n'
    .. '    -preprocess(img): takes a single img and prepares it for the network\n'
    .. '    -processBatch(paths, outputs): returns [loss, grad_outputs]\n'
  )
  cmd:option('-processor_opts', '', 'additional options for the processor')
  cmd:option('-nThreads', 8, 'number of threads')
  cmd:option('-gpu', '', 'comma separated list of GPUs to use')
  cmd:option('-nGPU', 4, 'number of GPU to use. Ignored if gpu is set')
  cmd:option('-batchSize', 32, 'batch size')
  cmd:option('-cache_every', 20, 'save model every n epochs')
end

function defineTrainingOptions(cmd)
  cmd:option('-LR', 0.001, 'learning rate')
  cmd:option('-momentum', 0.9, 'momentum')
  cmd:option('-epochs', 50, 'num epochs')
  cmd:option('-epochSize', -1, 'num batches per epochs')
  cmd:option('-val', '', 'validation data')
  cmd:option('-valSize', -1, 'num batches to validate')
  cmd:option('-val_every', 20, 'run validation every n epochs')
  cmd:option('-optimState', '', 'optimState to resume from')
end

function processArgs(cmd)
  local opt = cmd:parse(arg or {})
  if opt.processor == '' then
    error('A processor must be supplied.')
  end
  opt.processor = dofile(opt.processor).new(opt)

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

  return opt
end

function setJobs(n)
  numJobs = n
  jobsDone = 0
  xlua.progress(jobsDone, numJobs)
end

function jobDone()
  jobsDone = jobsDone + 1
  xlua.progress(jobsDone, numJobs)
end

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

function imageTableToTensor(T) -- Assumes 3 channels
  for k, v in pairs(T) do
    T[k] = v:reshape(1, v:size(1), v:size(2), v:size(3))
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
