function defineBaseOptions(cmd)
  cmd:argument(
    '-processor',
    'REQUIRED. lua file that preprocesses input and handles output. '
    .. 'Functions that can be defined:\n'
    .. '    -preprocess(img, opt): takes a single img and prepares it for the network\n'
    .. '    -evaluateBatch(pathNames, outputs): returns (grad_outputs, loss, #correct)\n'
  )
  cmd:option('-processor_opts', '', 'additional options for the processor')
  cmd:option('-batchSize', 32, 'batch size')
  cmd:option('-epochSize', -1, 'num batches per epochs. -1 means run all available data once')
  cmd:option('-nThreads', 8, 'number of threads')
  cmd:option('-nGPU', 1, 'number of GPU to use. Set to 0 to use CPU')
end

function defineTrainingOptions(cmd)
  cmd:option('-LR', 0.001, 'learning rate')
  cmd:option('-momentum', 0.9, 'momentum')
  cmd:option('-weightDecay', 0.0005, 'weight decay')
  cmd:option('-epochs', 50, 'num epochs')
  cmd:option('-update_every', 1, 'update model with sgd every n batches')
  cmd:option('-cache_every', 20, 'save model every n epochs. Set to -1 or a value >epochs to disable')
  cmd:option('-val', '', 'validation data')
  cmd:option('-valSize', -1, 'num batches to validate. -1 means run all available data once')
  cmd:option('-val_every', 20, 'run validation every n epochs')
  cmd:option('-noUseDataParallelTable', false, 'dont create model using DataParallelTable (only applies to .lua models)')
  cmd:option('-optimState', '', 'optimState to resume from')
end

function processArgs(cmd)
  local opt = cmd:parse(arg or {})
  nGPU = opt.nGPU
  noUseDataParallelTable = opt.noUseDataParallelTable

  if not opt.update_every then opt.update_every = 1 end
  opt.batchCount = opt.batchSize * opt.update_every

  torch.setnumthreads(opt.nThreads)
  local Threads = require 'threads'
  Threads.serialization('threads.sharedserialize')
  threads = Threads(
    opt.nThreads,
    function()
      package.path = package.path .. ';/home/jshen/scripts/?.lua'
      package.path = package.path .. ';/home/nvesdapu/opencv/?.lua'
    end,
    function()
      torch.setdefaulttensortype('torch.FloatTensor')
      require 'cunn'
      require 'cudnn'
      cv = require 'cv'
      require 'cv.imgcodecs'
      require 'dpnn'
      require 'fbnn'
      require 'image'
      require 'paths'
      require 'utils'
    end
  )

  if opt.processor == '' then
    error('A processor must be supplied.')
  end
  opt.processor = requirePath(opt.processor)(opt)

  return opt
end

function requirePath(path)
  package.path = package.path .. ';' .. paths.dirname(path) .. '/?.lua'
  return require(paths.basename(path, 'lua'))
end

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

function tableToBatchTensor(T) -- Assumes 3 dimensions
  for k, v in pairs(T) do
    local sz = v:size():totable()
    table.insert(sz, 1, 1)
    T[k] = v:view(torch.LongStorage(sz))
  end
  return torch.cat(T, 1)
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

function convertTensorToSVMLight(labels, tensor, append)
  assert(tensor:size(1) == labels:nElement())
  local data = append or {}
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

local class = require 'class'
function findModuleByName(model, name)
  if class.istype(model, 'Model') then
    return findModuleByName(model.model, name)
  end
  for i=1,#model.modules do
    if model.modules[i].name == name then
      return model.modules[i]
    end
  end
  return nil
end

function printOutputSizes(model)
  if class.istype(model, 'Model') then
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
  local function closure(input)
    return f(input, arg)
  end
  return closure
end
