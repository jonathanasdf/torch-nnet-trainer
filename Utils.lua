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
  cmd:option('-replicateModel', false, 'Replicate model across threads')
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
  if opts.processor == '' then
    error('A processor must be supplied.')
  end

  opts.pid = require("posix").getpid("pid")
  print("Process", opts.pid, "started!")

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

  local opt = opts
  local setup = {
    function()
      package.path = package.path .. ';/home/jshen/scripts/?.lua'
      package.path = package.path .. ';/home/nvesdapu/scripts/?.lua'

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
    end,
    function()
      require 'features'
      require 'Model'
      require 'Utils'

      min = math.min
      max = math.max
      floor = math.floor
      ceil = math.ceil
    end,
    function()
      cudnn.benchmark = true
      opts = opt
      nGPU = opts.nGPU
      nThreads = opts.nThreads
      requirePath(opts.processor)
    end
  }

  if opts.nThreads > 0 then
    torch.setnumthreads(opts.nThreads)
    local Threads = require 'threads'
    Threads.serialization('threads.sharedserialize')
    threads = Threads(opts.nThreads, unpack(setup))
  else
    threads = {}
    threads.addjob = function(self, f1, f2, ...)
      local r = {f1(...)}
      if f2 then f2(unpack(r)) end
    end
    threads.synchronize = function() end
    for i=1,#setup do
      setup[i]()
    end
  end
end

function augmentThreadState(...)
  local funcs = {...}
  if nThreads == 0 then
    for j=1,#funcs do
      funcs[j]()
    end
    return
  end
  local specific = threads:specific()
  threads:specific(true)
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

function cat(T1, T2, dim)
  if T1:nElement() == 0 then return T2:clone() end
  if T2:nElement() == 0 then return T1:clone() end
  return torch.cat(T1, T2, dim)
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

function maskToLongTensor(mask)
  assert(mask:dim() == 1)
  local idx = torch.linspace(1, mask:size(1), mask:size(1)):long()
  return idx[mask:eq(1)]
end

function string:split(sep)
  sep = sep or ','
  local fields = {}
  local pattern = string.format('([^%s]+)', sep)
  self:gsub(pattern, function(c) fields[#fields+1] = c end)
  return fields
end

function readAll(file)
    local f = io.open(file, "rb")
    local content = f:read("*all")
    f:close()
    return content
end

function os.capture(cmd, raw)
  local f = assert(io.popen(cmd, 'r'))
  local s = assert(f:read('*a'))
  f:close()
  if raw then return s end
  s = string.gsub(s, '^%s+', '')
  s = string.gsub(s, '%s+$', '')
  return s
end

function runMatlab(cmd)
  return os.capture('matlab -nodisplay -nosplash -nodesktop -r "try, ' .. cmd .. ', catch ME, disp(getReport(ME)); exit, end, exit" | tail -n +10')
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

function rgbToHsl(r, g, b)
  local max, min = math.max(r, g, b), math.min(r, g, b)
  local h, s, l

  l = (max + min) / 2

  if max == min then
    h, s = 0, 0 -- achromatic
  else
    local d = max - min
    if l > 0.5 then s = d / (2 - max - min) else s = d / (max + min) end
    if max == r then
      h = (g - b) / d
      if g < b then h = h + 6 end
    elseif max == g then h = (b - r) / d + 2
    elseif max == b then h = (r - g) / d + 4
    end
    h = h / 6
  end

  return h, s, l
end

function hslToRgb(h, s, l)
  local r, g, b

  if s == 0 then
    r, g, b = l, l, l -- achromatic
  else
    function hue2rgb(p, q, t)
      if t < 0   then t = t + 1 end
      if t > 1   then t = t - 1 end
      if t < 1/6 then return p + (q - p) * 6 * t end
      if t < 1/2 then return q end
      if t < 2/3 then return p + (q - p) * (2/3 - t) * 6 end
      return p
    end

    local q
    if l < 0.5 then q = l * (1 + s) else q = l + s - l * s end
    local p = 2 * l - q

    r = hue2rgb(p, q, h + 1/3)
    g = hue2rgb(p, q, h)
    b = hue2rgb(p, q, h - 1/3)
  end

  return r, g, b
end

function rgbToHsv(r, g, b)
  local max, min = math.max(r, g, b), math.min(r, g, b)
  local h, s, v
  v = max

  local d = max - min
  if max == 0 then s = 0 else s = d / max end

  if max == min then
    h = 0 -- achromatic
  else
    if max == r then h = (g - b) / d
    elseif max == g then h = (b - r) / d + 2
    elseif max == b then h = (r - g) / d + 4
    end
    h = h / 6
    if h < 0 then h = h + 1 end
  end

  return h, s, v
end

function hsvToRgb(h, s, v)
  local r, g, b

  if s == 0 then
    return v, v, v -- achromatic
  end

  h = h * 6
  local i = math.floor(h);
  local f = h - i;
  local p = v * (1 - s);
  local q = v * (1 - f * s);
  local t = v * (1 - (1 - f) * s);

  if i == 0 then r, g, b = v, t, p
  elseif i == 1 then r, g, b = q, v, p
  elseif i == 2 then r, g, b = p, v, t
  elseif i == 3 then r, g, b = p, q, v
  elseif i == 4 then r, g, b = t, p, v
  elseif i == 5 then r, g, b = v, p, q
  end

  return r, g, b
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
