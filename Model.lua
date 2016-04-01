require 'cudnn'
require 'cunn'
require 'dpnn'
require 'paths'

require 'DataLoader'
require 'Utils'
require 'resnet'

local M, Parent = torch.class('Model', 'nn.Decorator')

function M:__init(path)
  if nGPU == 0 then
    self.backend = 'nn'
  else
    self.backend = 'cudnn'
  end

  if path then
    self:load(path)
  end
  Parent.__init(self, self.model)

  print('=> Model')
  print(self.model)

  self.parameters, self.gradParameters = self:getParameters()
  print('Total parameters: ', self.gradParameters:size(1))
end

local function loadSavedModel(filename, backend)
  local model
  if paths.extname(filename) == 'caffemodel' then
    require 'loadcaffe'
    model = loadcaffe.load(paths.dirname(filename) .. '/deploy.prototxt', filename, backend)
  else
    model = torch.load(filename)
  end
  return model
end

function M:load(path)
  assert(paths.filep(path), 'File not found: ' .. path)
  if paths.extname(path) == 'lua' then
    print('Creating model from file: ' .. path)
    self.model = paths.dofile(path)
    if self.backend == 'cudnn' then
      cudnn.convert(self.model, cudnn)
    elseif self.backend ~= 'nn' then
      error('Unsupported backend')
    end
  else
    print('Loading model from file: ' .. path)
    self.model = loadSavedModel(path, self.backend)
  end
  if nGPU > 0 then
    self.model = self.model:cuda()
  end
end

function M:forward(inputs, deterministic)
  if deterministic then
    self:evaluate()
  else
    self:training()
  end
  return self.model:forward(inputs)
end

local function updateModel(model, gradParameters)
    opts.trainIter = opts.trainIter + 1
    if gradParameters then
      model.gradParameters:add(gradParameters)
    end
    if opts.trainIter % opts.updateEvery == 0 then
      opts.processor:updateModel()
    end
    jobDone()
end

local function accValResults(model, loss, cnt, stats)
  opts.processor:accStats(stats)
  model.validLoss = model.validLoss + loss
  model.validCount = model.validCount + cnt
  jobDone()
end

function M:train(trainFn, valFn)
  if not(opts.input) then
    error('Input must be defined for training.')
  end
  if not trainFn then
    trainFn = opts.processor.train
  end
  if not valFn then
    valFn = opts.processor.test
  end

  local trainLoader = DataLoader{path = opts.input}

  local validLoader
  if opts.val ~= '' then
    validLoader = DataLoader{path = opts.val, randomize = true}
  end

  if opts.optimState ~= '' then
    opts.optimState = torch.load(opts.optimState)
  else
    opts.optimState = {
      learningRate = opts.LR,
      learningRateDecay = 0.0,
      momentum = opts.momentum,
      dampening = 0.0,
      nesterov = true,
      weightDecay = opts.weightDecay
    }
  end

  local signal = require("posix.signal")
  signal.signal(signal.SIGINT, function(signum)
    if opts.output and opts.output ~= '' then
      self:save(opts.backupdir .. opts.basename .. '.interrupt')
    end
    os.exit(128 + signum)
  end)

  self:zeroGradParameters()
  opts.trainIter = 0
  for epoch=1,opts.epochs do
    print('==> training epoch # ' .. epoch)

    trainLoader:runAsync(opts.batchSize,
                         opts.epochSize,
                         true, --shuffle
                         bindPost(opts.processor.preprocessFn, true),
                         trainFn,
                         bind(updateModel, self))

    if opts.val ~= '' and epoch % opts.valEvery == 0 then
      self.validLoss = 0
      self.validCount = 0
      opts.processor:resetStats()
      validLoader:runAsync(opts.valBatchSize,
                           opts.valSize,
                           false, --don't shuffle
                           bindPost(opts.processor.preprocessFn, false),
                           valFn,
                           bind(accValResults, self))
      self.validLoss = self.validLoss / self.validCount
      print(string.format('  Validation loss: %.6f', self.validLoss))
      opts.processor:printStats()
    end

    if opts.cacheEvery ~= -1 and epoch % opts.cacheEvery == 0 and
       opts.output and opts.output ~= '' then
      self:save(opts.backupdir .. opts.basename .. '.cached')
    end
  end

  if opts.output and opts.output ~= '' then
    self:save(opts.output)
  end
end

function M:save(filename)
  self:clearState()
  torch.save(filename, self.model)
  opts.optimState.dfdx = nil
  torch.save(filename .. '.optimState', opts.optimState)
end

return M
