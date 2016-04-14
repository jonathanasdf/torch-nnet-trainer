require 'cudnn'
require 'cunn'
require 'dpnn'
require 'gnuplot'
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
    cudnn.benchmark = true
  end

  if path then
    self:load(path)
  end
  Parent.__init(self, self.model)

  print('=> Model')
  print(self.model)

  self.params, self.gradParams = self:getParameters()
  print('Total parameters: ', self.gradParams:size(1))
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

local function updateModel(model, gradParams, loss, cnt, stats)
  opts.processor:accStats(stats)
  model.loss = model.loss + loss
  model.count = model.count + cnt
  if gradParams then
    model.gradParams:add(gradParams)
  end
  if opts.epoch % opts.updateEvery == 0 then
    opts.processor:updateModel()
  end
  jobDone()
end

local function accValResults(model, loss, cnt, stats)
  opts.processor:accStats(stats)
  model.loss = model.loss + loss
  model.count = model.count + cnt
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

  local trainLoader = DataLoader{inputs = opts.input, weights = opts.inputWeights}

  local validLoader
  if opts.val ~= '' then
    validLoader = DataLoader{inputs = opts.val, randomize = true}
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
  self.trainLoss = torch.Tensor(opts.epochs)
  self.valLoss = torch.Tensor(opts.epochs)
  for epoch=1,opts.epochs do
    opts.epoch = epoch
    augmentThreadState(function()
      opts.epoch = epoch
    end)
    print('==> training epoch # ' .. epoch)

    self.loss = 0
    self.count = 0
    opts.processor:resetStats()
    trainLoader:runAsync(opts.batchSize,
                         opts.epochSize,
                         true, --randomSample
                         bindPost(opts.processor.preprocessFn, true),
                         trainFn,
                         bind(updateModel, self))
    self.loss = self.loss / self.count
    self.trainLoss[epoch] = self.loss
    print(string.format('  Training loss: %.6f', self.loss))
    printlog(opts.processor:processStats('train'))

    if opts.val ~= '' and epoch % opts.valEvery == 0 then
      self.loss = 0
      self.count = 0
      opts.processor:resetStats()
      validLoader:runAsync(opts.valBatchSize,
                           opts.valSize,
                           false, --randomSample
                           bindPost(opts.processor.preprocessFn, false),
                           valFn,
                           bind(accValResults, self))
      self.loss = self.loss / self.count
      self.valLoss[epoch] = self.loss
      print(string.format('  Validation loss: %.6f', self.loss))
      print(opts.processor:processStats('val'))
      print()
    end

    if opts.logdir then
      gnuplot.figure(opts.lossGraph)
      local valX
      if opts.val ~= '' and epoch >= opts.valEvery then
         valX = torch.range(opts.valEvery, epoch, opts.valEvery):long()
      end
      local trainX = torch.range(1, epoch):long()
      if valX then
        gnuplot.plot({'train', trainX, self.trainLoss:index(1, trainX), '+-'}, {'val', valX, self.valLoss:index(1, valX), '+-'})
      else
        gnuplot.plot({'train', trainX, self.trainLoss:index(1, trainX), '+-'})
      end
      gnuplot.plotflush()
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
  augmentThreadState(function()
    model:clearState()
  end)
  torch.save(filename, self.model)
  opts.optimState.dfdx = nil
  torch.save(filename .. '.optimState', opts.optimState)
end

return M
