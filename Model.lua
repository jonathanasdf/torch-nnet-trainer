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
    require 'cudnn'
    require 'cunn'
    self.backend = 'cudnn'
    cudnn.benchmark = true
  end

  if path then
    self:load(path)
  end
  setDropout(self.model, opts.dropout)
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

function M:get(index)
  return self.model:get(index)
end

function M:forward(inputs, deterministic)
  if deterministic then
    self:evaluate()
  else
    self:training()
  end
  return self.model:forward(inputs)
end

local function updateModel(loss, cnt, stats)
  _model.trainIter = _model.trainIter + 1
  _model.loss = _model.loss + loss
  _model.count = _model.count + cnt
  _processor:accStats(stats)
  if _model.trainIter % opts.updateEvery == 0 then
    _processor:updateModel()
  end
  jobDone()
end

local function accValResults(loss, cnt, stats)
  _processor:accStats(stats)
  _model.loss = _model.loss + loss
  _model.count = _model.count + cnt
  jobDone()
end

function M:train(trainFn, valFn)
  if not(opts.input) then
    error('Input must be defined for training.')
  end
  if _model ~= self then
    error('You can only train the main model.')
  end
  if not trainFn then
    trainFn = _processor.train
  end
  if not valFn then
    valFn = _processor.test
  end

  local trainLoader = DataLoader{inputs = opts.input, weights = opts.inputWeights}

  local validLoader
  if opts.val ~= '' then
    validLoader = DataLoader{inputs = opts.val, randomize = true}
  end

  if opts.optimState ~= '' then
    opts.optimState = torch.load(opts.optimState)
    opts.LR = opts.optimState.learningRate
    opts.LRDecay = opts.optimState.learningRateDecay
    opts.momentum = opts.optimState.momentum
    opts.weightDecay = opts.optimState.weightDecay
    opts.nesterov = opts.optimState.nesterov
  else
    opts.optimState = {
      learningRate = opts.LR,
      learningRateDecay = opts.LRDecay,
      momentum = opts.momentum,
      weightDecay = opts.weightDecay
    }
    if opts.nesterov == 1 then
      opts.optimState.dampening = 0.0
      opts.optimState.nesterov = true
    end
  end

  local signal = require("posix.signal")
  signal.signal(signal.SIGINT, function(signum)
    print("Interrupt!")
    if opts.output and opts.output ~= '' then
      self:save(opts.backupdir .. opts.basename .. '.interrupt')
    end
    os.exit(-1)
  end)

  self.trainIter = 0
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
    _processor:resetStats()
    trainLoader:runAsync(opts.batchSize,
                         opts.epochSize,
                         true, --randomSample
                         bindPost(_processor.preprocessFn, true),
                         trainFn,
                         updateModel)
    self.loss = self.loss / self.count
    self.trainLoss[epoch] = self.loss
    print(string.format('  Training loss: %.6f', self.loss))
    logprint(_processor:processStats('train'))

    if opts.val ~= '' and epoch % opts.valEvery == 0 then
      self.loss = 0
      self.count = 0
      _processor:resetStats()
      validLoader:runAsync(opts.valBatchSize,
                           opts.valSize,
                           false, --randomSample
                           bindPost(_processor.preprocessFn, false),
                           valFn,
                           accValResults)
      self.loss = self.loss / self.count
      self.valLoss[epoch] = self.loss
      print(string.format('  Validation loss: %.6f', self.loss))
      print(_processor:processStats('val'))
      print()
    end

    if opts.LRDropEvery ~= -1 and epoch % opts.LRDropEvery == 0 then
      opts.LR = opts.LR / opts.LRDropFactor
      opts.optimState.learningRate = opts.LR
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

      if opts.cacheEvery ~= -1 and epoch % opts.cacheEvery == 0 then
        local cachename = opts.backupdir .. opts.basename .. '.cached'
        self:save(cachename)
        if opts.keepCaches then
          os.execute('cp ' .. cachename .. ' ' .. opts.cachedir .. 'epoch' .. epoch .. '.t7')
        end
        augmentThreadState(function()
          _model:clearState()
        end)
      end
    end
  end

  if opts.output and opts.output ~= '' then
    self:save(opts.output)
  end
end

function M:save(filename)
  self:clearState()
  torch.save(filename, self.model:clone():float())
  opts.optimState.dfdx = nil
  torch.save(filename .. '.optimState', opts.optimState)
end

return M
