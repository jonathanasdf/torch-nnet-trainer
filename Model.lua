require 'cudnn'
require 'cunn'
require 'dpnn'
require 'gnuplot'
require 'optim'
require 'paths'

require 'DataLoader'
require 'Utils'

cudnn.benchmark = true

local M, Parent = torch.class('Model', 'nn.Decorator')

function M:__init(specStr)
  local args = specStr:split(' ');
  if #args < 2 then
    error('Model specifications must be in the form: <model> <processor> [-options].')
  end
  local path = table.remove(args, 1)
  local processorPath = table.remove(args, 1)

  assert(paths.filep(path), 'Cannot find model ' .. path)
  assert(paths.filep(processorPath), 'Cannot find processor ' .. processorPath)

  self:load(path)

  local processorOpts = table.concat(args, ' ')
  self.processor = requirePath(processorPath).new(self, processorOpts)

  setDropout(self.model, processorOpts.dropout)
  self.model:zeroGradParameters()
  Parent.__init(self, self.model)

  print('=> Model')
  print(self.model)

  self.params, self.gradParams = self:getParameters()
  print('Total parameters: ', self.gradParams:size(1))
end

local function loadSavedModel(filename)
  local model
  if paths.extname(filename) == 'caffemodel' then
    require 'loadcaffe'
    model = loadcaffe.load(paths.dirname(filename) .. '/deploy.prototxt', filename, 'cudnn')
  else
    model = torch.load(filename)
  end
  return model
end

function M:load(path)
  assert(paths.filep(path), 'File not found: ' .. path)
  if paths.extname(path) == 'lua' then
    print('Creating model from file: ' .. path)
    self.model = requirePath(path)
    cudnn.convert(self.model, cudnn)
  else
    print('Loading model from file: ' .. path)
    self.model = loadSavedModel(path)
  end
  self.model = self.model:cuda()
end

function M:save(filename)
  self:clearState()
  torch.save(filename, self.model)
  opts.optimState.dfdx = nil
  torch.save(filename .. '.optimState', opts.optimState)
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

function M:backward(inputs, gradOutputs, gradLayer)
  if gradLayer then
    -- feed gradients through a specific layer
    for i=gradLayer,2,-1 do
      gradOutputs = self:get(i):backward(self:get(i-1).output, gradOutputs)
    end
    self:get(1):backward(inputs, gradOutputs)
  else
    self.model:backward(inputs, gradOutputs)
  end
end

function M:updateModel(loss, cnt)
  self.trainIter = self.trainIter + 1
  self.loss = self.loss + loss
  self.count = self.count + cnt
  if self.trainIter % opts.updateEvery == 0 then
    optim.sgd(function()
                return 0, self.gradParams
              end,
              self.params,
              opts.optimState)
    self:zeroGradParameters()
  end
end

function M:accValResults(loss, cnt)
  self.loss = self.loss + loss
  self.count = self.count + cnt
end

function M:run(dataloader, batchSize, epochSize, randomSample, workerFn, resultHandler, startBatch)
  if batchSize == -1 then
    batchSize = dataloader:size()
  end

  if epochSize == -1 then
    epochSize = math.ceil(dataloader:size() * 1.0 / batchSize)
  end
  epochSize = math.min(epochSize, math.ceil(dataloader:size() * 1.0 / batchSize))

  startBatch = startBatch or 1
  if startBatch > epochSize then
    return
  end

  local jobSize = epochSize - startBatch + 1
  local jobsDone = 0
  xlua.progress(jobsDone, jobSize)

  local indexStart = (startBatch-1) * batchSize + 1
  for i=startBatch,epochSize do
    collectgarbage()
    local indexEnd = math.min(indexStart + batchSize - 1, dataloader:size())
    local pathNames = randomSample and dataloader:sample(batchSize) or dataloader:get(indexStart, indexEnd)

    resultHandler(workerFn(pathNames))

    jobsDone = jobsDone + 1
    xlua.progress(jobsDone, jobSize)

    indexStart = indexEnd + 1
    if indexStart > dataloader:size() then
      break
    end
  end
end

function M:train(trainFn, valFn)
  if not(opts.input) then
    error('Input must be defined for training.')
  end
  if not trainFn then
    trainFn = self.processor.trainFn
  end
  if not valFn then
    valFn = self.processor.testFn
  end

  local trainLoader = DataLoader{inputs = opts.input, weights = opts.inputWeights}

  local validLoader
  if opts.val ~= '' then
    validLoader = DataLoader{inputs = opts.val, randomize = true}
  end

  local pathNames = trainLoader:sample(opts.batchSize)
  local inputs = self.processor:loadAndPreprocessInputs(pathNames)

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
    print('==> training epoch # ' .. epoch)

    setPhase('train')
    self.loss = 0
    self.count = 0
    self.processor:resetStats()
    self:run(trainLoader,
             opts.batchSize,
             opts.epochSize,
             true,  -- randomSample
             trainFn,
             bind(self.updateModel, self))
    self.loss = self.loss / self.count
    self.trainLoss[epoch] = self.loss
    print(string.format('  Training loss: %.6f', self.loss))
    logprint(self.processor:getStats())

    if opts.val ~= '' and epoch % opts.valEvery == 0 then
      setPhase('val')
      self.loss = 0
      self.count = 0
      self.processor:resetStats()
      self:run(validLoader,
               opts.valBatchSize,
               opts.valSize,
               false,  -- randomSample
               valFn,
               bind(self.accValResults, self))
      self.loss = self.loss / self.count
      self.valLoss[epoch] = self.loss
      print(string.format('  Validation loss: %.6f', self.loss))
      print(self.processor:getStats())
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

        local pathNames = trainLoader:sample(opts.batchSize)
        local inputs = self.processor:loadAndPreprocessInputs(pathNames)
      end
    end
  end

  if opts.output and opts.output ~= '' then
    self:save(opts.output)
  end
end

return M
