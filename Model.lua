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
    error('Model specifications must be in the form: <model> [-options] <processor> [-options].')
  end
  local path = table.remove(args, 1)
  local modelOpts = {}
  local processorPath = table.remove(args, 1)
  while string.sub(processorPath, -4) ~= '.lua' do
    modelOpts[#modelOpts+1] = processorPath
    processorPath = table.remove(args, 1)
  end

  assert(paths.filep(path), 'Cannot find model ' .. path)
  assert(paths.filep(processorPath), 'Cannot find processor ' .. processorPath)

  nn.DataParallelTable.deserializeNGPUs = opts.nGPU
  self:load(path, table.concat(modelOpts, ' '))
  self.processor = requirePath(processorPath).new(self, table.concat(args, ' '))

  self.module:zeroGradParameters()
  Parent.__init(self, self.module)
  self:cuda()

  print('=> Model')
  print(self.module)

  self.params, self.gradParams = self:getParameters()
  print('Total parameters: ', self.gradParams:dim() > 0 and self.gradParams:size(1) or 0)
end

local function makeDataParallelTable(model, nGPU)
   if nGPU > 1 then
      local gpus = torch.range(1, nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            require 'Model'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil
      model = dpt:cuda()
   end
   return model
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

function M:load(path, modelOpts)
  assert(paths.filep(path), 'File not found: ' .. path)
  if paths.extname(path) == 'lua' then
    print('Creating model from file: ' .. path)
    self.module = requirePath(path)(modelOpts)
    cudnn.convert(self.module, cudnn)
  else
    print('Loading model from file: ' .. path)
    self.module = loadSavedModel(path)
  end
  if torch.type(self.module) ~= 'nn.DataParallelTable' then
    self.module = makeDataParallelTable(self.module, opts.nGPU)
  end
end

function M:save(filename)
  self:clearState()
  torch.save(filename, self.module)
  opts.dfdx = nil
  torch.save(filename .. '.optimstate', opts)
end

function M:get(index)
  return self.module:get(index)
end

function M:forward(inputs, deterministic)
  if deterministic then
    self:evaluate()
  else
    self:training()
  end
  return Parent.forward(self, inputs)
end

function M:backward(input, gradOutput, gradLayer)
  if gradLayer then
    -- feed gradients through a specific layer
    local currentGradOutput = gradOutput
    local currentModule = self:get(gradLayer)
    for i=gradLayer-1,1,-1 do
      local previousModule = self:get(i)
      currentGradOutput = self.module:rethrowErrors(currentModule, i+1, 'backward', previousModule.output, currentGradOutput)
      currentModule.gradInput = currentGradOutput
      currentModule = previousModule
    end
   currentGradOutput = self.module:rethrowErrors(currentModule, 1, 'backward', input, currentGradOutput)
   self.module.gradInput = currentGradOutput
   self.gradInput = currentGradOutput
   return currentGradOutput
  else
    return Parent.backward(self, input, gradOutput)
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
              opts)
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

function M:Train(trainFn, valFn)
  if not(opts.input) then
    error('Input must be defined for training.')
  end
  if not trainFn then
    trainFn = bind(self.processor.train, self.processor)
  end
  if not valFn then
    valFn = bind(self.processor.test, self.processor)
  end

  local trainLoader = DataLoader{inputs = opts.input, weights = opts.inputWeights}

  local validLoader
  if opts.val ~= '' then
    validLoader = DataLoader{inputs = opts.val}
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
    trainLoader:shuffle()
    self.loss = 0
    self.count = 0
    self.processor:resetStats()
    self:run(trainLoader,
             opts.batchSize,
             opts.epochSize,
             opts.epochSize ~= -1,  -- randomSample
             trainFn,
             bind(self.updateModel, self))
    self.loss = self.loss / self.count
    self.trainLoss[epoch] = self.loss
    print(string.format('  Training loss: %.6f', self.trainLoss[epoch]))
    print(self.processor:getStats())

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
      self.valLoss[epoch] = self.loss * opts.valLossMultiplier
      print(string.format('  Validation loss: %.6f', self.valLoss[epoch]))
      print(self.processor:getStats())
      print()
    end

    if opts.LRDropEvery ~= -1 and epoch % opts.LRDropEvery == 0 then
      opts.learningRate = opts.learningRate / opts.LRDropFactor
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
          local cachefile = opts.cachedir .. 'epoch' .. epoch .. '.t7'
          print('Saving cache ' .. cachefile)
          os.execute('cp ' .. cachename .. ' ' .. cachefile)
          if not paths.filep(cachefile) then
            print('ERROR COPYING FILE TO CACHE?')
          end
        end
      end
    end
  end

  if opts.output and opts.output ~= '' then
    self:save(opts.output)
  end
end

return M
