require 'cunn'
require 'cudnn'
require 'dpnn'
require 'paths'

require 'dataLoader'
require 'resnet'
require 'utils'

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

  print('=> Model')
  print(self.model)

  self.parameters, self.gradParameters = self.model:getParameters()
  print('Total parameters: ', self.gradParameters:size(1))
end

function M:forward(inputs, deterministic)
  if deterministic then
    self:evaluate()
  else
    self:training()
  end
  return Parent.forward(self, inputs)
end

local function updateModel(model, gradParameters)
    opts.train_iter = opts.train_iter + 1
    if gradParameters then
      model.gradParameters:add(gradParameters)
    end
    if opts.train_iter % opts.update_every == 0 then
      opts.processor:updateModel()
    end
    jobDone()
end

local function accValResults(model, loss, cnt, ...)
  opts.processor:accStats(...)
  model.valid_loss = model.valid_loss + loss
  model.valid_count = model.valid_count + cnt
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

  local train_loader = DataLoader{path = opts.input}

  local valid_loader
  if opts.val ~= '' then
    valid_loader = DataLoader{path = opts.val, randomize = true}
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
      self:save(opts.basename .. '.interrupt')
      opts.optimState.dfdx = nil
      torch.save(opts.basename .. '.interrupt.optimState', opts.optimState)
    end
    os.exit(128 + signum)
  end)

  self:zeroGradParameters()
  opts.train_iter = 0
  for epoch=1,opts.epochs do
    print('==> training epoch # ' .. epoch)

    train_loader:runAsync(opts.batchSize,
                          opts.epochSize,
                          true, --shuffle
                          bind_post(opts.processor.preprocessFn, true),
                          trainFn,
                          bind(updateModel, self))

    if opts.val ~= '' and epoch % opts.val_every == 0 then
      self.valid_loss = 0
      self.valid_count = 0
      opts.processor:resetStats()
      valid_loader:runAsync(opts.valBatchSize,
                            opts.valSize,
                            false, --don't shuffle
                            bind_post(opts.processor.preprocessFn, false),
                            valFn,
                            bind(accValResults, self))
      self.valid_loss = self.valid_loss / self.valid_count
      print(string.format('  Validation loss: %.6f', self.valid_loss))
      opts.processor:printStats()
    end

    if opts.cache_every ~= -1 and epoch % opts.cache_every == 0 and
       opts.output and opts.output ~= '' then
      self:save(opts.basename .. '.cached')
      opts.optimState.dfdx = nil
      torch.save(opts.basename .. '.cached.optimState', opts.optimState)
    end
  end

  if opts.output and opts.output ~= '' then
    self:save(opts.output)
    opts.optimState.dfdx = nil
    torch.save(opts.output .. '.optimState', opts.optimState)
  end
end

function M:save(filename)
  self:clearState()
  torch.save(filename, self.model)
end

return M
