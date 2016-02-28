function defineTrainingOptions(cmd)
  cmd:option('-LR', 0.01, 'learning rate')
  cmd:option('-momentum', 0.9, 'momentum')
  cmd:option('-batchSize', 32, 'batch size')
  cmd:option('-epochs', 50, 'num epochs')
  cmd:option('-epochSize', -1, 'num batches per epochs')
end

local function trainBatch(model, opt, updates, paths, inputs)
  local parameters, _ = model:getParameters()
  local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
  }
  if opt.nGPU > 0 then
    inputs = inputs:cuda()
  end
  new_parameters, _ = optim.sgd(updates(model, inputs), parameters, optimState)
  parameters, _ = model:getParameters()
  parameters:copy(new_parameters)

  if model.needsSync then
    model:syncParameters()
  end
end

function train(model, loader, opt, updates)
  local batchSize = opt.batchSize
  if batchSize == -1 then
    batchSize = loader:size()
  end

  local epochSize = opt.epochSize
  if epochSize == -1 then
    epochSize = math.ceil(loader:size() / batchSize)
  end
  epochSize = math.min(epochSize, math.ceil(loader:size() / batchSize))

  for epoch=1,opt.epochs do
    print("==> training epoch # " .. epoch)
    local batchNumber = 0
  
    model.model:training()
    cutorch.synchronize()
    local tm = torch.Timer()
    loader:runAsync(batchSize, 
                    epochSize, 
                    batchSize * epochSize < loader:size(), --shuffle
                    opt.nThreads, 
                    bind(trainBatch, model.model, opt, updates)) 
    cutorch.synchronize()
    print(string.format('Epoch [%d]: Total Time(s): %.2f', epoch, tm:time().real))

    collectgarbage()
    if opt.output and opt.output ~= "/dev/null" then
      model:saveDataParallel(opt.output .. ".cached")
    end
  end
end
