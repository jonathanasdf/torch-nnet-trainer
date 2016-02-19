local function trainBatch(model, opt, updates, batchNumber, paths, inputs)
  local parameters, _ = model:getParameters()
  local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
  }
  optim.sgd(updates(inputs), parameters, optimState)

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
    for i=1,opt.epochSize do
      loader:runAsync(batchSize, 
                      epochSize, 
                      true, --shuffle
                      opt.nThreads, 
                      bind(trainBatch, model.model, opt, updates)) 
    end 
    cutorch.synchronize()
    print(string.format('Epoch [%d]: Total Time(s): %.2f', epoch, tm:time().real))

    if opt.output then
      saveDataParallel(opt.output .. ".tmp", model.model)
    end
  end
end
