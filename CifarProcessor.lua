require 'hdf5'
local Processor = require 'Processor'
local M = torch.class('CifarProcessor', 'Processor')

function M:__init(model, processorOpts)
  Processor.__init(self, model, processorOpts)

  local f = hdf5.open('/file1/cifar10/data.h5', 'r')
  self.processorOpts.input = f:read('/input'):all()
  self.processorOpts.label = f:read('/label'):all()
  f:close()

  self.criterion = nn.CrossEntropyCriterion(nil, false):cuda()

  if opts.logdir then
    self.graph = gnuplot.pngfigure(opts.logdir .. 'acc.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('acc')
    gnuplot.grid(true)
    self.trainAcc = torch.Tensor(opts.epochs)
    self.valAcc = torch.Tensor(opts.epochs)
  end
end

function M:preprocess(path, pAugment)
  self:checkAugmentations(pAugment, nil)
  return processorOpts.input[tonumber(path)], nil
end

function M:getLabels(pathNames)
  local labels = torch.Tensor(#pathNames)
  for i=1,#pathNames do
    labels[i] = processorOpts.label[tonumber(pathNames[i])]
  end
  return labels:cuda()
end

function M:resetStats()
  self.stats = optim.ConfusionMatrix(10)
end

function M:updateStats(pathNames, outputs, labels)
  self.stats:batchAdd(outputs, labels)
end

function M:getStats()
  self.stats:updateValids()

  if self.graph then
    if opts.phase == 'train' then
      self.trainAcc[opts.epoch] = self.stats.averageValid
    elseif opts.phase == 'val' then
      self.valAcc[opts.epoch] = self.stats.averageValid
    end

    gnuplot.figure(self.graph)
    local x = torch.range(1, opts.epoch):long()
    local x2 = torch.range(opts.valEvery, opts.epoch):long()
    if opts.epoch >= opts.valEvery then
      gnuplot.plot({'train', x, self.trainAcc:index(1, x), '-'}, {'val', x2, self.valAcc:index(1, x2), '-'})
    else
      gnuplot.plot({'train', x, self.trainAcc:index(1, x), '-'})
    end
    gnuplot.plotflush()
  end
  return tostring(self.stats)
end

return M
