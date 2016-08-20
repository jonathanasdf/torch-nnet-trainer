local Processor = require 'Processor'
require 'TrueNLLCriterion'
local M = torch.class('CifarProcessor', 'Processor')

function M:__init(model, processorOpts)
  Processor.__init(self, model, processorOpts)

  local data = torch.load('/file1/cifar10/data.t7')
  self.processorOpts.input = data.data
  self.processorOpts.label = data.labels

  self.criterion = nn.TrueNLLCriterion(nil, false):cuda()

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
  return self.processorOpts.input[tonumber(path)]:cuda(), nil
end

function M:getLabels(pathNames)
  local labels = torch.Tensor(#pathNames)
  for i=1,#pathNames do
    labels[i] = self.processorOpts.label[tonumber(pathNames[i])]
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
    if opts.epoch >= opts.valEvery then
      local x2 = torch.range(opts.valEvery, opts.epoch):long()
      gnuplot.plot({'train', x, self.trainAcc:index(1, x), '-'}, {'val', x2, self.valAcc:index(1, x2), '-'})
    else
      gnuplot.plot({'train', x, self.trainAcc:index(1, x), '-'})
    end
    gnuplot.plotflush()
  end
  return tostring(self.stats)
end

return M
