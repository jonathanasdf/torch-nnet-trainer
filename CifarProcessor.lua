require 'hdf5'
local Processor = require 'Processor'
local M = torch.class('CifarProcessor', 'Processor')

function M:__init(model, processorOpts)
  Processor.__init(self, model, processorOpts)

  local f = hdf5.open('/file/cifar10/data.h5', 'r')
  self.processorOpts.input = f:read('/input'):all()
  self.processorOpts.label = f:read('/label'):all()
  f:close()
  self.preprocessFn = bindPost(self.preprocess, self.processorOpts)

  self.criterion = nn.CrossEntropyCriterion(nil, false)
  if nGPU > 0 then
    require 'cutorch'
    self.criterion = self.criterion:cuda()
  end

  if opts.logdir then
    self.graph = gnuplot.pngfigure(opts.logdir .. 'acc.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('acc')
    gnuplot.grid(true)
    self.trainAcc = torch.Tensor(opts.epochs)
    self.valAcc = torch.Tensor(opts.epochs)
  end
end

function M.preprocess(path, isTraining, processorOpts)
  return processorOpts.input[tonumber(path)]
end

function M.getLabels(pathNames)
  local labels = torch.Tensor(#pathNames)
  if nGPU > 0 then labels = labels:cuda() end
  for i=1,#pathNames do
    labels[i] = processorOpts.label[tonumber(pathNames[i])]
  end
  return labels
end

function M.calcStats(pathNames, outputs, labels)
  return {outputs:clone(), labels}
end

function M:resetStats()
  self.stats = optim.ConfusionMatrix(10)
end

function M:accStats(new_stats)
  self.stats:batchAdd(new_stats[1], new_stats[2])
end

function M:processStats(phase)
  self.stats:updateValids()

  if self.graph then
    if phase == 'train' then
      self.trainAcc[opts.epoch] = self.stats.averageValid
    elseif phase == 'val' then
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
