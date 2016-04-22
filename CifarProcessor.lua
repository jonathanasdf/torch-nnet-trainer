require 'fbnn'
local Processor = require 'Processor'
local M = torch.class('CifarProcessor', 'Processor')

function M:__init(model, processorOpts)
  Processor.__init(self, model, processorOpts)

  self.criterion = nn.TrueNLLCriterion()
  self.criterion.sizeAverage = false
  if nGPU > 0 then
    require 'cutorch'
    self.criterion = self.criterion:cuda()
  end
end

function M.preprocess(path, isTraining, processorOpts)
  return image.load(path, 3)
end

function M.getLabels(pathNames)
  local labels = torch.Tensor(#pathNames)
  for i=1,#pathNames do
    labels[i] = tonumber(paths.basename(paths.dirname(pathNames[i])))
  end
  return labels
end

function M.calcStats(pathNames, outputs, labels)
  local _, i = torch.max(outputs, 2)
  return {i, labels}
end

function M:resetStats()
  self.stats = optim.ConfusionMatrix({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'})
end

function M:accStats(new_stats)
  self.stats:batchAdd(new_stats[1], new_stats[2])
end

function M:processStats(phase)
  return tostring(self.stats)
end

return M
