local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechRegionProposalEvaluator', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)

  if self.outputBoxes == '' then
    error('CaltechRegionProposalEvaluator requires outputBoxes.')
  end
  if not self.nonms then
    print('WARNING: you might want to set nonms.')
  end
end

function M:preprocess()
  return torch.Tensor(1):cuda()
end

function M:train()
  error('Cannot train CaltechRegionProposalEvaluator.')
end

function M:test(pathNames)
  local output = torch.Tensor(#pathNames)
  for i=1,#pathNames do
    local path = pathNames[i]
    local box = self.boxes[paths.basename(path)]
    output[i] = box[5] >= 0.5
  end
  self:printBoxes(pathNames, output)
  return 0, #pathNames
end

return M
