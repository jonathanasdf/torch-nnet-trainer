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
  self:outputBoxes(pathNames, torch.ones(#pathNames))
  return 0, #pathNames
end

return M
