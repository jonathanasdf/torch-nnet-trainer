local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechRegionProposalEvaluator', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)

  if self.processorOpts.drawROC == '' then
    error('CaltechRegionProposalEvaluator requires drawROC.')
  end
  if not self.processorOpts.nonms then
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
  self:drawROC(pathNames, torch.ones(#pathNames))
  return 0, #pathNames
end

return M
