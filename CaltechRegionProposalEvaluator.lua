local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechRegionProposalEvaluator', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)

  if self.processorOpts.drawROC == '' then
    error('CaltechRegionProposalEvaluator requires drawROC.')
  end
end

function M.preprocess(path, isTraining, processorOpts)
  return torch.Tensor(1)
end

function M.train()
  error('Cannot train CaltechRegionProposalEvaluator.')
end

function M:accStats(new_stats) end

function M.test(pathNames, inputs)
  _processor.drawROC(pathNames, torch.ones(#pathNames))
end

return M
