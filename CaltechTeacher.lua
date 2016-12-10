local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechTeacher', 'CaltechProcessor')

function M:__init(model, processorOpts)
  self.cmd:option('-fc', false, 'use fc outputs')
  CaltechProcessor.__init(self, model, processorOpts)
end

function M:getLoss(outputs, labels)
  return 0, 0
end

local t = torch.CudaTensor(1)
function M:loadInput(path, augmentations)
  return t, {}
end

function M:getLabel(path)
  local basepath = '/file1/caltech10x/teacheroutputs32/'
  local out
  local name = paths.basename(path, '.png')
  local file = basepath .. name
  if self.fc then
    file = file .. 'fc'
  end
  file = file .. '.t7'
  return torch.load(file):view(1, -1)
end

function M:forward(pathNames, inputs, deterministic)
  local _, labels = self:loadAndPreprocessInputs(pathNames)
  return CaltechProcessor.forward(self, pathNames, labels, deterministic)
end

function M:train()
  error('Cannot train CaltechTeacher.')
end

function M:updateStats(pathNames, outputs, labels) end

return M
