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
function M:preprocess(path, augmentations)
  return t, {}
end

function M:getLabels(pathNames, outputs)
  local basepath = '/file1/caltech10x/teacheroutputs/'
  local out = {}
  for i=1,#pathNames do
    local name = paths.basename(pathNames[i], '.png')
    local file = basepath .. name
    if self.fc then
      file = file .. 'fc'
    end
    file = file .. '.t7'
    out[#out+1] = torch.load(file):view(1, -1)
  end
  return torch.cat(out, 1)
end

function M:forward(pathNames, inputs, deterministic)
  return CaltechProcessor.forward(self, pathNames, self:getLabels(pathNames), deterministic)
end

function M:train()
  error('Cannot train CaltechTeacher.')
end

function M:updateStats(pathNames, outputs, labels) end

return M
