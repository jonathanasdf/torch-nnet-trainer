local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('SaveTeacherOutputs', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)
end

function M:test(pathNames)
  local loss, total = CaltechProcessor.test(self, pathNames)
  for i=1,#pathNames do
     torch.save('/file1/caltech10x/teacheroutputs/' .. paths.basename(pathNames[i], '.png') .. 'fc.t7',
                 self.model:get(17).output[i])
     --torch.save('/file1/caltech10x/teacheroutputs/' .. paths.basename(pathNames[i], '.png') .. '.t7',
     --           self.model:get(20).output[i])
  end
  return loss, total
end

return M
