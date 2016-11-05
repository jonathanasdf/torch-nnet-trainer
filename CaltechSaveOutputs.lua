local Transforms = require 'Transforms'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('SaveTeacherOutputs', 'CaltechProcessor')

function M:__init(model, processorOpts)
  CaltechProcessor.__init(self, model, processorOpts)
end

function M:test(pathNames)
  local loss, total = CaltechProcessor.test(self, pathNames)
  for i=1,#pathNames do
     torch.save('/file1/caltech10x/teacheroutputs32/' .. paths.basename(pathNames[i], '.png') .. 'fc.t7',
                 self.model:get(15).output[i])
     torch.save('/file1/caltech10x/teacheroutputs32/' .. paths.basename(pathNames[i], '.png') .. '.t7',
                self.model:get(18).output[i])
  end
  return loss, total
end

return M
