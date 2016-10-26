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

function M:printBoxes(pathNames, values)
  if self.outputBoxes ~= '' then
    for i=1,#pathNames do
      local path = pathNames[i]
      local set, video, id = path:match("set(.-)_V(.-)_I(.-)_")

      local filename = self.outputBoxes .. 'set' .. set .. '/V' .. video .. '/I' .. id .. '.txt'
      mkdir(filename)
      local file, err = io.open(filename, 'a')
      if not(file) then error(err) end

      local box = self.boxes[paths.basename(path)]
      file:write(box[1]-1, ' ',  box[2]-1, ' ', box[3], ' ', box[4], ' ', tostring(box[5] >= 0.5), '\n')
      file:close()
    end
  end
end

function M:test(pathNames)
  self:printBoxes(pathNames, nil)
  return 0, #pathNames
end

return M
