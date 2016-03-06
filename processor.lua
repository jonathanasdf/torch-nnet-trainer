local class = require 'class'
local M = class('Processor')

M.cmd = torch.CmdLine()
function M:__init(opt)
  self.opt = {}
  for k,v in pairs(opt) do
    self.opt[k] = v
  end

  local new_opts = self.cmd:parse(opt.processor_opts:split(' ='))
  for k,v in pairs(new_opts) do
    self.opt[k] = v
  end
end

function M.preprocess(img)
  return img
end

function M:processBatch(paths, outputs, calculateStats)
  error('ProcessBatch is not defined.')
end

function M:printStats() end

return M
