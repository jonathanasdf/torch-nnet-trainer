local class = require 'class'
require 'image'
require 'paths'

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

function M.preprocess(path)
  return image.load(path, 3)
end

function M:getLabels(pathNames)
  error('getLabels is not defined.')
end

function M:processBatch(pathNames, outputs, testPhase)
  error('processBatch is not defined.')
end

function M:printStats() end

return M
