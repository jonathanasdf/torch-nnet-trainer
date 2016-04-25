local Transforms = require 'Transforms'
local Processor = require 'Processor'
local M = torch.class('CaltechRegionProposalEvaluator', 'Processor')

function M:__init(model, processorOpts)
  self.cmd:option('-drawROC', '', 'set a directory to use for full evaluation')
  self.cmd:option('-name', 'Result', 'name to use on ROC graph')
  Processor.__init(self, model, processorOpts)

  if self.processorOpts.drawROC == '' then
    error('CaltechRegionProposalEvaluator requires drawROC.')
  end

  local matio = require 'matio'
  matio.use_lua_strings = true
  local boxes = {
    matio.load('/file/caltech10x/val/box.mat'),
    matio.load('/file/caltech10x/test/box.mat')
  }
  self.boxes = {}
  for i=1,#boxes do
    for j=1,#boxes[i].name_pos do
      self.boxes[boxes[i].name_pos[j]] = boxes[i].box_pos[j]
    end
    for j=1,#boxes[i].name_neg do
      self.boxes[boxes[i].name_neg[j]] = boxes[i].box_neg[j]
    end
  end

  if not(opts.testing) then
    error('drawROC can only be used with Forward.lua')
  end
  if nThreads > 1 then
    error('sorry, drawROC can only be used with nThreads <= 1')
  end
  if opts.epochSize ~= -1 then
    error('sorry, drawROC can only be used with epochSize == -1')
  end
  if not(opts.resume) or opts.resume == '' then
    if paths.dir(self.processorOpts.drawROC) ~= nil then
      error('drawROC directory exists! Aborting')
    end
    paths.mkdir(self.processorOpts.drawROC)
  end
end

function M.preprocess(path, isTraining, processorOpts)
  return torch.Tensor(1)
end

function M.train()
  error('Cannot train CaltechRegionProposalEvaluator.')
end

function M:processStats(phase)
  if self.processorOpts.drawROC ~= '' then
     local dir = processorOpts.drawROC .. '/res/'
     -- remove duplicate boxes
     for file, attr in dirtree(dir) do
       if attr.mode == 'file' then
         os.execute("gawk -i inplace '!a[$0]++' " .. file)
       end
     end

     local has = {}
     for i=1,#opts.input do
       if opts.input[i]:find('val') then has['val'] = 1 end
       if opts.input[i]:find('test') then has['test'] = 1 end
     end
     local dataName
     for k,_ in pairs(has) do
       if not(dataName) then
         dataName = "{'" .. k .. "'"
       else
         dataName = dataName .. ", '" .. k .. "'"
       end
     end
     dataName = dataName .. '}'
     local cmd = "cd /file/caltech; dbEval('" .. self.processorOpts.drawROC .. "', " .. dataName .. ", '" .. self.processorOpts.name .. "')"
     print(runMatlab(cmd))
     print(readAll(self.processorOpts.drawROC .. '/eval/RocReasonable.txt'))
  end
  return tostring(self.stats)
end

function M.test(pathNames, inputs)
  if processorOpts.drawROC ~= '' then
    for i=1,#pathNames do
      local path = pathNames[i]
      local dataset, set, video, id = path:match("/file/caltech10x/(.-)/.-/raw/set(.-)_V(.-)_I(.-)_.*")

      local filename = processorOpts.drawROC .. '/res/set' .. set .. '/V' .. video .. '/I' .. id .. '.txt'
      paths.mkdir(paths.dirname(filename))
      local file, err = io.open(filename, 'a')
      if not(file) then error(err) end

      local boxes = _processor.boxes[paths.basename(path)]
      file:write(boxes[1], ' ',  boxes[2], ' ', boxes[3]-boxes[1]+1, ' ', boxes[4]-boxes[2]+1, ' ', 1, '\n')
      file:close()
    end
  end
end

return M
