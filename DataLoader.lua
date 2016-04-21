require 'image'
require 'paths'
require 'xlua'

require 'Utils'

local initcheck = require 'argcheck'{
  pack=true,
  noordered=true,
  help=[[
    A data loader class for loading images optimized for extremely large datasets.
    Tested only on Linux (as it uses command-line linux utilities to scale up)
  ]],
  {name='inputs',
   type='table',
   check=function(inputs)
     local out = true
     for k,v in ipairs(inputs) do
       if type(v) ~= 'string' then
         print('inputs can only be of string input')
         out = false
       end
     end
     return out
   end,
   help='input directories with images'},

  {name='weights',
   type='table',
   check=function(weights)
     local out = true
     for k,v in ipairs(weights) do
       if type(v) ~= 'number' then
         print('weights can only be of number input')
         out = false
       end
     end
     return out
   end,
   help='input weights to sample with. Only used if randomSampling is used',
   default={}},

  {name='randomize',
   type='boolean',
   help='whether to shuffle all of the images once after reading them',
   default=false}
}

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(...)
  local args = initcheck(...)
  for k,v in pairs(args) do self[k] = v end

  -- Normalize weights if they exist
  if #self.weights ~= 0 then
    assert(#self.weights == #self.inputs)
    local sum = 0
    for i,w in ipairs(self.weights) do
      sum = sum + w
    end
    for i=1,#self.weights do
      self.weights[i] = self.weights[i] / sum
    end
  end

  ----------------------------------------------------------------------
  -- Options for the GNU find command
  local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
  local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
  for i=2,#extensionList do
    findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
  end

  -- find the image paths
  print('Finding images...')
  self.imageListFile = {}
  self.imageListHandle = {}
  self.lineOffset = {}
  self.numSamples = 0
  for i=1,#self.inputs do
    if paths.filep(self.inputs[i]) then
      self.imageListFile[i] = self.inputs[i]
    else
      self.imageListFile[i] = os.tmpname()
      local tmpfile = os.tmpname()
      local tmphandle = assert(io.open(tmpfile, 'w'))
      local command = 'find ' .. self.inputs[i] .. ' ' .. findOptions .. ' >>"' .. self.imageListFile[i] .. '" \n'
      tmphandle:write(command)
      io.close(tmphandle)
      os.execute('bash ' .. tmpfile)
      os.execute('rm -f ' .. tmpfile)

      --==========================================================================
      local length = tonumber(sys.fexecute('wc -l "' .. self.imageListFile[i] .. '" | ' .. 'cut -f1 -d" "'))
      assert(length > 0, 'Could not find any image file in the given input paths')
    end

    self.imageListHandle[i] = io.open(self.imageListFile[i])
    self.lineOffset[i] = {0}
    for line in self.imageListHandle[i]:lines() do
      self.lineOffset[i][#self.lineOffset[i]+1] = self.imageListHandle[i]:seek()
    end
    table.remove(self.lineOffset[i])

    self.numSamples = self.numSamples + #self.lineOffset[i]
  end
  print(self.numSamples ..  ' images found.')

  if self.randomize then
    self.shuffle = torch.randperm(self:size())
  end
end

function DataLoader:size(i)
  if not i then return self.numSamples end
  return #self.lineOffset[i]
end

function DataLoader:retrieve(indices)
  local quantity = type(indices) == 'table' and #indices or indices:nElement()
  local pathNames = {}
  for i=1,quantity do
    -- load the sample
    local index = self.shuffle and self.shuffle[indices[i]] or indices[i]
    local j = 1
    while index > self:size(j) do
      index = index - self:size(j)
      j = j + 1
    end
    self.imageListHandle[j]:seek('set', self.lineOffset[j][index])
    pathNames[#pathNames+1] = self.imageListHandle[j]:read()
  end
  return pathNames
end

-- samples with replacement
function DataLoader:sample(quantity)
  quantity = quantity or 1
  local indices = {}
  if #self.weights == 0 then
    indices = torch.ceil(torch.rand(quantity) * self:size())
  else
    local j = 1
    local index = 0
    for i=1,#self.inputs do
      local count = i < #self.inputs and self.weights[i] * quantity or quantity - #indices
      for j=1,count do
        indices[#indices+1] = index + math.ceil(torch.uniform() * self:size(i))
      end
      index = index + self:size(i)
    end
  end
  return self:retrieve(indices)
end

function DataLoader:get(start, endIncl)
  local indices
  if type(start) == 'number' then
    if type(endIncl) == 'number' then -- range of indices
      endIncl = math.min(endIncl, self:size())
      indices = torch.range(start, endIncl)
    else -- single index
      indices = {start}
    end
  elseif type(start) == 'table' then
    indices = start -- table
  elseif (type(start) == 'userdata' and start:nDimension() == 1) then
    indices = start -- tensor
  else
    error('Unsupported input types: ' .. type(start) .. ' ' .. type(endIncl))
  end
  return self:retrieve(indices)
end

function DataLoader.loadInputs(pathNames, preprocessFn, workerFn)
  collectgarbage()
  local first = preprocessFn(pathNames[1])
  local size = torch.LongStorage(first:dim() + 1)
  size[1] = #pathNames
  for i=1,first:dim() do
    size[i+1] = first:size(i)
  end
  local inputs = first.new(size)
  inputs[1] = first
  for i=2,#pathNames do
    inputs[i] = preprocessFn(pathNames[i])
  end
  if workerFn then
    return workerFn(pathNames, inputs)
  else
    return pathNames, inputs
  end
end

function DataLoader:runAsync(batchSize, epochSize, randomSample, preprocessFn, workerFn, resultHandler, startBatch)
  if batchSize == -1 then
    batchSize = self:size()
  end

  if epochSize == -1 then
    epochSize = math.ceil(self:size() * 1.0 / batchSize)
  end
  epochSize = math.min(epochSize, math.ceil(self:size() * 1.0 / batchSize))

  startBatch = startBatch or 1
  if startBatch > epochSize then return end
  startJobs(epochSize-startBatch+1)

  local indexStart = (startBatch-1) * batchSize + 1
  for i=startBatch,epochSize do
    local indexEnd = math.min(indexStart + batchSize - 1, self:size())
    local pathNames = randomSample and self:sample(batchSize) or self:get(indexStart, indexEnd)
    threads:addjob(self.loadInputs, resultHandler, pathNames, preprocessFn, workerFn)
    indexStart = indexEnd + 1
    if indexStart > self:size() then
      break
    end
  end
  threads:synchronize()
end

return DataLoader
