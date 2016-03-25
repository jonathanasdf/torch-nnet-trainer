require 'image'
require 'paths'
require 'xlua'

require 'utils'

local argcheck = require 'argcheck'
local initcheck = argcheck{
  pack=true,
  help=[[
    A data loader class for loading images optimized for extremely large datasets.
    Tested only on Linux (as it uses command-line linux utilities to scale up)
  ]],
  {name='path',
   type='string',
   help='path of directories with images'},

  {name='randomize',
   type='boolean',
   help='whether to shuffle all of the images once after reading them',
   default=false}
}

local dataLoader = torch.class('DataLoader')

local min = math.min
local ceil = math.ceil

function DataLoader:__init(...)
  local args = initcheck(...)
  for k,v in pairs(args) do self[k] = v end

  ----------------------------------------------------------------------
  -- Options for the GNU find command
  local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
  local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
  for i=2,#extensionList do
    findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
  end

  -- find the image paths
  print('Finding images...')
  local imageList = os.tmpname()
  local tmpfile = os.tmpname()
  local tmphandle = assert(io.open(tmpfile, 'w'))
  local command = 'find ' .. args.path .. ' ' .. findOptions .. ' >>"' .. imageList .. '" \n'
  tmphandle:write(command)
  io.close(tmphandle)
  os.execute('bash ' .. tmpfile)
  os.execute('rm -f ' .. tmpfile)

  --==========================================================================
  local length = tonumber(sys.fexecute('wc -l "' .. imageList .. '" | ' .. 'cut -f1 -d" "'))
  assert(length > 0, 'Could not find any image file in the given input paths')

  self.imagePaths = io.open(imageList)
  self.lineOffset = {0}
  for line in self.imagePaths:lines() do
    self.lineOffset[#self.lineOffset+1] = self.imagePaths:seek()
  end
  table.remove(self.lineOffset)

  self.numSamples = #self.lineOffset
  print(self.numSamples ..  ' images found.')

  if self.randomize then
    self.shuffle = torch.randperm(self.numSamples)
  end
end

function DataLoader:size()
  return self.numSamples
end

function DataLoader:retrieve(indices)
  local pathNames = {}
  local quantity = type(indices) == 'table' and #indices or indices:nElement()
  for i=1,quantity do
    -- load the sample
    self.imagePaths:seek('set', self.lineOffset[indices[i]])
    pathNames[#pathNames+1] = self.imagePaths:read()
  end
  return pathNames
end

-- samples with replacement
function DataLoader:sample(quantity)
  quantity = quantity or 1
  local indices = {}
  for i=1,quantity do
    indices[#indices+1] = ceil(torch.uniform() * self:size())
  end
  return self:retrieve(indices)
end

function DataLoader:get(start, end_incl, perm)
  local indices
  if type(start) == 'number' then
    if type(end_incl) == 'number' then -- range of indices
      end_incl = min(end_incl, self:size())
      indices = torch.range(start, end_incl);
    else -- single index
      indices = {start}
    end
  elseif type(start) == 'table' then
    indices = start -- table
  elseif (type(start) == 'userdata' and start:nDimension() == 1) then
    indices = start -- tensor
  else
    error('Unsupported input types: ' .. type(start) .. ' ' .. type(end_incl))
  end
  if perm then
    indices = perm:index(1, indices:long())
  end
  return self:retrieve(indices)
end

function DataLoader.loadInputs(pathNames, preprocessFn, workerFn)
  collectgarbage()
  local inputs = {}
  for j=1,#pathNames do
    inputs[#inputs+1] = preprocessFn(pathNames[j])
  end
  inputs = tableToBatchTensor(inputs)
  if workerFn then
    return workerFn(pathNames, inputs)
  else
    return pathNames, inputs
  end
end

function DataLoader:runAsync(batchSize, epochSize, shuffle, preprocessFn, workerFn, resultHandler)
  if batchSize == -1 then
    batchSize = self:size()
  end

  if epochSize == -1 then
    epochSize = ceil(self:size() * 1.0 / batchSize)
  end
  epochSize = min(epochSize, ceil(self:size() * 1.0 / batchSize))

  local perm = self.shuffle
  if shuffle then
    perm = torch.randperm(self:size())
  end

  startJobs(epochSize)
  for i=1,epochSize do
    local indexStart = (i-1) * batchSize + 1
    local indexEnd = (indexStart + batchSize - 1)
    local pathNames = self:get(indexStart, indexEnd, perm)
    threads:addjob(self.loadInputs, resultHandler, pathNames, preprocessFn, workerFn)
  end
  threads:synchronize()
end

return dataLoader
