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

  {name='preprocessor',
   type='function',
   help='applied to image (ex: jittering). It takes the image as input'},

  {name='randomize',
   type='boolean',
   help='whether to shuffle all of the images once after reading them',
   default=false}
}

local dataLoader = torch.class('DataLoader')

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
    indices[#indices+1] = math.ceil(torch.uniform() * self:size())
  end
  return self:retrieve(indices)
end

function DataLoader:get(start, end_incl, source)
  local indices
  if type(start) == 'number' then
    if type(end_incl) == 'number' then -- range of indices
      end_incl = math.min(end_incl, self:size())
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
  if source then
    indices = source:index(1, indices:long())
  end
  return self:retrieve(indices)
end

function DataLoader:runAsync(batchSize, epochSize, shuffle, resultHandler)
  if batchSize == -1 then
    batchSize = self:size()
  end

  if epochSize == -1 then
    epochSize = math.ceil(self:size() * 1.0 / batchSize)
  end
  epochSize = math.min(epochSize, math.ceil(self:size() * 1.0 / batchSize))

  local jobsDone = 0
  xlua.progress(jobsDone, epochSize)

  local runFn = function(pathNames, preprocessor)
    collectgarbage()
    local inputs = {}
    for j=1,#pathNames do
      inputs[#inputs+1] = preprocessor(pathNames[j])
    end
    return pathNames, tableToBatchTensor(inputs)
  end

  local doneFn = function(pathNames, inputs)
    resultHandler(pathNames, inputs)
    jobsDone = jobsDone + 1
    xlua.progress(jobsDone, epochSize)
  end

  local perm = self.shuffle
  if shuffle then
    perm = torch.randperm(self:size())
  end
  for i=1,epochSize do
    local indexStart = (i-1) * batchSize + 1
    local indexEnd = (indexStart + batchSize - 1)
    local pathNames = self:get(indexStart, indexEnd, perm)
    threads:addjob(runFn, doneFn, pathNames, self.preprocessor)
  end
  threads:synchronize()
end

return dataLoader
