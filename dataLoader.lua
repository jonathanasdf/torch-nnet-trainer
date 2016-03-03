require 'paths'
require 'xlua'
require 'image'
require 'cutorch'
require 'utils'
local ffi = require 'ffi'
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
   help='applied to image (ex: jittering). It takes the image as input',
   opt = true},

  {name='verbose',
   type='boolean',
   help='Verbose mode during initialization',
   default = false},
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
  self.lineOffset = {}
  table.insert(self.lineOffset, self.imagePaths:seek())
  for line in self.imagePaths:lines() do
    table.insert(self.lineOffset, self.imagePaths:seek())
  end
  table.remove(self.lineOffset)

  self.numSamples = #self.lineOffset
  print(self.numSamples ..  ' images found.')
end

function DataLoader:size()
  return self.numSamples
end

function DataLoader:retrieve(indices)
  local paths = {}
  local quantity = type(indices) == 'table' and #indices or indices:nElement()
  for i=1,quantity do
    -- load the sample
    self.imagePaths:seek('set', self.lineOffset[indices[i]])
    local path = self.imagePaths:read()
    table.insert(paths, path)
  end
  return paths
end

-- samples with replacement
function DataLoader:sample(quantity)
  quantity = quantity or 1
  local indices = {}
  for i=1,quantity do
    local index = math.ceil(torch.uniform() * self:size())
    table.insert(indices, index)
  end
  return self:retrieve(indices)
end

function DataLoader:get(i1, i2)
  local indices
  if type(i1) == 'number' then
    if type(i2) == 'number' then -- range of indices
      i2 = math.min(i2, self:size())
      indices = torch.range(i1, i2);
    else -- single index
      indices = {i1}
    end
  elseif type(i1) == 'table' then
    indices = i1 -- table
  elseif (type(i1) == 'userdata' and i1:nDimension() == 1) then
    indices = i1 -- tensor
  else
    error('Unsupported input types: ' .. type(i1) .. ' ' .. type(i2))
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

  setJobs(epochSize)
  for i=1,epochSize do
    local paths
    if shuffle then
      paths = self:sample(batchSize)
    else
      local indexStart = (i-1) * batchSize + 1
      local indexEnd = (indexStart + batchSize - 1)
      paths = self:get(indexStart, indexEnd)
    end

    threads:addjob(
      function(preprocessor)
        collectgarbage()
        local inputs = {}
        for j=1,#paths do
          table.insert(inputs, preprocessor(image.load(paths[j], 3, 'float')))
        end
        return imageTableToTensor(inputs)
      end,
      function(inputs)
        resultHandler(paths, inputs)
      end,
      self.preprocessor
    )
  end
  threads:synchronize()
end

return dataLoader
