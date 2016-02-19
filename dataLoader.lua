require 'paths'
require 'xlua'
require 'image'
require 'cutorch'
require 'utils'
local ffi = require 'ffi'
local argcheck = require 'argcheck'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')
torch.setdefaulttensortype('torch.FloatTensor')

local initcheck = argcheck{
  pack=true,
  help=[[
    A data loader class for loading images optimized for extremely large datasets.
    Tested only on Linux (as it uses command-line linux utilities to scale up)
  ]],
  {name="path",
   type="string",
   help="path of directories with images"},

  {name="preprocessor",
   type="function",
   help="applied to image (ex: for lighting jitter). It takes the image as input",
   opt = true},

  {name="verbose",
   type="boolean",
   help="Verbose mode during initialization",
   default = false},
}

local dataLoader = torch.class('dataLoader')

function dataLoader:__init(...)
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
  print('load the concatenated list of sample paths to self.imagePath')
  local maxPathLength = tonumber(sys.fexecute("wc -L '" 
        .. imageList .. "' | " 
        .. "cut -f1 -d' '")) + 1
  local length = tonumber(sys.fexecute("wc -l '" 
        .. imageList .. "' | " 
        .. "cut -f1 -d' '"))
  assert(length > 0, "Could not find any image file in the given input paths")
  assert(maxPathLength > 0, "paths of files are length 0?")

  self.imagePath = torch.CharTensor()
  self.imagePath:resize(length, maxPathLength):fill(0)
  local s_data = self.imagePath:data()
  local count = 0
  for line in io.lines(imageList) do
    ffi.copy(s_data, line)
    s_data = s_data + maxPathLength
    if self.verbose and count % 10000 == 0 then 
      xlua.progress(count, length) 
    end 
    count = count + 1
  end
  if self.verbose then 
    xlua.progress(length, length) 
  end

  self.numSamples = self.imagePath:size(1)
  print(self.numSamples ..  ' samples found.')

  os.execute('rm -f "' .. imageList .. '"')
end

function dataLoader:loadImage(path)
  local path = ffi.string(torch.data(path))
  local img = image.load(path, 3, 'float')
  if self.preprocessor then img = self.preprocessor(img) end 
  return img
end

function dataLoader:size(class, list)
  return self.numSamples
end

function dataLoader:tableToTensor(table)
  for k, v in pairs(table) do
    table[k] = v:reshape(1, v:size(1), v:size(2), v:size(3))
  end
  return torch.cat(table, 1):cuda()
end 

-- samples with replacement
function dataLoader:sample(quantity)
  quantity = quantity or 1
  local paths = {}
  local data = {}
  for i=1,quantity do
    local index = math.ceil(torch.uniform() * self:size())
    path = self.imagePath[index]
    table.insert(paths, path) 
    table.insert(data, self:loadImage(path))
  end
  return paths, self:tableToTensor(data)
end

function dataLoader:get(i1, i2)
  local indices, quantity
  if type(i1) == 'number' then
    if type(i2) == 'number' then -- range of indices
      i2 = math.min(i2, self:size())
      indices = torch.range(i1, i2); 
      quantity = i2 - i1 + 1;
    else -- single index 
      indices = {i1}; quantity = 1 
    end 
  elseif type(i1) == 'table' then
    indices = i1; quantity = #i1;      -- table
  elseif (type(i1) == 'userdata' and i1:nDimension() == 1) then
  else
    error('Unsupported input types: ' .. type(i1) .. ' ' .. type(i2))    
  end
  assert(quantity > 0)

  -- now that indices has been initialized, get the samples
  local paths = {}
  local data = {}
  for i=1,quantity do
    -- load the sample
    path = self.imagePath[indices[i]]
    table.insert(paths, path) 
    table.insert(data, self:loadImage(path))
  end
  return paths, self:tableToTensor(data)
end

function dataLoader:runAsync(batchSize, epochSize, shuffle, nThreads, resultHandler)
  local jobDone = 0
  threads = Threads(
    nThreads,
    function()
      require 'dataLoader'
    end,
    function(threadid)
      loader = self
    end
  )

  for i=1,epochSize do
    threads:addjob(
      function()
        if loader.verbose and jobDone % math.floor(epochSize/1000) == 0 then 
          xlua.progress(jobDone, epochSize) 
        end
        jobDone = jobDone + 1
        if shuffle then
          return i, loader:sample(batchSize)
        else
          local indexStart = (i-1) * batchSize + 1
          local indexEnd = (indexStart + batchSize - 1)
          return i, loader:get(indexStart, indexEnd)
        end
      end,
      resultHandler
    )
  end

  threads:synchronize()
  if self.verbose then 
    xlua.progress(epochSize, epochSize) 
  end
end
