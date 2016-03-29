cv = require 'cv'
require 'cv.cudawarping'
require 'cv.imgcodecs'
require 'fbnn'
local Processor = require 'Processor'
local M = torch.class('ImageNetProcessor', 'Processor')

function M:__init()
  self.cmd:option('-cropSize', 224, 'What size to crop to.')
  Processor.__init(self)

  assert(self.processorOpts.cropSize <= 256)
  self.processorOpts.meanPixel = {}
  self.processorOpts.meanPixel[1] = torch.Tensor{103.939, 116.779, 123.68}:view(1, 1, 3)
  if nGPU > 0 then
    self.processorOpts.meanPixel[1] = self.processorOpts.meanPixel[1]:cuda()
    for i=2,nGPU do
      cutorch.setDevice(i)
      self.processorOpts.meanPixel[i] = self.processorOpts.meanPixel[1]:clone()
    end
    cutorch.setDevice(1)
  end

  self.words = {}
  self.lookup = {}
  local n = 1
  for line in io.lines('/file/imagenet/ILSVRC2012_devkit_t12/words.txt') do
    self.words[#self.words+1] = string.sub(line,11)
    self.lookup[string.sub(line, 1, 9)] = n
    n = n + 1
  end

  self.val = {}
  for line in io.lines('/file/imagenet/ILSVRC2012_devkit_t12/data/WNID_validation_ground_truth.txt') do
    self.val[#self.val+1] = self.lookup[line]
  end

  self.criterion = nn.TrueNLLCriterion()
  self.criterion.sizeAverage = false
  if nGPU > 0 then
    require 'cutorch'
    self.criterion = self.criterion:cuda()
  end
end

function M.preprocess(path, isTraining, processorOpts)
  local img = cv.imread{path, cv.IMREAD_COLOR}:float()
  if nGPU > 0 then
    img = img:cuda()
  end

  -- find the smaller dimension, and resize it to 256
  if nGPU > 0 then
    if img:size(2) < img:size(1) then
      img = cv.cuda.resize{img, {256 * img:size(1) / img:size(2), 256}}
    else
      img = cv.cuda.resize{img, {256, 256 * img:size(2) / img:size(1)}}
    end
  else
    img = img:permute(3, 1, 2)
    if img:size(3) < img:size(2) then
      img = image.scale(img, 256, 256 * img:size(2) / img:size(3))
    else
      img = image.scale(img, 256 * img:size(3) / img:size(2), 256)
    end
    img = img:permute(2, 3, 1)
  end

  local sz = processorOpts.cropSize
  local iW = img:size(2)
  local iH = img:size(1)
  local w1 = math.ceil((iW-sz)/2)
  local h1 = math.ceil((iH-sz)/2)
  img = img[{{h1, h1+sz-1}, {w1, w1+sz-1}}] -- center patch
  return img:csub(processorOpts.meanPixel[gpu]:expandAs(img)):permute(3, 1, 2)
end

function M.getLabels(pathNames)
  local labels = torch.Tensor(#pathNames)
  if nGPU > 0 then labels = labels:cuda() end
  for i=1,#pathNames do
    local name = pathNames[i]
    local filename = paths.basename(name)
    if name:find('train') then
      labels[i] = processor.lookup[string.sub(filename, 1, 9)]
    else
      assert(name:find('val'))
      labels[i] = processor.val[tonumber(string.sub(filename, -12, -5))]
    end
  end
  return labels
end

function M.calcStats(pathNames, outputs, labels)
  local top1 = 0
  local top5 = 0
  for i=1,#pathNames do
    local prob, classes = (#pathNames == 1 and outputs or outputs[i]):view(-1):sort(true)
    local result = 'predicted classes for ' .. paths.basename(pathNames[i]) .. ': '
    for j=1,5 do
      local color = ''
      if classes[j] == labels[i] then
        if j == 1 then top1 = top1 + 1 end
        top5 = top5 + 1
        color = '\27[33m'
      end
      result = result .. color .. '(' .. math.floor(prob[j]*100 + 0.5) .. '%) ' .. processor.words[classes[j]] .. '\27[0m; '
    end
    if labels[i] ~= -1 then
      result = result .. '\27[36mground truth: ' .. processor.words[labels[i]] .. '\27[0m'
    end
    --print(result)
  end
  return top1, top5, #pathNames
end

function M:resetStats()
  self.top1 = 0
  self.top5 = 0
  self.total = 0
end

function M:accStats(...)
  a, b, c = ...
  self.top1 = self.top1 + a
  self.top5 = self.top5 + b
  self.total = self.total + c
end

function M:printStats()
  print('  Top 1 accuracy: ' .. self.top1 .. '/' .. self.total .. ' = ' .. (self.top1*100.0/self.total) .. '%')
  print('  Top 5 accuracy: ' .. self.top5 .. '/' .. self.total .. ' = ' .. (self.top5*100.0/self.total) .. '%')
end

return M
