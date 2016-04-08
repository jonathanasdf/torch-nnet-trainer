cv = require 'cv'
require 'cv.cudawarping'
require 'cv.imgcodecs'
require 'fbnn'
local Processor = require 'Processor'
local M = torch.class('ImageNetProcessor', 'Processor')

function M:__init()
  self.cmd:option('-resize', 256, 'What size to crop to.')
  self.cmd:option('-cropSize', 224, 'What size to crop to.')
  self.cmd:option('-inceptionPreprocessing', false, 'Preprocess for inception models (RGB, [-1, 1))')
  self.cmd:option('-caffePreprocessing', false, 'Preprocess for caffe models (BGR, [0, 255])')
  Processor.__init(self)
  assert(self.processorOpts.cropSize <= self.processorOpts.resize)

  local synset = '/file/imagenet/ILSVRC2012_devkit_t12/words.txt'
  if self.processorOpts.inceptionPreprocessing then
    synset = '/file/imagenet/inception_synset.txt'
  elseif self.processorOpts.caffePreprocessing then
    self.processorOpts.meanPixel = torch.Tensor{103.939, 116.779, 123.68}:view(3, 1, 1)
  else
    self.processorOpts.meanPixel = torch.Tensor{0.485, 0.456, 0.406}:view(3, 1, 1)
    self.processorOpts.std = torch.Tensor{0.229, 0.224, 0.225}:view(3, 1, 1)
  end

  self.words = {}
  self.lookup = {}
  local n = 1
  for line in io.lines(synset) do
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
  local img = image.load(path, 3)

  -- find the smaller dimension, and resize it to processorOpts.resize
  local sz = processorOpts.resize
  if img:size(3) < img:size(2) then
    img = image.scale(img, sz, sz * img:size(2) / img:size(3))
  else
    img = image.scale(img, sz * img:size(3) / img:size(2), sz)
  end

  sz = processorOpts.cropSize
  img = image.crop(img, 'c', sz, sz)

  if processorOpts.inceptionPreprocessing then
    img = (img * 255 - 128) / 128
  elseif processorOpts.caffePreprocessing then
    img = (convertRGBBGR(img) * 255):csub(processorOpts.meanPixel:expandAs(img))
  else
    img = img:csub(processorOpts.meanPixel:expandAs(img)):cdiv(processorOpts.std:expandAs(img))
  end

  if nGPU > 0 then
    img = img:cuda()
  end
  return img
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
  return {top1, top5, #pathNames}
end

function M:resetStats()
  self.stats = {}
  self.stats.top1 = 0
  self.stats.top5 = 0
  self.stats.total = 0
end

function M:accStats(new_stats)
  self.stats.top1 = self.stats.top1 + new_stats[1]
  self.stats.top5 = self.stats.top5 + new_stats[2]
  self.stats.total = self.stats.total + new_stats[3]
end

function M:printStats()
  print('  Top 1 accuracy: ' .. self.stats.top1 .. '/' .. self.stats.total .. ' = ' .. (self.stats.top1*100.0/self.stats.total) .. '%')
  print('  Top 5 accuracy: ' .. self.stats.top5 .. '/' .. self.stats.total .. ' = ' .. (self.stats.top5*100.0/self.stats.total) .. '%')
end

return M
