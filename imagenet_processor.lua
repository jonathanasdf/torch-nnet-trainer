local class = require 'class'
require 'fbnn'
local Processor = require 'processor'

local M = class('ImageNetProcessor', 'Processor')

local cropSize
function M:__init(opt)
  self.cmd:option('-cropSize', 224, 'What size to crop to.')
  Processor.__init(self, opt)
  cropSize = self.opt.cropSize

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

function M.preprocess(path)
  local img = image.load(path, 3)

  -- find the smaller dimension, and resize it to 256
  if img:size(3) < img:size(2) then
     img = image.scale(img, 256, 256 * img:size(2) / img:size(3))
  else
     img = image.scale(img, 256 * img:size(3) / img:size(2), 256)
  end

  local iW = img:size(3)
  local iH = img:size(2)
  local w1 = math.ceil((iW-cropSize)/2)
  local h1 = math.ceil((iH-cropSize)/2)
  img = image.crop(img, w1, h1, w1+cropSize, h1+cropSize) -- center patch
  local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(255.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end

function M:getLabels(pathNames)
  local labels = torch.Tensor(#pathNames):fill(-1)
  for i=1,#pathNames do
    local name = pathNames[i]
    local filename = paths.basename(name)
    if name:find('train') then
      labels[i] = self.lookup[string.sub(filename, 1, 9)]
    elseif name:find('val') then
      labels[i] = self.val[tonumber(string.sub(filename, -12, -5))]
    end
  end
  return labels
end

local top1 = 0
local top5 = 0
local total = 0
function M:testBatch(pathNames, outputs)
  local labels = self:getLabels(pathNames)

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
      result = result .. color .. '(' .. math.floor(prob[j]*100 + 0.5) .. '%) ' .. self.words[classes[j]] .. '\27[0m; '
    end
    if labels[i] ~= -1 then
      result = result .. '\27[36mground truth: ' .. self.words[labels[i]] .. '\27[0m'
    end
    --print(result)

    total = total + 1
  end
end

function M:printStats()
  print('Top 1 accuracy: ' .. top1 .. '/' .. total .. ' = ' .. (top1*100.0/total) .. '%')
  print('Top 5 accuracy: ' .. top5 .. '/' .. total .. ' = ' .. (top5*100.0/total) .. '%')
end

return M
