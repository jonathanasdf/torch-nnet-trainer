local class = require 'class'
require 'fbnn'
local Processor = require 'processor'

local M = class('ImageNetProcessor', 'Processor')

function M:__init(opt)
  Processor.__init(self, opt)

  self.words = {}
  self.lookup = {}
  local n = 1
  for line in io.lines('/file/imagenet/ILSVRC2012_devkit_t12/words.txt') do
    table.insert(self.words, string.sub(line,11))
    self.lookup[string.sub(line, 1, 9)] = n
    n = n + 1
  end

  self.val = {}
  for line in io.lines('/file/imagenet/ILSVRC2012_devkit_t12/data/WNID_validation_ground_truth.txt') do
    table.insert(self.val, self.lookup[line])
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

  local p = 224
  local iW = img:size(3)
  local iH = img:size(2)
  local w1 = math.ceil((iW-p)/2)
  local h1 = math.ceil((iH-p)/2)
  img = image.crop(img, w1, h1, w1+p, h1+p) -- center patch
  local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(255.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end

local top1 = 0
local top5 = 0
local total = 0
function M:processBatch(pathNames, outputs, calculateStats)
  local labels = torch.Tensor(#pathNames)
  for i=1,#pathNames do
    local name = pathNames[i]
    local filename = paths.basename(name)
    local label = -1
    if name:find('train') then
      label = self.lookup[string.sub(filename, 1, 9)]
    elseif name:find('val') then
      label = self.val[tonumber(string.sub(filename, -12, -5))]
    end
    labels[i] = label

    if calculateStats then
      local prob, classes = (#pathNames == 1 and outputs or outputs[i]):view(-1):sort(true)
      local result = 'predicted classes for ' .. filename .. ': '
      for j=1,5 do
        local color = ''
        if classes[j] == label then
          if j == 1 then top1 = top1 + 1 end
          top5 = top5 + 1
          color = '\27[33m'
        end
        result = result .. color .. '(' .. math.floor(prob[j]*100 + 0.5) .. '%) ' .. self.words[classes[j]] .. '\27[0m; '
      end
      if label ~= -1 then
        result = result .. '\27[36mground truth: ' .. self.words[label] .. '\27[0m'
      end
      --print(result)

      total = total + 1
    end
  end

  if nGPU > 0 then
    labels = labels:cuda()
  end
  local loss = self.criterion:forward(outputs, labels)
  local grad_outputs = self.criterion:backward(outputs, labels)
  return loss, grad_outputs
end

function M:printStats()
  print('Top 1 accuracy: ' .. top1 .. '/' .. total .. ' = ' .. (top1*100.0/total) .. '%')
  print('Top 5 accuracy: ' .. top5 .. '/' .. total .. ' = ' .. (top5*100.0/total) .. '%')
end

return M
