local class = require 'class'
cv = require 'cv'
require 'cv.imgcodecs'
require 'fbnn'
require 'svm'

local Processor = require 'processor'

local M = class('CaltechProcessor', 'Processor')

function M:__init(opt)
  self.cmd:option('-svm', '', 'SVM to use')
  self.cmd:option('-layer', 'fc7', 'layer to use as SVM input')
  Processor.__init(self, opt)

  self.criterion = nn.TrueNLLCriterion()
  self.criterion.sizeAverage = false
  if nGPU > 0 then
    require 'cutorch'
    self.criterion = self.criterion:cuda()
  end
end

function M.preprocess(path)
  local img = cv.imread{path, cv.IMREAD_COLOR}:float():transpose(3, 1, 2)
  local mean_pixel = torch.FloatTensor{103.939, 116.779, 123.68}:view(3, 1, 1):expandAs(img)
  return img - mean_pixel
end

function M:getLabels(pathNames)
  local labels = torch.Tensor(#pathNames)
  for i=1,#pathNames do
    labels[i] = pathNames[i]:find('neg') and 1 or 2
  end
  return labels
end

local pos_correct = 0
local neg_correct = 0
local pos_total = 0
local neg_total = 0
function M:processBatch(pathNames, outputs, testPhase)
  local labels = self:getLabels(pathNames)

  if testPhase then
    assert(self.opt.svm ~= '', 'for testing, a trained svm must be used.')

    if not(self.svm_model) then
      self.svm_model = torch.load(self.opt.svm)
    end
    local data = convertTensorToSVMLight(labels, findModuleByName(self.model, self.opt.layer).output)
    local pred, _, _ = liblinear.predict(data, self.svm_model, '-q')

    for i=1,#pathNames do
      if labels[i] == 2 then
        pos_total = pos_total + 1
        if pred[i] == labels[i] then
          pos_correct = pos_correct + 1
        end
      else
        neg_total = neg_total + 1
        if pred[i] == labels[i] then
          neg_correct = neg_correct + 1
        end
      end
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
  print('Accuracy: ' .. (pos_correct + neg_correct) .. '/' .. (pos_total + neg_total) .. ' = ' .. ((pos_correct + neg_correct)*100.0/(pos_total + neg_total)) .. '%')
  print('Positive Accuracy: ' .. pos_correct .. '/' .. pos_total .. ' = ' .. (pos_correct*100.0/pos_total) .. '%')
  print('Negative Accuracy: ' .. neg_correct .. '/' .. neg_total .. ' = ' .. (neg_correct*100.0/neg_total) .. '%')
end

return M
