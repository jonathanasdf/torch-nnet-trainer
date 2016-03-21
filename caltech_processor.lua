local class = require 'class'
cv = require 'cv'
require 'cv.imgcodecs'
require 'fbnn'
require 'svm'

local Processor = require 'processor'

local M = class('CaltechProcessor', 'Processor')

function M:__init(opt)
  self.cmd:option('-imageSize', 227, 'What to resize to.')
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

function M.preprocess(path, opt, isTraining)
  local img = cv.imread{path, cv.IMREAD_COLOR}:float():transpose(3, 1, 2)
  local mean_pixel = torch.FloatTensor{103.939, 116.779, 123.68}:view(3, 1, 1):expandAs(img)
  return image.scale(img - mean_pixel, opt.imageSize, opt.imageSize)
end

function M:getLabels(pathNames)
  local labels = torch.Tensor(#pathNames)
  for i=1,#pathNames do
    labels[i] = pathNames[i]:find('neg') and 1 or 2
  end
  return labels
end

function M:resetStats()
  self.pos_correct = 0
  self.neg_correct = 0
  self.pos_total = 0
  self.neg_total = 0
end

function M:testBatch(pathNames, inputs)
  local outputs = self.model:forward(inputs, true)
  local labels = self:getLabels(pathNames)
  local pred

  if self.opt.svm == '' then
    _, pred = torch.max(outputs, 2)
    pred = pred:squeeze()
  else
    if not(self.svm_model) then
      self.svm_model = torch.load(self.opt.svm)
    end
    local data = convertTensorToSVMLight(labels, findModuleByName(self.model, self.opt.layer).output)
    pred = liblinear.predict(data, self.svm_model, '-q')
  end

  for i=1,#pathNames do
    if labels[i] == 2 then
      self.pos_total = self.pos_total + 1
      if pred[i] == labels[i] then
        self.pos_correct = self.pos_correct + 1
      end
    else
      self.neg_total = self.neg_total + 1
      if pred[i] == labels[i] then
        self.neg_correct = self.neg_correct + 1
      end
    end
  end

  local loss = self.criterion:forward(outputs, labels)
  return self.pos_correct + self.neg_correct, self.pos_total + self.neg_total, loss
end

function M:printStats()
  print('Accuracy: ' .. (self.pos_correct + self.neg_correct) .. '/' .. (self.pos_total + self.neg_total) .. ' = ' .. ((self.pos_correct + self.neg_correct)*100.0/(self.pos_total + self.neg_total)) .. '%')
  print('Positive Accuracy: ' .. self.pos_correct .. '/' .. self.pos_total .. ' = ' .. (self.pos_correct*100.0/self.pos_total) .. '%')
  print('Negative Accuracy: ' .. self.neg_correct .. '/' .. self.neg_total .. ' = ' .. (self.neg_correct*100.0/self.neg_total) .. '%')
end

return M
