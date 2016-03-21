local class = require 'class'
cv = require 'cv'
require 'cv.imgcodecs'
require 'fbnn'
matio = require 'matio'
require 'svm'

local Processor = require 'processor'

local M = class('CaltechProcessor', 'Processor')

local function defineSlidingWindowOptions(cmd)
  cmd:option('-windowSizeX', 30, 'width of sliding window')
  cmd:option('-windowSizeY', 50, 'height of sliding window')
  cmd:option('-windowStrideX', 15, 'horizontal stride of sliding window')
  cmd:option('-windowStrideY', 25, 'vertical stride of sliding window')
  cmd:option('-windowIOU', 0.5, 'what IOU to count as a positive example')
end

function M:__init(opt)
  self.cmd:option('-imageSize', 113, 'What to resize to.')
  self.cmd:option('-svm', '', 'SVM to use')
  self.cmd:option('-layer', 'fc7', 'layer to use as SVM input')
  defineSlidingWindowOptions(self.cmd)
  Processor.__init(self, opt)

  self.bboxes = {
    matio.load('/file/caltech/val/box.mat', 'box'),
    matio.load('/file/caltech/test/box.mat', 'box')
  }

  self.criterion = nn.TrueNLLCriterion()
  self.criterion.sizeAverage = false
  if nGPU > 0 then
    require 'cutorch'
    self.criterion = self.criterion:cuda()
  end
end

function M.preprocess(path, isTraining, opt)
  local img = cv.imread{path, cv.IMREAD_COLOR}:float():permute(3, 1, 2)
  local mean_pixel = torch.FloatTensor{103.939, 116.779, 123.68}:view(3, 1, 1):expandAs(img)
  img = img - mean_pixel
  if isTraining then
    img = image.scale(img, opt.imageSize, opt.imageSize)
  end
  return img
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

local function slidingWindow(path, img, bboxes, opt)
  local min = math.min
  local max = math.max
  local floor = math.floor
  local ceil = math.ceil

  local sz = opt.imageSize
  local sx = sz / opt.windowSizeX
  local sy = sz / opt.windowSizeY
  local dx = max(1, floor(opt.windowStrideX * sx))
  local dy = max(1, floor(opt.windowStrideY * sy))
  local w = ceil(img:size(3) * sx)
  local h = ceil(img:size(2) * sy)
  img = image.scale(img, w, h)

  local patches = {}
  local nPatches = 0
  local nCols = 0
  for j=1,h-sz+1,dy do
    for k=1,w-sz+1,dx do
      nPatches = nPatches+1
      patches[nPatches] = img[{{}, {j, j+sz-1}, {k, k+sz-1}}]
      if j == 1 then nCols = nCols + 1 end
    end
  end
  patches = tableToBatchTensor(patches)

  local labels = torch.ones(nPatches)
  local index = tonumber(paths.basename(path, 'png'))
  if bboxes[index] and bboxes[index]:nElement() ~= 0 then
    local SA = sz*sz
    for i=1,bboxes[index]:size(1) do
      local XB1 = bboxes[index][i][1] * sx
      local XB2 = (bboxes[index][i][1]+bboxes[index][i][3]) * sx
      local YB1 = bboxes[index][i][2] * sy
      local YB2 = (bboxes[index][i][2]+bboxes[index][i][4]) * sy
      local SB = bboxes[index][i][3]*bboxes[index][i][4]*sx*sy

      local left = max(0, ceil((XB1-sz)/dx))
      local right = floor(XB2/dx)
      local top = max(0, ceil((YB1-sz)/dy))
      local bottom = floor(YB2/dy)
      for j=top,bottom do
        for k=left,right do
          local XA1 = k*dx
          local XA2 = k*dx+sz
          local YA1 = j*dy
          local YA2 = j*dy+sz

          local SI = max(0, min(XA2, XB2) - max(XA1, XB1)) *
                     max(0, min(YA2, YB2) - max(YA1, YB1))
          local SU = SA + SB - SI
          -- TODO: actually use IOU
          if SB > 225*sx*sy and SI/SB > opt.windowIOU then
            labels[j*nCols+k+1] = 2
          end
        end
      end
    end
  end
  return patches, labels
end

local min = math.min
function M:testBatch(pathNames, inputs)
  local loss = 0
  local forwardFn = function(patches, labels)
    for i=1,labels:size(1),self.opt.batchSize do
      local j = min(i+self.opt.batchSize-1, labels:size(1))
      local outputs = self.model:forward(patches[{{i,j}}], true)

      local pred
      if self.opt.svm == '' then
        _, pred = torch.max(outputs, 2)
        pred = pred:squeeze()
      else
        if not(self.svm_model) then
          self.svm_model = torch.load(self.opt.svm)
        end
        local data = convertTensorToSVMLight(labels[{{i,j}}], findModuleByName(self.model, self.opt.layer).output)
        pred = liblinear.predict(data, self.svm_model, '-q')
      end

      for k=1,j-i do
        if labels[k+i] == 2 then
          self.pos_total = self.pos_total + 1
          if pred[k] == 2 then
            self.pos_correct = self.pos_correct + 1
          end
        else
          self.neg_total = self.neg_total + 1
          if pred[k] == 1 then
            self.neg_correct = self.neg_correct + 1
          end
        end
      end
      loss = loss + self.criterion:forward(outputs, labels[{{i,j}}])
    end
  end
  for k=1,#pathNames do
    threads:addjob(slidingWindow, forwardFn, pathNames[k], inputs[k],
        self.bboxes[pathNames[k]:find('val') and 1 or 2], self.opt)
  end
  return loss, self.pos_total + self.neg_total
end

function M:printStats()
  print('  Accuracy: ' .. (self.pos_correct + self.neg_correct) .. '/' .. (self.pos_total + self.neg_total) .. ' = ' .. ((self.pos_correct + self.neg_correct)*100.0/(self.pos_total + self.neg_total)) .. '%')
  print('  Positive Accuracy: ' .. self.pos_correct .. '/' .. self.pos_total .. ' = ' .. (self.pos_correct*100.0/self.pos_total) .. '%')
  print('  Negative Accuracy: ' .. self.neg_correct .. '/' .. self.neg_total .. ' = ' .. (self.neg_correct*100.0/self.neg_total) .. '%')
end

return M
