cv = require 'cv'
require 'cv.cudawarping'
require 'cv.imgcodecs'
require 'fbnn'
matio = require 'matio'
require 'svm'

local Processor = require 'Processor'
local M = torch.class('CaltechProcessor', 'Processor')

local function defineSlidingWindowOptions(cmd)
  cmd:option('-windowSizeX', 30, 'width of sliding window')
  cmd:option('-windowSizeY', 50, 'height of sliding window')
  cmd:option('-windowStrideX', 15, 'horizontal stride of sliding window')
  cmd:option('-windowStrideY', 25, 'vertical stride of sliding window')
  cmd:option('-windowScales', 2, 'how many times to downscale window (0 = no downscaling)')
  cmd:option('-windowDownscaling', 0.75, 'what percent to downscale window')
  cmd:option('-windowIOU', 0.5, 'what IOU to count as a positive example')
end

function M:__init()
  self.cmd:option('-imageSize', 113, 'What to resize to.')
  self.cmd:option('-svm', '', 'SVM to use')
  self.cmd:option('-layer', 'fc7', 'layer to use as SVM input')
  defineSlidingWindowOptions(self.cmd)
  Processor.__init(self)

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

function M:initializeThreads()
  augmentThreadState(function()
    require 'svm'
  end)
  Processor.initializeThreads(self)
end

function M.preprocess(path, isTraining, processorOpts)
  local img = cv.imread{path, cv.IMREAD_COLOR}:float()
  if nGPU > 0 then
    img = img:cuda()
  end
  if isTraining and (img:size(1) ~= processorOpts.imageSize or img:size(2) ~= processorOpts.imageSize) then
    if nGPU > 0 then
      img = cv.cuda.resize{img, {processorOpts.imageSize, processorOpts.imageSize}}
    else
      img = img:permute(3, 1, 2)
      img = image.scale(img, processorOpts.imageSize, processorOpts.imageSize)
      img = img:permute(2, 3, 1)
    end
  end
  return img:csub(processorOpts.meanPixel[gpu]:expandAs(img)):permute(3, 1, 2)
end

function M.getLabels(pathNames)
  local labels = torch.Tensor(#pathNames)
  if nGPU > 0 then labels = labels:cuda() end
  for i=1,#pathNames do
    labels[i] = pathNames[i]:find('neg') and 1 or 2
  end
  return labels
end

function M.calcStats(pathNames, outputs, labels)
  local pred
  if processor.processorOpts.svm == '' then
    _, pred = torch.max(outputs, 2)
    pred = pred:squeeze()
  else
    if not(processor.processorOpts.svmmodel) then
      processor.processorOpts.svmmodel = torch.load(processor.processorOpts.svm)
    end
    local data = convertTensorToSVMLight(labels, outputs)
    pred = liblinear.predict(data, processor.processorOpts.svmmodel, '-q')
  end

  local posCorrect = 0
  local negCorrect = 0
  local posTotal = 0
  local negTotal = 0
  for i=1,labels:size(1) do
    if labels[i] == 2 then
      posTotal = posTotal + 1
      if pred[i] == 2 then
        posCorrect = posCorrect + 1
      end
    else
      negTotal = negTotal + 1
      if pred[i] == 1 then
        negCorrect = negCorrect + 1
      end
    end
  end
  return posCorrect, negCorrect, posTotal, negTotal
end

function M:resetStats()
  self.posCorrect = 0
  self.negCorrect = 0
  self.posTotal = 0
  self.negTotal = 0
end

function M:accStats(...)
  a, b, c, d = ...
  self.posCorrect = self.posCorrect + a
  self.negCorrect = self.negCorrect + b
  self.posTotal = self.posTotal + c
  self.negTotal = self.negTotal + d
end

function M:printStats()
  print('  Accuracy: ' .. (self.posCorrect + self.negCorrect) .. '/' .. (self.posTotal + self.negTotal) .. ' = ' .. ((self.posCorrect + self.negCorrect)*100.0/(self.posTotal + self.negTotal)) .. '%')
  print('  Positive Accuracy: ' .. self.posCorrect .. '/' .. self.posTotal .. ' = ' .. (self.posCorrect*100.0/self.posTotal) .. '%')
  print('  Negative Accuracy: ' .. self.negCorrect .. '/' .. self.negTotal .. ' = ' .. (self.negCorrect*100.0/self.negTotal) .. '%')
end

local function findSlidingWindows(path, img, bboxes, scale)
  local min = math.min
  local max = math.max
  local floor = math.floor
  local ceil = math.ceil

  local sz = processor.processorOpts.imageSize
  local sizex = ceil(processor.processorOpts.windowSizeX * scale)
  local sizey = ceil(processor.processorOpts.windowSizeY * scale)
  local sx = max(1, floor(processor.processorOpts.windowStrideX * scale))
  local sy = max(1, floor(processor.processorOpts.windowStrideY * scale))
  local h = img:size(2)
  local w = img:size(3)
  local c = img:size(1)
  if img.getDevice then
    img = img:permute(2, 3, 1)
  end

  local nPatches = ceil((h-sizey+1)/sy)*ceil((w-sizex+1)/sx)
  local patches = {}

  local count = 0
  local nCols = 0
  for j=1,h-sizey+1,sy do
    for k=1,w-sizex+1,sx do
      count = count + 1
      if j == 1 then nCols = nCols + 1 end
      if img.getDevice then
        assert(c == 3)
        local window = img[{{j, j+sizey-1}, {k, k+sizex-1}}]:clone()
        patches[count] = cv.cuda.resize{window, {sz, sz}}:permute(3, 1, 2)
      else
        patches[count] = image.scale(img[{{}, {j, j+sizey-1}, {k, k+sizex-1}}], sz, sz)
      end
    end
  end
  assert(count == nPatches)
  patches = tableToBatchTensor(patches)
  if nGPU > 0 and not(patches.getDevice) then patches = patches:cuda() end

  local labels = torch.ones(nPatches)
  if bboxes:nElement() ~= 0 then
    local SA = sizex*sizey
    for i=1,bboxes:size(1) do
      local XB1 = bboxes[i][1]
      local XB2 = bboxes[i][1]+bboxes[i][3]
      local YB1 = bboxes[i][2]
      local YB2 = bboxes[i][2]+bboxes[i][4]
      local SB = bboxes[i][3]*bboxes[i][4]

      local left = max(0, ceil((XB1-sizex)/sx))
      local right = floor(XB2/sx)
      local top = max(0, ceil((YB1-sizey)/sy))
      local bottom = floor(YB2/sy)
      for j=top,bottom do
        for k=left,right do
          local XA1 = k*sx
          local XA2 = k*sx+sizex
          local YA1 = j*sy
          local YA2 = j*sy+sizey

          local SI = max(0, min(XA2, XB2) - max(XA1, XB1)) *
                     max(0, min(YA2, YB2) - max(YA1, YB1))
          local SU = SA + SB - SI
          if SI/SU > processor.processorOpts.windowIOU then
            labels[j*nCols+k+1] = 2
          end
        end
      end
    end
  end
  return patches, labels
end

function M.forward(inputs, deterministic)
  if not(deterministic) then
    return model:forward(inputs)
  else
    --print(#inputs)
    local outputs = model:forward(inputs, true)
    --print(#outputs)
    if processor.processorOpts.svm ~= '' then
      outputs = findModuleByName(model, processor.processorOpts.layer).output
    end
    return outputs
  end
end

function M.test(pathNames, inputs)
  local aggLoss = 0
  local aggTotal = 0
  local aggStats = {}
  local nInputs = #pathNames
  for i=1,nInputs do
    local path = pathNames[i]
    local bboxes = processor.bboxes[path:find('val') and 1 or 2]
    local index = tonumber(paths.basename(path, 'png'))
    local scale = 1
    for s=0,processor.processorOpts.windowScales do
      local patches, labels = findSlidingWindows(path, inputs[i], bboxes[index], scale)
      local nPatches = labels:size(1)
      for j=1,nPatches,opts.batchSize do
        local k = j+opts.batchSize-1
        if k > nPatches then k = nPatches end
        local res = {processor.testWithLabels(nil, patches[{{j, k}}], labels[{{j, k}}])}
        aggLoss = aggLoss + res[1]
        aggTotal = aggTotal + res[2]
        for l=1,#res-2 do
          if not(aggStats[l]) then aggStats[l] = 0 end
          aggStats[l] = aggStats[l] + res[l+2]
        end
      end
      scale = scale * processor.processorOpts.windowDownscaling
    end
  end
  collectgarbage()
  return aggLoss, aggTotal, unpack(aggStats)
end

return M
