cv = require 'cv'
require 'cv.cudawarping'
require 'cv.imgcodecs'
require 'draw'
require 'fbnn'
matio = require 'matio'
require 'svm'

local Processor = require 'Processor'
local M = torch.class('CaltechProcessor', 'Processor')

local function defineSlidingWindowOptions(cmd)
  cmd:option('-windowSizeX', 30, 'width of sliding window')
  cmd:option('-windowSizeY', 50, 'height of sliding window')
  cmd:option('-windowStrideX', 0.5, 'horizontal stride of sliding window as fraction of width')
  cmd:option('-windowStrideY', 0.5, 'vertical stride of sliding window as fraction of height')
  cmd:option('-windowScales', 2, 'how many times to downscale window (0 = no downscaling)')
  cmd:option('-windowDownscaling', 0.75, 'what percent to downscale window')
  cmd:option('-windowIOU', 0.5, 'what IOU to count as a positive example')
end

function M:__init()
  self.cmd:option('-imageSize', 113, 'What to resize to.')
  self.cmd:option('-negativesWeight', 1, 'Relative weight of negative examples')
  self.cmd:option('-svm', '', 'SVM to use')
  self.cmd:option('-layer', 'fc7', 'layer to use as SVM input')
  self.cmd:option('-drawBoxes', '', 'set a directory to save images of losses')
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

  local w = self.processorOpts.negativesWeight
  local weights = torch.Tensor{w/(1+w), 1/(1+w)} * 2
  self.criterion = nn.TrueNLLCriterion(weights)
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
    local color
    if labels[i] == 2 then
      posTotal = posTotal + 1
      if pred[i] == 2 then
        posCorrect = posCorrect + 1
        color = {0, 1.0, 0} -- green - true positive
      else
        color = {1.0, 0, 0} -- red - false negative
      end
    else
      negTotal = negTotal + 1
      if pred[i] == 1 then
        negCorrect = negCorrect + 1
      else
        color = {0, 0, 1.0} -- blue - false positive
      end
    end
    if processor.processorOpts.drawBoxes ~= '' and color then
      draw.rectangle(processor.currentImage, processor.windowX[i], processor.windowY[i], processor.windowX[i] + processor.windowSizeX - 1, processor.windowY[i] + processor.windowSizeY - 1, 1, color)
    end
  end
  return {posCorrect, negCorrect, posTotal, negTotal}
end

function M:resetStats()
  self.stats = {}
  self.stats.posCorrect = 0
  self.stats.negCorrect = 0
  self.stats.posTotal = 0
  self.stats.negTotal = 0
end

function M:accStats(new_stats)
  self.stats.posCorrect = self.stats.posCorrect + new_stats[1]
  self.stats.negCorrect = self.stats.negCorrect + new_stats[2]
  self.stats.posTotal = self.stats.posTotal + new_stats[3]
  self.stats.negTotal = self.stats.negTotal + new_stats[4]
end

function M:printStats()
  print('  Accuracy: ' .. (self.stats.posCorrect + self.stats.negCorrect) .. '/' .. (self.stats.posTotal + self.stats.negTotal) .. ' = ' .. ((self.stats.posCorrect + self.stats.negCorrect)*100.0/(self.stats.posTotal + self.stats.negTotal)) .. '%')
  print('  Positive Accuracy: ' .. self.stats.posCorrect .. '/' .. self.stats.posTotal .. ' = ' .. (self.stats.posCorrect*100.0/self.stats.posTotal) .. '%')
  print('  Negative Accuracy: ' .. self.stats.negCorrect .. '/' .. self.stats.negTotal .. ' = ' .. (self.stats.negCorrect*100.0/self.stats.negTotal) .. '%')
end

local function findSlidingWindows(outputPatches, outputLabels, img, bboxes, scale, start)
  local min = math.min
  local max = math.max
  local floor = math.floor
  local ceil = math.ceil

  local sz = processor.processorOpts.imageSize
  local sizex = ceil(processor.processorOpts.windowSizeX * scale)
  processor.windowSizeX = sizex
  local sizey = ceil(processor.processorOpts.windowSizeY * scale)
  processor.windowSizeY = sizey
  local sx = max(1, floor(processor.processorOpts.windowStrideX * sizex))
  local sy = max(1, floor(processor.processorOpts.windowStrideY * sizey))
  local h = img:size(2)
  local w = img:size(3)
  local c = img:size(1)
  local nPatches = ceil((h-sizey+1)/sy)*ceil((w-sizex+1)/sx)
  if start > nPatches then
    return false
  end
  local finish = min(start + opts.batchSize - 1, nPatches)
  if finish == nPatches - 1 then finish = nPatches - 2 end
  local count = finish - start + 1

  outputPatches:resize(count, c, sz, sz)
  outputLabels:resize(count):fill(1)

  if img.getDevice then
    img = img:permute(2, 3, 1)
  end

  local id = 0
  local cnt = 0
  local nCols = 0
  processor.windowX = {}
  processor.windowY = {}
  for j=1,h-sizey+1,sy do
    for k=1,w-sizex+1,sx do
      id = id + 1
      if j == 1 then nCols = nCols + 1 end
      if id >= start and id <= finish then
        cnt = cnt + 1
        if img.getDevice then
          assert(c == 3)
          local window = img[{{j, j+sizey-1}, {k, k+sizex-1}}]:clone()
          outputPatches[cnt] = cv.cuda.resize{window, {sz, sz}}:permute(3, 1, 2)
        else
          outputPatches[cnt] = image.scale(img[{{}, {j, j+sizey-1}, {k, k+sizex-1}}], sz, sz):cuda()
        end
        processor.windowX[cnt] = k
        processor.windowY[cnt] = j
      end
    end
  end

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
          id = j*nCols+k+1
          if id >= start and id <= finish then
            local XA1 = k*sx
            local XA2 = k*sx+sizex
            local YA1 = j*sy
            local YA2 = j*sy+sizey

            local SI = max(0, min(XA2, XB2) - max(XA1, XB1)) *
                       max(0, min(YA2, YB2) - max(YA1, YB1))
            local SU = SA + SB - SI
            if SI/SU > processor.processorOpts.windowIOU then
              outputLabels[id-start+1] = 2
            end
          end
        end
      end
    end
  end
  return true
end

function M.forward(inputs, deterministic)
  local outputs
  if not(deterministic) then
    outputs = model:forward(inputs)
  else
    outputs = model:forward(inputs, true)
  end
  if processor.processorOpts.svm ~= '' then
    outputs = findModuleByName(model, processor.processorOpts.layer).output
  end
  return outputs
end

function M.test(pathNames, inputs)
  local min = math.min
  local aggLoss = 0
  local aggTotal = 0
  local aggStats = {}
  local patches = torch.Tensor()
  local labels = torch.Tensor()
  if nGPU > 0 then
    patches = patches:cuda()
    labels = labels:cuda()
  end
  local outdir
  if processor.processorOpts.drawBoxes ~= '' then
    outdir = processor.processorOpts.drawBoxes .. '/'
    paths.mkdir(outdir)
  end
  local nInputs = #pathNames
  for i=1,nInputs do
    local path = pathNames[i]
    processor.currentImage = image.load(path, 3)
    local bboxes = processor.bboxes[path:find('val') and 1 or 2]
    local index = tonumber(paths.basename(path, 'png'))
    local scale = 1
    for s=0,processor.processorOpts.windowScales do
      local start = 1
      while true do
        if not findSlidingWindows(patches, labels, inputs[i], bboxes[index], scale, start) then break end
        local loss, total, stats = processor.testWithLabels(nil, patches, labels)
        aggLoss = aggLoss + loss
        aggTotal = aggTotal + total
        for l=1,#stats do
          if not(aggStats[l]) then aggStats[l] = 0 end
          aggStats[l] = aggStats[l] + stats[l]
        end
        start = start + labels:size(1)
      end
      scale = scale * processor.processorOpts.windowDownscaling
    end
    if outdir and bboxes[index]:nElement() ~= 0 then
      for j=1,bboxes[index]:size(1) do
        local bbox = bboxes[index][j]
        local XB1 = bbox[1]
        local XB2 = bbox[1]+bbox[3]
        local YB1 = bbox[2]
        local YB2 = bbox[2]+bbox[4]
        draw.rectangle(processor.currentImage, XB1, YB1, XB2, YB2, 1, {1.0, 1.0, 0})
      end
      image.save(outdir .. paths.basename(path), processor.currentImage)
    end
  end
  collectgarbage()
  return aggLoss, aggTotal, aggStats
end

return M
