require 'draw'
require 'fbnn'
matio = require 'matio'
require 'optim'
require 'svm'

local Transforms = require 'Transforms'
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
  cmd:option('-posMinBoxSize', 225, 'minimum ground truth box size to be counted as a positive example')
end

function M:__init()
  self.cmd:option('-imageSize', 113, 'What to resize to.')
  self.cmd:option('-inceptionPreprocessing', false, 'Preprocess for inception models (RGB, [-1, 1))')
  self.cmd:option('-caffePreprocessing', false, 'Preprocess for caffe models (BGR, [0, 255])')
  self.cmd:option('-flip', 0.5, 'Probability to do horizontal flip (for training)')
  self.cmd:option('-negativesWeight', 1, 'Relative weight of negative examples')
  self.cmd:option('-svm', '', 'SVM to use')
  self.cmd:option('-layer', 'fc7', 'layer to use as SVM input')
  self.cmd:option('-drawBoxes', '', 'set a directory to save images of losses')
  defineSlidingWindowOptions(self.cmd)
  Processor.__init(self)

  if self.processorOpts.inceptionPreprocessing then
    -- no mean normalization
  elseif self.processorOpts.caffePreprocessing then
    self.processorOpts.meanPixel = torch.Tensor{103.939, 116.779, 123.68}:view(3, 1, 1)
  else
    self.processorOpts.meanPixel = torch.Tensor{0.485, 0.456, 0.406}:view(3, 1, 1)
    self.processorOpts.std = torch.Tensor{0.229, 0.224, 0.225}:view(3, 1, 1)
  end

  self.bboxes = {
    matio.load('/file/caltech/val/box.mat', 'box'),
    matio.load('/file/caltech/test/box.mat', 'box')
  }

  if opts.logdir and opts.epochs then
    self.trainGraph = gnuplot.pngfigure(opts.logdir .. 'train.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('acc')
    gnuplot.grid(true)
    self.trainPosAcc = torch.Tensor(opts.epochs)
    self.trainNegAcc = torch.Tensor(opts.epochs)
    self.trainAcc = torch.Tensor(opts.epochs)

    if opts.val then
      self.valGraph = gnuplot.pngfigure(opts.logdir .. 'val.png')
      gnuplot.xlabel('epoch')
      gnuplot.ylabel('acc')
      gnuplot.grid(true)
      self.valPosAcc = torch.Tensor(opts.epochs)
      self.valNegAcc = torch.Tensor(opts.epochs)
      self.valAcc = torch.Tensor(opts.epochs)
    end
  end

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
  local img = image.load(path, 3)
  if isTraining then
    if img:size(2) ~= processorOpts.imageSize or img:size(3) ~= processorOpts.imageSize then
      img = image.scale(img, processorOpts.imageSize, processorOpts.imageSize)
    end
    img = Transforms.HorizontalFlip(processorOpts.flip)(img)
    img = Transforms.ColorJitter{brightness = 0.4, contrast = 0.4, saturation = 0.4}(img)
  end

  if processorOpts.inceptionPreprocessing then
    img = (img * 255 - 128) / 128
  elseif processorOpts.caffePreprocessing then
    img = (convertRGBBGR(img) * 255):csub(processorOpts.meanPixel:expandAs(img))
  else
    img = img:csub(processorOpts.meanPixel:expandAs(img)):cdiv(processorOpts.std:expandAs(img))
  end
  return img
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
  else
    if not(processor.processorOpts.svmmodel) then
      processor.processorOpts.svmmodel = torch.load(processor.processorOpts.svm)
    end
    local data = convertTensorToSVMLight(labels, outputs)
    pred = liblinear.predict(data, processor.processorOpts.svmmodel, '-q')
  end
  pred = pred:view(-1)

  for i=1,labels:size(1) do
    local color
    if labels[i] == 2 then
      if pred[i] == 2 then
        color = {0, 1.0, 0} -- green - true positive
      else
        color = {1.0, 0, 0} -- red - false negative
      end
    else
      if pred[i] == 2 then
        color = {0, 0, 1.0} -- blue - false positive
      end
    end
    if processor.processorOpts.drawBoxes ~= '' and color then
      draw.rectangle(processor.currentImage, processor.windowX[i], processor.windowY[i], processor.windowX[i] + processor.windowSizeX - 1, processor.windowY[i] + processor.windowSizeY - 1, 1, color)
    end
  end
  return {pred, labels}
end

function M:resetStats()
  self.stats = optim.ConfusionMatrix({'no person', 'person'})
end

function M:accStats(new_stats)
  self.stats:batchAdd(new_stats[1], new_stats[2])
end

function M:processStats(phase)
  self.stats:updateValids()
  if phase == 'train' and self.trainGraph then
    self.trainPosAcc[opts.epoch] = self.stats.valids[1]
    self.trainNegAcc[opts.epoch] = self.stats.valids[2]
    self.trainAcc[opts.epoch] = self.stats.averageValid

    local x = torch.range(1, opts.epoch):long()
    gnuplot.figure(self.trainGraph)
    gnuplot.plot({'pos', x, self.trainPosAcc:index(1, x), '+-'}, {'neg', x, self.trainNegAcc:index(1, x), '+-'}, {'overall', x, self.trainAcc:index(1, x), '-'})
    gnuplot.plotflush()
  elseif phase == 'val' and self.valGraph and opts.epoch >= opts.valEvery then
    self.valPosAcc[opts.epoch] = self.stats.valids[1]
    self.valNegAcc[opts.epoch] = self.stats.valids[2]
    self.valAcc[opts.epoch] = self.stats.averageValid

    local x = torch.range(opts.valEvery, opts.epoch, opts.valEvery):long()
    gnuplot.figure(self.valGraph)
    gnuplot.plot({'pos', x, self.valPosAcc:index(1, x), '+-'}, {'neg', x, self.valNegAcc:index(1, x), '+-'}, {'overall', x, self.valAcc:index(1, x), '-'})
    gnuplot.plotflush()
  end
  return tostring(self.stats)
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
          local window = img[{{j, j+sizey-1}, {k, k+sizex-1}}]:contiguous()
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

      if SB >= processor.processorOpts.posMinBoxSize then
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
  end
  return true
end

function M.forward(inputs, deterministic)
  local outputs = model:forward(inputs, deterministic)
  if processor.processorOpts.svm ~= '' then
    outputs = findModuleByName(model, processor.processorOpts.layer).output
  end
  outputs = outputs:view(inputs:size(1), -1)
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
    if outdir then
      processor.currentImage = image.load(path, 3)
    end
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
          if not(aggStats[l]) then
            aggStats[l] = stats[l]:clone()
          else
            aggStats[l] = cat({aggStats[l], stats[l]}, 1)
          end
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
