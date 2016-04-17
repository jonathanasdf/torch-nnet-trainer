require 'draw'
require 'fbnn'
local nms = require 'nms'
require 'optim'
require 'svm'

local Transforms = require 'Transforms'
local Processor = require 'Processor'
local M = torch.class('CaltechProcessor', 'Processor')

local function defineSlidingWindowOptions(cmd)
  cmd:option('-windowSizeX', 41, 'width of sliding window')
  cmd:option('-windowSizeY', 100, 'height of sliding window')
  cmd:option('-windowStride', 0.3, 'stride of sliding window as fraction of size')
  cmd:option('-windowScales', 2, 'how many times to downscale window (0 = no downscaling)')
  cmd:option('-windowDownscaling', 0.5, 'what percent to downscale window')
  cmd:option('-posMinBoxSize', 225, 'minimum ground truth box size to be counted as a positive example')
end

function M:__init()
  self.cmd:option('-imageSize', 113, 'What to resize to.')
  self.cmd:option('-inceptionPreprocessing', false, 'Preprocess for inception models (RGB, [-1, 1))')
  self.cmd:option('-caffePreprocessing', false, 'Preprocess for caffe models (BGR, [0, 255])')
  self.cmd:option('-flip', 0.5, 'Probability to do horizontal flip (for training)')
  self.cmd:option('-overlap', 0.5, 'what IOU to count as a positive example, and for NMS')
  self.cmd:option('-negativesWeight', 1, 'Relative weight of negative examples')
  self.cmd:option('-svm', '', 'SVM to use')
  self.cmd:option('-layer', 'fc7', 'layer to use as SVM input')
  self.cmd:option('-drawBoxes', '', 'set a directory to save images of losses')
  self.cmd:option('-drawBoxThreshold', 0.95, 'score threshold to count as positive')
  self.cmd:option('-fullEval', '', 'set a directory to use for full evaluation')
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

  local matio = require 'matio'
  self.bboxes = {
    matio.load('/file/caltech/val/box.mat', 'box'),
    matio.load('/file/caltech/test/box.mat', 'box')
  }

  self.mappings = {{}, {}}
  for line in io.lines('/file/caltech/val/mapping.txt') do
    local s = line:split()
    self.mappings[1][s[1]] = s[2]
  end
  for line in io.lines('/file/caltech/test/mapping.txt') do
    local s = line:split()
    self.mappings[2][s[1]] = s[2]
  end

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

  if self.processorOpts.drawBoxes ~= '' then
    paths.mkdir(self.processorOpts.drawBoxes)
  end
  if self.processorOpts.fullEval ~= '' then
    if opts.epochSize ~= -1 then
      error('fullEval requires epochSize == -1')
    end
    if not(opts.resume) or opts.resume == '' then
      if paths.dir(self.processorOpts.fullEval) ~= nil then
        error('fullEval directory exists! Aborting')
      end
      paths.mkdir(self.processorOpts.fullEval)
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
  if processorOpts.svm == '' then
    pred = outputs[{{},{2}}]:float()
  else
    if not(processorOpts.svmmodel) then
      processorOpts.svmmodel = torch.load(processorOpts.svm)
    end
    local data = convertTensorToSVMLight(labels, outputs)
    pred = liblinear.predict(data, processorOpts.svmmodel, '-q')
  end
  return {pred, labels}
end

function M:resetStats()
  self.stats = optim.ConfusionMatrix({'no person', 'person'})
end

function M:accStats(new_stats)
  self.stats:batchAdd(torch.ge(new_stats[1], self.processorOpts.overlap) + 1, new_stats[2])
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

local function maxOverlap(bboxes, coords)
  local min = math.min
  local max = math.max
  local m = 0
  if bboxes:nElement() ~= 0 then
    local SA = (coords[3]-coords[1]+1)*(coords[4]-coords[2]+1)
    for i=1,bboxes:size(1) do
      local XB1 = bboxes[i][1]
      local XB2 = bboxes[i][1]+bboxes[i][3]-1
      local YB1 = bboxes[i][2]
      local YB2 = bboxes[i][2]+bboxes[i][4]-1
      local SB = bboxes[i][3]*bboxes[i][4]

      if SB >= processorOpts.posMinBoxSize then
        local SI = max(0, min(coords[3], XB2) - max(coords[1], XB1) + 1) *
                   max(0, min(coords[4], YB2) - max(coords[2], YB1) + 1)
        local SU = SA + SB - SI
        if SI/SU >= m then
          m = SI/SU
        end
      end
    end
  end
  return m
end

local function findSlidingWindows(outputPatches, outputCoords, outputLabels, img, bboxes, scale, start)
  local min = math.min
  local max = math.max
  local floor = math.floor
  local ceil = math.ceil

  local sz = processorOpts.imageSize
  local sizex = ceil(processorOpts.windowSizeX * scale)
  processor.windowSizeX = sizex
  local sizey = ceil(processorOpts.windowSizeY * scale)
  processor.windowSizeY = sizey
  local sx = max(1, floor(processorOpts.windowStride * sizex))
  local sy = max(1, floor(processorOpts.windowStride * sizey))
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
  outputCoords:resize(count, 4)
  outputLabels:resize(count)

  if img.getDevice then
    img = img:permute(2, 3, 1)
  end

  local id = 0
  local cnt = 0
  local nCols = 0
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
        outputCoords[cnt] = torch.Tensor{k, j, k+sizex-1, j+sizey-1}
        outputLabels[cnt] = maxOverlap(bboxes, outputCoords[cnt]) >= processorOpts.overlap and 2 or 1
      end
    end
  end
  return true
end

function M.forward(inputs, deterministic)
  local outputs = model:forward(inputs, deterministic)
  if processorOpts.svm ~= '' then
    outputs = findModuleByName(model, processorOpts.layer).output
  end
  outputs = outputs:view(inputs:size(1), -1)
  return outputs
end

function M.test(pathNames, inputs)
  local min = math.min
  local aggLoss = 0
  local aggTotal = 0
  local aggStats = {torch.Tensor(), torch.Tensor()}
  local coords = torch.Tensor()
  local patches = torch.Tensor()
  local labels = torch.Tensor()
  if nGPU > 0 then
    patches = patches:cuda()
    labels = labels:cuda()
  end
  local nInputs = #pathNames
  for i=1,nInputs do
    local boxes = torch.Tensor()
    local path = pathNames[i]
    local bboxes = processor.bboxes[path:find('val') and 1 or 2]
    local index = tonumber(paths.basename(path, 'png'))
    local scale = 1
    for s=0,processorOpts.windowScales do
      local start = 1
      while true do
        if not findSlidingWindows(patches, coords, labels, inputs[i], bboxes[index], scale, start) then break end
        local loss, total, stats = processor.testWithLabels(nil, patches, labels)
        boxes = cat(boxes, cat(coords, stats[1]:view(-1, 1), 2), 1)
        aggLoss = aggLoss + loss
        aggTotal = aggTotal + total
        for l=1,#stats do
          aggStats[l] = cat(aggStats[l], stats[l], 1)
        end
        start = start + labels:size(1)
      end
      scale = scale * processorOpts.windowDownscaling
    end

    if processorOpts.drawBoxes or processorOpts.fullEval then
      local I = nms(boxes, processorOpts.overlap)
      boxes = boxes:index(1, I)
    end

    if processorOpts.drawBoxes then
      local img = image.load(path, 3)
      for j=1,boxes:size(1) do
        local label = maxOverlap(bboxes[index], boxes[j]) >= processorOpts.overlap and 2 or 1
        local p = boxes[j][5]
        local t = processorOpts.drawBoxThreshold
        local color
        if label == 2 then
          if p >= t then
            -- true positive - green
            local h, s, l = rgbToHsl(0, 1, 0)
            l = l*(0.1 + (p-t)/(1-t)*0.9)
            color = {hslToRgb(h, s, l)}
          else
            -- false negative - red
            local h, s, l = rgbToHsl(1, 0, 0)
            l = l*(1-p)
            color = {hslToRgb(h, s, l)}
          end
        else
          if p >= t then
            -- false positive - blue
            local h, s, l = rgbToHsl(0, 0, 1)
            l = l*(0.1 + (p-t)/(1-t)*0.9)
            color = {hslToRgb(h, s, l)}
          end
        end
        if color then
          draw.rectangle(img, boxes[j][1], boxes[j][2], boxes[j][3], boxes[j][4], 1, color)
        end
      end
      if bboxes[index]:nElement() ~= 0 then
        -- ground truth - yellow
        for j=1,bboxes[index]:size(1) do
          local bbox = bboxes[index][j]
          local XB1 = bbox[1]
          local XB2 = bbox[1]+bbox[3]-1
          local YB1 = bbox[2]
          local YB2 = bbox[2]+bbox[4]-1
          draw.rectangle(img, XB1, YB1, XB2, YB2, 1, {1.0, 1.0, 0})
        end
      end
      image.save(processorOpts.drawBoxes .. '/' .. paths.basename(path), img)
    end

    if processorOpts.fullEval then
      local mapping = processor.mappings[path:find('val') and 1 or 2]
      local filename = processorOpts.fullEval .. '/res/' .. mapping[paths.basename(path)]
      paths.mkdir(paths.dirname(filename))
      local file, err = io.open(filename, 'w')
      if not(file) then error(err) end
      for j=1,boxes:size(1) do
        file:write(boxes[j][1], ' ',  boxes[j][2], ' ',
                   boxes[j][3]-boxes[j][1]+1, ' ', boxes[j][4]-boxes[j][2]+1, ' ',
                   boxes[j][5], '\n')
      end
      file:close()
    end
  end
  collectgarbage()
  return aggLoss, aggTotal, aggStats
end

return M
