require 'fbnn'

local Transforms = require 'Transforms'
local Processor = require 'Processor'
local M = torch.class('CaltechProcessor', 'Processor')

local function defineSlidingWindowOptions(cmd)
  cmd:option('-testImageWidth', 640, 'width of input test image')
  cmd:option('-testImageHeight', 480, 'height of input test image')
  cmd:option('-windowHeight', 200, 'height of sliding window')
  cmd:option('-windowAspectRatio', 0.41, 'aspect ratio of sliding window')
  cmd:option('-windowStride', 0.3, 'stride of sliding window as fraction of size')
  cmd:option('-windowScales', 3, 'number of sliding window scales (1 = no downscaling)')
  cmd:option('-windowDownscaling', 0.5, 'downscaling factor')
  cmd:option('-posMinHeight', 50, 'minimum ground truth box height to be counted as a positive example')
end

function M:__init(model, processorOpts)
  self.cmd:option('-imageSize', 113, 'input patch size')
  self.cmd:option('-inceptionPreprocessing', false, 'preprocess for inception models (RGB, [-1, 1))')
  self.cmd:option('-caffePreprocessing', false, 'preprocess for caffe models (BGR, [0, 255])')
  self.cmd:option('-flip', 0.5, 'probability to do horizontal flip (for training)')
  self.cmd:option('-overlap', 0.5, 'what IOU to count as a positive example, and for NMS')
  self.cmd:option('-negativesWeight', 1, 'relative weight of negative examples')
  self.cmd:option('-svm', '', 'SVM to use')
  self.cmd:option('-layer', 'fc7', 'layer to use as SVM input')
  self.cmd:option('-drawBoxes', '', 'set a directory to save images of losses')
  self.cmd:option('-drawBoxesThreshold', 0.995, 'score threshold to count as positive')
  self.cmd:option('-drawROC', '', 'set a directory to use for full evaluation')
  self.cmd:option('-name', 'Result', 'name to use on ROC graph')
  defineSlidingWindowOptions(self.cmd)
  Processor.__init(self, model, processorOpts)

  if self.processorOpts.inceptionPreprocessing then
    -- no mean normalization
  elseif self.processorOpts.caffePreprocessing then
    self.processorOpts.meanPixel = torch.Tensor{103.939, 116.779, 123.68}:view(3, 1, 1)
  else
    self.processorOpts.meanPixel = torch.Tensor{0.485, 0.456, 0.406}:view(3, 1, 1)
    self.processorOpts.std = torch.Tensor{0.229, 0.224, 0.225}:view(3, 1, 1)
  end

  local w = self.processorOpts.negativesWeight
  local weights = torch.Tensor{w/(1+w), 1/(1+w)} * 2
  self.criterion = nn.TrueNLLCriterion(weights)
  self.criterion.sizeAverage = false
  if nGPU > 0 then
    self.criterion = self.criterion:cuda()
  end

  local matio = require 'matio'
  self.bboxes = {
    matio.load('/file/caltech/val/box.mat', 'box'),
    matio.load('/file/caltech/test/box.mat', 'box')
  }
  for i=1,#self.bboxes do
    for j=1,#self.bboxes[i] do
      local bboxes = self.bboxes[i][j]
      if bboxes:nElement() > 0 then
        local keep = maskToLongTensor(bboxes[{{}, 4}]:ge(self.processorOpts.posMinHeight))
        if keep:nElement() == 0 then
          self.bboxes[i][j] = self.bboxes[i][j].new()
        else
          self.bboxes[i][j] = bboxes:index(1, keep)
          bboxes = self.bboxes[i][j]
          bboxes[{{}, 3}] = bboxes[{{}, 1}] + bboxes[{{}, 3}] - 1
          bboxes[{{}, 4}] = bboxes[{{}, 2}] + bboxes[{{}, 4}] - 1
        end
      end
    end
  end

  local slidingWindows = {}
  local nWindows = 0
  local w = self.processorOpts.testImageWidth
  local h = self.processorOpts.testImageHeight
  local scale = 1
  for s=1,self.processorOpts.windowScales do
    local sizex = math.ceil(self.processorOpts.windowHeight * self.processorOpts.windowAspectRatio * scale)
    local sizey = math.ceil(self.processorOpts.windowHeight * scale)
    local stridex = math.max(1, math.floor(self.processorOpts.windowStride * sizex))
    local stridey = math.max(1, math.floor(self.processorOpts.windowStride * sizey))
    assert(w >= sizex)
    assert(h >= sizey)

    local windows = {}
    for j=1,h-sizey+1,stridey do
      for k=1,w-sizex+1,stridex do
        windows[#windows+1] = torch.Tensor{k, j, k+sizex-1, j+sizey-1}:view(1, 4)
      end
    end
    slidingWindows[s] = windows
    nWindows = nWindows + #windows
    scale = scale * self.processorOpts.windowDownscaling
  end
  print('Windows per image:', nWindows)
  assert(nWindows > 0)
  self.slidingWindows = torch.Tensor(nWindows, 4)
  self.slidingWindowScaleOffsets = {}
  local n = 0
  for i=1,self.processorOpts.windowScales do
    self.slidingWindowScaleOffsets[i] = n
    for j=1,#slidingWindows[i] do
      self.slidingWindows[n+j] = slidingWindows[i][j]
    end
    n = n + #slidingWindows[i]
  end
  self.slidingWindowScaleOffsets[self.processorOpts.windowScales+1] = n

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

  if self.processorOpts.drawROC ~= '' then
    if not(opts.testing) then
      error('drawROC can only be used with Forward.lua')
    end
    if not(opts.resume) or opts.resume == '' then
      if paths.dir(self.processorOpts.drawROC) ~= nil then
        error('drawROC directory exists! Aborting')
      end
      paths.mkdir(self.processorOpts.drawROC)
    end
  end
  if self.processorOpts.drawBoxes ~= '' then
    paths.mkdir(self.processorOpts.drawBoxes)
  end
end

function M:initializeThreads()
  Processor.initializeThreads(self)
  augmentThreadState(function()
    require 'svm'
  end)
end

function M.preprocess(path, isTraining, processorOpts)
  local img = image.load(path, 3)
  if isTraining then
    local sz = processorOpts.imageSize
    if img:size(2) ~= sz or img:size(3) ~= sz then
      img = image.scale(img, sz, sz)
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
    pred = outputs[{{},2}]
  else
    if not(processorOpts.svmmodel) then
      processorOpts.svmmodel = torch.load(processorOpts.svm)
    end
    local data = convertTensorToSVMLight(labels, outputs)
    pred = liblinear.predict(data, processorOpts.svmmodel, '-q')
  end
  return {pred:float(), labels}
end

function M:resetStats()
  self.stats = optim.ConfusionMatrix({'no person', 'person'})
end

function M:accStats(new_stats)
  self.stats:batchAdd(new_stats[1]:ge(0.5) + 1, new_stats[2])
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

  if self.processorOpts.drawROC ~= '' then
     local has = {}
     for i=1,#opts.input do
       if opts.input[i]:find('val') then has['val'] = 1 end
       if opts.input[i]:find('test') then has['test'] = 1 end
     end
     local dataName
     for k,_ in pairs(has) do
       if not(dataName) then
         dataName = "{'" .. k .. "'"
       else
         dataName = dataName .. ", '" .. k .. "'"
       end
     end
     dataName = dataName .. '}'
     local cmd = "cd /file/caltech; dbEval('" .. self.processorOpts.drawROC .. "', " .. dataName .. ", '" .. self.processorOpts.name .. "')"
     print(runMatlab(cmd))
     print(readAll(self.processorOpts.drawROC .. '/eval/RocReasonable.txt'))
  end
  return tostring(self.stats)
end

local function maxOverlap(bboxes, boxes)
  local out = boxes.new(boxes:size(1)):fill(0)
  if bboxes:nElement() == 0 then
    return out
  end

  local XA1 = boxes[{{}, 1}]
  local YA1 = boxes[{{}, 2}]
  local XA2 = boxes[{{}, 3}]
  local YA2 = boxes[{{}, 4}]
  local SA = torch.cmul(XA2-XA1+1, YA2-YA1+1)

  for i=1,bboxes:size(1) do
    local XB1 = bboxes[i][1]
    local YB1 = bboxes[i][2]
    local XB2 = bboxes[i][3]
    local YB2 = bboxes[i][4]
    local SB = (XB2-XB1+1)*(YB2-YB1+1)
    local SI = torch.cmax(torch.cmin(XA2, XB2) - torch.cmax(XA1, XB1) + 1, 0):cmul(
               torch.cmax(torch.cmin(YA2, YB2) - torch.cmax(YA1, YB1) + 1, 0))
    local SU = SA + SB - SI
    out:cmax(SI:cdiv(SU))
  end
  return out
end

function M.forward(inputs, deterministic)
  local outputs = _model:forward(inputs, deterministic)
  if processorOpts.svm ~= '' then
    outputs = findModuleByName(_model, processorOpts.layer).output
  end
  outputs = outputs:view(inputs:size(1), -1)
  return outputs
end

function M.test(pathNames, inputs)
  local aggLoss = 0
  local aggTotal = 0
  local aggStats = {torch.Tensor(), torch.Tensor()}
  local nInputs = #pathNames
  for i=1,nInputs do
    local path = pathNames[i]
    local c = inputs[i]:size(1)
    local w = processorOpts.testImageWidth
    local h = processorOpts.testImageHeight
    local sz = processorOpts.imageSize
    local index = tonumber(paths.basename(path, 'png'))
    local bboxes = _processor.bboxes[path:find('val') and 1 or 2][index]
    local labels = maxOverlap(bboxes, _processor.slidingWindows):ge(processorOpts.overlap) + 1
    local patches = torch.Tensor()
    if nGPU > 0 then
      labels = labels:cuda()
      patches = patches:cuda()
    end
    local scores = torch.Tensor()

    local scale = 1
    for s=1,processorOpts.windowScales do
      collectgarbage()
      local sizex = ceil(processorOpts.windowHeight * processorOpts.windowAspectRatio * scale)
      local sizey = ceil(processorOpts.windowHeight * scale)
      local sx = sz / sizex
      local sy = sz / sizey
      local input = image.scale(inputs[i], ceil(sx*w), ceil(sy*h))

      local start = _processor.slidingWindowScaleOffsets[s]
      local nWindows = _processor.slidingWindowScaleOffsets[s+1] - start
      for j=1,nWindows,opts.batchSize do
        local count = min(j+opts.batchSize-1, nWindows) - j + 1
        if patches:dim() == 0 or count ~= patches:size(1) then
          patches:resize(count, c, sz, sz)
        end
        for k=1,count do
          local window = _processor.slidingWindows[start+j+k-1]
          local x = min(ceil((window[1]-1)*sx+1), input:size(3)-sz+1)
          local y = min(ceil((window[2]-1)*sy+1), input:size(2)-sz+1)
          patches[k] = input[{{}, {y, y+sz-1}, {x, x+sz-1}}]
        end
        local loss, total, stats = _processor.testWithLabels(path, patches, labels[{{start+j, start+j+count-1}}])
        scores = cat(scores, stats[1], 1)
        aggLoss = aggLoss + loss
        aggTotal = aggTotal + total
        for l=1,#stats do
          aggStats[l] = cat(aggStats[l], stats[l], 1)
        end
      end
      scale = scale * processorOpts.windowDownscaling
    end

    local boxes = _processor.slidingWindows
    if processorOpts.drawBoxes ~= '' or processorOpts.drawROC ~= '' then
      local nms = require 'nms'
      local I = nms(boxes, scores, processorOpts.overlap)
      boxes = boxes:index(1, I)
      labels = labels:index(1, I)
      scores = scores:index(1, I)
    end

    if processorOpts.drawBoxes ~= '' then
      local img = image.load(path, 3)
      img = image.scale(img, w, h)
      for j=1,boxes:size(1) do
        local bbox = boxes[j]
        local p = scores[j]
        local t = processorOpts.drawBoxesThreshold
        local color
        if labels[j] == 2 then
          if p >= t then
            -- true positive - green
            local h, s = rgbToHsl(0, 1, 0)
            local l = 0.3 + (1-(p-t)/(1-t))*0.4
            color = {hslToRgb(h, s, l)}
          else
            -- false negative - red
            local h, s = rgbToHsl(1, 0, 0)
            local l = 0.3 + p/t*0.4
            color = {hslToRgb(h, s, l)}
          end
        else
          if p >= t then
            -- false positive - blue
            local h, s = rgbToHsl(0, 0, 1)
            local l = 0.3 + (1-(p-t)/(1-t))*0.4
            color = {hslToRgb(h, s, l)}
          end
        end
        if color then
          draw.rectangle(img, bbox[1], bbox[2], bbox[3], bbox[4], 1, color)
        end
      end
      if bboxes:nElement() ~= 0 then
        -- ground truth - yellow
        for j=1,bboxes:size(1) do
          local bbox = bboxes[j]
          if bbox[4]-bbox[2]+1 >= processorOpts.posMinHeight then
            draw.rectangle(img, bbox[1], bbox[2], bbox[3], bbox[4], 1, {1, 1, 0})
          end
        end
      end
      image.save(processorOpts.drawBoxes .. '/' .. paths.basename(path), img)
    end

    if processorOpts.drawROC ~= '' then
      local mapping = _processor.mappings[path:find('val') and 1 or 2]
      local filename = processorOpts.drawROC .. '/res/' .. mapping[paths.basename(path)]
      paths.mkdir(paths.dirname(filename))
      local file, err = io.open(filename, 'w')
      if not(file) then error(err) end
      for j=1,boxes:size(1) do
        file:write(boxes[j][1], ' ',  boxes[j][2], ' ', boxes[j][3]-boxes[j][1]+1, ' ', boxes[j][4]-boxes[j][2]+1, ' ', scores[j], '\n')
      end
      file:close()
    end
  end
  return aggLoss, aggTotal, aggStats
end

return M
