require 'fbnn'

local Transforms = require 'Transforms'
local Processor = require 'Processor'
local M = torch.class('CaltechProcessor', 'Processor')

function M:__init(model, processorOpts)
  self.cmd:option('-imageSize', 113, 'input patch size')
  self.cmd:option('-inceptionPreprocessing', false, 'preprocess for inception models (RGB, [-1, 1))')
  self.cmd:option('-caffePreprocessing', false, 'preprocess for caffe models (BGR, [0, 255])')
  self.cmd:option('-flip', 0.5, 'probability to do horizontal flip (for training)')
  self.cmd:option('-negativesWeight', 1, 'relative weight of negative examples')
  self.cmd:option('-svm', '', 'SVM to use')
  self.cmd:option('-layer', 'fc7', 'layer to use as SVM input')
  self.cmd:option('-drawROC', '', 'set a directory to use for full evaluation')
  self.cmd:option('-name', 'Result', 'name to use on ROC graph')
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
  self.criterion = nn.TrueNLLCriterion(weights, false)
  if nGPU > 0 then
    self.criterion = self.criterion:cuda()
  end

  local matio = require 'matio'
  matio.use_lua_strings = true
  local boxes = {
    matio.load('/file/caltech10x/val/box.mat'),
    matio.load('/file/caltech10x/test5/box.mat')
  }
  self.boxes = {}
  for i=1,#boxes do
    for j=1,#boxes[i].name_pos do
      self.boxes[boxes[i].name_pos[j]] = boxes[i].box_pos[j]
    end
    for j=1,#boxes[i].name_neg do
      self.boxes[boxes[i].name_neg[j]] = boxes[i].box_neg[j]
    end
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
    if opts.epochSize ~= -1 then
      error('sorry, drawROC can only be used with epochSize == -1')
    end
    if nThreads > 1 then
      error('sorry, drawROC can only be used with nThreads <= 1')
    end
    if not(opts.resume) or opts.resume == '' then
      if paths.dir(self.processorOpts.drawROC) ~= nil then
        error('drawROC directory exists! Aborting')
      end
      paths.mkdir(self.processorOpts.drawROC)
    end
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

  local sz = processorOpts.imageSize
  if img:size(2) ~= sz or img:size(3) ~= sz then
    img = image.scale(img, sz, sz)
  end

  if isTraining then
    img = Transforms.HorizontalFlip(processorOpts.flip)(img)
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
    pred = outputs[{{},2}]:clone()
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
     print("Preparing data for drawing ROC...")
     local dir = processorOpts.drawROC .. '/res/'
     -- remove duplicate boxes
     for file, attr in dirtree(dir) do
       if attr.mode == 'file' then
         os.execute("gawk -i inplace '!a[$0]++' " .. file)
       end
     end

     local has = {}
     for i=1,#opts.input do
       if opts.input[i]:find('val') then has['val'] = 1 end
       if opts.input[i]:find('test') then has['test'] = 1 end
     end

     -- make sure all files exist
     if has['val'] then
       os.execute('cat /file/caltech10x/val/files.txt | awk \'{print "' .. dir .. '"$0}\' | xargs touch')
     end
     if has['test'] then
       os.execute('cat /file/caltech10x/test/files.txt | awk \'{print "' .. dir .. '"$0}\' | xargs touch')
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
     print("Running MATLAB script...")
     print(runMatlab(cmd))
     print(readAll(self.processorOpts.drawROC .. '/eval/RocReasonable.txt'))
  end
  return tostring(self.stats)
end

function M.forward(inputs, deterministic)
  local outputs = _model:forward(inputs, deterministic)
  if processorOpts.svm ~= '' then
    outputs = findModuleByName(_model, processorOpts.layer).output
  end
  return outputs
end

function M.drawROC(pathNames, values)
  if processorOpts.drawROC ~= '' then
    for i=1,#pathNames do
      local path = pathNames[i]
      local dataset, set, video, id = path:match("/file/caltech10x/(.-)/.-/raw/set(.-)_V(.-)_I(.-)_.*")

      local filename = processorOpts.drawROC .. '/res/set' .. set .. '/V' .. video .. '/I' .. id .. '.txt'
      paths.mkdir(paths.dirname(filename))
      local file, err = io.open(filename, 'a')
      if not(file) then error(err) end

      local boxes = _processor.boxes[paths.basename(path)]
      file:write(boxes[1]-1, ' ',  boxes[2]-1, ' ', boxes[3]-boxes[1]+1, ' ', boxes[4]-boxes[2]+1, ' ', values[i], '\n')
      file:close()
    end
  end
end

function M.test(pathNames, inputs)
  local loss, total, stats = Processor.test(pathNames, inputs)
  _processor.drawROC(pathNames, stats[1])
  return loss, total, stats
end

return M
