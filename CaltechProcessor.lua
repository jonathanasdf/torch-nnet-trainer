require 'nms'
local Transforms = require 'Transforms'
local Processor = require 'Processor'
local M = torch.class('CaltechProcessor', 'Processor')

function M:__init(model, processorOpts)
  self.cmd:option('-imageSize', 224, 'input patch size')
  self.cmd:option('-inceptionPreprocessing', false, 'preprocess for inception models (RGB, [-1, 1))')
  self.cmd:option('-caffePreprocessing', false, 'preprocess for caffe models (BGR, [0, 255])')
  self.cmd:option('-flip', 0, 'probability to do horizontal flip (for training)')
  self.cmd:option('-negativesWeight', 1, 'relative weight of negative examples')
  self.cmd:option('-outputBoxes', '', 'set a directory to output boxes to')
  self.cmd:option('-drawROC', '', 'set a directory to use for full evaluation')
  self.cmd:option('-boxes', '', 'file name to bounding box mapping for drawROC')
  self.cmd:option('-name', 'Result', 'name to use on ROC graph')
  self.cmd:option('-nonms', false, 'dont apply nms to boxes for drawROC')
  Processor.__init(self, model, processorOpts)

  if self.inceptionPreprocessing then
    -- no mean normalization
  elseif self.caffePreprocessing then
    self.meanPixel = torch.Tensor{103.939, 116.779, 123.68}:view(3, 1, 1)
  else
    self.meanPixel = torch.Tensor{0.485, 0.456, 0.406}:view(3, 1, 1)
    self.std = torch.Tensor{0.229, 0.224, 0.225}:view(3, 1, 1)
  end

  local w = self.negativesWeight
  local weights = torch.Tensor{w/(1+w), 1/(1+w)} * 2
  self.criterion = nn.CrossEntropyCriterion(weights, false):cuda()

  if self.drawROC ~= '' then
    if self.outputBoxes == '' then
      self.outputBoxes = self.drawROC .. '/res/'
    end
  end
  if self.outputBoxes ~= '' then
    self:initDrawROC()
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

      if self.drawROC ~= '' then
        self.valROCGraph = gnuplot.pngfigure(opts.logdir .. 'val-logavgmiss.png')
        gnuplot.xlabel('epoch')
        gnuplot.ylabel('log-average miss rate')
        gnuplot.grid(true)
        self.valROC = torch.Tensor(opts.epochs)
      end
    end
  end
end

function M:initDrawROC()
  if string.sub(self.outputBoxes, 1, 1) ~= '/' then
      self.outputBoxes = paths.concat(paths.cwd(), self.outputBoxes)
  end
  if string.sub(self.outputBoxes, -1) ~= '/' then
      self.outputBoxes = self.outputBoxes .. '/'
  end
  if self.drawROC ~= '' and string.sub(self.drawROC, 1, 1) ~= '/' then
      self.drawROC = paths.concat(paths.cwd(), self.drawROC)
  end

  opts.drawROCInputs = {}
  local inputs
  if opts.phase == 'test' then
    if self.drawROC ~= '' and opts.epochSize ~= -1 then
      error('drawROC can only be used with epochSize == -1')
    end
    inputs = opts.input
  else -- val
    if opts.val == '' then
      error('drawROC specified without validation data?')
    end
    if self.drawROC ~= '' and opts.valSize ~= -1 then
      error('drawROC can only be used with valSize == -1')
    end
    inputs = opts.val
  end
  for i=1,#inputs do
    if inputs[i]:find('train') then opts.drawROCInputs['train'] = 1 end
    if inputs[i]:find('val') then opts.drawROCInputs['val'] = 1 end
    if inputs[i]:find('test') then opts.drawROCInputs['test'] = 1 end
  end

  if not(opts.resume) or opts.resume == '' then
    if paths.dir(self.outputBoxes) ~= nil or (self.drawROC ~= '' and paths.dir(self.drawROC) ~= nil) then
      local answer
      repeat
        print('Warning: drawROC directory exists! Continue (y/n)?')
        answer=io.read()
      until answer=="y" or answer=="n"
      if answer == "n" then
        error('drawROC directory exists! Aborting.')
      end
    else
      mkdir(self.outputBoxes)
      if self.drawROC ~= '' then
        mkdir(self.drawROC)
      end
    end
  end

  self:prepareBoxes()
end

function M:prepareBoxes()
  if self.boxes == '' then
    error('Please specify boxes file. Example: /file1/caltech10x/test/box.mat')
  end
  local matio = require 'matio'
  matio.use_lua_strings = true
  local boxes = {}
  for _, path in ipairs(self.boxes:split(';')) do
    boxes[#boxes+1] = matio.load(path)
  end
  self.boxes = {}
  for i=1,#boxes do
    for j=1,#boxes[i].name_pos do
      self.boxes[boxes[i].name_pos[j]] = boxes[i].box_pos[j]
    end
    for j=1,#boxes[i].name_neg do
      self.boxes[boxes[i].name_neg[j]] = boxes[i].box_neg[j]
    end
  end
end

function M:preprocess(path, augmentations)
  local augs = {}
  if augmentations ~= nil then
    for i=1,#augmentations do
      local name = augmentations[i][1]
      if name == 'hflip' then
        augs[#augs+1] = augmentations[i]
      end
    end
  else
    if opts.phase == 'train' then
      if self.flip ~= 0 then
        augs[#augs+1] = Transforms.HorizontalFlip(self.flip)
      end
    end
  end

  local img = image.load(path, 3)
  local sz = self.imageSize
  img = Transforms.Scale(sz, sz)[2](img)
  img = Transforms.Apply(augs, img)

  if self.inceptionPreprocessing then
    img = (img * 255 - 128) / 128
  elseif self.caffePreprocessing then
    img = (convertRGBBGR(img) * 255):csub(self.meanPixel:expandAs(img))
  else
    img = img:csub(self.meanPixel:expandAs(img)):cdiv(self.std:expandAs(img))
  end
  return img:cuda(), augs
end

function M:getLabels(pathNames, outputs)
  local labels = torch.Tensor(#pathNames)
  for i=1,#pathNames do
    if pathNames[i]:find('cyclist') then
      labels[i] = 3
    else
      labels[i] = pathNames[i]:find('neg') and 1 or 2
    end
  end
  return labels:cuda()
end

function M:resetStats()
  self.stats = optim.ConfusionMatrix({'no person', 'person'})
  if self.outputBoxes ~= '' then
    os.execute('rm -rf ' .. self.outputBoxes)
  end
end

function M:updateStats(pathNames, outputs, labels)
  self.stats:batchAdd(outputs, labels)
end

function M:getStats()
  local ROC
  if opts.phase ~= 'train' then
    if self.outputBoxes ~= '' then
      print("Printing boxes...")
      if self.nonms then
        -- remove duplicate boxes
        local total = 0
        for _, attr in dirtree(self.outputBoxes) do
          if attr.mode == 'file' and attr.size > 0 then
            total = total + 1
          end
        end
        local count = 0
        for filename, attr in dirtree(self.outputBoxes) do
          if attr.mode == 'file' and attr.size > 0 then
            os.execute("gawk -i inplace '!a[$0]++' " .. filename)
            count = count + 1
            xlua.progress(count, total)
          end
        end
      else
        -- do nms
        nmsCaltech(self.outputBoxes)
      end
    end

    if self.drawROC ~= '' then
      -- remove cache files
      os.execute('rm -f ' .. self.drawROC .. '/eval/dt*.mat')
      os.execute('rm -f ' .. self.drawROC .. '/eval/ev*.mat')

      local dataName
      for k,_ in pairs(opts.drawROCInputs) do
        if not(dataName) then
          dataName = "{'" .. k .. "'"
        else
          dataName = dataName .. ", '" .. k .. "'"
        end
      end
      dataName = dataName .. '}'
      local cmd
      if opts.phase == 'test' then
        cmd = "cd /file1/caltech; addpath(genpath('toolbox')); dbEval('" .. self.drawROC .. "', " .. dataName .. ", '" .. self.name .. "')"
      elseif opts.phase == 'val' then
        cmd = "cd /file1/caltech; addpath(genpath('toolbox')); dbEvalVal('" .. self.drawROC .. "', " .. dataName .. ")"
      end
      print("Running MATLAB script...")
      print(runMatlab(cmd))
      local result = readAll(self.drawROC .. '/eval/RocReasonable.txt')
      print(result)
      ROC = string.sub(result, 8)
    end
  end

  self.stats:updateValids()
  if opts.phase == 'train' and self.trainGraph then
    self.trainPosAcc[opts.epoch] = self.stats.valids[1]
    self.trainNegAcc[opts.epoch] = self.stats.valids[2]
    self.trainAcc[opts.epoch] = self.stats.averageValid

    local x = torch.range(1, opts.epoch):long()
    gnuplot.figure(self.trainGraph)
    gnuplot.plot({'pos', x, self.trainPosAcc:index(1, x), '+-'}, {'neg', x, self.trainNegAcc:index(1, x), '+-'}, {'overall', x, self.trainAcc:index(1, x), '-'})
    gnuplot.plotflush()
  elseif opts.phase == 'val' and opts.epoch >= opts.valEvery then
    if self.valGraph then
      self.valPosAcc[opts.epoch] = self.stats.valids[1]
      self.valNegAcc[opts.epoch] = self.stats.valids[2]
      self.valAcc[opts.epoch] = self.stats.averageValid

      local x = torch.range(opts.valEvery, opts.epoch, opts.valEvery):long()
      gnuplot.figure(self.valGraph)
      gnuplot.plot({'pos', x, self.valPosAcc:index(1, x), '+-'}, {'neg', x, self.valNegAcc:index(1, x), '+-'}, {'overall', x, self.valAcc:index(1, x), '-'})
      gnuplot.plotflush()
    end
    if self.valROCGraph then
      self.valROC[opts.epoch] = tonumber(ROC)

      local x = torch.range(opts.valEvery, opts.epoch, opts.valEvery):long()
      gnuplot.figure(self.valROCGraph)
      gnuplot.plot({'result', x, self.valROC:index(1, x), '+-'})
      gnuplot.plotflush()
    end
  end

  return tostring(self.stats)
end

function M:printBoxes(pathNames, values)
  if self.outputBoxes ~= '' then
    for i=1,#pathNames do
      local path = pathNames[i]
      local set, video, id = path:match("set(.-)_V(.-)_I(.-)_")

      local dirname = self.outputBoxes .. 'set' .. set .. '/V' .. video .. '/'
      local filename = dirname .. 'I' .. id .. '.txt'
      if not paths.dirp(dirname) then
        mkdir(dirname)
      end
      local file, err = io.open(filename, 'a')
      if not(file) then error(err) end

      local box = self.boxes[paths.basename(path)]
      file:write(box[1]-1, ' ',  box[2]-1, ' ', box[3], ' ', box[4], ' ', values[i][2], '\n')
      file:close()
    end
  end
end

function M:test(pathNames)
  local loss, total = Processor.test(self, pathNames)
  self:printBoxes(pathNames, self.model.output)
  return loss, total
end

return M
