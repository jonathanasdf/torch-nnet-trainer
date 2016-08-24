require 'TrueNLLCriterion'
local Transforms = require 'Transforms'
local Processor = require 'Processor'
local M = torch.class('CaltechProcessor', 'Processor')

function M:__init(model, processorOpts)
  self.cmd:option('-imageSize', 113, 'input patch size')
  self.cmd:option('-inceptionPreprocessing', false, 'preprocess for inception models (RGB, [-1, 1))')
  self.cmd:option('-caffePreprocessing', false, 'preprocess for caffe models (BGR, [0, 255])')
  self.cmd:option('-flip', 0.5, 'probability to do horizontal flip (for training)')
  self.cmd:option('-negativesWeight', 1, 'relative weight of negative examples')
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
  self.criterion = nn.TrueNLLCriterion(weights, false):cuda()

  local matio = require 'matio'
  matio.use_lua_strings = true
  local boxes = {
    matio.load('/file1/caltech10x/val/box.mat'),
    matio.load('/file1/caltech10x/test5/box.mat')
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

  if self.processorOpts.drawROC ~= '' then
    if string.sub(self.processorOpts.drawROC, 1, 1) ~= '/' then
      self.processorOpts.drawROC = paths.concat(paths.cwd(), self.processorOpts.drawROC)
    end
    if opts.testing then
      if opts.epochSize ~= -1 then
        error('sorry, drawROC can only be used with epochSize == -1')
      end
    else -- val
      if opts.val == '' then
        error('drawROC specified without validation data?')
      elseif opts.valSize ~= -1 then
        error('sorry, drawROC can only be used with valSize == -1')
      end
    end

    if not(opts.resume) or opts.resume == '' then
      if paths.dir(self.processorOpts.drawROC) ~= nil then
        local answer
        repeat
          print('Warning: drawROC directory exists! Continue (y/n)?')
          answer=io.read()
        until answer=="y" or answer=="n"
        if answer == "n" then
          error('drawROC directory exists! Aborting.')
        end
      else
        paths.mkdir(self.processorOpts.drawROC)
      end
      -- make sure all files exist
      print('Preparing drawROC directory.')
      self.processorOpts.drawROCDir = self.processorOpts.drawROC .. '/res/'
      os.execute('cat /file1/caltech10x/val/dirs.txt | awk \'{print "' .. self.processorOpts.drawROCDir .. '"$0}\' | xargs mkdir -p')
      os.execute('cat /file1/caltech10x/test5/dirs.txt | awk \'{print "' .. self.processorOpts.drawROCDir .. '"$0}\' | xargs mkdir -p')
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

      if self.processorOpts.drawROC ~= '' then
        self.valROCGraph = gnuplot.pngfigure(opts.logdir .. 'val-logavgmiss.png')
        gnuplot.xlabel('epoch')
        gnuplot.ylabel('log-average miss rate')
        gnuplot.grid(true)
        self.valROC = torch.Tensor(opts.epochs)
      end
    end
  end
end

function M:preprocess(path, augmentations)
  local augs = {}
  if augmentations ~= nil then
    for i=1,#augmentations do
      local name = augmentations[i][1]
      if name == 'scale' or name == 'hflip' then
        augs[#augs+1] = augmentations[i]
      end
    end
  else
    local sz = self.processorOpts.imageSize
    augs[#augs+1] = Transforms.Scale(sz, sz)

    if opts.phase == 'train' then
      if self.processorOpts.flip ~= 0 then
        augs[#augs+1] = Transforms.HorizontalFlip(self.processorOpts.flip)
      end
    end
  end

  local img = image.load(path, 3)
  Transforms.Apply(augs, img)

  if self.processorOpts.inceptionPreprocessing then
    img = (img * 255 - 128) / 128
  elseif self.processorOpts.caffePreprocessing then
    img = (convertRGBBGR(img) * 255):csub(self.processorOpts.meanPixel:expandAs(img))
  else
    img = img:csub(self.processorOpts.meanPixel:expandAs(img)):cdiv(self.processorOpts.std:expandAs(img))
  end

  self:checkAugmentations(augmentations, augs)
  return img:cuda(), augs
end

function M:getLabels(pathNames)
  local labels = torch.Tensor(#pathNames)
  for i=1,#pathNames do
    labels[i] = pathNames[i]:find('neg') and 1 or 2
  end
  return labels:cuda()
end

function M:resetStats()
  self.stats = optim.ConfusionMatrix({'no person', 'person'})
  if self.processorOpts.drawROC ~= '' then
    os.execute('cat /file1/caltech10x/val/files.txt | awk \'{print "' .. self.processorOpts.drawROCDir .. '"$0}\' | xargs truncate -s 0')
    os.execute('cat /file1/caltech10x/test5/files.txt | awk \'{print "' .. self.processorOpts.drawROCDir .. '"$0}\' | xargs truncate -s 0')
  end
end

function M:updateStats(pathNames, outputs, labels)
  self.stats:batchAdd(outputs, labels)
end

function M:processStats()
  local ROC
  if opts.phase ~= 'train' and self.processorOpts.drawROC ~= '' then
    print("Preparing data for drawing ROC...")
    -- remove duplicate boxes
    for file, attr in dirtree(self.processorOpts.drawROCDir) do
      if attr.mode == 'file' and attr.size > 0 then
        os.execute("gawk -i inplace '!a[$0]++' " .. file)
      end
    end

    -- remove cache files
    os.execute('rm -f ' .. self.processorOpts.drawROC .. '/eval/dt*.mat')
    os.execute('rm -f ' .. self.processorOpts.drawROC .. '/eval/ev*.mat')

    local has = {}
    local inputs
    if opts.phase == 'test' then
      inputs = opts.input
    elseif opts.phase == 'val' then
      inputs = opts.val
    end

    for i=1,#inputs do
      if inputs[i]:find('val') then has['val'] = 1 end
      if inputs[i]:find('test') then has['test'] = 1 end
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
    local cmd
    if opts.phase == 'test' then
      cmd = "cd /file1/caltech; dbEval('" .. self.processorOpts.drawROC .. "', " .. dataName .. ", '" .. self.processorOpts.name .. "')"
    elseif opts.phase == 'val' then
      cmd = "cd /file1/caltech; dbEvalVal('" .. self.processorOpts.drawROC .. "', " .. dataName .. ")"
    end
    print("Running MATLAB script...")
    print(runMatlab(cmd))
    local result = readAll(self.processorOpts.drawROC .. '/eval/RocReasonable.txt')
    print(result)
    ROC = string.sub(result, 8)
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

function M:drawROC(pathNames, values)
  if self.processorOpts.drawROC ~= '' then
    for i=1,#pathNames do
      local path = pathNames[i]
      local dataset, set, video, id = path:match("/file1/caltech10x/(.-)/.-/raw/set(.-)_V(.-)_I(.-)_.*")

      local filename = self.processorOpts.drawROCDir .. 'set' .. set .. '/V' .. video .. '/I' .. id .. '.txt'
      local file, err = io.open(filename, 'a')
      if not(file) then error(err) end

      local boxes = self.boxes[paths.basename(path)]
      file:write(boxes[1]-1, ' ',  boxes[2]-1, ' ', boxes[3]-boxes[1]+1, ' ', boxes[4]-boxes[2]+1, ' ', values[i], '\n')
      file:close()
    end
  end
end

function M:test(pathNames)
  local loss, total = Processor.test(self, pathNames)
  self:drawROC(pathNames, self.model.model.output[{{}, 2}])
  return loss, total
end

return M
