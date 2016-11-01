require 'nms'
local Transforms = require 'Transforms'
local Processor = require 'Processor'
local M = torch.class('KITTIProcessor', 'Processor')

function M:__init(model, processorOpts)
  self.cmd:option('-imageSize', 224, 'input patch size')
  self.cmd:option('-classWeights', '1;1;1', 'relative weight of neg;pedestrian;cyclist')
  self.cmd:option('-drawROC', '', 'set a directory to use for full evaluation')
  self.cmd:option('-boxes', '', 'file name to bounding box mapping for drawROC')
  self.cmd:option('-nonms', false, 'dont apply nms to boxes for drawROC')
  Processor.__init(self, model, processorOpts)

  self.meanPixel = torch.Tensor{0.485, 0.456, 0.406}:view(3, 1, 1)
  self.std = torch.Tensor{0.229, 0.224, 0.225}:view(3, 1, 1)

  self.classWeights = self.classWeights:split(';')
  for i=1,#self.classWeights do
    self.classWeights[i] = tonumber(self.classWeights[i])
  end
  self.criterion = nn.CrossEntropyCriterion(torch.Tensor(self.classWeights), false):cuda()

  self.softmax = nn.SoftMax():cuda()

  if self.drawROC ~= '' then
    if string.sub(self.drawROC, -1) ~= '/' then
        self.drawROC = self.drawROC .. '/'
    end
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
  if string.sub(self.drawROC, 1, 1) ~= '/' then
      self.drawROC = paths.concat(paths.cwd(), self.drawROC)
  end

  if opts.phase == 'test' then
    if opts.epochSize ~= -1 then
      error('drawROC can only be used with epochSize == -1')
    end
  else -- val
    if opts.val == '' then
      error('drawROC specified without validation data?')
    end
    if opts.valSize ~= -1 then
      error('drawROC can only be used with valSize == -1')
    end
  end

  if not(opts.resume) or opts.resume == '' then
    if paths.dir(self.drawROC) ~= nil then
      local answer
      repeat
        print('Warning: drawROC directory exists! Continue (y/n)?')
        answer=io.read()
      until answer=="y" or answer=="n"
      if answer == "n" then
        error('drawROC directory exists! Aborting.')
      end
    else
      mkdir(self.drawROC)
    end
  end

  if self.boxes == '' then
    error('Please specify boxes file. Example: /home/nvesdapu/box.txt')
  end
  local boxes = {}
  for l in io.lines(self.boxes) do
    local s = l:split(' ')
    boxes[s[1]] = {s[2], s[3], s[4], s[5]}
  end
  self.boxes = boxes
end

function M:preprocess(path, augmentations)
  local img = image.load(path, 3)
  local sz = self.imageSize
  img = Transforms.Scale(sz, sz)[2](img)
  img = img:csub(self.meanPixel:expandAs(img)):cdiv(self.std:expandAs(img))
  return img:cuda(), {}
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
  self.stats = optim.ConfusionMatrix({'no person', 'person', 'cyclist'})
  if self.drawROC ~= '' then
    os.execute('rm -rf ' .. self.drawROC)
    os.execute('mkdir -p ' .. self.drawROC)
  end
end

function M:updateStats(pathNames, outputs, labels)
  self.stats:batchAdd(outputs, labels)
end

function M:getStats()
  local ROC
  if opts.phase ~= 'train' then
    if self.drawROC ~= '' then
      print("Printing boxes...")
      if self.nonms then
        -- remove duplicate boxes
        local total = 0
        for _, attr in dirtree(self.drawROC) do
          if attr.mode == 'file' and attr.size > 0 then
            total = total + 1
          end
        end
        local count = 0
        for filename, attr in dirtree(self.drawROC) do
          if attr.mode == 'file' and attr.size > 0 then
            os.execute("gawk -i inplace '!a[$0]++' " .. filename)
            count = count + 1
            xlua.progress(count, total)
          end
        end
      else
        -- do nms
        nmsKITTI(self.drawROC)
      end

      print("Running MATLAB script...")
      print(runMatlab("cd /home/nvesdapu; eval_KITTI('" .. self.drawROC .. "')"))
      ROC = readAll(self.drawROC .. '/result.txt')
      print(ROC)
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
  if self.drawROC ~= '' then
    local classNames = {'DontCare', 'Pedestrian', 'DontCare'}
    local _, class = torch.max(values, 2)
    for i=1,#pathNames do
      local path = pathNames[i]
      local filename = self.drawROC .. string.sub(paths.basename(path), 1, 6) .. '.txt'
      local file, err = io.open(filename, 'a')
      if not(file) then error(err) end

      local box = self.boxes[paths.basename(path)]
      file:write(string.format("%s %f %f %f %f %f\n", classNames[class[i][1]], box[1], box[2], box[3], box[4], values[i][class[i][1]]))
      file:close()
    end
  end
end

function M:test(pathNames)
  local loss, total = Processor.test(self, pathNames)
  local output = self.softmax:forward(self.model.output)
  self:printBoxes(pathNames, output)
  return loss, total
end

return M
