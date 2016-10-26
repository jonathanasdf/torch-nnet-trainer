local Transforms = require 'Transforms'
local Processor = require 'Processor'
local M = torch.class('ImageNetProcessor', 'Processor')

function M:__init(model, processorOpts)
  self.cmd:option('-resize', 256, 'what size to resize to before cropping')
  self.cmd:option('-cropSize', 224, 'what size to crop')
  self.cmd:option('-inceptionPreprocessing', false, 'preprocess for inception models (RGB, [-1, 1))')
  self.cmd:option('-caffePreprocessing', false, 'preprocess for caffe models (BGR, [0, 255])')
  Processor.__init(self, model, processorOpts)
  assert(self.cropSize <= self.resize)

  self.criterion = nn.CrossEntropyCriterion(nil, false):cuda()

  local synset = '/file1/imagenet/ILSVRC2012_devkit_t12/words.txt'
  if self.inceptionPreprocessing then
    synset = '/file1/imagenet/inception_synset.txt'
  elseif self.caffePreprocessing then
    self.meanPixel = torch.Tensor{103.939, 116.779, 123.68}:view(3, 1, 1)
  else
    self.meanPixel = torch.Tensor{0.485, 0.456, 0.406}:view(3, 1, 1)
    self.std = torch.Tensor{0.229, 0.224, 0.225}:view(3, 1, 1)
  end

  self.words = {}
  self.lookup = {}
  local n = 1
  for line in io.lines(synset) do
    self.words[#self.words+1] = string.sub(line,11)
    self.lookup[string.sub(line, 1, 9)] = n
    n = n + 1
  end

  self.val = {}
  for line in io.lines('/file1/imagenet/ILSVRC2012_devkit_t12/data/WNID_validation_ground_truth.txt') do
    self.val[#self.val+1] = self.lookup[line]
  end

  if opts.logdir then
    self.trainGraph = gnuplot.pngfigure(opts.logdir .. 'train.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('acc')
    gnuplot.grid(true)
    self.trainTop1 = torch.Tensor(opts.epochs)
    self.trainTop5 = torch.Tensor(opts.epochs)

    if opts.val then
      self.valGraph = gnuplot.pngfigure(opts.logdir .. 'val.png')
      gnuplot.xlabel('epoch')
      gnuplot.ylabel('acc')
      gnuplot.grid(true)
      self.valTop1 = torch.Tensor(opts.epochs)
      self.valTop5 = torch.Tensor(opts.epochs)
    end
  end
end

function M:preprocess(path, augmentations)
  local img = image.load(path, 3)

  local augs = {}
  augs[#augs+1] = Transforms.ScaleKeepAspect(self.resize)
  augs[#augs+1] = Transforms.CenterCrop(self.cropSize)
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
    local name = pathNames[i]
    local filename = paths.basename(name)
    if name:find('train') then
      labels[i] = self.lookup[string.sub(filename, 1, 9)]
    else
      assert(name:find('val'))
      labels[i] = self.val[tonumber(string.sub(filename, -12, -5))]
    end
  end
  return labels:cuda()
end

function M:resetStats()
  self.stats = {}
  self.stats.top1 = 0
  self.stats.top5 = 0
  self.stats.total = 0
end

function M:updateStats(pathNames, outputs, labels)
  for i=1,#pathNames do
    local prob, classes = (#pathNames == 1 and outputs or outputs[i]):view(-1):sort(true)
    local result = 'predicted classes for ' .. paths.basename(pathNames[i]) .. ': '
    for j=1,5 do
      local color = ''
      if classes[j] == labels[i] then
        if j == 1 then
          self.stats.top1 = self.stats.top1 + 1
        end
        self.stats.top5 = self.stats.top5 + 1
        color = '\27[33m'
      end
      result = result .. color .. '(' .. math.floor(prob[j]*100 + 0.5) .. '%) ' .. self.words[classes[j]] .. '\27[0m; '
    end
    if labels[i] ~= -1 then
      result = result .. '\27[36mground truth: ' .. self.words[labels[i]] .. '\27[0m'
    end
    --print(result)
  end
  self.stats.total = self.stats.total + #pathNames
end

function M:getStats()
  local output = ''
  output = output .. '  Top 1 accuracy: ' .. self.stats.top1 .. '/' .. self.stats.total .. ' = ' .. (self.stats.top1*100.0/self.stats.total) .. '%\n'
  output = output .. '  Top 5 accuracy: ' .. self.stats.top5 .. '/' .. self.stats.total .. ' = ' .. (self.stats.top5*100.0/self.stats.total) .. '%'

  if opts.phase == 'train' and self.trainGraph then
    self.trainTop1[opts.epoch] = self.stats.top1/self.stats.total
    self.trainTop5[opts.epoch] = self.stats.top5/self.stats.total

    local x = torch.range(1, opts.epoch):long()
    gnuplot.figure(self.trainGraph)
    gnuplot.plot({'top1', x, self.trainTop1:index(1, x), '+-'}, {'top5', x, self.trainTop5:index(1, x), '+-'})
    gnuplot.plotflush()
  elseif opts.phase == 'val' and self.valGraph and opts.epoch >= opts.valEvery then
    self.valTop1[opts.epoch] = self.stats.top1/self.stats.total
    self.valTop5[opts.epoch] = self.stats.top5/self.stats.total

    local x = torch.range(opts.valEvery, opts.epoch, opts.valEvery):long()
    gnuplot.figure(self.valGraph)
    gnuplot.plot({'top1', x, self.trainTop1:index(1, x), '+-'}, {'top5', x, self.trainTop5:index(1, x), '+-'})
    gnuplot.plotflush()
  end
  return output
end

return M
