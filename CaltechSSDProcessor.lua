local Processor = require 'Processor'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechSSDProcessor', 'CaltechProcessor')

function M:__init(model, processorOpts)
  self.cmd:option('-classWeight', 1, 'weight for classification criterion')
  self.cmd:option('-posNegWeight', '', 'semicolon separated weights for pos/neg')
  self.cmd:option('-bboxWeight', 1, 'weight for bbox regression criterion')

  CaltechProcessor.__init(self, model, processorOpts)

  local boxesFile = '/file1/caltechrpn/ssd_boxes.txt'
  self.nBoxes = 0
  for _ in io.lines(boxesFile) do
    self.nBoxes = self.nBoxes + 1
  end
  self.boxes = torch.Tensor(self.nBoxes, 4)
  local df = torch.DiskFile(boxesFile, 'r')
  df:readFloat(self.boxes:storage())
  df:close()
  self.boxes = self.boxes:cuda()

  self.gt = torch.load('/file1/caltech10x/gt_ssd.t7')

  if self.posNegWeight and self.posNegWeight ~= '' then
    self.posNegWeight = torch.Tensor(self.posNegWeight:split(';'))
    for i=1,self.posNegWeight:size(1) do
      self.posNegWeight[i] = tonumber(self.posNegWeight[i])
    end
  else
    self.posNegWeight = nil
  end

  self.classCriterion = nn.CrossEntropyCriterion(self.posNegWeight, false):cuda()
  self.smoothL1Criterion = nn.SmoothL1Criterion():cuda()
  self.smoothL1Criterion.sizeAverage = false
  self.criterion = nn.ParallelCriterion()
    :add(self.classCriterion, self.classWeight)
    :add(self.smoothL1Criterion, self.bboxWeight)
  self.criterion.sizeAverage = false
end

function M:prepareBoxes() end

function M:preprocess(path, augmentations)
  local img = image.load(path, 3)
  if self.inceptionPreprocessing then
    img = (img * 255 - 128) / 128
  elseif self.caffePreprocessing then
    img = (convertRGBBGR(img) * 255):csub(self.meanPixel:expandAs(img))
  else
    img = img:csub(self.meanPixel:expandAs(img)):cdiv(self.std:expandAs(img))
  end
  return img:cuda(), {}
end

-- outputs = {scores, offsets}
function M:getLabels(pathNames, outputs)
  local n = self.nBoxes
  local pos = torch.zeros(#pathNames, n):cuda()
  local offsets = torch.zeros(#pathNames, n, 4):cuda()
  for i=1,#pathNames do
    local name = paths.basename(pathNames[i], '.jpg')
    local box = self.gt[name]
    if box then
      for j=1,box:size(1) do
        pos[i][box[j][1]] = 1;
        offsets[i][box[j][1]] = box[j][{{2,5}}];
      end
    end
  end
  -- pos is 0/1 but torch needs 1/2 for classes
  pos = pos + 1
  return {pos:view(-1), offsets}
end

function M:resetStats()
  CaltechProcessor.resetStats(self)
  self.classLoss = 0
  self.bboxLoss = 0
  self.count = 0
end

function M:updateStats(pathNames, outputs, labels)
  self.stats:batchAdd(outputs[1], labels[1])
  self.classLoss = self.classLoss + self.classCriterion:forward(outputs[1], labels[1])
  self.bboxLoss = self.bboxLoss + self.smoothL1Criterion:forward(outputs[2], labels[2])
  self.count = self.count + #pathNames
end

function M:getStats()
  print("Classification loss: ", self.classLoss / self.count)
  print("Bounding boxes loss: ", self.bboxLoss / self.count)
  return CaltechProcessor.getStats(self)
end

-- values = {scores, offsets}
function M:printBoxes(pathNames, values)
  if self.outputBoxes ~= '' then
    local n = self.nBoxes
    for i=1,#pathNames do
      local path = pathNames[i]
      local set, video, id = path:match("/set(.-)_V(.-)_I(.-)%.")

      local filename = self.outputBoxes .. 'set' .. set .. '/V' .. video .. '/I' .. id .. '.txt'
      mkdir(filename)
      local file, err = io.open(filename, 'a')
      if not(file) then error(err) end

      for j=1,n do
        if self.model.output[1][(i-1)*n+j][2] > self.model.output[1][(i-1)*n+j][1] then
          local box = self.boxes[j] + self.model.output[2][i][j]
          file:write(string.format("%f %f %f %f %f\n", box[1]-box[3]/2, box[2]-box[4]/2, box[3], box[4], self.model.output[1][(i-1)*n+j][2]))
        end
      end
      file:close()
    end
  end
end

function M:test(pathNames)
  local loss, total = Processor.test(self, pathNames)
  self:printBoxes(pathNames, nil)
  return loss, total
end

return M
