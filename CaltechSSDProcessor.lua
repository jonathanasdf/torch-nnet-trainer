local Processor = require 'Processor'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechFullImageProcessor', 'CaltechProcessor')

function M:__init(model, processorOpts)
  self.cmd:option('-criterionWeights', '', 'semicolon separated weights for pos/neg')

  CaltechProcessor.__init(self, model, processorOpts)

  local boxesFile = '/file1/caltechrpn/boxes.txt'
  self.processorOpts.nBoxes = 0
  for _ in io.lines(boxesFile) do
    self.processorOpts.nBoxes = self.processorOpts.nBoxes + 1
  end
  self.boxes = torch.Tensor(self.processorOpts.nBoxes, 4)
  local df = torch.DiskFile(boxesFile, 'r')
  df:readFloat(self.boxes:storage())
  df:close()

  if self.processorOpts.criterionWeights and self.processorOpts.criterionWeights ~= '' then
    self.processorOpts.criterionWeights = torch.Tensor(self.processorOpts.criterionWeights:split(';'))
    for i=1,self.processorOpts.criterionWeights:size(1) do
      self.processorOpts.criterionWeights[i] = tonumber(self.processorOpts.criterionWeights[i])
    end
  else
    self.processorOpts.criterionWeights = nil
  end

  local smoothL1Criterion = nn.SmoothL1Criterion()
  smoothL1Criterion.sizeAverage = false
  self.criterion = nn.ParallelCriterion()
    :add(nn.CrossEntropyCriterion(self.processorOpts.criterionWeights, false))
    :add(smoothL1Criterion)
    :cuda()
  self.criterion.sizeAverage = false
end

function M:prepareBoxes()
end

function M:preprocess(path, augmentations)
  local img = image.load(path, 3)
  if self.processorOpts.inceptionPreprocessing then
    img = (img * 255 - 128) / 128
  elseif self.processorOpts.caffePreprocessing then
    img = (convertRGBBGR(img) * 255):csub(self.processorOpts.meanPixel:expandAs(img))
  else
    img = img:csub(self.processorOpts.meanPixel:expandAs(img)):cdiv(self.processorOpts.std:expandAs(img))
  end
  return img:cuda(), {}
end

-- outputs = {scores, boxes}
-- Returned labels = {highest iou > 0.5, offsets}
local check_gt = requirePath('/data/rpn/datasets/check_gt.lua')
function M:getLabels(pathNames, outputs)
  local boxes = outputs[1]
  local n = self.processorOpts.nBoxes
  local pos = torch.zeros(#pathNames, n):cuda()
  local offsets = torch.zeros(#pathNames, n, 4):cuda()
  for i=1,#pathNames do
    closest[i], labels[i] = check_gt(paths.basename(pathNames[i]), boxes[i])
  end
  -- pos is {0, 1} but in torch we need it to be 1-indexed
  pos = pos + 1
  return {pos:view(-1), offsets}
end

function M:updateStats(pathNames, outputs, labels)
  self.stats:batchAdd(outputs[1], labels[1])
end

function M:outputBoxes(pathNames, values)
  if self.processorOpts.outputBoxes ~= '' then
    for i=1,#pathNames do
      local path = pathNames[i]
      local set, video, id = path:match("/set(.-)_V(.-)_I(.-)%.")

      local filename = self.processorOpts.outputBoxes .. 'set' .. set .. '/V' .. video .. '/I' .. id .. '.txt'
      local file, err = io.open(filename, 'a')
      if not(file) then error(err) end

      local boxes = self.model.output[1][i]
      assert(values:size(1) == #pathNames * self.processorOpts.nBoxes)
      for j=1,self.processorOpts.nBoxes do
        file:write(boxes[j][1]-1, ' ', boxes[j][2]-1, ' ', boxes[j][3]-boxes[j][1]+1, ' ', boxes[j][4]-boxes[j][2]+1, ' ', values[(i-1)*n+j], '\n')
      end
      file:close()
    end
  end
end

function M:test(pathNames)
  local loss, total = Processor.test(self, pathNames)
  self:outputBoxes(pathNames, self.model.output)
  return loss, total
end

return M
