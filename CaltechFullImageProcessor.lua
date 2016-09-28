local Processor = require 'Processor'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechFullImageProcessor', 'CaltechProcessor')

function M:__init(model, processorOpts)
  self.cmd:option('-boxesPerImage', 100, 'Number of boxes output per image.')
  self.cmd:option('-criterionWeights', '', 'semicolon separated weights for pos/neg')

  CaltechProcessor.__init(self, model, processorOpts)

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
    :add(smoothL1Criterion)
    :add(nn.CrossEntropyCriterion(self.processorOpts.criterionWeights, false))
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

-- Assumes outputs is a table with {boxes, scores for Y/N classes}
-- Returned labels = {closest_box, overlap > 0.5}
local check_gt = requirePath('/data/rpn/datasets/check_gt.lua')
function M:getLabels(pathNames, outputs)
  local boxes = outputs[1]
  local n = self.processorOpts.boxesPerImage
  local closest = torch.CudaTensor(#pathNames, n, 4)
  local labels = torch.CudaTensor(#pathNames, n)
  for i=1,#pathNames do
    closest[i], labels[i] = check_gt(paths.basename(pathNames[i]), boxes[i])
  end
  -- labels is {0, 1} but in torch we need it to be 1-indexed
  labels = labels + 1
  return {closest, labels:view(-1)}
end

function M:updateStats(pathNames, outputs, labels)
  self.stats:batchAdd(outputs[2], labels[2])
end

-- values length is batchSize * boxesPerImage
function M:drawROC(pathNames, values)
  if self.processorOpts.drawROC ~= '' then
    local n = self.processorOpts.boxesPerImage;
    for i=1,#pathNames do
      local path = pathNames[i]
      local set, video, id = path:match("set(.-)_V(.-)_I(.-)%.")

      local filename = self.processorOpts.drawROCDir .. 'set' .. set .. '/V' .. video .. '/I' .. id .. '.txt'
      local file, err = io.open(filename, 'a')
      if not(file) then error(err) end

      local boxes = self.model.output[1][i]
      assert(boxes:size(1) == n)
      assert(values:size(1) == #pathNames * n)
      for j=1,n do
        file:write(boxes[j][1]-1, ' ', boxes[j][2]-1, ' ', boxes[j][3]-boxes[j][1]+1, ' ', boxes[j][4]-boxes[j][2]+1, ' ', values[(i-1)*n+j], '\n')
      end
      file:close()
    end
  end
end

function M:test(pathNames)
  local loss, total = Processor.test(self, pathNames)
  self:drawROC(pathNames, self.model.output[2][{{}, 2}])
  return loss, total
end

return M
