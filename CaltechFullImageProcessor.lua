require 'TrueNLLCriterion'
local Processor = require 'Processor'
local M = torch.class('CaltechFullImageProcessor', 'CaltechProcessor')

function M:__init(model, processorOpts)
  self.cmd:option('-boxesPerImage', 100, 'Number of boxes output per image.')
  Processor.__init(self, model, processorOpts)

  if self.processorOpts.drawROC ~= '' then
    error('Sorry, drawROC does not work with this processor.')
  end

  local smoothL1Criterion = nn.SmoothL1Criterion()
  smoothL1Criterion.sizeAverage = false
  self.criterion = nn.ParallelCriterion()
    :add(smoothL1Criterion)
    :add(nn.TrueNLLCriterion(nil, false))
    :cuda()
  self.criterion.sizeAverage = false
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
function M:getLabels(pathNames, outputs)
  local boxes = outputs[1]
  local n = self.processorOpts.boxesPerImage
  local check_gt = requirePath('/data/rpn/datasets/check_gt.lua')
  local closest = torch.CudaTensor(#pathNames, 4*n)
  local labels = torch.CudaTensor(#pathNames, n)
  for i=1,#pathNames do
    closest[i], labels[i] = check_gt(pathNames[i], boxes[i])
  end
  return {closest, labels:view(-1)}
end

function M:updateStats(pathNames, outputs, labels)
  self.stats:batchAdd(outputs[2], labels[2])
end

return M
