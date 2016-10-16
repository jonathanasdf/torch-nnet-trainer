local Processor = require 'Processor'
local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechSSDProcessor', 'CaltechProcessor')

function M:__init(model, processorOpts)
  self.cmd:option('-criterionWeights', '', 'semicolon separated weights for pos/neg')

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

  if self.criterionWeights and self.criterionWeights ~= '' then
    self.criterionWeights = torch.Tensor(self.criterionWeights:split(';'))
    for i=1,self.criterionWeights:size(1) do
      self.criterionWeights[i] = tonumber(self.criterionWeights[i])
    end
  else
    self.criterionWeights = nil
  end

  local smoothL1Criterion = nn.SmoothL1Criterion()
  smoothL1Criterion.sizeAverage = false
  self.criterion = nn.ParallelCriterion()
    :add(nn.CrossEntropyCriterion(self.criterionWeights, false))
    :add(smoothL1Criterion)
    :cuda()
  self.criterion.sizeAverage = false
end

function M:prepareBoxes()
end

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
  return {pos, offsets}
end

function M:updateStats(pathNames, outputs, labels)
  self.stats:batchAdd(outputs[1], labels[1])
end

-- values = {scores, offsets}
function M:printBoxes(pathNames, values)
  if self.outputBoxes ~= '' then
    for i=1,#pathNames do
      local path = pathNames[i]
      local set, video, id = path:match("/set(.-)_V(.-)_I(.-)%.")

      local filename = self.outputBoxes .. 'set' .. set .. '/V' .. video .. '/I' .. id .. '.txt'
      local file, err = io.open(filename, 'a')
      if not(file) then error(err) end

      for j=1,self.nBoxes do
        if self.model.output[1][i][j] > 0 then
          local box = self.boxes[j] + self.model.output[2][i][j]
          file:write(box[1]-box[3]/2, ' ', box[2]-box[4]/2, ' ', box[3], ' ', box[4], ' ', self.model.output[1][i][j], '\n')
        end
      end
      file:close()
    end
  end
end

return M
