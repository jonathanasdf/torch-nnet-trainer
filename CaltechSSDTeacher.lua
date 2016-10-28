local CaltechProcessor = require 'CaltechProcessor'
local M = torch.class('CaltechSSDTeacher', 'CaltechProcessor')

function M:__init(model, processorOpts)
  self.cmd:option('-provider', '/file1/mscnn/ssd.t7', 'Teacher SSD data file.')

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

  self.data = torch.load(self.provider)
end

function M:getLoss(outputs, labels)
  return 0, 0
end

-- Only called by TrainStudentModel.lua
function M:getStudentLoss(student, studentOutputs, teacherOutputs)
  local classLoss, classGrad = CaltechProcessor.getStudentLoss(self, student, studentOutputs[1], teacherOutputs[1])
  local bboxLoss = student.processor.smoothL1Criterion:forward(studentOutputs[2], teacherOutputs[2]);
  local bboxGrad = student.processor.smoothL1Criterion:backward(studentOutputs[2], teacherOutputs[2]);
  return classLoss * student.processor.classWeight + bboxLoss * student.processor.bboxWeight,
         {classGrad * student.processor.classWeight, bboxGrad * student.processor.bboxWeight}
end

function M:prepareBoxes() end

local t = torch.CudaTensor(1)
function M:preprocess(path, augmentations)
  return t, {}
end

function M:getLabels(pathNames, outputs)
  local n = self.nBoxes
  local scores = torch.zeros(#pathNames, n):cuda()
  local offsets = torch.zeros(#pathNames, n, 4):cuda()
  for i=1,#pathNames do
    local name = paths.basename(pathNames[i], '.jpg')
    local box = self.data[name]
    if box then
      for j=1,box:size(1) do
        scores[i][box[j][1]] = box[j][6];
        offsets[i][box[j][1]] = box[j][{{2,5}}];
      end
    end
  end
  scores = scores:view(-1)
  return {scores, offsets}
end

function M:forward(pathNames, inputs, deterministic)
  local labels = self:getLabels(pathName)
  labels[1] = torch.cat(labels[1], torch.add(torch.mul(labels[1], -1), 1))
  return CaltechProcessor.forward(self, pathNames, labels, deterministic)
end

function M:train()
  error('Cannot train CaltechSSDTeacher.')
end

function M:updateStats(pathNames, outputs, labels) end

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
        if self.model.output[1][(i-1)*n+j] > 0 then
          local box = self.boxes[j] + self.model.output[2][i][j]
          file:write(box[1]-box[3]/2, ' ', box[2]-box[4]/2, ' ', box[3], ' ', box[4], ' ', self.model.output[1][(i-1)*n+j], '\n')
        end
      end
      file:close()
    end
  end
end

return M
