local CaltechSSDProcessor = require 'CaltechSSDProcessor'
local M = torch.class('CaltechSSDTeacher', 'CaltechSSDProcessor')

function M:__init(model, processorOpts)
  self.cmd:option('-provider', '/file1/mscnn/ssd.t7', 'Teacher SSD data file.')

  CaltechSSDProcessor.__init(self, model, processorOpts)

  self.data = torch.load(self.provider)
end

function M:getLoss(outputs, labels)
  return 0, 0
end

-- Only called by TrainStudentModel.lua
function M:getStudentLoss(student, studentOutputs, teacherOutputs)
  local classLoss, classGrad = CaltechSSDProcessor.getStudentLoss(self, student, studentOutputs[1], teacherOutputs[1])
  local bboxLoss = student.processor.smoothL1Criterion:forward(studentOutputs[2], teacherOutputs[2]);
  local bboxGrad = student.processor.smoothL1Criterion:backward(studentOutputs[2], teacherOutputs[2]);
  return classLoss * student.processor.classWeight + bboxLoss * student.processor.bboxWeight,
         {classGrad * student.processor.classWeight, bboxGrad * student.processor.bboxWeight}
end

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
  return CaltechSSDProcessor.forward(self, pathNames, labels, deterministic)
end

function M:train()
  error('Cannot train CaltechSSDTeacher.')
end

return M