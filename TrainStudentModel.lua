package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'Model'
require 'Utils'

local cmd = torch.CmdLine()
cmd:argument('-teacher', 'teacher model to load')
cmd:argument('-student', 'student model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained student model')
defineBaseOptions(cmd)      -- defined in Utils.lua
defineTrainingOptions(cmd)  -- defined in Utils.lua
cmd:option('-useSameInputs', false, 'the teacher and student use the same inputs, so no need to load it twice')
cmd:option('-matchLayer', 1, 'which layers to match outputs, counting from the end. Defaults to last layer')
cmd:option('-teacherMatchLayer', -1, 'which teacher layer to match, counting from the end. Defaults to -1 which means use matchLayer')
cmd:option('-copyTeacherLayers', false, 'whether to copy teacher layers after the matchLayer')
cmd:option('-useMSE', false, 'use mean squared error instead of soft cross entropy')
cmd:option('-T', 2, 'temperature for soft cross entropy')
cmd:option('-lambda', 0.5, 'hard target relative weight')
cmd:option('-dropoutBayes', 1, 'forward multiple time to achieve dropout as Bayesian approximation')
cmd:option('-useCOV', false, 'use Vishnu\'s covariance weighted error when using dropoutBayes')
processArgs(cmd)

if opts.nGPU > 1 then
  error('TrainStudentModel can only use nGPU = 1.')
end

local teacher = Model(opts.teacher)
local student = Model(opts.student)

local studentContainer = student
while #studentContainer.modules == 1 and studentContainer.modules[1].modules do
  studentContainer = studentContainer.modules[1]
end
local studentLayer = #studentContainer.modules - opts.matchLayer + 1

if opts.teacherMatchLayer == -1 then
  opts.teacherMatchLayer = opts.matchLayer
end
local teacherContainer = teacher
while #teacherContainer.modules == 1 and teacherContainer.modules[1].modules do
  teacherContainer = teacherContainer.modules[1]
end
local teacherLayer = math.max(1, #teacherContainer.modules - opts.teacherMatchLayer + 1)

if opts.copyTeacherLayers then
  for i=studentLayer+1,#studentContainer.modules do
    studentContainer:remove()
  end
  for i=teacherLayer+1,#teacherContainer.modules do
    studentContainer:add(teacherContainer:get(i):clone())
  end
  student.params, student.gradParams = student:getParameters()
end

local function train(pathNames)
  local studentInputs, augmentations = student.processor:loadAndPreprocessInputs(pathNames)

  local teacherInputs = studentInputs
  if not(opts.useSameInputs) then
    teacherInputs = teacher.processor:loadAndPreprocessInputs(pathNames, augmentations)
  end

  local teacherLayerOutputs, outputs, variance=1
  if opts.dropoutBayes > 1 then
    teacher.processor:forward(pathNames, teacherInputs)
    local mean = teacherContainer:get(teacherLayer).output
    local sumsqr = torch.cmul(mean, mean)
    outputs = mean.new(opts.dropoutBayes, mean:size(1), mean:size(2))
    outputs[1] = mean

    local start
    for l=1,teacherLayer do
      if hasDropout(teacherContainer:get(l)) then
        start = l-1
        break
      end
    end
    if start == nil then
      start = teacherLayer
    end
    local out_init = teacherContainer:get(start).output
    for i=2,opts.dropoutBayes do
      local out = out_init
      for j=start+1,teacherLayer do
        out = teacherContainer:get(j):forward(out)
      end
      mean = mean + out
      sumsqr = sumsqr + torch.cmul(out, out)
      outputs[i] = out
    end
    sumsqr = sumsqr / opts.dropoutBayes
    mean = mean / opts.dropoutBayes
    variance = sumsqr - torch.cmul(mean, mean)
    variance = torch.cinv(torch.exp(-variance) + 1)*2 - 0.5  --pass into sigmoid function
    teacherLayerOutputs = mean
  else
    teacher.processor:forward(pathNames, teacherInputs, true)
    teacherLayerOutputs = teacherContainer:get(teacherLayer).output
  end

  local studentOutputs = student.processor:forward(pathNames, studentInputs)
  local studentLayerOutputs = studentContainer:get(studentLayer).output

  if opts.useCOV then
    error('useCOV is broken right now.')
    local cov = teacherLayerOutputs.new(teacherLayerOutputs:size(1), teacherLayerOutputs:size(2), teacherLayerOutputs:size(2)):zero()
    for i=1,teacherLayerOutputs:size(1) do
      for j=1,opts.dropoutBayes do
        local diff = (outputs[j][i] - teacherLayerOutputs[i]):view(-1, 1)
        cov[i] = cov[i] + diff * diff:t()
      end
      cov[i] = (cov[i] / (opts.dropoutBayes - 1)):inverse()
    end
    softCriterion.invcov = cov
  end

  local loss, softGradOutputs = teacher.processor:getStudentLoss(student, studentLayerOutputs, teacherLayerOutputs)
  if opts.dropoutBayes > 1 and not opts.useCOV then
    softGradOutputs = torch.cdiv(softGradOutputs, variance)
  end
  if type(softGradOutputs) == 'table' then
    for i=1,#softGradOutputs do
      softGradOutputs[i] = softGradOutputs[i] / opts.batchCount
    end
  else
    softGradOutputs = softGradOutputs / opts.batchCount
  end
  student.processor:backward(studentInputs, softGradOutputs, studentLayer)

  -- Hard labels
  local labels = student.processor:getLabels(pathNames, studentOutputs)

  if student.processor.criterion and opts.lambda ~= 0 then
    local hardLoss, hardGradOutputs = student.processor:getLoss(studentOutputs, labels)
    loss = loss + hardLoss * opts.lambda
    if type(hardGradOutputs) == 'table' then
      for i=1,#hardGradOutputs do
        hardGradOutputs[i] = hardGradOutputs[i] * opts.lambda / opts.batchCount
      end
    else
      hardGradOutputs = hardGradOutputs * opts.lambda / opts.batchCount
    end
    student.processor:backward(studentInputs, hardGradOutputs)
  end

  student.processor:updateStats(pathNames, studentOutputs, labels)
  return loss, #pathNames
end

student:train(train)
print("Done!\n")
