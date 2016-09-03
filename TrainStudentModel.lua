package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'Model'
require 'MSECovCriterion'
require 'SoftCrossEntropyCriterion'
require 'Utils'

local cmd = torch.CmdLine()
cmd:argument('-teacher', 'teacher model to load')
cmd:argument('-student', 'student model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained student model')
defineBaseOptions(cmd)      -- defined in Utils.lua
defineTrainingOptions(cmd)  -- defined in Utils.lua
cmd:option('-useSameInputs', false, 'the teacher and student use the same inputs, so no need to load it twice')
cmd:option('-matchLayer', 2, 'which layers to match outputs, counting from the end. Defaults to second last layer (i.e. input to SoftMax layer)')
cmd:option('-useMSE', false, 'use mean squared error instead of soft cross entropy')
cmd:option('-T', 2, 'temperature for soft cross entropy')
cmd:option('-lambda', 0.5, 'hard target relative weight')
cmd:option('-dropoutBayes', 1, 'forward multiple time to achieve dropout as Bayesian approximation')
cmd:option('-useCOV', false, 'use Vishnu\'s covariance weighted error when using dropoutBayes')
processArgs(cmd)

local softCriterion
if opts.useCOV then
  if opts.dropoutBayes == 1 then
    error('useCOV requires dropoutBayes. Please set useMSE if not using dropout.')
  end
  softCriterion = MSECovCriterion(false)
elseif opts.useMSE then
  softCriterion = nn.MSECriterion(false)
else
  softCriterion = SoftCrossEntropyCriterion(opts.T, false)
end
softCriterion = softCriterion:cuda()

local teacher = Model(opts.teacher)
local student = Model(opts.student)
local studentLayer = #student.module.modules - opts.matchLayer + 1
for i=studentLayer+1,#student.module.modules do
  student.module:remove()
end
local teacherLayer = #teacher.module.modules - opts.matchLayer + 1
for i=teacherLayer+1,#teacher.module..modules do
  student.module:add(teacher.module:get(i):clone())
end
student.params, student.gradParams = student:getParameters()

local function train(pathNames)
  if softCriterion.sizeAverage ~= false or student.processor.criterion.sizeAverage ~= false then
    error('this function assumes criterion.sizeAverage == false because we divide through by batchCount.')
  end

  local labels = student.processor:getLabels(pathNames)
  local studentInputs, augmentations = student.processor:loadAndPreprocessInputs(pathNames)

  local teacherInputs = studentInputs
  if not(opts.useSameInputs) then
    teacherInputs = teacher.processor:loadAndPreprocessInputs(pathNames, augmentations)
  end

  local teacherLayerOutputs, outputs, variance=1
  if opts.dropoutBayes > 1 then
    teacher:training()
    teacher:forward(teacherInputs)
    local mean = teacher:get(teacherLayer).output
    local sumsqr = torch.cmul(mean, mean)
    outputs = mean.new(opts.dropoutBayes, mean:size(1), mean:size(2))
    outputs[1] = mean

    local l
    for l=1,teacherLayer do
      if hasDropout(teacher:get(l)) then
        break
      end
    end
    local out_init = teacher:get(l-1).output
    for i=2,opts.dropoutBayes do
      local out = out_init
      for j=l,teacherLayer do
        out = teacher:get(j):forward(out)
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
    teacher:forward(teacherInputs, true)
    teacherLayerOutputs = teacher:get(teacherLayer).output
  end

  local studentOutputs = student.processor:forward(studentInputs)
  local studentLayerOutputs = student:get(studentLayer).output

  if opts.useCOV then
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

  local softLoss = softCriterion:forward(studentLayerOutputs, teacherLayerOutputs)
  local softGradOutputs = softCriterion:backward(studentLayerOutputs, teacherLayerOutputs)
  if opts.dropoutBayes > 1 and not(opts.useCOV) then
    softGradOutputs = torch.cdiv(softGradOutputs, variance)
  end
  student.processor:backward(studentInputs, softGradOutputs / opts.batchCount, studentLayer)

  -- Hard labels
  local hardLoss = student.processor.criterion:forward(studentOutputs, labels)*opts.lambda
  local hardGradOutputs = student.processor.criterion:backward(studentOutputs, labels)*opts.lambda
  student.processor:backward(studentInputs, hardGradOutputs / opts.batchCount)

  student.processor:updateStats(pathNames, studentOutputs, labels)

  return softLoss + hardLoss, labels:size(1)
end

student:train(train)
print("Done!\n")
