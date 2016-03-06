package.path = package.path .. ';/home/jshen/scripts/?.lua'
torch.setdefaulttensortype('torch.FloatTensor')

require 'cutorch'
require 'cudnn'
require 'model'
require 'paths'
require 'utils'
require 'SoftCrossEntropyCriterion'

local cmd = torch.CmdLine()
cmd:argument('-teacher', 'teacher model to load')
cmd:argument('-student', 'student model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained student model')
defineBaseOptions(cmd)     --defined in utils.lua
defineTrainingOptions(cmd) --defined in train.lua
cmd:option('-T', 4, 'temperature')

local opt = processArgs(cmd)
assert(paths.filep(opt.teacher), 'Cannot find teacher model ' .. opt.teacher)
assert(paths.filep(opt.student), 'Cannot find student model ' .. opt.student)

local teacher = Model(opt.teacher)
if #teacher.model:findModules('cudnn.SoftMax') ~= 0 or
   #teacher.model:findModules('nn.SoftMax') ~= 0 then
  teacher.model:remove()
end

local student = Model(opt.student)
local hasSoftmax = false
if #student.model:findModules('cudnn.SoftMax') ~= 0 or
   #student.model:findModules('nn.SoftMax') ~= 0 then
  hasSoftmax = true
end

local criterion = SoftCrossEntropyCriterion(opt.T)
if nGPU > 0 then
  criterion = criterion:cuda()
end

local function updates(teacher, student, paths, inputs)
  local logits = teacher:forward(inputs, true)
  local student_logits = student:forward(inputs)
  if hasSoftmax then
    student_logits = student.model.modules[#student.model.modules-1].output
  end
  local loss = criterion:forward(student_logits, logits)
  local grad_outputs = criterion:backward(student_logits, logits)

  student:zeroGradParameters()
  if hasSoftmax then
    student.model.modules[#student.model.modules-1]:backward(inputs, grad_outputs)
  else
    student:backward(inputs, grad_outputs)
  end

  return function(x)
    return loss, student.gradParameters
  end
end

student:train(opt, bind(updates, teacher))
student:save(opt.output)
