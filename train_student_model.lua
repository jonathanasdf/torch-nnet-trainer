package.path = package.path .. ';/home/jshen/scripts/?.lua'
torch.setdefaulttensortype('torch.FloatTensor')

require 'model'
require 'nn'
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
cmd:option('-teacher_processor', '', 'alternate processor for teacher model')
cmd:option('-hintLayer', '', 'which hint layer to use. Defaults to last non-softmax layer')
cmd:option('-T', 2, 'temperature')
cmd:option('-lambda', 0.5, 'hard target relative weight')

local opt = processArgs(cmd)
assert(paths.filep(opt.teacher), 'Cannot find teacher model ' .. opt.teacher)
assert(paths.filep(opt.student), 'Cannot find student model ' .. opt.student)

local teacher = Model(opt.teacher)
if #teacher.model:findModules('cudnn.SoftMax') ~= 0 or
   #teacher.model:findModules('nn.SoftMax') ~= 0 then
  teacher.model:remove()
end
if opt.teacher_processor ~= '' then
  opt.teacher_processor = requirePath(opt.teacher_processor)(opt)
  opt.teacher_processor.model = teacher
end

local student = Model(opt.student)
local hasSoftmax = false
if #student.model:findModules('cudnn.SoftMax') ~= 0 or
   #student.model:findModules('nn.SoftMax') ~= 0 then
  hasSoftmax = true
end
opt.processor.model = student

local criterion = SoftCrossEntropyCriterion(opt.T)
if nGPU > 0 then
  criterion = criterion:cuda()
end

local function trainBatch(student, pathNames, student_inputs)
  local teacher_inputs = student_inputs
  if opt.teacher_processor ~= '' then
    _, teacher_inputs = DataLoader.LoadInputs(pathNames, opt.teacher_processor.preprocessFn)
  end
  local logits = teacher:forward(teacher_inputs, true)
  if opt.hintLayer ~= '' then
    logits = findModuleByName(teacher, opt.hintLayer).output
  end

  local student_outputs = student:forward(student_inputs)
  local student_logits = student_outputs
  if hasSoftmax then
    student_logits = student.model.modules[#student.model.modules-1].output
  end

  criterion:forward(student_logits, logits)
  local soft_grad_outputs = criterion:backward(student_logits, logits)*opt.T*opt.T

  if hasSoftmax then
    -- Assumes that model structure is Sequential(Sequential(everything_else)):add(SoftMax())
    student.model.modules[#student.model.modules-1]:backward(student_inputs, soft_grad_outputs)
  else
    student:backward(student_inputs, soft_grad_outputs)
  end

  local hard_grad_outputs = opt.processor:evaluateBatch(pathNames, student_outputs)*opt.lambda
  student:backward(student_inputs, hard_grad_outputs)
end

student:train(opt, trainBatch)
