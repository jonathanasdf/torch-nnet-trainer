package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'cutorch'
require 'cudnn'
require 'model'
require 'paths'
require 'optim'
require 'dataLoader'
require 'utils'
require 'SoftCrossEntropyCriterion'

torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:argument('-teacher', 'teacher model to load')
cmd:argument('-student', 'student model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained student model')
defineBaseOptions(cmd)     --defined in utils.lua
defineTrainingOptions(cmd) --defined in train.lua
cmd:option('-T', 4, 'temperature')

local opt = processArgs(cmd) 
assert(paths.filep(opt.teacher), "Cannot find teacher model " .. opt.teacher)
assert(paths.filep(opt.student), "Cannot find student model " .. opt.student)

local loader = DataLoader{
  path = opt.input,
  preprocessor = opt.processor.preprocess,
  verbose = true
}

local teacher = Model{gpu=opt.gpu, nGPU=opt.nGPU}:load(opt.teacher)
if #teacher.model:findModules('cudnn.SoftMax') ~= 0 or 
   #teacher.model:findModules('nn.SoftMax') ~= 0 then
  teacher.model:remove()
end

local student = Model{gpu=opt.gpu, nGPU=opt.nGPU}:load(opt.student)
if #student.model:findModules('cudnn.SoftMax') ~= 0 or 
   #student.model:findModules('nn.SoftMax') ~= 0 then
  student.model:remove()
end

local criterion = SoftCrossEntropyCriterion(opt.T)
if opt.nGPU > 0 then
  criterion = criterion:cuda()
end

local function updates(teacher, student, paths, inputs)
  local parameters, grad_parameters = student:getParameters()
  
  return function(x)
    grad_parameters:zero()

    local logits = teacher:forward(inputs, true)
    local student_logits = student:forward(inputs)
    local loss = criterion:forward(student_logits, logits)
    local grad_outputs = criterion:backward(student_logits, logits)

    student:backward(inputs, grad_outputs)
    return loss, grad_parameters
  end
end

student:train(loader, opt, bind(updates, teacher))
if student.model.backend == 'cudnn' then
  student.model:add(cudnn.SoftMax())
else
  student.model:add(nn.SoftMax())
end
student:saveDataParallel(opt.output)
