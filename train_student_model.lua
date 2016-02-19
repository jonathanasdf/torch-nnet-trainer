require 'cutorch'
require 'cudnn'
require 'model'
require 'paths'
require 'optim'
require 'train'
require 'utils'
require 'SoftCrossEntropyCriterion'

local cmd = torch.CmdLine()
cmd:argument('-teacher', 'teacher model to load')
cmd:argument('-student', 'student model to train')
cmd:argument('-output', 'path to save trained student model')
cmd:argument('-input', 'input file or folder')
cmd:argument('-processor',
             'lua file that preprocesses input and handles output. '
             .. 'Functions that can be defined:\n'
             .. '    -preprocess(img): takes a single img and prepares it for the network')
cmd:option('-T', 4, 'temperature')
cmd:option('-LR', 0.01, 'learning rate')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-batchSize', 32, 'batch size')
cmd:option('-epochs', 50, 'num epochs')
cmd:option('-epochSize', -1, 'num batches per epochs')
cmd:option('-nThreads', 8, 'number of threads')
cmd:option('-nGPU', 4, 'number of GPU to use')
cmd:option('-gpu', 1, 'default GPU to use')
local opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.gpu)

assert(paths.filep(opt.teacher), "Cannot find teacher model " .. opt.teacher)
assert(paths.filep(opt.student), "Cannot find student model " .. opt.student)

local processor = dofile(opt.processor)

paths.dofile('dataLoader.lua')
local loader = dataLoader{
  path = opt.input,
  preprocessor = processor.preprocess,
  verbose = true
}

local teacher = Model{nGPU=opt.nGPU, gpu=opt.gpu}
teacher:load(opt.teacher)

local student = Model{nGPU=opt.nGPU, gpu=opt.gpu}
student:load(opt.student)
if #student.model:findModules('cudnn.SoftMax') ~= 0 then
  student.model:remove()
end

local criterion = SoftCrossEntropyCriterion(opt.T):cuda()

local function updates(teacher, student, inputs)
  local parameters, grad_parameters = student:getParameters()
  
  return function(x)
    if parameters ~= x then
      parameters:copy(x)
    end

    teacher:forward(inputs, true)
    local logits = findModuleByName(teacher.model, 'fc8').output
    local student_logits = student:forward(inputs)
    
    local err = criterion:forward(student_logits, logits)
    local grad_outputs = criterion:backward(student_logits, logits)

    student:zeroGradParameters()
    student:backward(inputs, grad_outputs)
    return err, grad_parameters
  end
end

train(student, loader, opt, bind(updates, teacher, student))
student.model:add(cudnn.SoftMax())
saveDataParallel(opt.output, student.model)
