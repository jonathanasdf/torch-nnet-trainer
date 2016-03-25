package.path = package.path .. ';/home/jshen/scripts/?.lua'

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
cmd:option('-teacher_processor_opts', '', 'alternate processor options for teacher model')
cmd:option('-hintLayer', '', 'which hint layer to use. Defaults to last non-softmax layer')
cmd:option('-T', 2, 'temperature')
cmd:option('-lambda', 0.5, 'hard target relative weight')
processArgs(cmd)

assert(paths.filep(opts.teacher), 'Cannot find teacher model ' .. opts.teacher)
assert(paths.filep(opts.student), 'Cannot find student model ' .. opts.student)

local student = Model(opts.student)
local hasSoftmax = false
if #student:findModules('cudnn.SoftMax') ~= 0 or
   #student:findModules('nn.SoftMax') ~= 0 then
  hasSoftmax = true
end
opts.processor.model = student
opts.processor:initializeThreads()

local local_teacher = Model(opts.teacher)
if #local_teacher:findModules('cudnn.SoftMax') ~= 0 or
   #local_teacher:findModules('nn.SoftMax') ~= 0 then
  local_teacher.model:remove()
end
local local_teacher_processor
if opts.teacher_processor ~= '' then
  if opts.teacher_processor_opts ~= '' then
    opts.processor_opts = opts.teacher_processor_opts
  end
  local_teacher_processor = requirePath(opts.teacher_processor).new()
end

local criterion = SoftCrossEntropyCriterion(opts.T)
if nGPU > 0 then
  criterion = criterion:cuda()
end


local specific = threads:specific()
threads:specific(true)
if nGPU > 0 then assert(cutorch.getDevice() == 1) end
local nDevices = math.max(nGPU, 1)
for device=1,nDevices do
  if device ~= 1 then
    cutorch.setDevice(device)
    local_teacher = local_teacher:clone()
  end
  local teacher_mutex_id = (require 'threads').Mutex():id()
  for i=device-1,nThreads,nDevices do if i > 0 then
    threads:addjob(i,
      function()
        require 'SoftCrossEntropyCriterion'
      end
    )
    threads:addjob(i,
      function()
        teacher = local_teacher
        if opts.teacher_processor ~= '' then
          teacher_processor = local_teacher_processor
        end
        teacher_mutex = (require 'threads').Mutex(teacher_mutex_id)
        soft_criterion = criterion:clone()
      end
    )
  end end
end
if nGPU > 0 then cutorch.setDevice(1) end
threads:specific(specific)


local function train(pathNames, student_inputs)
  if nGPU > 0 and not(student_inputs.getDevice) then student_inputs = student_inputs:cuda() end
  local labels = processor.getLabels(pathNames)

  local teacher_inputs = student_inputs
  if teacher_processor then
    _, teacher_inputs = DataLoader.loadInputs(pathNames, bind_post(teacher_processor.preprocessFn, true))
    if nGPU > 0 and not(teacher_inputs.getDevice) then teacher_inputs = teacher_inputs:cuda() end
  end

  teacher_mutex:lock()
  local logits = teacher:forward(teacher_inputs, true)
  if opts.hintLayer ~= '' then
    logits = findModuleByName(teacher, opts.hintLayer).output:clone()
  end
  teacher_mutex:unlock()

  mutex:lock()

  local student_outputs = model:forward(student_inputs)
  local student_logits = student_outputs
  if hasSoftmax then
    student_logits = model.model.modules[#model.model.modules-1].output
  end

  soft_criterion:forward(student_logits, logits)
  local soft_grad_outputs = soft_criterion:backward(student_logits, logits)*opts.T*opts.T

  if hasSoftmax then
    -- Assumes that model structure is Sequential(Sequential(everything_else)):add(SoftMax())
    model.model.modules[#model.model.modules-1]:backward(student_inputs, soft_grad_outputs)
  else
    model:backward(student_inputs, soft_grad_outputs)
  end

  -- Hard labels
  processor.criterion:forward(student_outputs, labels)
  local hard_grad_outputs = processor.criterion:backward(student_outputs, labels)*opts.lambda
  model:backward(student_inputs, hard_grad_outputs)

  mutex:unlock()
end

student:train(train)
print("Done!")
