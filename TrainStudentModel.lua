package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'paths'

require 'Model'
require 'SoftCrossEntropyCriterion'
require 'Utils'

local cmd = torch.CmdLine()
cmd:argument('-teacher', 'teacher model to load')
cmd:argument('-student', 'student model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained student model')
defineBaseOptions(cmd)     --defined in utils.lua
defineTrainingOptions(cmd) --defined in train.lua
cmd:option('-teacherProcessor', '', 'alternate processor for teacher model')
cmd:option('-teacherProcessorOpts', '', 'alternate processor options for teacher model')
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

local localTeacher = Model(opts.teacher)
if #localTeacher:findModules('cudnn.SoftMax') ~= 0 or
   #localTeacher:findModules('nn.SoftMax') ~= 0 then
  localTeacher.model:remove()
end
local localTeacherProcessor
if opts.teacherProcessor ~= '' then
  if opts.teacherProcessorOpts ~= '' then
    opts.processorOpts = opts.teacherProcessorOpts
  end
  localTeacherProcessor = requirePath(opts.teacherProcessor).new()
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
    localTeacher = localTeacher:clone()
  end
  local teacherMutexId = (require 'threads').Mutex():id()
  for i=device-1,nThreads,nDevices do if i > 0 then
    threads:addjob(i,
      function()
        require 'SoftCrossEntropyCriterion'
      end
    )
    threads:addjob(i,
      function()
        teacher = localTeacher
        if opts.teacherProcessor ~= '' then
          teacherProcessor = localTeacherProcessor
        end
        teacherMutex = (require 'threads').Mutex(teacherMutexId)
        softCriterion = criterion:clone()
      end
    )
  end end
end
if nGPU > 0 then cutorch.setDevice(1) end
threads:specific(specific)


local function train(pathNames, studentInputs)
  if nGPU > 0 and not(studentInputs.getDevice) then studentInputs = studentInputs:cuda() end
  local labels = processor.getLabels(pathNames)

  local teacherInputs = studentInputs
  if teacherProcessor then
    _, teacherInputs = DataLoader.loadInputs(pathNames, bindPost(teacherProcessor.preprocessFn, true))
    if nGPU > 0 and not(teacherInputs.getDevice) then teacherInputs = teacherInputs:cuda() end
  end

  teacherMutex:lock()
  local logits = teacher:forward(teacherInputs, true)
  if opts.hintLayer ~= '' then
    logits = findModuleByName(teacher, opts.hintLayer).output:clone()
  end
  teacherMutex:unlock()

  mutex:lock()

  local studentOutputs = model:forward(studentInputs)
  local studentLogits = studentOutputs
  if hasSoftmax then
    studentLogits = model.model.modules[#model.model.modules-1].output
  end

  softCriterion:forward(studentLogits, logits)
  local softGradOutputs = softCriterion:backward(studentLogits, logits)*opts.T*opts.T

  if hasSoftmax then
    -- Assumes that model structure is Sequential(Sequential(everything_else)):add(SoftMax())
    model.model.modules[#model.model.modules-1]:backward(studentInputs, softGradOutputs)
  else
    model:backward(studentInputs, softGradOutputs)
  end

  -- Hard labels
  processor.criterion:forward(studentOutputs, labels)
  local hardGradOutputs = processor.criterion:backward(studentOutputs, labels)*opts.lambda
  model:backward(studentInputs, hardGradOutputs)

  mutex:unlock()
end

student:train(train)
print("Done!")
