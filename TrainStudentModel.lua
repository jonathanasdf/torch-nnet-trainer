package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'paths'
cv = require 'cv'
require 'cv.cudawarping'
require 'cv.imgcodecs'

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
cmd:option('-teacherProcessor', '', 'alternate processor for teacher model. Only the preprocess function is used')
cmd:option('-teacherProcessorOpts', '', 'alternate processor options for teacher model')
cmd:option('-matchLayer', 2, 'which layers to match outputs, counting from the end. Defaults to second last layer (i.e. input to SoftMax layer)')
cmd:option('-useMSE', false, 'use mean squared error instead of soft cross entropy')
cmd:option('-T', 2, 'temperature for soft cross entropy')
cmd:option('-lambda', 0.5, 'hard target relative weight')
processArgs(cmd)

assert(paths.filep(opts.teacher), 'Cannot find teacher model ' .. opts.teacher)
assert(paths.filep(opts.student), 'Cannot find student model ' .. opts.student)

local criterion = opts.useMSE and nn.MSECriterion(false) or SoftCrossEntropyCriterion(opts.T, false)
if nGPU > 0 then
  criterion = criterion:cuda()
end

local teacher = Model(opts.teacher)
local teacherProcessor
if opts.teacherProcessor ~= '' then
  teacherProcessor = requirePath(opts.teacherProcessor).new(teacher, opts.teacherProcessorOpts)
end

local student = Model(opts.student)
local studentProcessor = requirePath(opts.processor).new(student, opts.processorOpts)
studentProcessor.studentLayer = #student.model.modules - opts.matchLayer + 1
for i=studentProcessor.studentLayer+1,#student.model.modules do
  student.model:remove()
end
studentProcessor.teacherLayer = #teacher.model.modules - opts.matchLayer + 1
for i=studentProcessor.teacherLayer+1,#teacher.model.modules do
  student.model:add(teacher.model:get(i):clone())
end
studentProcessor.lambda = opts.lambda
studentProcessor:initializeThreads()


if nThreads == 0 then
  _teacher = teacher
  _teacherProcessor = teacherProcessor
  teacherMutex = {}
  teacherMutex.lock = function() end
  teacherMutex.unlock = function() end
  softCriterion = criterion
else
  local specific = threads:specific()
  threads:specific(true)
  if nGPU > 0 then assert(cutorch.getDevice() == 1) end
  local nDevices = math.max(nGPU, 1)
  local localTeacher = teacher
  local localSoftCriterion = criterion
  for device=1,nDevices do
    if device ~= 1 then
      cutorch.setDevice(device)
      localTeacher = localTeacher:clone()
      localSoftCriterion = localSoftCriterion:clone()
    end
    local teacherMutexId = (require 'threads').Mutex():id()
    for i=device-1,nThreads,nDevices do if i > 0 then
      threads:addjob(i,
        function()
          require 'SoftCrossEntropyCriterion'
          if opts.teacherProcessor ~= '' then
            requirePath(opts.teacherProcessor)
          end
        end
      )
      threads:addjob(i,
        function()
          _teacher = localTeacher
          if teacherProcessor then
            _teacherProcessor = teacherProcessor
          end
          teacherMutex = (require 'threads').Mutex(teacherMutexId)
          softCriterion = localSoftCriterion
        end
      )
    end end
  end
  if nGPU > 0 then cutorch.setDevice(1) end
  threads:specific(specific)
end


local function train(pathNames, studentInputs)
  if softCriterion.sizeAverage ~= false or _processor.criterion.sizeAverage ~= false then
    error('this function assumes criterion.sizeAverage == false because we divide through by batchCount.')
  end

  local labels = _processor.getLabels(pathNames)
  if nGPU > 0 and not(studentInputs.getDevice) then studentInputs = studentInputs:cuda() end
  if nGPU > 0 and not(labels.getDevice) then labels = labels:cuda() end

  local teacherInputs = studentInputs
  if _teacherProcessor then
    _, teacherInputs = DataLoader.loadInputs(pathNames, bindPost(_teacherProcessor.preprocessFn, true))
    if nGPU > 0 and not(teacherInputs.getDevice) then teacherInputs = teacherInputs:cuda() end
  end

  teacherMutex:lock()
  _teacher:forward(teacherInputs, true)
  local teacherLayerOutputs = _teacher.model:get(_processor.teacherLayer).output:clone()
  teacherMutex:unlock()

  mutex:lock()

  local studentOutputs = _processor.forward(studentInputs)
  local studentLayerOutputs = _model.model:get(_processor.studentLayer).output

  local softLoss = softCriterion:forward(studentLayerOutputs, teacherLayerOutputs)
  local softGradOutputs = softCriterion:backward(studentLayerOutputs, teacherLayerOutputs)
  _processor.backward(studentInputs, softGradOutputs / opts.batchCount, _processor.studentLayer)

  -- Hard labels
  local hardLoss = _processor.criterion:forward(studentOutputs, labels)*_processor.lambda
  local stats = _processor.calcStats(pathNames, studentOutputs, labels)
  local hardGradOutputs = _processor.criterion:backward(studentOutputs, labels)*_processor.lambda
  _processor.backward(studentInputs, hardGradOutputs / opts.batchCount)

  mutex:unlock()

  return softLoss + hardLoss, labels:size(1), stats
end

student:train(train)
print("Done!\n")
