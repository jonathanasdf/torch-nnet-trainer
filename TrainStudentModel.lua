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
cmd:option('-teacherProcessor', '', 'alternate processor for teacher model')
cmd:option('-teacherProcessorOpts', '', 'alternate processor options for teacher model')
cmd:option('-matchLayer', 2, 'which layers to match outputs, counting from the end. Defaults to second last layer')
cmd:option('-useMSE', false, 'Use mean squared error instead of soft cross entropy')
cmd:option('-T', 2, 'temperature for soft cross entropy')
cmd:option('-lambda', 0.5, 'hard target relative weight')
processArgs(cmd)

assert(paths.filep(opts.teacher), 'Cannot find teacher model ' .. opts.teacher)
assert(paths.filep(opts.student), 'Cannot find student model ' .. opts.student)

local criterion = opts.useMSE and nn.MSECriterion() or SoftCrossEntropyCriterion(opts.T)
criterion.sizeAverage = false
if nGPU > 0 then
  criterion = criterion:cuda()
end

local localTeacher = Model(opts.teacher)
local localTeacherProcessor
if opts.teacherProcessor ~= '' then
  if opts.teacherProcessorOpts ~= '' then
    opts.processorOpts = opts.teacherProcessorOpts
  end
  localTeacherProcessor = requirePath(opts.teacherProcessor).new()
end

local student = Model(opts.student)
opts.processor.model = student

opts.processor.studentLayer = #student.model.modules - opts.matchLayer + 1
for i=opts.processor.studentLayer+1,#student.model.modules do
  student.model:remove()
end
opts.processor.teacherLayer = #localTeacher.model.modules - opts.matchLayer + 1
for i=opts.processor.teacherLayer+1,#localTeacher.model.modules do
  student.model:add(localTeacher.model:get(i):clone())
end

opts.processor:initializeThreads()


if nThreads == 0 then
  teacher = localTeacher
  teacherProcessor = localTeacherProcessor
  teacherMutex = {}
  teacherMutex.lock = function() end
  teacherMutex.unlock = function() end
  softCriterion = criterion
else
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
end


local function train(pathNames, studentInputs)
  if nGPU > 0 and not(studentInputs.getDevice) then studentInputs = studentInputs:cuda() end
  local labels = processor.getLabels(pathNames)

  local teacherInputs = studentInputs
  if teacherProcessor then
    _, teacherInputs = DataLoader.loadInputs(pathNames, bindPost(teacherProcessor.preprocessFn, true))
    if nGPU > 0 and not(teacherInputs.getDevice) then teacherInputs = teacherInputs:cuda() end
  end

  teacherMutex:lock()
  teacher:forward(teacherInputs, true)
  local teacherLayerOutputs = teacher.model:get(processor.teacherLayer).output:clone()
  teacherMutex:unlock()

  mutex:lock()
  local studentOutputs = model:forward(studentInputs)
  local studentLayerOutputs = model.model:get(processor.studentLayer).output

  local softLoss = softCriterion:forward(studentLayerOutputs, teacherLayerOutputs)
  local softGradOutputs = softCriterion:backward(studentLayerOutputs, teacherLayerOutputs)
  local softGradParams = processor.backward(studentInputs, softGradOutputs, processor.studentLayer)

  -- Hard labels
  local hardLoss = processor.criterion:forward(studentOutputs, labels)*opts.lambda
  local stats = processor.calcStats(pathNames, studentOutputs, labels)
  local hardGradOutputs = processor.criterion:backward(studentOutputs, labels)*opts.lambda
  local hardGradParams = processor.backward(studentInputs, hardGradOutputs)

  local gradParams
  if softGradParams then
    gradParams = softGradParams + hardGradParams
  end
  mutex:unlock()

  return gradParams, softLoss + hardLoss, labels:size(1), stats
end

student:train(train)
print("Done!\n")
