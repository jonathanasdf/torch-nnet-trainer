package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'paths'
require 'svm'

require 'Model'
require 'Utils'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained model')
cmd:argument(
    '-processor',
    'REQUIRED. lua file that does the heavy lifting. ' ..
    'See processor.lua for functions that can be defined.\n'
  )
defineBaseOptions(cmd)     --defined in utils.lua
cmd:option('-processorOpts', '', 'additional options for the processor')
defineBaseOptions(cmd)
cmd:option('-layer', 'fc7', 'layer to train svm from')
processArgs(cmd)

assert(paths.filep(opts.model), 'Cannot find model ' .. opts.model)

local model = Model(opts.model)
local processor = requirePath(opts.processor).new(model, opts.processorOpts)
processor:initializeThreads()

local function getData(pathNames, inputs)
  if nGPU > 0 and not(inputs.getDevice) then inputs = inputs:cuda() end
  local labels = _processor.getLabels(pathNames)

  mutex:lock()
  _processor.forward(inputs, true)
  local outputs = findModuleByName(_model, opts.layer).output:clone()
  mutex:unlock()

  return convertTensorToSVMLight(labels, outputs)
end

local data = {}
local function accumulateData(arr)
  for i=1,#arr do
    data[#data+1] = arr[i]
  end
  jobDone()
end

DataLoader{inputs = opts.input}:runAsync(
  opts.batchSize,
  opts.epochSize,
  true,           -- randomSample,
  bindPost(processor.preprocessFn, true),
  getData,
  accumulateData)

local svmmodel = liblinear.train(data, '-s 0 -B 1')
torch.save(opts.output, svmmodel)
print("Done!\n")
