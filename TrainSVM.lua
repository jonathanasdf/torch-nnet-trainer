package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'paths'
require 'svm'

require 'Model'
require 'Utils'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained model')
--defined in utils.lua
defineBaseOptions(cmd)
cmd:option('-layer', 'fc7', 'layer to train svm from')
processArgs(cmd)

assert(paths.filep(opts.model), 'Cannot find model ' .. opts.model)

local model = Model(opts.model)
opts.processor.model = model
opts.processor:initializeThreads()

local function getData(pathNames, inputs)
  local labels = opts.processor.getLabels(pathNames)

  mutex:lock()
  model:forward(inputs, true)
  local outputs = findModuleByName(model, opts.layer).output:clone()
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

DataLoader{path = opts.input}:runAsync(
  opts.batchSize,
  opts.epochSize,
  true,           -- shuffle,
  bindPost(opts.processor.preprocessFn, true),
  getData,
  accumulateData)

local svmmodel = liblinear.train(data, '-s 2 -B 1')
torch.save(opts.output, svmmodel)
