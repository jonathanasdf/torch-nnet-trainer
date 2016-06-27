package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'svm'

require 'Model'
require 'Utils'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained model')
defineBaseOptions(cmd)  -- defined in Utils.lua
cmd:option('-layer', 'fc7', 'layer to train svm from')
processArgs(cmd)
setPhase('test')

local model = Model(opts.model)

local function getData(pathNames)
  local labels = model.processor:getLabels(pathNames)
  local inputs = model.processor:loadAndPreprocessInputs(pathNames)
  model.processor:forward(inputs, true)
  local outputs = findModuleByName(model, opts.layer).output:clone()

  return convertTensorToSVMLight(labels, outputs)
end

local data = {}
local function accumulateData(arr)
  for i=1,#arr do
    data[#data+1] = arr[i]
  end
end

local dataloader = DataLoader{inputs = opts.input}

model:run(dataloader,
          opts.batchSize,
          opts.epochSize,
          true,           -- randomSample,
          getData,
          accumulateData)

local svmmodel = liblinear.train(data, '-s 0 -B 1')
torch.save(opts.output, svmmodel)
print("Done!\n")
