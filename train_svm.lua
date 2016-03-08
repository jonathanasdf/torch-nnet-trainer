package.path = package.path .. ';/home/jshen/scripts/?.lua'
torch.setdefaulttensortype('torch.FloatTensor')

require 'svm'

require 'model'
require 'paths'
require 'utils'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained model')
--defined in utils.lua
defineBaseOptions(cmd)
cmd:option('-layer', 'fc7', 'layer to train svm from')
-- Additional processor functions:
--   -getLabels(pathNames): return labels for examples

local opt = processArgs(cmd)
assert(paths.filep(opt.model), 'Cannot find model ' .. opt.model)

local model = Model(opt.model)
local class = require 'class'
local data = {}

local function accumulateData(pathNames, inputs)
  local labels = opt.processor:getLabels(pathNames)

  model:forward(inputs, true)
  local outputs = findModuleByName(model, opt.layer).output
  convertTensorToSVMLight(labels, outputs, data)
end

DataLoader{
  path = opt.input,
  preprocessor = opt.processor.preprocess
}:runAsync(opt.batchSize,
           opt.epochSize,
           true,           -- shuffle,
           accumulateData) -- resultHandler

local svm_model = liblinear.train(data, '-s 2 -B 1')
torch.save(opt.output, svm_model)
