require 'cutorch'
require 'model'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to load')
cmd:argument('-input', 'input file or folder')
cmd:argument('-processor',
             'lua file that preprocesses input and handles output. '
             .. 'Functions that can be defined:\n'
             .. '    -preprocess(img): takes a single img and prepares it for the network\n'
             .. '    -processOutputs(outputs): is called once with the outputs from each batchSize')
cmd:option('-batchSize', -1, 'batch size')
cmd:option('-nThreads', 8, 'number of threads')
cmd:option('-nGPU', 4, 'number of GPU to use')
cmd:option('-gpu', 1, 'default GPU to use')
local opt = cmd:parse(arg or {})

assert(paths.filep(opt.model), "Cannot find model " .. opt.model)

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.gpu)

local processor = dofile(opt.processor)

paths.dofile('dataLoader.lua')
local loader = dataLoader{
  path = opt.input,
  preprocessor = processor.preprocess,
  verbose = true
}

local model = Model{nGPU=opt.nGPU, gpu=opt.gpu}
model:load(opt.model)

local batchSize = opt.batchSize
if batchSize == -1 then
  batchSize = loader:size()
end 

local function testBatch(batchNumber, paths, inputs)
  processor.processOutputs(model:forward(inputs, true), paths)
end

loader:runAsync(batchSize, 
                math.ceil(loader:size() / batchSize), -- epochSize
                false,                                -- don't shuffle,
                opt.nThreads,
                testBatch)                            -- resultHandler
