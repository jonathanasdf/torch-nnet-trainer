package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'cutorch'
require 'model'
require 'dataLoader'

torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to load')
cmd:argument('-input', 'input file or folder')
cmd:argument('-processor',
             'lua file that preprocesses input and handles output. '
             .. 'Functions that can be defined:\n'
             .. '    -preprocess(img): takes a single img and prepares it for the network\n'
             .. '    -processOutputs(outputs): is called once with the outputs from each batchSize\n'
             .. '    -printStats(): is called once at the very end')
defineBaseOptions(cmd)     --defined in utils.lua
cmd:option('-batchSize', 16, 'batch size')
local opt = cmd:parse(arg or {})

assert(paths.filep(opt.model), "Cannot find model " .. opt.model)

local processor = dofile(opt.processor)

local loader = DataLoader{
  path = opt.input,
  preprocessor = processor.preprocess,
  verbose = true
}

local model = Model{gpu=opt.gpu, nGPU=opt.nGPU}
model:load(opt.model)

local batchSize = opt.batchSize
if batchSize == -1 then
  batchSize = loader:size()
end 

local function testBatch(paths, inputs)
  processor.processOutputs(model:forward(inputs, true), paths)
end

loader:runAsync(batchSize, 
                math.ceil(loader:size() / batchSize), -- epochSize
                false,                                -- don't shuffle,
                opt.nThreads,
                testBatch)                            -- resultHandler

processor.printStats()
