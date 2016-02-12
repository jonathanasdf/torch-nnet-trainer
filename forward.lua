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
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')

processor = dofile(opt.processor)

paths.dofile('dataLoader.lua')
loader = dataLoader{
  path = opt.input,
  preprocessor = processor.preprocess,
  verbose = true
}

model = paths.dofile('model.lua')
model:load(opt.model)

batchSize = opt.batchSize
if batchSize == -1 then
  batchSize = loader:size()
end 

local function testBatch(batchNumber, names, inputs)
  model:test(inputs, processor.processOutputs, names)
end

loader:runAsync(batchSize, testBatch, opt.nThreads)
