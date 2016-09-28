package.path = package.path .. ';/home/jshen/scripts/?.lua'
torch.setdefaulttensortype('torch.FloatTensor')

require 'image'

local cmd = torch.CmdLine()
cmd:argument('-image', 'base image')
cmd:argument('-boxes', 'boxes for image')
cmd:argument('-output', 'output')
opts = cmd:parse(arg or {})

local img = image.load(opts.image)

local lines = 0
for _ in io.lines(opts.boxes) do
  lines = lines + 1
end
local boxes = torch.Tensor(lines, 5)
local df = torch.DiskFile(opts.boxes, 'r')
df:readFloat(boxes:storage())
df:close()
boxes[{{}, {3, 4}}] = boxes[{{}, {3, 4}}] + boxes[{{}, {1, 2}}];
boxes[{{}, {1, 2}}] = boxes[{{}, {1, 2}}] + 1;

for i=1,boxes:size(1) do
  image.drawRect(img, boxes[i][1], boxes[i][2], boxes[i][3], boxes[i][4], {inplace = true})
end
image.save(opts.output, img)
