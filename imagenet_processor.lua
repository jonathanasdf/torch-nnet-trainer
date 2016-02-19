require 'image'
require 'paths'

local M = {}

local words = {}
for line in io.lines'/file/jshen/data/ILSVRC2012_devkit_t12/words.txt' do
  table.insert(words, string.sub(line,11))
end

function M.preprocess(img)
  -- find the smaller dimension, and resize it to 256 
  if img:size(3) < img:size(2) then
     img = image.scale(img, 256, 256 * img:size(2) / img:size(3))
  else
     img = image.scale(img, 256 * img:size(3) / img:size(2), 256)
  end

  local p = 224
  local iW = img:size(3)
  local iH = img:size(2)
  local w1 = math.ceil((iW-p)/2)
  local h1 = math.ceil((iH-p)/2)
  img = image.crop(img, w1, h1, w1+p, h1+p) -- center patch
  local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(255.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end

function M.processOutputs(outputs, ...)
  local pathNames = (...)
  for i=1,outputs:size(1) do
    local val, classes = outputs[i]:view(-1):sort(true) 
    local name = ""
    for j=1,pathNames[i]:size(1) do
      if pathNames[i][j] ~= 0 then
        name = name .. string.char(pathNames[i][j])
      end
    end
    local result = 'predicted classes for ' .. paths.basename(name) .. ': '
    for j=1,5 do
      result = result .. "(" .. math.floor(val[j]*100 + 0.5) .. "%) " .. words[classes[j]] .. "; "
    end
    print(result)
  end
end

return M
