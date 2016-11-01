require 'Utils'
function nms(boxes, scores, overlap)
  local keep = {}

  local x1 = boxes[{{},1}]
  local y1 = boxes[{{},2}]
  local x2 = boxes[{{},3}]
  local y2 = boxes[{{},4}]
  local area = torch.cmul(x2-x1+1, y2-y1+1)

  local _, I = scores:sort()
  local less = torch.ByteTensor(boxes:size(1)):fill(0)
  for b=1,boxes:size(1) do
    local i = I[b]
    less[i] = 1

    local inter = torch.cmax(torch.cmin(x2, x2[i]) - torch.cmax(x1, x1[i]) + 1, 0):cmul(
                  torch.cmax(torch.cmin(y2, y2[i]) - torch.cmax(y1, y1[i]) + 1, 0))
    local union = area + area[i] - inter
    if torch.cmax(less, torch.cdiv(inter, union):lt(overlap):byte()):all() then
      keep[#keep+1] = i
    end
  end
  return torch.LongTensor(keep)
end

function nmsCaltech(dir)
  for filename, attr in dirtree(dir) do
    if attr.mode == 'file' and attr.size > 0 then
      local lines = 0
      for _ in io.lines(filename) do
        lines = lines + 1
      end
      local boxes = torch.FloatTensor(lines, 5)
      local df = torch.DiskFile(filename, 'r')
      df:readFloat(boxes:storage())
      df:close()
      boxes[{{}, {3, 4}}] = boxes[{{}, {3, 4}}] + boxes[{{}, {1, 2}}];
      boxes[{{}, {1, 2}}] = boxes[{{}, {1, 2}}] + 1;
      local indexes = nms(boxes[{{}, {1, 4}}], boxes[{{}, 5}], 0.5)

      local file, err = io.open(filename, 'w')
      if not(file) then error(err) end

      for i=1, indexes:size(1) do
        local box = boxes[indexes[i]]
        file:write(box[1]-1, ' ',  box[2]-1, ' ', box[3]-box[1]+1, ' ', box[4]-box[2]+1, ' ', box[5], '\n')
      end
      file:close()
    end
  end
end

function nmsKITTI(dir)
  for filename, attr in dirtree(dir) do
    if attr.mode == 'file' and attr.size > 0 then
      local lines = 0
      for _ in io.lines(filename) do
        lines = lines + 1
      end
      local classes = {}
      local boxes = torch.FloatTensor(lines, 5)
      lines = 0
      for l in io.lines(filename) do
        lines = lines + 1
        local s = l:split(' ')
        classes[lines] = s[1]
        boxes[lines][1] = tonumber(s[2])
        boxes[lines][2] = tonumber(s[3])
        boxes[lines][3] = tonumber(s[4])
        boxes[lines][4] = tonumber(s[5])
        boxes[lines][5] = tonumber(s[6])
      end
      local indexes = nms(boxes[{{}, {1, 4}}], boxes[{{}, 5}], 0.5)

      local file, err = io.open(filename, 'w')
      if not(file) then error(err) end

      for i=1, indexes:size(1) do
        local box = boxes[indexes[i]]
        file:write(classes[indexes[i]], ' ', box[1], ' ',  box[2], ' ', box[3], ' ', box[4], ' ', box[5], '\n')
      end
      file:close()
    end
  end
end
