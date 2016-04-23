require 'Utils'
local function nms(boxes, scores, overlap)
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

return nms
