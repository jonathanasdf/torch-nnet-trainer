local SoftCrossEntropyCriterion, parent = torch.class('SoftCrossEntropyCriterion', 'nn.Criterion')

-- Input: Logits. Calculates cross entropy loss L=-sum(sm(input, T) log sm(target, T)) 
-- where sm(z, T) = exp(z/T) / sum(exp(z/T))
function SoftCrossEntropyCriterion:__init(temperature)
   parent.__init(self)
   self.temperature = temperature
end

function SoftCrossEntropyCriterion:updateOutput(input, target)
  input = input:squeeze() / self.temperature
  target = (type(target) == 'number' and target or target:squeeze()) / self.temperature
    
  if input:dim() == 1 then
    self.output = -torch.dot(torch.exp(input) / torch.exp(input):sum(), target - math.log(torch.exp(target):sum()))
  elseif input:dim() == 2 then  
    local d = torch.exp(input):sum(2)
    local losses = torch.cmul(torch.cdiv(torch.exp(input), d:expandAs(input)), torch.exp(target):sum(2):log():expandAs(target) - target):sum(2)
    self.output = losses:sum() / losses:size(1)
  else
    error('matrix or vector expected')
  end
  return self.output
end

local function calcGrad(input, target, temperature)
  local h = torch.exp(input):resize(input:size(1), 1) / torch.exp(input):sum()
  local c = target - math.log(torch.exp(target):sum())
  return torch.cmul(c, ((h*h:t()):sum(2)-h)):resize(input:size())
end

function SoftCrossEntropyCriterion:updateGradInput(input, target)
  input = input:squeeze()
  target = type(target) == 'number' and target or target:squeeze()
  
  self.gradInput:resizeAs(input)
  if input:dim() == 1 then
    self.gradInput = calcGrad(input, target, self.temperature)
  elseif input:dim() == 2 then
    for i=1,input:size(1) do
      self.gradInput[i] = calcGrad(input[i], target[i], self.temperature)
    end
  else
    error('matrix or vector expected')
  end
  return self.gradInput
end

return SoftCrossEntropyCriterion
