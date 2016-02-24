local SoftCrossEntropyCriterion, parent = torch.class('SoftCrossEntropyCriterion', 'nn.Criterion')

-- Input: Logits. Calculates cross entropy loss L=-sum(sm(target, T) log sm(inputs, T)) 
-- where sm(z, T) = exp(z/T) / sum(exp(z/T))
function SoftCrossEntropyCriterion:__init(temperature)
  parent.__init(self)
  self.temperature = temperature
end

local function sm(input)
  if input:dim() == 1 then
    return torch.exp(input) / torch.exp(input):sum()
  elseif input:dim() == 2 then  
    return torch.cdiv(torch.exp(input), torch.exp(input):sum(2):expandAs(input))
  else
    error('matrix or vector expected')
  end
end

function SoftCrossEntropyCriterion:updateOutput(input, target)
  input = input:squeeze() / self.temperature
  target = (type(target) == 'number' and target or target:squeeze()) / self.temperature
    
  if input:dim() == 1 then
    self.output = -torch.dot(sm(target), input - math.log(torch.exp(input):sum()))
  elseif input:dim() == 2 then  
    local losses = -torch.cmul(sm(target), input - torch.exp(input):sum(2):log():expandAs(input)):sum(2)
    self.output = losses:sum() / losses:size(1)
  else
    error('matrix or vector expected')
  end
  return self.output
end

local function calcGrad(input, target, temperature)
  return 
end

function SoftCrossEntropyCriterion:updateGradInput(input, target)
  input = input:squeeze() / self.temperature
  target = type(target) == 'number' and target or target:squeeze() / self.temperature
  local y = sm(target)
  self.gradInput:resizeAs(input)
  if input:dim() == 1 then
    self.gradInput = (sm(input) * y:sum() - y) / self.temperature
  elseif input:dim() == 2 then
    self.gradInput = (torch.cmul(sm(input), y:sum(2):expandAs(input)) - y) / self.temperature / input:size(1)
  else
    error('matrix or vector expected')
  end
  return self.gradInput
end

return SoftCrossEntropyCriterion
