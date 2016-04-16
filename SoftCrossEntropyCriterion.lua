local SoftCrossEntropyCriterion, parent = torch.class('SoftCrossEntropyCriterion', 'nn.Criterion')

-- Input: Logits. Calculates cross entropy loss L=-sum(sm(target, T) log sm(inputs, T))
-- where sm(z, T) = exp(z/T) / sum(exp(z/T))
function SoftCrossEntropyCriterion:__init(temperature)
  parent.__init(self)
  self.temperature = temperature
  self.sizeAverage = true
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
  input = input / self.temperature
  target = target / self.temperature

  if input:dim() == 1 then
    self.output = -torch.dot(sm(target), input - math.log(torch.exp(input):sum()))
  elseif input:dim() == 2 then
    self.output = -(torch.cmul(sm(target), input - torch.exp(input):sum(2):log():expandAs(input)):sum(2)):sum()
    if self.sizeAverage then
      self.output = self.output / input:size(1)
    end
  else
    error('matrix or vector expected. input size: ' .. tostring(input:size()))
  end
  return self.output
end

local function calcGrad(input, target, temperature)
  return
end

function SoftCrossEntropyCriterion:updateGradInput(input, target)
  self.gradInput:resizeAs(input)

  input = input / self.temperature
  target = target / self.temperature

  local y = sm(target)
  if input:dim() == 1 then
    self.gradInput:viewAs((sm(input) * y:sum() - y) / self.temperature, self.gradInput)
  elseif input:dim() == 2 then
    self.gradInput:viewAs((torch.cmul(sm(input), y:sum(2):expandAs(input)) - y) / self.temperature, self.gradInput)
    if self.sizeAverage then
      self.gradInput = self.gradInput / input:size(1)
    end
  else
    error('matrix or vector expected. input size: ' .. tostring(input:size()))
  end
  return self.gradInput * self.temperature * self.temperature
end

return SoftCrossEntropyCriterion
