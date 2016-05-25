local MSECovCriterion, parent = torch.class('MSECovCriterion', 'nn.Criterion')

-- Input: x, mu. Calculates (x-mu)^T*cov^-1*(x-mu)
function MSECovCriterion:__init(sizeAverage)
  parent.__init(self)
  if sizeAverage ~= nil then
    self.sizeAverage = sizeAverage
  else
    self.sizeAverage = true
  end
end

function MSECovCriterion:updateOutput(input, target)
  if input:dim() > 2 then
    error('matrix or vector expected. input size: ' .. tostring(input:size()))
  end
  if input:dim() == 1 then
    input = input:view(1, -1)
  end

  self.output = 0
  for i=1,input:size(1) do
    local diff = (input[i] - target[i]):view(-1, 1)
    self.output = self.output + (diff:t() * self.invcov[i] * diff):sum()
  end
  if self.sizeAverage then
    self.output = self.output / input:size(1)
  end
  return self.output
end

function MSECovCriterion:updateGradInput(input, target)
  if input:dim() > 2 then
    error('matrix or vector expected. input size: ' .. tostring(input:size()))
  end
  if input:dim() == 1 then
    input = input:view(1, -1)
  end

  self.gradInput:resizeAs(input)
  for i=1,input:size(1) do
    local diff = (input[i] - target[i]):view(-1, 1)
    self.gradInput[i] = self.invcov[i] * diff * 2
  end
  if self.sizeAverage then
    self.gradInput = self.gradInput / input:size(1)
  end
  return self.gradInput
end

return MSECovCriterion
