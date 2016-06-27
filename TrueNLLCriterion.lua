-- Copyright 2004-present Facebook. All Rights Reserved.

--[[
`TrueNLLCriterion` computes the negative log-loss criterion directly.
]]
local TrueNLLCriterion, parent = torch.class('nn.TrueNLLCriterion',
                                             'nn.Criterion')

-- For numerical stability
local eps = 0.00000001

function TrueNLLCriterion:__init(weights, sizeAverage)
   parent.__init(self)
   if weights then
     assert(weights:dim() == 1, "weights input should be 1-D Tensor")
     self.weights = weights
   end
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end
end

function TrueNLLCriterion:updateOutput(input, target)
   if input:dim() == 1 then
      local w = 1
      if self.weights then w = self.weights[target] end
      self.output = -math.log(input[target]*w + eps)
   elseif input:dim() == 2 then
      local output = 0
      for i=1,target:size(1) do
         local w = 1
         if self.weights then w = self.weights[target[i]] end
         output = output - math.log(input[i][target[i]]*w + eps)
      end
      if self.sizeAverage then
         output = output / target:size(1)
      end
      self.output = output
   else
      error('matrix or vector expected')
   end
   return self.output
end

function TrueNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()

   if input:dim() == 1 then
      local w = 1
      if self.weights then w = self.weights[target] end
      self.gradInput[target] = -1 / (input[target]*w + eps)
   else
      local z = -1
      if self.sizeAverage then
         z = z / target:size(1)
      end
      local gradInput = self.gradInput
      for i=1,target:size(1) do
         local w = 1
         if self.weights then w = self.weights[target[i]] end
         gradInput[i][target[i]] = z / (input[i][target[i]]*w + eps)
      end
   end

   return self.gradInput
end
