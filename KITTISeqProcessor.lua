require 'rnn'
local Transforms = require 'Transforms'
local KITTIProcessor = require 'KITTIProcessor'
local M = torch.class('KITTISeqProcessor', 'KITTIProcessor')

function M:__init(model, processorOpts)
  self.cmd:option('-seqlen', 5, 'length of sequences to process. Set -1 to use entire sequence')
  KITTIProcessor.__init(self, model, processorOpts)

  self.criterion = nn.SequencerCriterion(self.criterion, self.criterion.sizeAverage):cuda()

  self.lengths = {}
  self.lengths['00'] = 4541
  self.lengths['01'] = 1101
  self.lengths['02'] = 4661
  self.lengths['03'] = 801
  self.lengths['04'] = 271
  self.lengths['05'] = 2761
  self.lengths['06'] = 1101
  self.lengths['07'] = 1101
  self.lengths['08'] = 4071
  self.lengths['09'] = 1591
  self.lengths['10'] = 1201

  if self.saveOutput ~= '' and self.seqlen ~= -1 then
    error('saveOutput can only be used with seqlen == -1')
  end
end

function M:preprocess(path, augmentations)
  local base = paths.dirname(paths.dirname(path))
  local seq = base:sub(-2)
  local id = tonumber(path:sub(-10, -5))

  local a = self.seqlen == -1 and id or torch.random(1, self.lengths[seq]-self.seqlen+1)
  local b = self.seqlen == -1 and id or start + self.seqlen - 1
  local inputs = {}
  local labels = {}
  for i=a,b do
    self:loadImage(base, seq, i)
    inputs[#inputs+1] = self.images[seq][i]
    label[#labels+1] = self:getLabel(path .. '/image_2/' .. string.format('%06d', i-1) .. '.png')
  end
  return batchConcat(inputs):cuda(), batchConcat(labels):cuda(), {}
end

function M:loadAndPreprocessInputs(pathNames, augmentations)
  local inputs, labels, augs = KITTIProcessor.loadAndPreprocessInputs(self, pathNames, augmentations)
  -- batch x seq x data to seq x batch x data
  return inputs:transpose(1, 2), labels, augs
end

return M
