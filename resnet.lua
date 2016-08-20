require 'dpnn'

local function shortcut(nInputPlane, nOutputPlane, stride)
  assert(nOutputPlane >= nInputPlane)
  if stride == 1 and nInputPlane == nOutputPlane then
    return nn.Identity()
  else
    -- Strided, zero-padded identity shortcut
    local m = nn.Sequential()
    m:add(nn.SpatialAveragePooling(1, 1, stride, stride))
    m:add(nn.Padding(2, nOutputPlane - nInputPlane))
    return m
  end
end

-- Typically shareGradInput uses the same gradInput storage for all modules
-- of the same type. This is incorrect for some SpatialBatchNormalization
-- modules in this network b/c of the in-place CAddTable. This marks the
-- module so that it's shared only with other modules with the same key
local function ShareGradInput(module, key)
   assert(key)
   module.__shareGradInputKey = key
   return module
end

local BasicResidualModule, Parent = torch.class('nn.BasicResidualModule', 'nn.Decorator')
function BasicResidualModule:__init(nInputPlane, n, stride)
  self.nInputPlane = nInputPlane
  self.n = n
  self.stride = stride

  self.module = nn.Sequential()
  self.module:add(ShareGradInput(nn.SpatialBatchNormalization(nInputPlane), 'preact'))
  self.module:add(nn.ReLU(true))

  self.block = nn.Sequential()
  self.block:add(nn.SpatialConvolution(nInputPlane, n, 3, 3, stride, stride, 1, 1))
  self.block:add(nn.SpatialBatchNormalization(n))
  self.block:add(nn.ReLU(true))
  self.block:add(nn.SpatialConvolution(n, n, 3, 3, 1, 1, 1, 1))

  self.shortcut = shortcut(nInputPlane, n, stride)

  self.module:add(nn.ConcatTable():add(self.block):add(self.shortcut))
  self.module:add(nn.CAddTable(true))

  Parent.__init(self, self.module)
end

--function BasicResidualModule:__tostring__()
--  return string.format('%s(%d, %d, %d)', torch.type(self), self.nInputPlane, self.n, self.stride)
--end


local BottleneckResidualModule, Parent = torch.class('nn.BottleneckResidualModule', 'nn.Decorator')
function BottleneckResidualModule:__init(nInputPlane, nSqueeze, nExpand, stride)
  self.nInputPlane = nInputPlane
  self.nSqueeze = nSqueeze
  self.nExpand = nExpand
  self.stride = stride

  self.module = nn.Sequential()
  self.module:add(ShareGradInput(nn.SpatialBatchNormalization(nInputPlane), 'preact'))
  self.module:add(nn.ReLU(true))

  self.block = nn.Sequential()
  self.block:add(nn.SpatialConvolution(nInputPlane, nSqueeze, 1, 1, 1, 1))
  self.block:add(nn.SpatialBatchNormalization(nSqueeze))
  self.block:add(nn.ReLU(true))
  self.block:add(nn.SpatialConvolution(nSqueeze, nSqueeze, 3, 3, stride, stride, 1, 1))
  self.block:add(nn.SpatialBatchNormalization(nSqueeze))
  self.block:add(nn.ReLU(true))
  self.block:add(nn.SpatialConvolution(nSqueeze, nExpand, 1, 1, 1, 1))

  self.shortcut = shortcut(nInputPlane, nExpand, stride)

  self.module:add(nn.ConcatTable():add(self.block):add(self.shortcut))
  self.module:add(nn.CAddTable(true))

  Parent.__init(self, self.module)
end

function BottleneckResidualModule:__tostring__()
  return string.format('%s(%d, %d, %d, %d)', torch.type(self), self.nInputPlane, self.nSqueeze, self.nExpand, self.stride)
end


local FireResidualModule, Parent = torch.class('nn.FireResidualModule', 'nn.Decorator')
function FireResidualModule:__init(nInputPlane, s1x1, e1x1, e3x3)
  self.nInputPlane = nInputPlane
  self.s1x1 = s1x1
  self.e1x1 = e1x1
  self.e3x3 = e3x3

  local fireModule = nn.FireModule(nInputPlane, s1x1, e1x1, e3x3, 'ReLU', true)
  local m = fireModule.modules[1]
  assert(torch.type(m.modules[#m.modules]) == 'nn.ReLU')
  m:remove()
  assert(torch.type(m.modules[#m.modules]) == 'nn.SpatialBatchNormalization')
  m:remove()

  self.module = nn.Sequential()
  self.module:add(ShareGradInput(nn.SpatialBatchNormalization(nInputPlane), 'preact'))
  self.module:add(nn.ReLU(true))

  self.block = nn.Sequential()
  self.block:add(fireModule)

  self.shortcut = shortcut(nInputPlane, e1x1+e3x3, 1)

  self.module = nn.Sequential()
  self.module:add(nn.ConcatTable():add(self.block):add(self.shortcut))
  self.module:add(nn.CAddTable(true))

  Parent.__init(self, self.module)
end

function FireResidualModule:__tostring__()
  return string.format('%s(%d, %d, %d, %d)', torch.type(self), self.nInputPlane, self.s1x1, self.e1x1, self.e3x3)
end
