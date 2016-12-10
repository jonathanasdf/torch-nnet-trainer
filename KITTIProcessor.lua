local Transforms = require 'Transforms'
local Processor = require 'Processor'
local M = torch.class('KITTIProcessor', 'Processor')

function M:__init(model, processorOpts)
  self.cmd:option('-imageSize', 224, 'input patch size')
  self.cmd:option('-beta', 1, 'weight for quaternion loss')
  self.cmd:option('-saveOutput', '', 'directory to write output poses')
  Processor.__init(self, model, processorOpts)

  self.images = {}
  self.meanPixel = torch.Tensor{0.485, 0.456, 0.406}:view(3, 1, 1)
  self.std = torch.Tensor{0.229, 0.224, 0.225}:view(3, 1, 1)
  self.gt = torch.load('/file1/kitti_odom/dataset/poses/ground_truth.t7')
  for k,v in pairs(self.gt) do
    local t = v.new(v:size(1), 4, 4)
    t[{{}, {1, 3}, {}}] = v:view(-1, 3, 4)
    t[{{}, 4, {}}] = torch.repeatTensor(v.new{0, 0, 0, 1}:view(1, 1, 4), v:size(1), 1, 1)
    self.gt[k] = t:cuda()
    self.images[k] = {}
  end

  self.criterion = nn.ParallelCriterion()
  self.criterion:add(nn.MSECriterion(false), self.beta)
  self.criterion:add(nn.MSECriterion(false))
  self.criterion.sizeAverage = false
  self.criterion:cuda()

  if self.saveOutput ~= '' then
    if opts.phase == 'test' and opts.epochSize ~= -1 then
      error('saveOutput can only be used with epochSize == -1')
    elseif opts.phase == 'val' and opts.valSize ~= -1 then
      error('saveOutput can only be used with valSize == -1')
    end
    if string.sub(self.saveOutput, -1) ~= '/' then
      self.saveOutput = self.saveOutput .. '/'
    end
    self.results = {}
  end

  if opts.logdir and opts.epochs then
    self.rGraph = gnuplot.pngfigure(opts.logdir .. 'r.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('MSE')
    gnuplot.grid(true)
    self.rTrain = torch.Tensor(opts.epochs)
    if opts.val then
      self.rVal = torch.Tensor(opts.epochs)
    end

    self.tGraph = gnuplot.pngfigure(opts.logdir .. 't.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('MSE')
    gnuplot.grid(true)
    self.tTrain = torch.Tensor(opts.epochs)
    if opts.val then
      self.tVal = torch.Tensor(opts.epochs)
    end
  end
end

function M:loadImage(base, seq, id)
  if self.images[seq][id] == nil then
    local path = string.format('%06d', id-1) .. '.png'
    local img = torch.cat({
      image.load(base .. '/image_2/' .. path, 3),
      image.load(base .. '/image_3/' .. path, 3)}, 1)
    local sz = self.imageSize
    img = Transforms.Scale(sz, sz)[2](img)
    img = img:csub(torch.repeatTensor(self.meanPixel, 2, 1, 1):expandAs(img))
             :cdiv(torch.repeatTensor(self.std, 2, 1, 1):expandAs(img))
    self.images[seq][id] = img
  end
end

function M:loadInput(path, augmentations)
  local base = paths.dirname(paths.dirname(path))
  local seq = base:sub(-2)
  local id = tonumber(path:sub(-10, -5))

  self:loadImage(base, seq, id)
  self:loadImage(base, seq, id+1)

  local input = torch.cat({self.images[seq][id], self.images[seq][id+1]}, 1)
  return input:cuda(), {}
end

local function getQuaternion(r)
  local q = r.new(4) -- {w, x, y, z}
  local trace = r[1][1] + r[2][2] + r[3][3]
  if trace > 0 then
    S = math.sqrt(trace + 1) * 2 -- S=4*q[1]
    q[1] = 0.25 * S
    q[2] = (r[3][2] - r[2][3]) / S
    q[3] = (r[1][3] - r[3][1]) / S
    q[4] = (r[2][1] - r[1][2]) / S
  elseif (r[1][1] > r[2][2]) and (r[1][1] > r[3][3]) then
    S = math.sqrt(1 + r[1][1] - r[2][2] - r[3][3]) * 2 -- S=4*q[2]
    q[1] = (r[3][2] - r[2][3]) / S
    q[2] = 0.25 * S
    q[3] = (r[1][2] + r[2][1]) / S
    q[4] = (r[1][3] + r[3][1]) / S
  elseif (r[2][2] > r[3][3]) then
    S = math.sqrt(1 + r[2][2] - r[1][1] - r[3][3]) * 2 -- S=4*q[3]
    q[1] = (r[1][3] - r[3][1]) / S
    q[2] = (r[1][2] + r[2][1]) / S
    q[3] = 0.25 * S
    q[4] = (r[2][3] + r[3][2]) / S
  else
    S = math.sqrt(1.0 + r[3][3] - r[1][1] - r[2][2]) * 2 -- S=4*q[4]
    q[1] = (r[2][1] - r[1][2]) / S
    q[2] = (r[1][3] + r[3][1]) / S
    q[3] = (r[2][3] + r[3][2]) / S
    q[4] = 0.25 * S
  end
  return q
end

function M:getPose(q, t)
  local qx = q[2] + q[2]
  local qy = q[3] + q[3]
  local qz = q[4] + q[4]
  local qwx = qx * q[1]
  local qwy = qy * q[1]
  local qwz = qz * q[1]
  local qxx = qx * q[2]
  local qxy = qy * q[2]
  local qxz = qz * q[2]
  local qyy = qy * q[3]
  local qyz = qz * q[3]
  local qzz = qz * q[4]
  return q.new{
    {1 - (qyy + qzz), qxy - qwz, qxz + qwy, t[1]},
    {qxy + qwz, 1 - (qxx + qzz), qyz - qwx, t[2]},
    {qxz - qwy, qyz + qwx, 1 - (qxx + qyy), t[3]},
    {0, 0, 0, 1}}
end

local function quaternionMul(a, b)
  local r = a.new(4)
  r[1] = b[1]*a[1]-b[2]*a[2]-b[3]*a[3]-b[4]*a[4]
  r[2] = b[1]*a[2]+b[2]*a[1]-b[3]*a[4]+b[4]*a[3]
  r[3] = b[1]*a[3]+b[2]*a[4]+b[3]*a[1]-b[4]*a[2]
  r[4] = b[1]*a[4]-b[2]*a[3]+b[3]*a[2]+b[4]*a[1]
  return r
end

local function quaternionDiff(b, a)
  local ainv = a.new(4)
  ainv[1] = a[1]
  ainv[2] = -a[2]
  ainv[3] = -a[3]
  ainv[4] = -a[4]
  return quaternionMul(b, ainv)
end

function M:getTransform(seq, id)
  local a = self.gt[seq][id]
  local b = self.gt[seq][id+1]
  local diff = torch.mm(torch.inverse(a), b)
  return getQuaternion(diff), diff[{{1,3}, 4}]
  --return quaternionDiff(getQuaternion(b), getQuaternion(a)),
  --       b[{{}, 4}] - a[{{}, 4}]
end

function M:getLabel(path)
  local seq = paths.dirname(paths.dirname(path)):sub(-2)
  local id = tonumber(path:sub(-10, -5))
  local r, t = self:getTransform(seq, id)
  return {r, t}
end

function M:resetStats()
  self.count = 0
  self.rloss = 0
  self.tloss = 0
  if self.saveOutput ~= '' then
    os.execute('rm -rf ' .. self.saveOutput)
    os.execute('mkdir -p ' .. self.saveOutput)
  end
end

function M:updateStats(pathNames, outputs, labels)
  self.count = self.count + #pathNames
  self.rloss = self.rloss + self.criterion.criterions[1].output
  self.tloss = self.tloss + self.criterion.criterions[2].output
  if self.rloss ~= self.rloss or self.tloss ~= self.tloss then
    print("ERROR: NaN in loss!")
  end

  if self.saveOutput ~= '' then
    for i=1,#pathNames do
      local path = pathNames[i]
      local seq = paths.dirname(paths.dirname(path)):sub(-2)
      local id = tonumber(path:sub(-10, -5))
      if self.results[seq] == nil then
        self.results[seq] = {}
        --self.results[seq][1] = {outputs[1].new{1, 0, 0, 0}, outputs[2].new{0, 0, 0}}
        self.results[seq][1] = outputs[1].new{{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}
      end
      --self.results[seq][id+1] = {outputs[1][i] / outputs[1][i]:norm(), outputs[2][i]}
      self.results[seq][id+1] = self:getPose(outputs[1][i], outputs[2][i])
    end
  end
end

function M:getStats()
  if self.saveOutput ~= '' then
    for k,v in pairs(self.results) do
      local f = io.open(self.saveOutput .. k .. '.txt', 'w')
      for i=1,#v do
        if i > 1 then
          --v[i][1] = quaternionMul(v[i-1][1], v[i][1])
          --v[i][2] = v[i-1][2] + v[i][2]
          v[i] = torch.mm(v[i-1], v[i])
        end
        --local pose = self:getPose(v[i][1], v[i][2])
        local pose = torch.view(v[i], -1)
        f:write(pose[1])
        for j=2,12 do
          f:write(' ' .. pose[j])
        end
        f:write('\n')
      end
      f:close()
    end
  end

  if self.rGraph then
    if opts.phase == 'train' then
      self.rTrain[opts.epoch] = self.rloss / self.count
      self.tTrain[opts.epoch] = self.tloss / self.count

      gnuplot.figure(self.rGraph)
      local x = torch.range(1, opts.epoch):long()
      if self.rVal and opts.epoch-1 >= opts.valEvery then
        local xval = torch.range(opts.valEvery, opts.epoch-1, opts.valEvery):long()
        gnuplot.plot({'train', x, self.rTrain:index(1, x), '+-'},
                     {'val', xval, self.rVal:index(1, xval), '+-'})
      else
        gnuplot.plot({'train', x, self.rTrain:index(1, x), '+-'})
      end
      gnuplot.plotflush()
    elseif opts.phase == 'val' and opts.epoch >= opts.valEvery then
      self.rVal[opts.epoch] = self.rloss / self.count
      self.tVal[opts.epoch] = self.tloss / self.count

      gnuplot.figure(self.rGraph)
      local x = torch.range(1, opts.epoch):long()
      local xval = torch.range(opts.valEvery, opts.epoch, opts.valEvery):long()
      gnuplot.plot({'train', x, self.rTrain:index(1, x), '+-'},
                   {'val', xval, self.rVal:index(1, xval), '+-'})
      gnuplot.plotflush()
    end
  end

  return string.format('Average rotation loss: %f translation loss: %f', self.rloss / self.count, self.tloss / self.count)
end

return M
