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
    t[{{}, {1,3}, {}}] = torch.view(v, -1, 3, 4)
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

function M:preprocess(path, augmentations)
  local base = paths.dirname(paths.dirname(path))
  local seq = base:sub(-2)
  local id = tonumber(path:sub(-10, -5))

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

  if self.images[seq][id+1] == nil then
    local path = string.format('%06d', id) .. '.png'
    local img = torch.cat({
      image.load(base .. '/image_2/' .. path, 3),
      image.load(base .. '/image_3/' .. path, 3)}, 1)
    local sz = self.imageSize
    img = Transforms.Scale(sz, sz)[2](img)
    img = img:csub(torch.repeatTensor(self.meanPixel, 2, 1, 1):expandAs(img))
             :cdiv(torch.repeatTensor(self.std, 2, 1, 1):expandAs(img))
    self.images[seq][id+1] = img
  end

  local input = torch.cat({self.images[seq][id], self.images[seq][id+1]}, 1)
  return input:cuda(), {}
end

local function getQuaternion(pose)
  local r = pose:view(4, 4)
  local q = torch.Tensor(4) -- {w, x, y, z}
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

function M:getLabels(pathNames, outputs)
  local r = outputs[1].new(#pathNames, 4)
  local t = outputs[1].new(#pathNames, 3)
  for i=1,#pathNames do
    local path = pathNames[i]
    local seq = paths.dirname(paths.dirname(path)):sub(-2)
    local id = tonumber(path:sub(-10, -5))
    local cur = self.gt[seq][id+1]
    local prev = self.gt[seq][id]
    local diff = torch.mm(cur, torch.inverse(prev))
    r[i] = getQuaternion(diff)
    t[i] = diff[{{1,3}, 4}]
  end
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

local function getPose(q, t)
  q = q / q:norm()
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

function M:updateStats(pathNames, outputs, labels)
  self.count = self.count + #pathNames
  self.rloss = self.rloss + self.criterion.criterions[1].output * self.criterion.weights[1]
  self.tloss = self.tloss + self.criterion.criterions[2].output * self.criterion.weights[2]
  if self.rloss ~= self.rloss or self.tloss ~= self.tloss then
    print("ERROR: NaN in loss!")
  end

  if self.saveOutput ~= '' then
    for i=1,#pathNames do
      local path = pathNames[i]
      local seq = paths.dirname(paths.dirname(path)):sub(-2)
      local id = tonumber(path:sub(-10, -5))
      if self.results[seq] == nil then
        self.results[seq] = outputs[1].new(self.gt[seq]:size(1), 4, 4)
        self.results[seq][1] = outputs[1].new{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}
      end
      self.results[seq][id+1] = getPose(outputs[1][i], outputs[2][i])
    end
  end
end

function M:getStats()
  if self.saveOutput ~= '' then
    for k,v in pairs(self.results) do
      local f = io.open(self.saveOutput .. k .. '.txt', 'w')
      for i=1,v:size(1) do
        if i > 1 then
          v[i] = torch.mm(v[i], v[i-1])
        end
        f:write(v[i][1][1])
        for j=1,3 do
          for k=1,4 do
            if j ~= 1 or k ~= 1 then
              f:write(' ' .. v[i][j][k])
            end
          end
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
