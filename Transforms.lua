--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

require 'image'

local M = {}

function M.Compose(transforms)
   return function(input)
      local augmentations = {}
      for _, transform in ipairs(transforms) do
         input, augmentation = transform(input)
         for k,v in pairs(augmentation) do
           assert(augmentations[k] == nil, 'duplicate augmentation')
           augmentations[k] = v
         end
      end
      return input, augmentations
   end
end

function M.ColorNormalize(meanstd)
   return function(img)
      img = img:clone()
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
         img[i]:div(meanstd.std[i])
      end
      return img, {cnorm = meanstd}
   end
end

function M.Scale(targetW, targetH, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(input)
      local w, h = input:size(3), input:size(2)
      local result
      if w == targetW and h == targetH then
         result = input
      else
         result = image.scale(input, targetW, targetH, interpolation)
      end
      return result, {scale = {targetW, targetH, interpolation}}
   end
end

-- Scales the smaller edge to size
function M.ScaleKeepAspect(size, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(input)
      local w, h = input:size(3), input:size(2)
      local targetW, targetH, result
      if (w <= h and w == size) or (h <= w and h == size) then
         targetW = w
         targetH = h
         result = input
      else
         if w < h then
            targetW = size
            targetH = h/w * size
         else
            targetW = w/h * size
            targetH = size
         end
         result = image.scale(input, targetW, targetH, interpolation)
      end
      return result, {scale = {targetW, targetH, interpolation}}
   end
end

function M.Crop(corners, padding)
   padding = padding or 0

   return function(input)
      if padding > 0 then
         local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
         temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
         input = temp
      end

      return image.crop(input, corners[1], corners[2], corners[3], corners[4]), {crop = {corners, padding}}
   end
end

-- Crop to centered rectangle
function M.CenterCrop(size)
   return function(input)
      local w1 = math.ceil((input:size(3) - size)/2)
      local h1 = math.ceil((input:size(2) - size)/2)
      return image.crop(input, w1, h1, w1 + size, h1 + size),  -- center patch
             {crop = {{w1, h1, w1 + size, h1 + size}, 0}}
   end
end

-- Random crop from larger image with optional zero padding
function M.RandomCrop(size, padding)
   padding = padding or 0

   return function(input)
      if padding > 0 then
         local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
         temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
         input = temp
      end

      local w, h = input:size(3), input:size(2)
      if w == size and h == size then
         return input, {crop = {{0, 0, w, h}, padding}}
      end

      local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
      local out = image.crop(input, x1, y1, x1 + size, y1 + size)
      assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
      return out, {crop = {{x1, y1, x1 + size, y1 + size}, padding}}
   end
end

-- Four corner patches and center crop from image and its horizontal reflection
function M.TenCrop(size)
   local centerCrop = M.CenterCrop(size)

   return function(input)
      local w, h = input:size(3), input:size(2)

      local output = {}
      for _, img in ipairs{input, image.hflip(input)} do
         table.insert(output, centerCrop(img))
         table.insert(output, image.crop(img, 0, 0, size, size))
         table.insert(output, image.crop(img, w-size, 0, w, size))
         table.insert(output, image.crop(img, 0, h-size, size, h))
         table.insert(output, image.crop(img, w-size, h-size, w, h))
      end

      -- View as mini-batch
      for i, img in ipairs(output) do
         output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
      end

      return input.cat(output, 1), {tencrop = size}
   end
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function M.RandomScale(minSize, maxSize)
   return function(input)
      local w, h = input:size(3), input:size(2)

      local targetSz = torch.random(minSize, maxSize)
      local targetW, targetH = targetSz, targetSz
      if w < h then
         targetH = torch.round(h / w * targetW)
      else
         targetW = torch.round(w / h * targetH)
      end

      return image.scale(input, targetW, targetH, 'bicubic'),
             {scale = {targetW, targetH, 'bicubic'}}
   end
end

function M.HorizontalFlip(prob)
   return function(input)
      local flipped = 0
      if torch.uniform() < prob then
         input = image.hflip(input)
         flipped = 1
      end
      return input, {hflip = flipped}
   end
end

function M.Rotation(deg)
   return function(input)
      if deg ~= 0 then
         input = image.rotate(input, (torch.uniform() - 0.5) * deg * math.pi / 180, 'bilinear')
      end
      return input, {rot = deg}
   end
end

-- Lighting noise (AlexNet-style PCA-based noise)
function M.Lighting(alphastd, eigval, eigvec)
   return function(input)
      if alphastd == 0 then
         return input, {lighting = torch.zeros(3)}
      end

      local alpha = torch.Tensor(3):normal(0, alphastd)
      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()

      input = input:clone()
      for i=1,3 do
         input[i]:add(rgb[i])
      end
      return input, {lighting = rgb}
   end
end

local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function M.Saturation(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input, {saturation = alpha}
   end
end

function M.Brightness(var)
   local gs

   return function(input)
      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input, {brightness = alpha}
   end
end

function M.Contrast(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)
      gs:fill(gs[1]:mean())

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input, {contrast = alpha}
   end
end

function M.ColorJitter(opt)
   local brightness = opt.brightness or 0
   local contrast = opt.contrast or 0
   local saturation = opt.saturation or 0

   local ts = {}
   if brightness ~= 0 then
      table.insert(ts, M.Brightness(brightness))
   end
   if contrast ~= 0 then
      table.insert(ts, M.Contrast(contrast))
   end
   if saturation ~= 0 then
      table.insert(ts, M.Saturation(saturation))
   end

   if #ts == 0 then
      return function(input) return input, {} end
   end

   return Compose(ts)
end

return M
