--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--  Transforms are given as an ordered list of {'name', fn}
--

require 'image'

local M = {}

function M.Apply(transforms, img)
  for i=1,#transforms do
    img = transforms[i][2](img)
  end
  return img
end

function M.ColorNormalize(meanstd)
   return {'cnorm', function(img)
      img = img:clone()
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
         img[i]:div(meanstd.std[i])
      end
      return img
   end}
end

function M.Scale(targetW, targetH, interpolation)
   interpolation = interpolation or 'bicubic'
   return {'scale', function(input)
      local w, h = input:size(3), input:size(2)
      local result
      if w == targetW and h == targetH then
         result = input
      else
         result = image.scale(input, targetW, targetH, interpolation)
      end
      return result
   end}
end

-- Scales the smaller edge to size
function M.ScaleKeepAspect(size, interpolation)
   interpolation = interpolation or 'bicubic'
   return {'scale', function(input)
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
      return result
   end}
end

function M.Crop(corners, pad, mode)
   pad = pad or 0
   mode = mode or 'zero'
   return {'crop', function(input)
     if pad > 0 then
       local module
       if mode == 'reflection' then
         module = nn.SpatialReflectionPadding(pad, pad, pad, pad):float()
       elseif mode == 'zero' then
         module = nn.SpatialZeroPadding(pad, pad, pad, pad):float()
       else
         error('unknown mode ' .. mode)
       end
       input = module:forward(input)
     end
     return image.crop(input, corners[1], corners[2], corners[3], corners[4])
   end}
end

-- Crop to centered rectangle
function M.CenterCrop(size)
   return {'crop', function(input)
      local w1 = math.ceil((input:size(3) - size)/2)
      local h1 = math.ceil((input:size(2) - size)/2)
      return image.crop(input, w1, h1, w1 + size, h1 + size)  -- center patch
   end}
end

-- Random crop from larger image with optional padding
function M.RandomCrop(size, pad, mode)
   pad = pad or 0
   local a, b = math.random(), math.random()
   return {'crop', function(input)
     local w, h = input:size(3) + 2*pad, input:size(2) + 2*pad
     local x1, y1 = math.floor(a * (w - size + 1)), math.floor(b * (h - size + 1))
     return M.Crop({x1, y1, x1 + size, y1 + size}, pad, mode)[2](input)
   end}
end

-- Random crop a percentage of the entire image
function M.RandomCropPercent(minSizePercent)
   local p = math.random() * (1 - minSizePercent) + minSizePercent
   local a, b = math.random(), math.random()
   return {'crop', function(input)
      local w, h = input:size(3), input:size(2)
      local pw, ph = math.ceil(p * w), math.ceil(p * h)
      local x1, y1 = math.floor(a * (w - pw)), math.floor(b * (h - ph))
      local out = image.crop(input, x1, y1, x1 + pw, y1 + ph)
      assert(out:size(2) == ph and out:size(3) == pw, 'wrong crop size')
      return out
   end}
end

-- Four corner patches and center crop from image and its horizontal reflection
function M.TenCrop(size)
   local centerCrop = M.CenterCrop(size)[2]
   return {'tencrop', function(input)
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

      return input.cat(output, 1)
   end}
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function M.RandomScale(minSize, maxSize)
   local targetSz = torch.random(minSize, maxSize)
   return {'scale', function(input)
      local w, h = input:size(3), input:size(2)
      local targetW, targetH = targetSz, targetSz
      if w < h then
         targetH = torch.round(h / w * targetW)
      else
         targetW = torch.round(w / h * targetH)
      end
      return image.scale(input, targetW, targetH, 'bicubic')
   end}
end

function M.HorizontalFlip(prob)
   local r = torch.uniform()
   return {'hflip', function(input)
      if r < prob then
         input = image.hflip(input)
      end
      return input
   end}
end

function M.Rotation(deg)
   return {'rot', function(input)
      if deg ~= 0 then
         input = image.rotate(input, (torch.uniform() - 0.5) * deg * math.pi / 180, 'bilinear')
      end
      return input
   end}
end

-- Lighting noise (AlexNet-style PCA-based noise)
function M.Lighting(alphastd, eigval, eigvec)
   local alpha = torch.Tensor(3):normal(0, alphastd)
   return {'lighting', function(input)
      if alphastd == 0 then
         return input, {lighting = torch.zeros(3)}
      end

      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()

      input = input:clone()
      for i=1,3 do
         input[i]:add(rgb[i])
      end
      return input
   end}
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
   local alpha = 1.0 + torch.uniform(-var, var)
   return {'saturation', function(input)
      local gs = gs or input.new()
      grayscale(gs, input)
      blend(input, gs, alpha)
      return input
   end}
end

function M.Brightness(var)
   local alpha = 1.0 + torch.uniform(-var, var)
   return {'brightness', function(input)
      local gs = gs or input.new()
      gs:resizeAs(input):zero()
      blend(input, gs, alpha)
      return input
   end}
end

function M.Contrast(var)
   local alpha = 1.0 + torch.uniform(-var, var)
   return {'contrast', function(input)
      local gs = gs or input.new()
      grayscale(gs, input)
      gs:fill(gs[1]:mean())
      blend(input, gs, alpha)
      return input
   end}
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
   return ts
end

return M
