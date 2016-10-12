# unnamed torch nnet training framework

Not very polished, for personal use. Lots of hardcoded paths and small changes to dependency packages.

## Dependencies

cudnn, dpnn, image, optim

## Example

th Train.lua "models/cifar10_simple.lua -layers 32,32,32,32,32,64,64,128,128,256,256 -pool 3,8 CifarProcessor.lua -flip 0.5 -minCropPercent 0.8" /file/cifar10/trainval.txt cifar10_simple.t7 -val /file1/cifar10/test.txt -valSize -1 -valEvery 1 -batchSize 128 -epochSize -1 -epochs 210 -learningRate 0.2 -LRDropEvery 20 -LRDropFactor 2
