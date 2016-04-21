#!/bin/bash -e
function joinStrings { local IFS="$1"; shift; echo "$*"; }
gpu=(1,2)
nGPU=${#gpu[*]}
export CUDA_VISIBLE_DEVICES=$(joinStrings , "${gpu[@]}")
set -x

# Imagenet train, TrainStudentModel, forward with nThreads 0
th /home/jshen/scripts/Train.lua /home/jshen/models/VGG/vggbn.lua /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/ImagenetProcessor.lua -val /file/imagenet/ILSVRC2012_img_val -valSize 2 -valEvery 1 -epochs 1 -epochSize 1 -nThreads 0 -nGPU 1
th /home/jshen/scripts/TrainStudentModel.lua /home/jshen/models/VGG/VGG_ILSVRC_16_layers.caffemodel /tmp/test.t7 /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/ImagenetProcessor.lua -epochs 1 -epochSize 1 -nThreads 0 -nGPU 1
th /home/jshen/scripts/Forward.lua /tmp/test.t7 "/file/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_0000000*.JPEG" /home/jshen/scripts/ImagenetProcessor.lua -nThreads 0 -nGPU 1

# Imagenet train, TrainStudentModel, forward with nThreads 1
th /home/jshen/scripts/Train.lua /home/jshen/models/VGG/vggbn.lua /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/ImagenetProcessor.lua -val /file/imagenet/ILSVRC2012_img_val -valSize 2 -valEvery 1 -epochs 1 -epochSize 1 -nThreads 1 -nGPU 1
th /home/jshen/scripts/TrainStudentModel.lua /home/jshen/models/VGG/VGG_ILSVRC_16_layers.caffemodel /tmp/test.t7 /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/ImagenetProcessor.lua -epochs 1 -epochSize 1 -nThreads 1 -nGPU 1
th /home/jshen/scripts/Forward.lua /tmp/test.t7 "/file/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_0000000*.JPEG" /home/jshen/scripts/ImagenetProcessor.lua -nThreads 1 -nGPU 1

# Imagenet train, TrainStudentModel, forward with nThreads 4
th /home/jshen/scripts/Train.lua /home/jshen/models/VGG/vggbn.lua /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/ImagenetProcessor.lua -val /file/imagenet/ILSVRC2012_img_val -valSize 2 -valEvery 1 -epochs 2 -epochSize 1 -nGPU 1 -nThreads 4
th /home/jshen/scripts/TrainStudentModel.lua /home/jshen/models/VGG/VGG_ILSVRC_16_layers.caffemodel /tmp/test.t7 /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/ImagenetProcessor.lua -epochs 1 -epochSize 1 -nGPU 1 -nThreads 4
th /home/jshen/scripts/Forward.lua /tmp/test.t7 "/file/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_0000000*.JPEG" /home/jshen/scripts/ImagenetProcessor.lua -nGPU 1 -nThreads 4

# Imagenet train, TrainStudentModel, forward with CPU
th /home/jshen/scripts/Train.lua /home/jshen/models/VGG/vggbn.lua /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/ImagenetProcessor.lua -val /file/imagenet/ILSVRC2012_img_val -valSize 1 -valEvery 1 -epochs 1 -epochSize 1 -batchSize 4 -nGPU -1
th /home/jshen/scripts/TrainStudentModel.lua /home/jshen/models/VGG/VGG_ILSVRC_16_layers.caffemodel /tmp/test.t7 /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/ImagenetProcessor.lua -epochs 1 -epochSize 1 -batchSize 4 -nGPU -1
th /home/jshen/scripts/Forward.lua /tmp/test.t7 "/file/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_0000000*.JPEG" /home/jshen/scripts/ImagenetProcessor.lua -nGPU -1

if (($nGPU >= 2)); then
  # Imagenet train, forward with nGPU 2 nThreads 2
  th /home/jshen/scripts/Train.lua /home/jshen/models/VGG/vggbn.lua /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/ImagenetProcessor.lua -val /file/imagenet/ILSVRC2012_img_val -valSize 2 -valEvery 1 -epochs 1 -epochSize 2 -nGPU 2 -nThreads 2
  th /home/jshen/scripts/Forward.lua /tmp/test.t7 "/file/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_0000000*.JPEG" /home/jshen/scripts/ImagenetProcessor.lua -batchSize 3 -nGPU 2 -nThreads 2
  
  # Imagenet train, forward with nGPU 2 nThreads 8
  th /home/jshen/scripts/Train.lua /home/jshen/models/VGG/vggbn.lua /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/ImagenetProcessor.lua -val /file/imagenet/ILSVRC2012_img_val -valSize 2 -valEvery 1 -epochs 1 -epochSize 2 -nGPU 2 -nThreads 8
  th /home/jshen/scripts/Forward.lua /tmp/test.t7 "/file/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_0000000*.JPEG" /home/jshen/scripts/ImagenetProcessor.lua -batchSize 3 -nGPU 2 -nThreads 8
fi

# Caltech train, forward with nThreads 0
th /home/jshen/scripts/Train.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test.t7 /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -val /file/caltech/val/base -valBatchSize 3 -valSize 1 -valEvery 1 -epochs 1 -epochSize 1 -nGPU 1 -nThreads 0
th /home/jshen/scripts/Forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227" -epochSize 1 -batchSize 3 -nGPU 1 -nThreads 0

# Caltech train, forward with nThreads 1
th /home/jshen/scripts/Train.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test.t7 /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -val /file/caltech/val/base -valBatchSize 3 -valSize 1 -valEvery 1 -epochs 1 -epochSize 1 -nGPU 1 -nThreads 1
th /home/jshen/scripts/Forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -epochSize 1 -batchSize 3 -nGPU 1 -nThreads 1

# Caltech train, forward with nThreads 4
th /home/jshen/scripts/Train.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test.t7 /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -val /file/caltech/val/base -valBatchSize 3 -valSize 1 -valEvery 1 -epochs 1 -epochSize 1 -nGPU 1 -nThreads 4
th /home/jshen/scripts/Forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -epochSize 1 -batchSize 3 -nGPU 1 -nThreads 4

# TrainSVM
th /home/jshen/scripts/TrainSVM.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test_svm.t7 /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227" -layer fc7 -epochSize 1 -nGPU 1 -nThreads 0
th /home/jshen/scripts/Forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-svm /tmp/test_svm.t7 -layer fc7 -imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -epochSize 1 -batchSize 3 -nGPU 1 -nThreads 1
th /home/jshen/scripts/Forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-svm /tmp/test_svm.t7 -layer fc7 -imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -epochSize 1 -batchSize 3 -nGPU 1 -nThreads 8

# Caltech hog luv train, TrainStudentModel, forward with nThreads 0
th /home/jshen/scripts/Train.lua /home/jshen/models/VGG/vggbn_10channels_2outputs.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" "" /home/jshen/scripts/CaltechHOGLUV.lua -processorOpts "-imageSize 227" -val /file/caltech/val/base -valSize 1 -valBatchSize 3 -valEvery 1 -epochs 1 -epochSize 1 -nGPU 1 -nThreads 0
th /home/jshen/scripts/TrainStudentModel.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel /home/jshen/models/VGG/vggbn_10channels_2outputs.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test.t7 /home/jshen/scripts/CaltechHOGLUV.lua -processorOpts "-imageSize 224 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -teacherProcessor /home/jshen/scripts/CaltechProcessor.lua -teacherProcessorOpts "-imageSize 227" -val /file/caltech/val/base -valSize 1 -valBatchSize 3 -valEvery 1 -epochs 1 -epochSize 1 -LR 0.01 -nGPU 1 -nThreads 0
th /home/jshen/scripts/Forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/CaltechHOGLUV.lua -processorOpts "-imageSize 227" -epochSize 1 -batchSize 3 -nGPU 1 -nThreads 0

# Caltech hog luv train, TrainStudentModel, forward with nThreads 1
th /home/jshen/scripts/Train.lua /home/jshen/models/VGG/vggbn_10channels_2outputs.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" "" /home/jshen/scripts/CaltechHOGLUV.lua -processorOpts "-imageSize 227" -val /file/caltech/val/base -valSize 1 -valBatchSize 3 -valEvery 1 -epochs 1 -epochSize 1 -nGPU 1 -nThreads 1
th /home/jshen/scripts/TrainStudentModel.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel /home/jshen/models/VGG/vggbn_10channels_2outputs.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test.t7 /home/jshen/scripts/CaltechHOGLUV.lua -processorOpts "-imageSize 224 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -teacherProcessor /home/jshen/scripts/CaltechProcessor.lua -teacherProcessorOpts "-imageSize 227" -val /file/caltech/val/base -valSize 1 -valBatchSize 3 -valEvery 1 -epochs 1 -epochSize 1 -LR 0.01 -nGPU 1 -nThreads 1
th /home/jshen/scripts/Forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/CaltechHOGLUV.lua -processorOpts "-imageSize 227" -epochSize 1 -batchSize 3 -nGPU 1 -nThreads 1

# Caltech hog luv train, TrainStudentModel, forward with nThreads 4
th /home/jshen/scripts/Train.lua /home/jshen/models/VGG/vggbn_10channels_2outputs.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" "" /home/jshen/scripts/CaltechHOGLUV.lua -processorOpts "-imageSize 227" -val /file/caltech/val/base -valSize 1 -valBatchSize 3 -valEvery 1 -epochs 1 -epochSize 1 -nGPU 1 -nThreads 4
th /home/jshen/scripts/TrainStudentModel.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel /home/jshen/models/VGG/vggbn_10channels_2outputs.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test.t7 /home/jshen/scripts/CaltechHOGLUV.lua -processorOpts "-imageSize 224 -windowSizeX 80 -windowSizeY 60 -windowScales 1" -teacherProcessor /home/jshen/scripts/CaltechProcessor.lua -teacherProcessorOpts "-imageSize 227" -val /file/caltech/val/base -valSize 1 -valBatchSize 3 -valEvery 1 -epochs 1 -epochSize 1 -LR 0.01 -nGPU 1 -nThreads 4
th /home/jshen/scripts/Forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/CaltechHOGLUV.lua -processorOpts "-imageSize 227 -windowSizeX 80 -windowSizeY 60 -windowScales 1" -epochSize 1 -batchSize 3 -nGPU 1 -nThreads 4

if (($nGPU >= 2)); then
  # Caltech train, TrainStudentModel, forward with nGPU 2 nThreads 2 
  th /home/jshen/scripts/Train.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test.t7 /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -val /file/caltech/val/base -valSize 2 -valBatchSize 3 -valEvery 1 -epochs 1 -epochSize 2 -nGPU 2 -nThreads 2
  th /home/jshen/scripts/TrainStudentModel.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel /home/jshen/models/SqueezeNet/squeezenet.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test.t7 /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -val /file/caltech/val/base -valSize 2 -valBatchSize 3 -valEvery 1 -epochs 1 -epochSize 2 -LR 0.01 -nGPU 2 -nThreads 2
  th /home/jshen/scripts/Forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -epochSize 2 -batchSize 3 -nGPU 2 -nThreads 2
  
  # Caltech train, TrainStudentModel, forward with nGPU 2 nThreads 8 
  th /home/jshen/scripts/Train.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test.t7 /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -val /file/caltech/val/base -valSize 2 -valBatchSize 3 -valEvery 1 -epochs 1 -epochSize 2 -nGPU 2 -nThreads 8
  th /home/jshen/scripts/TrainStudentModel.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel /home/jshen/models/SqueezeNet/squeezenet.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test.t7 /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -val /file/caltech/val/base -valSize 2 -valBatchSize 3 -valEvery 1 -epochs 1 -epochSize 2 -LR 0.01 -nGPU 2 -nThreads 8
  th /home/jshen/scripts/Forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -epochSize 2 -batchSize 3 -nGPU 2 -nThreads 8
  
  # Caltech train, forward replicateModel
  th /home/jshen/scripts/Train.lua /home/jshen/models/SqueezeNet/squeezenet.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test.t7 /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -val /file/caltech/val/base -valSize 2 -valBatchSize 3 -valEvery 1 -epochs 1 -epochSize 2 -nGPU 2 -nThreads 4 -replicateModel
  th /home/jshen/scripts/Forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/CaltechProcessor.lua -processorOpts "-imageSize 227 -windowSizeX 320 -windowSizeY 240 -windowScales 1" -epochSize 2 -batchSize 3 -nGPU 2 -nThreads 4 -replicateModel
fi
