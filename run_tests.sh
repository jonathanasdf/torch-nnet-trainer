#!/bin/sh -ex
export CUDA_VISIBLE_DEVICES=2,1
# Imagenet train, train_student_model, forward with nThreads 1
th /home/jshen/scripts/train.lua /home/jshen/models/VGG/vggbn.lua /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/imagenet_processor.lua -val /file/imagenet/ILSVRC2012_img_val -valSize 2 -val_every 1 -epochs 1 -epochSize 1 -nThreads 1 -nGPU 1
th /home/jshen/scripts/train_student_model.lua /home/jshen/models/VGG/VGG_ILSVRC_16_layers.caffemodel /tmp/test.t7 /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/imagenet_processor.lua -epochs 1 -epochSize 1 -nThreads 1 -nGPU 1
th /home/jshen/scripts/forward.lua /tmp/test.t7 "/file/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_0000000*.JPEG" /home/jshen/scripts/imagenet_processor.lua -nThreads 1 -nGPU 1

## Imagenet train, train_student_model, forward with nThreads 4
#th /home/jshen/scripts/train.lua /home/jshen/models/VGG/vggbn.lua /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/imagenet_processor.lua -val /file/imagenet/ILSVRC2012_img_val -valSize 2 -val_every 1 -epochs 2 -epochSize 1 -nGPU 1 -nThreads 4
#th /home/jshen/scripts/train_student_model.lua /home/jshen/models/VGG/VGG_ILSVRC_16_layers.caffemodel /tmp/test.t7 /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/imagenet_processor.lua -epochs 1 -epochSize 1 -nGPU 1 -nThreads 4
#th /home/jshen/scripts/forward.lua /tmp/test.t7 "/file/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_0000000*.JPEG" /home/jshen/scripts/imagenet_processor.lua -nGPU 1 -nThreads 4

# Imagenet train, train_student_model, forward with CPU
th /home/jshen/scripts/train.lua /home/jshen/models/VGG/vggbn.lua /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/imagenet_processor.lua -val /file/imagenet/ILSVRC2012_img_val -valSize 1 -val_every 1 -epochs 1 -epochSize 1 -batchSize 4 -nGPU -1
th /home/jshen/scripts/train_student_model.lua /home/jshen/models/VGG/VGG_ILSVRC_16_layers.caffemodel /tmp/test.t7 /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/imagenet_processor.lua -epochs 1 -epochSize 1 -batchSize 4 -nGPU -1
th /home/jshen/scripts/forward.lua /tmp/test.t7 "/file/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_0000000*.JPEG" /home/jshen/scripts/imagenet_processor.lua -nGPU -1

# Imagenet train, train_student_model, forward with nGPU 2 nThreads 2
th /home/jshen/scripts/train.lua /home/jshen/models/VGG/vggbn.lua /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/imagenet_processor.lua -val /file/imagenet/ILSVRC2012_img_val -valSize 2 -val_every 1 -epochs 1 -epochSize 2 -nGPU 2 -nThreads 2
th /home/jshen/scripts/train_student_model.lua /home/jshen/models/VGG/VGG_ILSVRC_16_layers.caffemodel /tmp/test.t7 /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/imagenet_processor.lua -epochs 1 -epochSize 2 -nGPU 2 -nThreads 2
th /home/jshen/scripts/forward.lua /tmp/test.t7 "/file/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_0000000*.JPEG" /home/jshen/scripts/imagenet_processor.lua -batchSize 3 -nGPU 2 -nThreads 2

## Imagenet train, train_student_model, forward with nGPU 2 nThreads 8
#th /home/jshen/scripts/train.lua /home/jshen/models/VGG/vggbn.lua /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/imagenet_processor.lua -val /file/imagenet/ILSVRC2012_img_val -valSize 2 -val_every 1 -epochs 1 -epochSize 2 -nGPU 2 -nThreads 8
#th /home/jshen/scripts/train_student_model.lua /home/jshen/models/VGG/VGG_ILSVRC_16_layers.caffemodel /tmp/test.t7 /file/imagenet/ILSVRC2012_img_val /tmp/test.t7 /home/jshen/scripts/imagenet_processor.lua -epochs 1 -epochSize 2 -nGPU 2 -nThreads 8
#th /home/jshen/scripts/forward.lua /tmp/test.t7 "/file/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_0000000*.JPEG" /home/jshen/scripts/imagenet_processor.lua -batchSize 3 -nGPU 2 -nThreads 8

# Caltech train, forward with nThreads 1
th /home/jshen/scripts/train.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel "/file/caltech/train/{pos/base,pos/IOU0.5,neg}/base" /tmp/test.t7 /home/jshen/scripts/caltech_processor.lua -processor_opts "-imageSize 227" -val /file/caltech/val/base -valBatchSize 2 -valSize 1 -val_every 1 -epochs 1 -epochSize 1 -nGPU 1 -nThreads 1
th /home/jshen/scripts/forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/caltech_processor.lua -processor_opts "-imageSize 227" -epochSize 1 -batchSize 2 -nGPU 1 -nThreads 1

## Caltech train, forward with nThreads 4
#th /home/jshen/scripts/train.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel "/file/caltech/train/{pos/base,pos/IOU0.5,neg}/base" /tmp/test.t7 /home/jshen/scripts/caltech_processor.lua -processor_opts "-imageSize 227" -val /file/caltech/val/base -valBatchSize 2 -valSize 1 -val_every 1 -epochs 1 -epochSize 1 -nGPU 1 -nThreads 4
#th /home/jshen/scripts/forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/caltech_processor.lua -processor_opts "-imageSize 227" -epochSize 1 -batchSize 2 -nGPU 1 -nThreads 4

# train_svm
th /home/jshen/scripts/train_svm.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel "/file/caltech/train/{pos/base,pos/IOU0.5,neg}/base" /tmp/test_svm.t7 /home/jshen/scripts/caltech_processor.lua -processor_opts "-imageSize 227" -layer fc7 -epochSize 1 -nGPU 1 -nThreads 1
th /home/jshen/scripts/forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/caltech_processor.lua -processor_opts "-svm /tmp/test_svm.t7 -layer fc7 -imageSize 227" -epochSize 1 -batchSize 2 -nGPU 1 -nThreads 1
#th /home/jshen/scripts/forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/caltech_processor.lua -processor_opts "-svm /tmp/test_svm.t7 -layer fc7 -imageSize 227" -epochSize 1 -batchSize 2 -nGPU 1 -nThreads 8

# Caltech hog luv train, train_student_model, forward with nThreads 1
th /home/jshen/scripts/train.lua /home/jshen/models/VGG/vggbn_10channels_2outputs.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg}/base" "" /home/jshen/scripts/caltech_hog_luv.lua -processor_opts "-imageSize 227" -val /file/caltech/val/base -valSize 1 -valBatchSize 2 -val_every 1 -epochs 1 -epochSize 1 -nGPU 1 -nThreads 1
th /home/jshen/scripts/train_student_model.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel /home/jshen/models/VGG/vggbn_10channels_2outputs.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg}/base" /tmp/test.t7 /home/jshen/scripts/caltech_hog_luv.lua -processor_opts "-imageSize 224" -teacher_processor /home/jshen/scripts/caltech_processor.lua -teacher_processor_opts "-imageSize 227" -val /file/caltech/val/base -valSize 1 -valBatchSize 2 -val_every 1 -epochs 1 -epochSize 1 -LR 0.01 -nGPU 1 -nThreads 1
th /home/jshen/scripts/forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/caltech_hog_luv.lua -processor_opts "-imageSize 227" -epochSize 1 -batchSize 2 -nGPU 1 -nThreads 1

## Caltech hog luv train, train_student_model, forward with nThreads 4
#th /home/jshen/scripts/train.lua /home/jshen/models/VGG/vggbn_10channels_2outputs.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg}/base" "" /home/jshen/scripts/caltech_hog_luv.lua -processor_opts "-imageSize 227" -val /file/caltech/val/base -valSize 1 -valBatchSize 2 -val_every 1 -epochs 1 -epochSize 1 -nGPU 1 -nThreads 4
#th /home/jshen/scripts/train_student_model.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel /home/jshen/models/VGG/vggbn_10channels_2outputs.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg}/base" /tmp/test.t7 /home/jshen/scripts/caltech_hog_luv.lua -processor_opts "-imageSize 224" -teacher_processor /home/jshen/scripts/caltech_processor.lua -teacher_processor_opts "-imageSize 227" -val /file/caltech/val/base -valSize 1 -valBatchSize 2 -val_every 1 -epochs 1 -epochSize 1 -LR 0.01 -nGPU 1 -nThreads 4
#th /home/jshen/scripts/forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/caltech_hog_luv.lua -processor_opts "-imageSize 227" -epochSize 1 -batchSize 2 -nGPU 1 -nThreads 4

# Caltech train, train_student_model, forward with nGPU 2 nThreads 2 
th /home/jshen/scripts/train.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel "/file/caltech/train/{pos/base,pos/IOU0.5,neg}/base" /tmp/test.t7 /home/jshen/scripts/caltech_processor.lua -processor_opts "-imageSize 227" -val /file/caltech/val/base -valSize 2 -valBatchSize 2 -val_every 1 -epochs 1 -epochSize 2 -nGPU 2 -nThreads 2
th /home/jshen/scripts/train_student_model.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel /home/jshen/models/SqueezeNet/squeezenet.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg}/base" /tmp/test.t7 /home/jshen/scripts/caltech_processor.lua -processor_opts "-imageSize 227" -val /file/caltech/val/base -valSize 2 -valBatchSize 2 -val_every 1 -epochs 1 -epochSize 2 -LR 0.01 -nGPU 2 -nThreads 2
th /home/jshen/scripts/forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/caltech_processor.lua -processor_opts "-imageSize 227" -epochSize 2 -batchSize 2 -nGPU 2 -nThreads 2

## Caltech train, train_student_model, forward with nGPU 2 nThreads 8 
#th /home/jshen/scripts/train.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel "/file/caltech/train/{pos/base,pos/IOU0.5,neg}/base" /tmp/test.t7 /home/jshen/scripts/caltech_processor.lua -processor_opts "-imageSize 227" -val /file/caltech/val/base -valSize 2 -valBatchSize 2 -val_every 1 -epochs 1 -epochSize 2 -nGPU 2 -nThreads 8
#th /home/jshen/scripts/train_student_model.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel /home/jshen/models/SqueezeNet/squeezenet.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg}/base" /tmp/test.t7 /home/jshen/scripts/caltech_processor.lua -processor_opts "-imageSize 227" -val /file/caltech/val/base -valSize 2 -valBatchSize 2 -val_every 1 -epochs 1 -epochSize 2 -LR 0.01 -nGPU 2 -nThreads 8
#th /home/jshen/scripts/forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/caltech_processor.lua -processor_opts "-imageSize 227" -epochSize 2 -batchSize 2 -nGPU 2 -nThreads 8

# Caltech train, forward replicateModel
th /home/jshen/scripts/train.lua /home/jshen/models/SqueezeNet/squeezenet.lua "/file/caltech/train/{pos/base,pos/IOU0.5,neg}/base" /tmp/test.t7 /home/jshen/scripts/caltech_processor.lua -processor_opts "-imageSize 227" -val /file/caltech/val/base -valSize 2 -valBatchSize 2 -val_every 1 -epochs 1 -epochSize 2 -nGPU 2 -nThreads 4 -replicateModel
th /home/jshen/scripts/forward.lua /tmp/test.t7 /file/caltech/val/base /home/jshen/scripts/caltech_processor.lua -processor_opts "-imageSize 227" -epochSize 2 -batchSize 2 -nGPU 2 -nThreads 4 -replicateModel
