#!/bin/bash -xe
cd /home/jshen/scripts/

# Imagenet train, TrainStudentModel, forward
th Train.lua "/home/jshen/models/VGG/vggbn.lua ImagenetProcessor.lua" /file1/imagenet/ILSVRC2012_img_val /tmp/test.t7 -val /file1/imagenet/ILSVRC2012_img_val -valSize 2 -valEvery 1 -epochs 1 -epochSize 1
th TrainStudentModel.lua "/home/jshen/models/VGG/VGG_ILSVRC_16_layers.caffemodel ImagenetProcessor.lua" "/tmp/test.t7 ImagenetProcessor.lua" /file1/imagenet/ILSVRC2012_img_val /tmp/test.t7 -epochs 1 -epochSize 1 -useSameInputs
th Forward.lua "/tmp/test.t7 ImagenetProcessor.lua" "/file1/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_0000000*.JPEG"

# Caltech train, forward
th Train.lua "/home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel CaltechProcessor.lua -imageSize 227" "/file1/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test.t7 -val /file1/caltech/val/base -valBatchSize 3 -valSize 1 -valEvery 1 -epochs 1 -epochSize 1
th Forward.lua "/tmp/test.t7 CaltechProcessor.lua -imageSize 227" /file1/caltech/val/base -epochSize 1 -batchSize 3

# TrainSVM
th TrainSVM.lua "/home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel CaltechProcessor.lua -imageSize 227" "/file1/caltech/train/{pos/base,pos/IOU0.5,neg/base,neg/IOU0.5}/base" /tmp/test_svm.t7 -layer fc7 -epochSize 1

# CaltechACF train, TrainStudentModel, forward
th Train.lua "/home/jshen/models/resnet/res3_acf.t7 CaltechACF.lua -imageSize 113" "/file1/caltech10x/train/pos/raw;/file1/caltech10x/train/neg/raw" "" -val "/file1/caltech10x/val/*/raw" -valSize 1 -valBatchSize 3 -valEvery 1 -epochs 1 -epochSize 1 -batchSize 3
th TrainStudentModel.lua "/home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel CaltechProcessor.lua -imageSize 227" "/home/jshen/models/resnet/res3_acf.t7 CaltechACF.lua -imageSize 113" "/file1/caltech10x/train/pos/raw;/file1/caltech10x/train/neg/raw" /tmp/test.t7 -val "/file1/caltech10x/val/*/raw" -valSize 1 -valBatchSize 3 -valEvery 1 -epochs 1 -epochSize 1 -batchSize 3
th Forward.lua "/tmp/test.t7 CaltechACF.lua -imageSize 113" "/file1/caltech10x/val/*/raw" -epochSize 1 -batchSize 3
