#!/bin/sh
set -e
th /home/jshen/scripts/train.lua /home/jshen/models/VGG/vggbn.lua /file/imagenet/ILSVRC2012_img_train /tmp/test.t7 /home/jshen/scripts/imagenet_processor.lua -val /file/imagenet/ILSVRC2012_img_val -valSize 2 -val_every 1 -epochs 2 -epochSize 1 -nGPU 1
th /home/jshen/scripts/train_student_model.lua /home/jshen/models/VGG/VGG_ILSVRC_16_layers.caffemodel /tmp/test.t7 /file/imagenet/ILSVRC2012_img_train /tmp/test.t7 /home/jshen/scripts/imagenet_processor.lua -epochs 1 -epochSize 1 -nGPU 1
th /home/jshen/scripts/forward.lua /tmp/test.t7 "/file/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_0000000*.JPEG" /home/jshen/scripts/imagenet_processor.lua -nGPU 1
th /home/jshen/scripts/train_svm.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel "/file/caltech/{crop/person,crop_neg}" /tmp/test_svm.t7 /home/jshen/scripts/caltech_processor.lua -layer fc7 -epochSize 1 -nGPU 1
th /home/jshen/scripts/train.lua /home/jshen/models/AlexNet_ped/caltech10x_warp_finetune.caffemodel "/file/caltech/{crop/person,crop_neg}" /tmp/test.t7 /home/jshen/scripts/caltech_processor.lua -epochs 1 -epochSize 1 -nGPU 1
th /home/jshen/scripts/forward.lua /tmp/test.t7 "/file/caltech/{crop/person,crop_neg}" /home/jshen/scripts/caltech_processor.lua -processor_opts "-svm /tmp/test_svm.t7 -layer fc7" -epochSize 1 -nGPU 1
