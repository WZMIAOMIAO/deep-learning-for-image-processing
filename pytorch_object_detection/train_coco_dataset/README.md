# 训练COCO2017数据集

## 该项目参考自pytorch官方torchvision模块中的源码(使用pycocotools处略有不同)
* https://github.com/pytorch/vision/tree/master/references/detection

## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.10.0
* pycocotools(Linux:```pip install pycocotools```; Windows:```pip install pycocotools-windows```(不需要额外安装vs))
* Ubuntu或Centos(不建议Windows)
* 最好使用GPU训练
* 详细环境配置见```requirements.txt```

## 文件结构：
```
  ├── backbone: 特征提取网络，可以根据自己的要求选择，这里是以VGG16为例
  ├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
  ├── train_utils: 训练验证相关模块（包括pycocotools）
  ├── my_dataset.py: 自定义dataset用于读取COCO2017数据集
  ├── train.py: 以resnet50做为backbone进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── transforms.py: 数据预处理（随机水平翻转图像以及bboxes、将PIL图像转为Tensor）
```

## 预训练权重下载地址（下载后放入项目根目录）：
* Resnet50 https://download.pytorch.org/models/resnet50-19c8e357.pth
* 注意，下载的预训练权重记得要重命名，比如在train.py中读取的是`resnet50.pth`文件，
  不是`resnet50-19c8e357.pth`
 
 
## 数据集，本例程使用的是COCO2017数据集
* COCO官网地址：https://cocodataset.org/
* 对数据集不了解的可以看下我写的博文：https://blog.csdn.net/qq_37541097/article/details/113247318
* 这里以下载coco2017数据集为例，主要下载三个文件：
    * `2017 Train images [118K/18GB]`：训练过程中使用到的所有图像文件
    * `2017 Val images [5K/1GB]`：验证过程中使用到的所有图像文件
    * `2017 Train/Val annotations [241MB]`：对应训练集和验证集的标注json文件
* 都解压到`coco2017`文件夹下，可得到如下文件结构：
```
├── coco2017: 数据集根目录
     ├── train2017: 所有训练图像文件夹(118287张)
     ├── val2017: 所有验证图像文件夹(5000张)
     └── annotations: 对应标注文件夹
              ├── instances_train2017.json: 对应目标检测、分割任务的训练集标注文件
              ├── instances_val2017.json: 对应目标检测、分割任务的验证集标注文件
              ├── captions_train2017.json: 对应图像描述的训练集标注文件
              ├── captions_val2017.json: 对应图像描述的验证集标注文件
              ├── person_keypoints_train2017.json: 对应人体关键点检测的训练集标注文件
              └── person_keypoints_val2017.json: 对应人体关键点检测的验证集标注文件夹
```

## 训练方法
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 若要使用单GPU训练直接使用train.py训练脚本
* 若要使用多GPU训练，使用`torchrun --nproc_per_node=8 train_multi_GPU.py`指令,`nproc_per_node`参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上`CUDA_VISIBLE_DEVICES=0,3`(例如我只要使用设备中的第1块和第4块GPU设备)
* `CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py`

## 注意事项
* 在使用训练脚本时，注意要将`--data-path`设置为自己存放`coco2017`文件夹所在的**根目录**
* 训练过程中保存的`results.txt`是每个epoch在验证集上的COCO指标，前12个值是COCO指标，后面两个值是训练平均损失以及学习率
* 在使用预测脚本时，要将`weights_path`设置为你自己生成的权重路径。
* 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改`--num-classes`、`--data-path`和`--weights-path`即可，其他代码尽量不要改动

## 本项目训练得到的权重(Faster R-CNN + Resnet50)
* 链接: https://pan.baidu.com/s/1iF-Yl_9TkFFeAy-JysfGSw  密码: d2d8
* COCO2017验证集mAP：
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.277
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.290
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.126
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.308
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.243
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.169
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.402
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.512
```

## 如果对Faster RCNN原理不是很理解可参考我的bilibili
* https://b23.tv/sXcBSP

## Faster RCNN框架图
![Faster R-CNN](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/raw/master/pytorch_object_detection/faster_rcnn/fasterRCNN.png) 
