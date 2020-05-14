# Faster R-CNN
## 环境配置：
* Python3.6或者3.7
* Pytorch1.5(注意：是1.5)
* pycocotools(Linux: pip install pycocotools;   
  Windows:pip install pycocotools-win(如果报错，需要Microsoft Visual C++ 14))
* Ubuntu或Centos(不建议Windows)
* 最好使用GPU训练

## 文件结构：
* ├── backbone: 特征提取网络，可以根据自己的要求选择
* ├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
* ├── train_utils: 训练验证相关模块（包括cocotools）
* ├── my_dataset.py: 自定义dataset用于读取VOC数据集
* ├── train_mobilenet.py: 以MobileNetV2做为backbone进行训练
* ├── train_resnet50_fpn.py: 以resnet50+FPN做为backbone进行训练
* ├── train_multi_GPU.py: 针对使用多GPU的用户使用
* ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
* ├── pascal_voc_classes.json: pascal_voc标签文件

 