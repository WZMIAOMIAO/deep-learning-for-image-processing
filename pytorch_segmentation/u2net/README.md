# U2-Net(Going Deeper with Nested U-Structure for Salient Object Detection)

## 该项目主要是来自官方的源码
* https://github.com/xuebinqin/U-2-Net

## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.10
* Ubuntu或Centos(Windows暂不支持多GPU训练)
* 建议使用GPU训练
* 详细环境配置见```requirements.txt```


## 官方权重
从官方转换得到的权重：
- `u2net_full.pth`下载链接: https://pan.baidu.com/s/1ojJZS8v3F_eFKkF3DEdEXA  密码: fh1v
- `u2net_lite.pth`下载链接: https://pan.baidu.com/s/1TIWoiuEz9qRvTX9quDqQHg  密码: 5stj

`u2net_full`在DUTS-TE上的验证结果(使用`validation.py`进行验证)：
```
MAE: 0.044
maxF1: 0.868
```
**注：**
- 这里的maxF1和原论文中的结果有些差异，经过对比发现差异主要来自post_norm，原仓库中会对预测结果进行post_norm，但在本仓库中将post_norm给移除了。
如果加上post_norm这里的maxF1为`0.872`，如果需要做该后处理可自行添加，post_norm流程如下，其中output为验证时网络预测的输出：
```python
ma = torch.max(output)
mi = torch.min(output)
output = (output - mi) / (ma - mi)
```
- 如果要载入官方提供的权重，需要将`src/model.py`中`ConvBNReLU`类里卷积的bias设置成True，因为官方代码里没有进行设置（Conv2d的bias默认为True）。
因为卷积后跟了BN，所以bias是没有用的，所以在本仓库中默认将bias设置为False。

## 训练记录(`u2net_full`)
训练指令：
```
torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.004 --amp
```
训练最终在DUTS-TE上的验证结果：
```
MAE: 0.047
maxF1: 0.859
```
训练过程详情可见results.txt文件，训练权重下载链接: https://pan.baidu.com/s/1df2jMkrjbgEv-r1NMaZCZg  密码: n4l6