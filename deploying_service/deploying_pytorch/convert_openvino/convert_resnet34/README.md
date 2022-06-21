本项目展示如何将Pytorch中的ResNet34网络转成Openvino的IR格式，并进行量化处理，具体使用流程如下：
1. 按照`requirements.txt`配置环境
2. 下载事先训练好的ResNet34权重（之前在花分类数据集上训练得到的）放在当前文件夹下。百度云链接: https://pan.baidu.com/s/1x4WFX1HynYcXLium3UaaFQ  密码: qvi6
3. 使用`convert_pytorch2onnx.py`将Resnet34转成ONNX格式
4. 在命令行中使用以下指令将ONNX转成IR格式：
```
mo  --input_model resnet34.onnx \
    --input_shape "[1,3,224,224]" \
    --mean_values="[123.675,116.28,103.53]" \
    --scale_values="[58.395,57.12,57.375]" \
    --data_type FP32 \
    --output_dir ir_output
```
5. 下载并解压花分类数据集，将`quantization_int8.py`中的`data_path`指向解压后的`flower_photos`
6. 使用`quantization_int8.py`量化模型