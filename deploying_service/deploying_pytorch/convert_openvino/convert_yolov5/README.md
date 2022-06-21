OpenVINO量化YOLOv5

1. 按照`requirements.txt`配置环境
2. 将YOLOv5转为ONNX
YOLOv5官方有提供导出ONNX以及OpenVINO的方法，但我这里仅导出成ONNX，这里以YOLOv5s为例
```
python export.py --weights yolov5s.pt --include onnx
```

3. ONNX转换为IR
使用OpenVINO的`mo`工具将ONNX转为OpenVINO的IR格式
```
mo  --input_model yolov5s.onnx \
    --input_shape "[1,3,640,640]" \
    --scale 255 \
    --data_type FP32 \
    --output_dir ir_output
```

4. 量化模型
使用`quantization_int8.py`进行模型的量化，量化过程中需要使用到COCO2017数据集，需要将`data_path`指向coco2017目录
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

5. benchmark
直接利用`benchmark_app`工具测试量化前后的`Throughput`，这里以`CPU: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz`设备为例
```
benchmark_app -m ir_output/yolov5s.xml -d CPU -api sync
```
output：
```
Latency:
    Median:     59.56 ms
    AVG:        63.30 ms
    MIN:        57.88 ms
    MAX:        99.89 ms
Throughput: 16.79 FPS
```

```
benchmark_app -m quant_ir_output/quantized_yolov5s.xml -d CPU -api sync
```
output:
```
Latency:
    Median:     42.97 ms
    AVG:        46.56 ms
    MIN:        41.18 ms
    MAX:        95.75 ms
Throughput: 23.27 FPS
```