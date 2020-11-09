## 多GPU启动指令

- ```python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_gpu_using_launch.py```
- 其中```nproc_per_node```为并行GPU的数量

## 训练时间对比
![training time](./training_time.png)

## 是否使用SyncBatchNorm
![syncbn](./syncbn.png)

## 单GPU与多GPU训练曲线
![accuracy](./accuracy.png)