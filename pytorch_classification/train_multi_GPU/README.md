## 多GPU启动指令

- ```python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py```
- 其中```nproc_per_node```为并行GPU的数量