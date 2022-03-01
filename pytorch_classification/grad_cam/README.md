## Grad-CAM
- Original Impl: [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- Grad-CAM简介: [https://b23.tv/1kccjmb](https://b23.tv/1kccjmb)
- 使用Pytorch实现Grad-CAM并绘制热力图: [https://b23.tv/n1e60vN](https://b23.tv/n1e60vN)

## 使用流程(替换成自己的网络)
1. 将创建模型部分代码替换成自己创建模型的代码，并载入自己训练好的权重
2. 根据自己网络设置合适的`target_layers`
3. 根据自己的网络设置合适的预处理方法
4. 将要预测的图片路径赋值给`img_path`
5. 将感兴趣的类别id赋值给`target_category`

