## keras functional api训练的模型权重与subclassed训练的模型权重能否混用 [tensorflow2.0.0]
强烈不建议混用，即使两个模型的名称结构完全一致也不要混用，里面有坑，用什么方法训练的模型就载入相应的模型权重


## 使用subclassed模型时无法使用model.summary() [tensorflow2.0.0]
subclassed模型在实例化时没有自动进行build操作（只有在开始训练时，才会自动进行build），如果需要使用summary操作，需要提前手动build  
model.build((batch_size, height, width, channel))


## 无法使用keras的plot_model(model, 'my_model.png')问题 [tensorflow2.0.0]
#### 在linux下你需要安装一些包：
* pip install pydot==1.2.3
* sudo apt-get install graphviz   
#### 在windows中，同样需要安装一些包（windows比较麻烦）：
* pip install pydot==1.2.3
* 安装graphviz，并添加相关环境变量  
参考连接：https://github.com/XifengGuo/CapsNet-Keras/issues/7