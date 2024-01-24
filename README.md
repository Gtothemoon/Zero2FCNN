# Zero2FCNN
本实验报告详细描述了在不使用深度学习库<sup>1</sup> 的情况下从零<sup>2</sup> 开始搭建一个简单的全连接神经网络（Fully Connected Neural Network，FCNN）的详细数学推导和完整计算过程（前向传播、梯度回传等），并提供了配套的实验代码。在拟合三角函数任务和 MNIST 数据集分类任务上对所搭建的 FCNN 进行了实验并报告了结果。

1. *如 Pytorch 库*
2. *但是使用 numpy、math 等必要的数值计算库*

## 使用说明
### 依赖环境
* Python 3.10.5
* `ZeroNet/requirements.txt`中的所有包

### 生成拟合三角函数所需的实验数据
直接运行文件`ZeroNet/create_data.py`，或进入项目路径下在终端输入命令：
```
python create_data.py
```

运行成功后即可在文件夹 `ZeroNet/data`下看到四个数据文件 `x_train.csv`、`y_train.csv`、`x_test.csv`、`y_test.csv`。

### 复现报告中的拟合三角函数任务实验结果
直接运行文件 `ZeroNet/ZeroNet.py`，或进入项目路径下在终端输入命令：
```
python ZeroNet.py
```

### 复现报告中的 MNIST 分类任务实验结果
直接运行文件 `ZeroNet/ZeroNet_mnist.py`，或进入项目路径下在终端输入命令：
```
python ZeroNet_mnist.py
```

### 注意事项
* SGD 优化器和 Adam 优化器可以简单地在初始化模型时进行替换
* 要复现报告实验中的loss曲线图和三角函数拟合预测图，可直接将画图的代码取消注释
* 要自定义新的超参数值，可直接在对应的变量上更改数值
