### 使用pytorch框架来分类imagenet数据集。

1.network.py:模型选择预训练resnet，直接在torchvision中加载，修改最后一层的分类数。训练时，只训练最后一层的参数，训练完成之后保存模型。

2.Test.py:用某一个样例来测试保存的模型，输出测试样例的类别