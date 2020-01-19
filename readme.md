### 使用pytorch框架来分类imagenet数据集。

1.network.py:模型选择预训练resnet，直接在torchvision中加载，修改最后一层的分类数。训练时，只训练最后一层的参数，训练完成之后保存模型。

2.Test.py:用某一个样例或者批量数据集(或者所有测试数据集)来测试保存的模型，输出测试样例的类别

其中批量测试以及测试所有的测试数据集结果如下：

```python
Test data load begin!
<class 'torch.utils.data.dataloader.DataLoader'>
Test data load done!
load model begin!
load model done!
Test acc in current batch:0.744140625
Test acc in current batch:0.720703125
Test acc in current batch:0.736328125
Test acc in current batch:0.705078125
Test acc in current batch:0.701171875
Test acc in current batch:0.703125
Test acc in current batch:0.73046875
Test acc in current batch:0.73046875
Test acc in current batch:0.748046875
Test acc in current batch:0.734375
Test acc in current batch:0.734375
Test acc in current batch:0.740234375
Test acc in current batch:0.701171875
Test acc in current batch:0.716796875
Test acc in current batch:0.7734375
Test acc in current batch:0.73046875
Test acc in current batch:0.76953125
Test acc in current batch:0.759765625
Test acc in current batch:0.7265625
Test acc in current batch:0.712890625
Test acc in current batch:0.7578125
Test acc in current batch:0.75
Test acc in current batch:0.744140625
Test acc in current batch:0.728515625
Test acc in current batch:0.748046875
Test acc in current batch:0.748046875
Test acc in current batch:0.728515625
Test acc in current batch:0.744140625
Test acc in current batch:0.7265625
Test acc in current batch:0.740234375
Test acc in current batch:0.732421875
Test acc in current batch:0.744140625
Test acc in current batch:0.767578125
Test acc in current batch:0.71875
Test acc in current batch:0.72265625
Test acc in current batch:0.71484375
Test acc in current batch:0.728515625
Test acc in current batch:0.716796875
Test acc in current batch:0.720703125
Test acc in current batch:0.703125
Test acc in current batch:0.720703125
Test acc in current batch:0.73046875
Test acc in current batch:0.7109375
Test acc in current batch:0.73828125
Test acc in current batch:0.732421875
Test acc in current batch:0.7421875
Test acc in current batch:0.755859375
Test acc in current batch:0.701171875
Test acc in current batch:0.697265625
Test acc in current batch:0.673828125
Test acc in current batch:0.744140625
Test acc in current batch:0.7421875
Test acc in current batch:0.681640625
Test acc in current batch:0.75
Test acc in current batch:0.73828125
Test acc in current batch:0.744140625
Test acc in current batch:0.748046875
Test acc in current batch:0.708984375
Test acc in current batch:0.70703125
Test acc in current batch:0.76171875
Test acc in current batch:0.71875
Test acc in current batch:0.712890625
Test acc in current batch:0.724609375
Test acc in current batch:0.7578125
Test acc in current batch:0.751953125
Test acc in current batch:0.724609375
Test acc in current batch:0.748046875
Test acc in current batch:0.75
Test acc in current batch:0.734375
Test acc in current batch:0.7421875
Test acc in current batch:0.7578125
Test acc in current batch:0.76953125
Test acc in current batch:0.73046875
Test acc in current batch:0.712890625
Test acc in current batch:0.744140625
Test acc in current batch:0.720703125
Test acc in current batch:0.728515625
Test acc in current batch:0.689453125
Test acc in current batch:0.75
Test acc in current batch:0.736328125
Test acc in current batch:0.732421875
Test acc in current batch:0.716796875
Test acc in current batch:0.73828125
Test acc in current batch:0.740234375
Test acc in current batch:0.73828125
Test acc in current batch:0.728515625
Test acc in current batch:0.73828125
Test acc in current batch:0.76953125
Test acc in current batch:0.748046875
Test acc in current batch:0.7265625
Test acc in current batch:0.724609375
Test acc in current batch:0.7265625
Test acc in current batch:0.701171875
Test acc in current batch:0.7421875
Test acc in current batch:0.748046875
Test acc in current batch:0.73046875
Test acc in current batch:0.71484375
Test acc in current batch:0.7232142857142857
final acc in Test data:0.7319093932215744
```

