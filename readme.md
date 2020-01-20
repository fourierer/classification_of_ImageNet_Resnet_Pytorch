### 使用pytorch框架来分类imagenet数据集。

1.network.py:模型选择预训练resnet，直接在torchvision中加载，修改最后一层的分类数。训练时，只训练最后一层的参数，训练完成之后保存模型。

2.Test.py:用某一个样例或者批量数据集(或者所有测试数据集)来测试保存的模型，输出测试样例的类别

3.用保存好的模型去分类测试集，在每个类别中分别建立两个文件夹，right和w rong文件夹，分别保存正确分类和错分样本

代码

(1)-(5)为network.py文件，(6)为Test.py文件，(7)为classify_test_data.py文件

(1)环境

```python
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定一块gpu为可见
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 指定四块gpu为可见
```

(2)读取数据：

```python
# #############创建数据加载器###################
print('data loaded begin!')
# 预处理，将各种预处理组合在一起
data_transform = transforms.Compose([
    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])

train_dataset = torchvision.datasets.ImageFolder(root='/home/momo/mnt/data2/datum/raw/train', transform=data_transform)
# 使用ImageFolder需要数据集存储的形式：每个文件夹存储一类图像
# ImageFolder第一个参数root : 在指定的root路径下面寻找图片
# 第二个参数transform: 对PIL Image进行转换操作,transform 输入是loader读取图片返回的对象
train_data = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
# 第一个参数train_dataset是上面自定义的数据形式
# 最后一个参数是线程数，>=1即可多线程预读数据

test_dataset = torchvision.datasets.ImageFolder(root='/home/momo/mnt/data2/datum/raw/val2', transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=4)

print(type(train_data))
print('data loaded done!')
# <class 'torch.utils.data.dataloader.DataLoader'>
```

(3)加载resnet模型

```python
# ##################创建网络模型###################
'''
这里选择从torch里面直接导入resnet，不搭建网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Torch.nn.Conv2d(in_channels，out_channels，kernel_size，stride = 1，
        # padding = 0，dilation = 1，groups = 1，bias = True)
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3),  # 16,298,298
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3),  # 32,296,296
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))  # 32,148,148
'''
'''
print('resnet model loaded begin!')
# 使用resnet50,进行预训练
#model = models.resnet50(pretrained=True)
print('resnet model loaded done!')
# 对于模型的每个权重，使其进行反向传播，即不固定参数
or param in model.parameters():
    param.requires_grad = True
'''

print('resnet model loaded begin!')
model = models.resnet50(pretrained=True)
print(model)
print('resnet model loaded done!')
# 对于模型的每个权重，使其不进行反向传播，即固定参数
for param in model.parameters():
    param.requires_grad = False
# 修改最后一层的参数，使其不固定，即不固定全连接层fc
for param in model.fc.parameters():
    param.requires_grad = True



# 修改最后一层的分类数
class_num = 1000  # imagenet的类别数是1000
channel_in = model.fc.in_features  # 获取fc层的输入通道数
model.fc = nn.Linear(channel_in, class_num)  # 最后一层替换
```

(4)训练及保存模型

```python
# ##############训练#################

# 在可见的gpu中，指定第一块卡训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), 1e-1)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 1e-1)

nums_epoch = 1  # 为了快速，只训练一个epoch

print('training begin!')
# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

for epoch in range(nums_epoch):
    train_loss = 0
    train_acc = 0
    model = model.train()
    print('Epoch ' + str(epoch+1) + ' begin!')
    for img, label in train_data:
        img = img.to(device)
        label = label.to(device)

        # 前向传播
        out = model(img)
        optimizer.zero_grad()
        loss = criterion(out, label)
        print('Train loss in current Epoch' + str(epoch+1) + ':' + str(loss))
        #print('BP begin!')
        # 反向传播
        loss.backward()
        #print('BP done!')
        optimizer.step()

        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
        print('Train accuracy in current Epoch' + str(epoch) + ':' + str(acc))

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    print('Epoch' + str(epoch+1)  + ' Train  done!')
    print('Epoch' + str(epoch+1)  + ' Test  begin!')
    # 每个epoch测一次acc和loss
    eval_loss = 0
    eval_acc = 0
    # 测试集不训练
    for img1, label1 in test_data:
        img1 = img1.to(device)
        label1 = label1.to(device)
        out = model(img1)

        loss = criterion(out, label1)
        # print('Test loss in current Epoch:' + str(loss))

        # 记录误差
        eval_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label1).sum().item()
        acc = num_correct / img1.shape[0]
        eval_acc += acc

    print('Epoch' + str(epoch+1)  + ' Test  done!')
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('Epoch {} Train Loss {} Train  Accuracy {} Test Loss {} Test Accuracy {}'.format(
        epoch + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data),
            eval_acc / len(test_data)))
    torch.save(model, '/home/momo/sun.zheng/pytorch_imagenet/model_f.pkl')
    print('model saved done!')
```

(5).结果(只给出测试部分结果)

使用四张TITAN XP的显卡，训练和测试的batch_size都设置为512，一个epoch大概5个小时左右(只要显存够，单卡和四卡对速度应该没影响)。

```python
Epoch 1 Train Loss 1.4981742842741315 Train  Accuracy 0.7642411056033459 Test Loss 1.1175287165203873 Test Accuracy 0.7334610741618075

......(剩下的没有跑完)
```



(6).针对上述模型进行测试

测试代码：

测试的时候，在模型加载之后，一定不能少了model.eval()，用于固定batch_norm和dropout等，否则batch_size会影响测试结果

```python
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
device = torch.device('cuda')
data_transform = transforms.Compose([
    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])

print('Test data load begin!')
test_dataset = torchvision.datasets.ImageFolder(root='/home/momo/mnt/data2/datum/raw/val2', transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
print(type(test_data))
print('Test data load done!')

print('load model begin!')
model = torch.load('/home/momo/sun.zheng/pytorch_imagenet/model_f.pkl')
model.eval()  # 固定batchnorm，dropout等，一定要有
model= model.to(device)
print('load model done!')


#测试单个图像属于哪个类别
'''
torch.no_grad()
img = Image.open('/home/momo/mnt/data2/datum/raw/val2/n01440764/ILSVRC2012_val_00026064.JPEG')
img = transform(img).unsqueeze(0)
img_= img.to(device)
outputs = net(img_)
_, predicted = torch.max(outputs,1)
print('this picture maybe:' + str(predicted))
'''
#批量测试准确率,并输出所有测试集的平均准确率
eval_acc = 0
torch.no_grad()
for img1, label1 in test_data:
    img1 = img1.to(device)
    label1 = label1.to(device)
    out = model(img1)

    _, pred = out.max(1)
    print(pred)
    print(label1)
    num_correct = (pred == label1).sum().item()
    acc = num_correct / img1.shape[0]
    print('Test acc in current batch:' + str(acc))
    eval_acc +=acc

print('final acc in Test data:' + str(eval_acc / len(test_data)))
```

如果测试单个图片，最后输出的是一个tensor，数值代表了类别。如：

```python
this picture maybe:tensor([494], device='cuda:0')
```

如果批量测试则输出每个batch的准确率以及所有测试集的平均准确率，测试结果如下：

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

(7)用保存的模型对测试集进行分类，在每个类别里面分为right和wrong文件夹，分别存储正确分类和错误分类的图像

```python
# 此脚本用于识别测试数据集中每一类，将分类正确的和错误的分别置于每一类文件中两个文件夹里面
import shutil
import os
import torch
import torchvision
import D  # 自己写的D.py，为了方便后续分类
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
#定义数据转换格式transform
device = torch.device('cuda')
data_transform = transforms.Compose([
    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std
])

#加载模型
print('load model begin!')
model = torch.load('/home/momo/sun.zheng/pytorch_imagenet/model_f.pkl')
model.eval()
model= model.to(device)
print('load model done!')


#从数据集中加载测试数据
test_dataset = D.ImageFolder(root='/home/momo/mnt/data2/datum/raw/val2', transform=data_transform)  # 这里使用自己写的D.py文件，ImageFolder不仅返回图片和标签，还返回图片的路径，方便后续方便保存
#test_dataset = torchvision.datasets.ImageFolder(root='/home/momo/mnt/data2/datum/raw/val2', transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

'''
路径/home/momo/sun.zheng/pytorch_imagenet/classify_test_data_result/val2里面是原始测试集，
路径/home/momo/sun.zheng/pytorch_imagenet/classify_test_data_result/test_result里面批量建立文件夹，每一类数据建立一个文件夹，
文件夹包括right和wrong两个文件夹，分别表示在保存的模型测试下，该类正确分类和错误分类的样本集合
'''


count = 0  # 当前类别测试图片个数，上限是1450000



for img1, label1, path1 in test_data:
    count = count + 1

    img11 = img1.squeeze()  # 此处的squeeze()去除size为1的维度，,将(1,3,224,224)的tensor转换为(3,224,224)的tensor
    #new_img1 = transforms.ToPILImage()(img11).convert('RGB')  # new_img1为PIL.Image类型
    img1 = img1.to(device)  # img1是tensor类型，规模是(1,3,224,224),gpu的tensor无法转换成PIL，所以在转换之后再放到gpu上
    label1 = label1.to(device)
    out = model(img1)
    _, pred = out.max(1)  # pred是类别数，tensor类型
    print(count)
    #print(path1[0])
    #print(type(path1[0]))

    if pred == label1:
        #将分对的图像放在right文件夹里面
        img_path = '/home/momo/sun.zheng/pytorch_imagenet/classify_test_data_result/test_result/' + str(label1[0]) + '/right/'
        folder = os.path.exists(img_path)
        if not folder:
            os.makedirs(img_path)
        shutil.copy(path1[0], img_path)
    else:
        #将错的图像放在wrong文件夹里面
        img_path = '/home/momo/sun.zheng/pytorch_imagenet/classify_test_data_result/test_result/' + str(label1[0]) + '/wrong/'
        folder = os.path.exists(img_path)
        if not folder:
            os.makedirs(img_path)
        shutil.copy(path1[0], img_path)
```

D.py文件(实际上是torch1.0.0的官方文档，更改ImageFolder类继承的DataFolder类，不仅仅返回图像和标签，也返回读取的图片的路径，后续用这个路径来复复制正确分类和错误分类的样本)：

```python
import torch.utils.data as data

from PIL import Image

import os
import os.path


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path  # 增加了path


    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
```





