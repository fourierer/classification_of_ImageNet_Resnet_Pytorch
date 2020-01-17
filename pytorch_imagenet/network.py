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


