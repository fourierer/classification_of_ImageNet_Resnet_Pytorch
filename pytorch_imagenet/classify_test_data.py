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
model.eval()  # 固定训练模型的batchnorm以及dropout等的参数
model= model.to(device)
print('load model done!')


#从数据集中加载测试数据
test_dataset = D.ImageFolder(root='/home/momo/mnt/data2/datum/raw/val2', transform=data_transform)  # 这里使用自己写的data.py文件，ImageFolder不仅返回图片和标签，还返回图片的路径，方便后续方便保存
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


