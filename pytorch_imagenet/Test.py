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


