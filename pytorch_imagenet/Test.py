import torch
from PIL import Image
from torchvision import transforms
device = torch.device('cuda')
transform = transforms.Compose([
    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])

def predict(img_path):
    net = torch.load('/home/momo/sun.zheng/pytorch_imagenet/model.pkl')
    net = net.to(device)
    torch.no_grad()
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    img_= img.to(device)
    outputs = net(img_)
    _, predicted = torch.max(outputs,1)
    print('this picture maybe:' + str(predicted))

if __name__ == '__main__':
    predict('/home/momo/mnt/data2/datum/raw/val2/n01440764/ILSVRC2012_val_00000293.JPEG')


