import torch.utils.data as data
import os
import cv2
from PIL import Image
import xlrd
import torch
import numpy as np
from torchvision import transforms

import random

# mul_target = 1
import img_to_gamma

retinopathyGrade = 1
two_class = 0
mul_target = 0
def up_sample(datas):
    len_ = len(datas)
    for i in range(len_):
        x,y = datas[i]
        if retinopathyGrade:
            upsamplelist = [0, 2, 1, 1]
        else:
            upsamplelist = [0, 12, 5]
        for j in range(upsamplelist[y]):
            datas.append((x, y))
    return datas

def tran_mul_target(target):
    if target == 0:
        return [1,0,0,0]
    elif target == 1:
        return [0,1,0,0]
    elif target == 2:
        return [0,1,1,0]
    elif target == 3:
        return [0,1,1,1]

def make_dataset(path):
    imgs=[]
    for dir in os.listdir(path):
        if os.path.isdir(path + '/' + dir):
            image_dir = path + '/' + dir
            for image in os.listdir(image_dir):
                image_path = image_dir + '/' + image  # 图像路径
                if image[-3:] == 'xls':
                    book = xlrd.open_workbook(image_path)
                    sheet = book.sheet_by_index(0)
                    image_name = [image_dir + '/'+ i for i in sheet.col_values(0)[1:]]
                    y_retinopathyGrade = sheet.col_values(2)[1:]
                    y_riskOfMacularEdema = sheet.col_values(3)[1:]
                    if retinopathyGrade:
                        y = y_retinopathyGrade
                    else:
                        y = y_riskOfMacularEdema
                    for x,y in zip(image_name,y):
                        if two_class:
                            if y==2 or y==3:
                                y = 1
                        y = int(y)
                        if mul_target:
                                y = tran_mul_target(y)
                        imgs.append((x,y))
    random.seed(1234)
    random.shuffle(imgs)
    if two_class:
        train_imgs = up_sample(imgs[:int(0.9*len(imgs))])
    else:
        train_imgs = up_sample(imgs[:int(0.7 * len(imgs))])
    test_imgs = imgs[int(0.7*len(imgs)):]
    return train_imgs,test_imgs

class Datasets(data.Dataset):
    def __init__(self, path = './images', is_color=True,
                 phase = 'train',
                 transform = None):
        self.is_color = is_color
        self.phase = phase
        if phase == "train":
            self.imgs = make_dataset(path)[0]
        elif phase == "test":
            self.imgs = make_dataset(path)[1]
        else:
            print("phase error")
        self.transform = transform
    def __getitem__(self, index):
        image_path, y = self.imgs[index]
        #x = cv2.imread(image_path,1 if self.is_color else 0)
        #if mul_target:
        # y_retinopathyGrade = torch.FloatTensor([y_retinopathyGrade])
        #    y_riskOfMacularEdema = torch.FloatTensor(y_riskOfMacularEdema)
        #################裁剪######################33
        x = cv2.imread(image_path, 0)
        #x = img_to_gamma.gamma_intensity_correction(x,0.7)
        # # x = cv2.imread(image_path,0)
        # crop = int((x.shape[1] - x.shape[0])/2)
        # x = x[0:x.shape[0],crop:crop+x.shape[0]]
        # if np.sum(x[crop:x.shape[0]-crop,0:int(1/2*x.shape[1])]) > np.sum(x[crop:x.shape[0]-crop,int(1/2*x.shape[1]):x.shape[1]]):
        #     x=x[int(1/8*x.shape[0]):int(7/8*x.shape[0]),int(2/5*x.shape[1]):int(5/6*x.shape[1])]
        # else:
        #     x=x[int(1/8*x.shape[0]):int(7/8*x.shape[0]),int(1/6*x.shape[0]):int(3/5*x.shape[1])]
        #     x=cv2.flip(x,1)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))  # 自适应直方图均衡化
        x = clahe.apply(x)
        x = cv2.merge((x,x,x))
        #######################直方图######################
        # b,g,r = cv2.split(x)
        # x=np.uint8(0.2*b+0.2*r+0.6*g)
        # cv2.imshow('0',cv2.resize(x,(512,512)))
        # print(y_retinopathyGrade)
        # cv2.waitKey(0)
        x=Image.fromarray(x)

        if self.phase == "train":
            if self.transform is not None:
                x = self.transform(x)
        elif self.phase == "test":
            img = x
            x = []
            for i in range(10):
                if self.transform is not None:
                    x.append(self.transform(img))
        return x, y
    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = Datasets(is_color=True,
                       phase='train',
                       transform=transforms.Compose([transforms.Resize((334,334)),
                                                     transforms.ToTensor(),
                                                     normalize, ]))
    test_dataset = Datasets(is_color=True,
                       phase='test',
                       transform=transforms.Compose([transforms.Resize(384),
                                                     transforms.ToTensor(),
                                                     normalize, ]))
    print("已读取%d张图像" % len(train_dataset))
    print("已读取%d张图像" % len(test_dataset))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=2, shuffle=(train_sampler is None),
                                               num_workers=0, pin_memory=True, sampler=train_sampler,drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=2, shuffle=False,
                                             num_workers=0, pin_memory=True,drop_last=True)

    for i,(input,target) in enumerate(train_loader):
       print(input.shape)













