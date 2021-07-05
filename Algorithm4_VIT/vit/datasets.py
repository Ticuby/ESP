import torch.utils.data as data
import os
import cv2
from PIL import Image
import xlrd
import torch
from torchvision import transforms
import random
import numpy as np

mul_target = 0
two_target = 0
def tran_mul_target(target):
    if target == 0:
        return [1,0,0,0]
    elif target == 1:
        return [0,1,0,0]
    elif target == 2:
        return [0,1,1,0]
    elif target == 3:
        return [0,1,1,1]

def up_sample(datas):
    len_ = len(datas)
    for i in range(len_):
        x,y = datas[i]
        # upsamplelist = [0, 2, 1, 1]
        upsamplelist = [0, 0, 0, 0]
        for j in range(upsamplelist[y]):
            datas.append((x, y))
    return datas

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
                    for x,y0,y1 in zip(image_name,y_retinopathyGrade,y_riskOfMacularEdema):
                        if two_target:
                            if y == 2 or y == 3:
                                y = 1
                        y = int(y0)
                        if mul_target:
                            y = tran_mul_target(y)
                        imgs.append((x, y))
    random.seed(1234)
    random.shuffle(imgs)
    if two_target:
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
        image_path, y, = self.imgs[index]
        x = cv2.imread(image_path)
        _, x, _ = cv2.split(x)
        # x = cv2.medianBlur(x,5)
        # cv2.imshow('0', cv2.resize(x, (512, 512)))
        # cv2.imshow('1',cv2.resize(x1,(512,512)))
        # print(y)
        # cv2.waitKey(0)

        crop = int((x.shape[1] - x.shape[0]) / 2)
        x = x[0:x.shape[0], crop:crop + x.shape[0]]
        # if np.sum(x[crop:x.shape[0] - crop, 0:int(1 / 2 * x.shape[1])]) > np.sum(
        #         x[crop:x.shape[0] - crop, int(1 / 2 * x.shape[1]):x.shape[1]]):
        #     x = x[int(1 / 8 * x.shape[0]):int(7 / 8 * x.shape[0]), int(2 / 5 * x.shape[1]):int(5 / 6 * x.shape[1])]
        # else:
        #     x = x[int(1 / 8 * x.shape[0]):int(7 / 8 * x.shape[0]), int(1 / 6 * x.shape[0]):int(3 / 5 * x.shape[1])]
        #     x = cv2.flip(x, 1)
        # b, g, r = cv2.split(x)
        # x = np.uint8(0.2 * b + 0.2 * r + 0.6 * g)
        # x = cv2.merge((x, x, x))


        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))  # 自适应直方图均衡化

        # planes = cv2.split(x)  # 将图片分为三个单通道，
        # for i in range(0, 3):
        #     # 可能是因为读取到的图片是三通到，而cv2.createCLAHE只能对单通道图片处理
        #     # 所有用cv2.split()将图片变为三个单通道，然后在应用cv2.createCLAHE处理
        #     planes[i] = clahe.apply(planes[i])
        x = clahe.apply(x)

        # planes[0] = np.uint8(0.2 * planes[0])
        # planes[1] = np.uint8(0.8 * planes[1])
        # planes[2] = np.uint8(0.2 * planes[2])
        x = cv2.merge((x,x,x))
        x = cv2.medianBlur(x, 5)

        # if np.sum(x[crop:x.shape[0] - crop, 0:int(1 / 2 * x.shape[1])]) > np.sum(
        #         x[crop:x.shape[0] - crop, int(1 / 2 * x.shape[1]):x.shape[1]]):
        #     x = x[int(1 / 8 * x.shape[0]):int(7 / 8 * x.shape[0]), int(2 / 5 * x.shape[1]):int(5 / 6 * x.shape[1])]
        # else:
        #     x = x[int(1 / 8 * x.shape[0]):int(7 / 8 * x.shape[0]), int(1 / 6 * x.shape[0]):int(3 / 5 * x.shape[1])]
        #     x = cv2.flip(x, 1)

        # cv2.imshow('0',cv2.resize(x,(512,512)))
        # print(y)
        # cv2.waitKey(0)
        x = Image.fromarray(x)
        if self.phase == "train":
            if self.transform is not None:
                x = self.transform(x)
        elif self.phase == "test":
            # img = x
            # x = []
            # for i in range(10):
            #     if self.transform is not None:
            #         x.append(self.transform(img))
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = Datasets(is_color=True,
                       phase='train',
                       transform=transforms.Compose([transforms.RandomResizedCrop(384),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.RandomVerticalFlip(),
                                                           transforms.RandomRotation(30),
                                                           transforms.ToTensor(),
                                                           # normalize,
                                                           ]),
                                        )
    test_dataset = Datasets(is_color=True,
                       phase='test',
                       transform=transforms.Compose([transforms.Resize(384),
                                                     transforms.ToTensor(),
                                                     # normalize,
                                                      ]),
                                  )
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













