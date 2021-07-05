import os
import cv2
import xlrd
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import torch


def getDataloader(path = './images', batchsize=128, img_size=224):
    data = np.zeros((1,3,img_size,img_size),dtype=np.float32)
    data_name = []
    target = []
    imagDict = {}
    for dir in os.listdir(path):
        if os.path.isdir(path+'/'+dir):
            image_dir = path + '/'+ dir
            for image in os.listdir(image_dir):
                image_path = image_dir + '/' + image
                if image[-3:] == 'xls':
                    book = xlrd.open_workbook(image_path)
                    sheet = book.sheet_by_index(0)
                    image_name = sheet.col_values(0)[1:]
                    Retinopathy_grade = sheet.col_values(2)[1:]
                    for i,j in zip(image_name, Retinopathy_grade):
                        imagDict[i] = j
                else:
                    x = cv2.imread(image_path)
                    x = cv2.resize(x, (img_size, img_size))
                    x = x / 255
                    x = x.astype(np.float32)
                    x = np.expand_dims(x, axis=0)
                    x = np.swapaxes(x, 2, 3)
                    x = np.swapaxes(x, 1, 2)
                    data = np.concatenate([data, x], axis=0)
                    data_name.append(image)
            for i in data_name:
                target.append(imagDict[i])
            data_name = []
            imagDict = {}

    target = np.array(target,dtype=np.int64)
    #打乱数据集
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(target)

    train = TensorDataset(torch.from_numpy(data[1:1001]), torch.from_numpy(target[:1000]))
    train_DL = DataLoader(train, batch_size=batchsize, shuffle=True, drop_last=True)
    val = TensorDataset(torch.from_numpy(data[1001:]), torch.from_numpy(target[1000:]))
    val_DL = DataLoader(val, batch_size=batchsize, shuffle=True, drop_last=True)


    return train_DL, val_DL

if __name__ == '__main__':
    # x, y = getDataloader()
    # for (image, target) in x:
    #     print(image.size(), target)
    i = 0
    acc1 = torch.tensor([1])
    # print("epoch:" + str(i) + 'train Acc@1 {:.3f}'.format(acc1.data))
    print(acc1.data.item())





