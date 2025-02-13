from torch.utils.data import Dataset,DataLoader
import os
import scipy.io as scio
import PIL.Image as Image
from torchvision import transforms
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
class MyDataset(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, folder,label_duiying,transform=None):
        label = label_duiying
        self.dictlabel={}
        for i in range(0,label.shape[0]):
            # self.dictlabel.setdefault(label[i][0][0][0][0],label[i][1][0][0]) yuan lai
            name=str(label[i][0])+'.jpg'
            self.dictlabel.setdefault(name, label[i][1])
        # label=one_hot(10,label[:,1])
        self.folder = folder
        self.transform = transform
        self.label = label
        self.axial = os.listdir(self.folder)




    def __getitem__(self, index):
        axial_index = self.axial[index]
        label = self.dictlabel[axial_index]
        axial_path = os.path.join(self.folder,axial_index)
        axial = Image.open(axial_path)


        if self.transform is not None:
            # axial = self.transform(axial)
            axial,label = self.transform(axial,label)
        return axial,label


    def __len__(self):
        return len(self.axial)

myTransforms = transforms.Compose([transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp


def transform(data, label):
    label = one_hot(10, label)
    resize=transforms.Resize(size=(64,64))
    totensor=transforms.ToTensor()
    data=resize(data)
    data=totensor(data)
    return data, label


#源域训练数据构建
Scouce_train_path = r''
# label_path = r''
df_source_train = pd.read_excel()
label_duiying_source_train = df_source_train.values
# 获取第二列数据，并转化为数组
Scouce_trainDataset = MyDataset(Scouce_train_path,label_duiying_source_train,transform=transform)
Scouce_train_loader = DataLoader(
    dataset=Scouce_trainDataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
)

#目标域训练数据构建
Target_train_path = r''
# label_path = r''
df_target_train = pd.read_excel()
label_duiying_target_train = df_target_train.values
# 获取第二列数据，并转化为数组
Target_trainDataset = MyDataset(Target_train_path,label_duiying_target_train,transform=transform)
Target_train_loader = DataLoader(
    dataset=Target_trainDataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
)



for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(
        zip(Scouce_train_loader, Target_train_loader)):
    if torch.cuda.is_available(): # Move images and labels to gpu if available
        im_source = Variable(im_source).cuda()
        label_source = Variable(label_source).cuda()
        im_target = Variable(im_target).cuda()
print('finish')
