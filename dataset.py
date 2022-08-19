import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image


class HorseDataset(Dataset):
    def __init__(self,root,index,resize=None):
        super(HorseDataset,self).__init__()
        self.root = root
        self.index=index
        self.resize=resize
        self.imgs = list(np.array(list(sorted(os.listdir(os.path.join(root, "horse")))))[index])
        self.masks = list(np.array(list(sorted(os.listdir(os.path.join(root, "mask")))))[index])
    
    def __getitem__(self, idx):
        # 载入图片
        img_path = self.root + '/' + 'horse' + '/' + self.imgs[idx]
        mask_path = self.root + '/' + 'mask' + '/' + self.masks[idx]

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        if self.resize is not None:
            img, mask = self.resize(img, mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)

#改变图片大小
class RESIZE(object):
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, img, mask):
        img = transforms.Resize((self.image_size, self.image_size), interpolation=Image.BILINEAR)(img)
        mask = transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)(mask)
        return img, mask

#像素归一化
class Totensor(object): 
    def __call__(self, img, mask):
        img = transforms.ToTensor()(img)
        mask = torch.from_numpy(np.array(mask))
        if not isinstance(mask, torch.LongTensor):
            mask = mask.float()
        return img, mask

#对houres和mask做处理
class Transform_Compose(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, mask):
        for trans in self.transform:
            img, mask = trans(img, mask)
        return img, mask
    
#划分训练集和测试集
def dataloder(root,batchsize,imagesize,ratio):
    idx=np.arange(327)
    a = int(327*ratio)
    np.random.shuffle(idx)
    train_idx = idx[:327-a]
    test_idx = idx[327-a:]
    train_transforms = Transform_Compose([RESIZE(image_size=imagesize),Totensor()])
    test_transforms = Transform_Compose([RESIZE(image_size=imagesize),Totensor()])
    
    train_data = HorseDataset(root,train_idx,train_transforms)
    test_data =  HorseDataset(root,test_idx,test_transforms)

    Train_data = DataLoader(train_data, shuffle = True, batch_size = batchsize, num_workers = 0, pin_memory = True)
    Test_data = DataLoader(test_data, shuffle = False, batch_size = batchsize, num_workers = 0, pin_memory = True)
    
    return Train_data,Test_data
