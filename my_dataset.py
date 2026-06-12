import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from astropy.io import fits
from PIL import Image
from skimage import io

class MyDataSet(Dataset):
    """
    统一的自定义数据集
    支持 'train' (读取预先切好的tensor) 和 'inference' (FITS原图按坐标滑动窗口切图) 两种模式
    """
    def __init__(self, data_source, mode='train', transform=None, size=(64, 64)):
        """
        :param data_source: 在 'train' 模式下，是包含路径和标签的 txt 文件路径；
                            在 'inference' 模式下，是包含 [labpath, fitpath, name] 的列表。
        :param mode: 'train' 或 'inference'
        :param transform: 数据增强或预处理
        :param size: 推理切图时的目标尺寸，默认 (64, 64)
        """
        self.mode = mode
        self.transform = transform
        self.size = size
        
        if self.mode == 'train':
            # ================= 训练模式初始化 =================
            imgs = []
            labs = []
            with open(data_source, 'r') as fh:
                for line in fh:
                    line = line.rstrip()
                    if not line:
                        continue
                    words = line.split()
                    imgs.append(words[0])
                    labs.append(int(words[1]))
            self.images_path = imgs
            self.images_class = labs
            
        elif self.mode == 'inference':
            # ================= 推理/滑动窗口模式初始化 =================
            self.num = []
            self.datt = []
            self.sum = 0
            self.imageData = []
            self.error = []
            
            for i, datatxt in enumerate(data_source):
                labpath, fitpath, name = datatxt
                data = []
                with open(labpath, "r") as f:
                    f.readline() # 跳过表头
                    for line in f.readlines():
                        line = line.strip('\n')
                        if not line:
                            continue
                        data.append(line.split())
                
                datt = np.array(data).astype('float')
                self.sum += len(datt)
                self.num.extend([i] * len(datt))
                
                if i == 0:
                    self.datt = datt
                else:
                    self.datt = np.vstack((self.datt, datt))
                    
                self.imageData.append(fits.getdata(fitpath, ext=0))
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Please use 'train' or 'inference'.")

    def __len__(self):
        if self.mode == 'train':
            return len(self.images_path)
        elif self.mode == 'inference':
            return self.sum

    def __getitem__(self, item):
        if self.mode == 'train':
            # ================= 训练模式加载 =================
            img = torch.load(self.images_path[item])        
            img = img.repeat(3, 1, 1) # 单通道转三通道
            label = self.images_class[item]

            if self.transform is not None:
                img = self.transform(img)
            return img, label
            
        elif self.mode == 'inference':
            # ================= 推理模式切图 =================
            x, y = self.datt[item][:2]
            xmin = int(x) - int(self.size[0] / 2)
            ymin = int(y) - int(self.size[1] / 2)
            xmax = int(x) + int(self.size[0] / 2)
            ymax = int(y) + int(self.size[1] / 2)

            imageData = self.imageData[self.num[item]]
            sliceImage = imageData[int(ymin):int(ymax), int(xmin):int(xmax)]
            
            try:
                # 归一化与缩放
                minn = np.min(sliceImage)
                sliceImage -= minn
                maxx = np.max(sliceImage)
                if maxx != 0:
                    sliceImage = sliceImage / maxx
                
                sliceImage = sliceImage[None, :, :]
                img = torch.from_numpy(sliceImage).float()
                img *= maxx
                img += minn
                
                # 尺寸校验与插值 (处理边界切图尺寸不足的情况)
                if img.shape[1] != self.size[0] or img.shape[2] != self.size[1]:
                    img = img.unsqueeze(0)
                    img = F.interpolate(img, size=self.size[0], mode='bilinear', align_corners=False)
                    img = img.squeeze(0)  
                
                img = img.repeat(3, 1, 1)
                label = self.datt[item][-1]
                
                if self.transform is not None:
                    img = self.transform(img)
                    
                return img, label
                
            except Exception as e:
                # 遇到边缘截断或其他错误时跳过，加载上一个数据
                self.error.append(item)
                return self.__getitem__(item - 1)

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        # 确保 label 是 long 类型以匹配交叉熵损失函数
        labels = labels.type(torch.long)
        return images, labels
