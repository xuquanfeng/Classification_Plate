import os
import numpy as np
from tqdm import tqdm
from astropy.io import fits

from functools import partial
import multiprocessing

import torch
import torch.nn.functional as F
from predict import print_acc

label = np.load('/home/yaowei/xqf/classify/label.npy')
# print(label)

source = '/home/yaowei/lenovo_1/xqf/temp'

size = (64,64)
i = 310

labpath,fitpath,nn = label[i]

print(label[i])
assert '/BJ' in labpath
# print(labpath)
data = []
with open(labpath, "r") as f:  # 打开文件
    f.readline()
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        data.append(line.split())
data = np.array(data).astype('float')

if not os.path.exists(source):
    os.mkdir(source)
imageData = fits.getdata(fitpath, ext=0)

def do(j):
    x,y = data[j][:2]
    xmin,ymin,xmax,ymax = int(x)-int(size[0]/2),int(y)-int(size[1]/2),int(x)+int(size[0]/2),int(y)+int(size[1]/2)
    sliceImage = imageData[int(ymin):int(ymax),int(xmin):int(xmax)]
    savepath = source + '/'+nn
    try:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
    except:
        pass
    savename = str(int(j))+".pth"
    if os.path.exists(savepath+"/"+savename):
        return 0
    try:
        minn = np.min(sliceImage)
        sliceImage -=  np.min(sliceImage)
        maxx = np.max(sliceImage)
        sliceImage = sliceImage/maxx
        sliceImage = sliceImage[None,:,:]
        img = torch.from_numpy(sliceImage).float()
        img *= maxx
        img += minn
    except:
        pass
    if img.shape[1] != size[0] or img.shape[2] != size[1]:
        img = img.unsqueeze(0)
        img = F.interpolate(img, size=size[0], mode='bilinear',align_corners=False)
        img = img.squeeze(0)
    torch.save(img, savepath+"/"+savename)
    train = open(source+'/'+nn+'.txt', 'a+')
    train.write(savepath+"/"+savename+' '+str(int(data[j][-1]))+ '\n')
    train.close()

print(len(data))
# exist = [i for i in range(len(data))]
exist = [i for i in range(50000)]
p = multiprocessing.Pool(10)
p.map(partial(do), exist)
p.close()
p.join()

print_acc(source, nn)
