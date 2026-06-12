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

aa = []
ab = -1
ac = []
for fe in label:
    ab +=1
    labpath,fitpath,nn = fe
    name = fitpath.split('/')
    if name[-3] not in aa:
        aa.append(name[-3])
        ac.append(ab)
for fe in range(len(aa)):
    print(aa[fe],"index is:",ac[fe])

# print(label[300])
# assert '/BJ' in labpath
# # print(labpath)

def deal_data(lll):
    labpath,fitpath,nn = lll
    source = '/home/yaowei/lenovo_1/xqf/temp'
    size = (64,64)
    data = []
    with open(labpath, "r") as f:  # 打开文件
        f.readline()
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            data.append(line.split())
    data = np.array(data).astype('float')

    # a = data[data[:,-1]>0.5]
    # a1 = np.arange(len(a))
    # np.random.shuffle(a1)
    # data1 = a[a1[:num]]
    # a = data[data[:,-1]<0.5]
    # a1 = np.arange(len(a))
    # np.random.shuffle(a1)
    # data2 = a[a1[:num]]
    # data = np.r_[data1,data2]
    if not os.path.exists(source):
        os.mkdir(source)
    imageData = fits.getdata(fitpath, ext=0)
    for j in tqdm(range(len(data))):
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
        if j < 100:
            if np.abs(len(os.listdir(savepath))-len(data))<100:
                break
        if os.path.exists(savepath+"/"+savename):
            continue
        # if j==52000:
        #     break
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
            continue
        if img.shape[1] != size[0] or img.shape[2] != size[1]:
            img = img.unsqueeze(0)
            img = F.interpolate(img, size=size[0], mode='bilinear',align_corners=False)
            img = img.squeeze(0)
        torch.save(img, savepath+"/"+savename)
        if j == 0:
            train = open(source+'/'+nn+'.txt', 'w')
            train.write(savepath+"/"+savename+' '+str(int(data[j][-1]))+ '\n')
            train.close()
        else:
            train = open(source+'/'+nn+'.txt', 'a+')
            train.write(savepath+"/"+savename+' '+str(int(data[j][-1]))+ '\n')
            train.close()
    return nn
# i = 300
# i = 310
i = 310
# i = 11310
# name = deal_data(label[i])
# print_acc('/home/yaowei/lenovo_1/xqf/temp', name)

ran = [i+300 for i in range(11)]
for fe in range(len(aa)-1):
    for i in range(11):
        ran.append(ac[fe+1]+i)
print(ran)

for i in ran:
    name = deal_data(label[i])
    print(i,name)
    print_acc('/home/yaowei/lenovo_1/xqf/temp', name)
