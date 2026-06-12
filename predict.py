import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision
from skimage import io
from model import swin_tiny_patch4_window7_224 as create_model
from my_dataset import MyDataSet
import time

def print_acc(source, name):
    # Hyper parameters
    batch_size = 64    #每次投喂数据量
    # root = '/home/yaowei/lenovo_1/xqf/data/'#数据集的文件夹
    # test_data = MyDataSet(datatxt=root + 'test.txt', transform=None)
    root = source
    test_data = MyDataSet(datatxt=root +"/"+ name +'.txt', transform=None)
    # test_data = MyDataset(datatxt=root + 'train.txt', transform=None)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = batch_size, shuffle=False,num_workers=20)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    net = create_model(num_classes=2).to(device)
    weights_dict = torch.load("./swint/weights/model_best.pth", map_location=device)
    net.load_state_dict(weights_dict, strict=False)

    test_acc = 0.
    net.eval()
        # with torch.no_grad():
    for batch_x, batch_y in tqdm(test_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        out = net(batch_x)
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        test_acc += num_correct.item()
    asd = 'Acc: {:.6f}'.format(test_acc / (len(test_data)))
    print(asd)