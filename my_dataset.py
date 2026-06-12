from PIL import Image
import torch
from torch.utils.data import Dataset
from skimage import io
    
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, datatxt, transform=None):
        fh = open(datatxt, 'r')
        imgs = []
        labs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append(words[0])
            labs.append(int(words[1]))
        self.images_path = imgs
        self.images_class = labs
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # img = io.imread(self.images_path[item])
        img = torch.load(self.images_path[item])        
        img = img.repeat(3, 1, 1)#.numpy()
        # img = Image.fromarray(img).convert('RGB')
        # img = img.resize((256,256))
        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels