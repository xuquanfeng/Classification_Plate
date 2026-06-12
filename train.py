import os
import argparse
import yaml

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import swin_tiny_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate


def load_config(config_path):
    """读取 YAML 配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 建立工程目录
    os.makedirs("runs", exist_ok=True)        # TensorBoard 日志目录
    os.makedirs("checkpoints", exist_ok=True) # 模型权重保存目录
    tb_writer = SummaryWriter(log_dir="runs/classification_plate")

    # 1. 载入并合并配置参数
    config = load_config(args.config) if args.config else {}
    data_path = args.data_path or config.get("data_path", "./dataset")
    pretrained = args.pretrained or config.get("pretrained", "")
    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 0.0001)
    num_classes = config.get("num_classes", 2)
    freeze_layers = config.get("freeze_layers", False)
    use_transform = config.get("use_transform", True)
    img_size = config.get("img_size", 224)

    # 2. 数据增强与预处理 (针对特定的 Application Scenario 调整)
    if use_transform:
        data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(int(img_size * 1.143)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.Resize(int(img_size * 1.143)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    else:
        data_transform = {"train": None, "val": None}

    # 3. 实例化数据集
    train_dataset = MyDataSet(datatxt=os.path.join(data_path, 'train.txt'),
                              transform=data_transform["train"])

    val_dataset = MyDataSet(datatxt=os.path.join(data_path, 'test.txt'),
                            transform=data_transform["val"])

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 4. 构建模型与权重加载
    model = create_model(num_classes=num_classes).to(device)

    if pretrained != "":
        assert os.path.exists(pretrained), f"weights file: '{pretrained}' not exist."
        weights_dict = torch.load(pretrained, map_location=device)["model"]
        # 删除有关分类类别的权重以便进行微调
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print("Loaded pretrained weights:", model.load_state_dict(weights_dict, strict=False))

    # 冻结除 head 外的所有层（适用于基础模型后的微调阶段）
    if freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print(f"training {name}")

    # 5. 优化器与训练循环
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=lr, weight_decay=5E-2)

    save_loss = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # 验证阶段
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        # 记录日志
        tags = ["loss/train", "acc/train", "loss/val", "acc/val", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 最佳模型保存策略
        if val_loss < save_loss:
            save_loss = val_loss
            torch.save(model.state_dict(), "./checkpoints/model_best.pth")
            print(f"Epoch {epoch}: Best model saved with val_loss: {save_loss:.4f}")

        # 周期性保存模型 (最后48个epoch中每10个保存一次)
        if epoch % 10 == 9 and epoch >= epochs - 48:
            torch.save(model.state_dict(), f"./checkpoints/model_{epoch}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Astrometric Registration Enhancer Training Script")
    
    # 核心架构参数 (与 README 对齐)
    parser.add_argument('--config', type=str, default='', 
                        help='Path to the yaml configuration file (e.g., configs/base_model.yaml)')
    parser.add_argument('--data_path', type=str, default='', 
                        help='Root directory of the dataset. Overrides config if specified.')
    parser.add_argument('--pretrained', type=str, default='', 
                        help='Path to pre-trained weights for fine-tuning. Overrides config if specified.')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
