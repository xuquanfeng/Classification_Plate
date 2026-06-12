import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import swin_tiny_patch4_window7_224 as create_model
from my_dataset import MyDataSet

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. 构建数据集输入格式
    # 对应 MyDataSet 推理模式的 data_source: [(labpath, fitpath, name)]
    data_list = [(args.input_catalog, args.image_path, "inference_plate")]
    
    print(f"[*] 加载原始星表: {args.input_catalog}")
    print(f"[*] 加载底片原图: {args.image_path}")
    
    # 注意：这里调用了上一步更新好的 MyDataSet，指定 mode='inference'
    dataset = MyDataSet(data_source=data_list, mode='inference', transform=None)
    
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             num_workers=nw)

    # 2. 加载模型与权重
    print(f"[*] 正在加载微调模型权重: {args.weights} ...")
    model = create_model(num_classes=args.num_classes).to(device)
    
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"权重文件不存在: {args.weights}")
        
    weights_dict = torch.load(args.weights, map_location=device)
    # 兼容是否带有 'model' key 的字典格式
    if "model" in weights_dict:
        weights_dict = weights_dict["model"]
    model.load_state_dict(weights_dict, strict=False)
    model.eval()

    all_preds = []
    all_labels = []

    # 3. 开始批量推理
    print("[*] 开始进行底片源清洗 (Sliding Window Inference)...")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Processing"):
            batch_x = batch_x.to(device)
            out = model(batch_x)
            pred = torch.max(out, 1)[1]
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.numpy())  # 若用于纯推理，这里的 label 是占位符或原始表里的列

    # 4. (可选) 打印评估指标 
    # 仅当传入了 --evaluate 参数，且原始星表里附带了真实标签时使用
    if args.evaluate:
        acc = accuracy_score(all_labels, all_preds)
        p = precision_score(all_labels, all_preds, zero_division=0)
        r = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print("\n" + "="*30)
        print("📊 验证集评估指标 (Evaluation Metrics)")
        print("="*30)
        print(f"Accuracy  : {acc:.6f}")
        print(f"Precision : {p:.6f}")
        print(f"Recall    : {r:.6f}")
        print(f"F1 Score  : {f1:.6f}")
        print(f"切图边缘越界错误数: {len(dataset.error)}")
        print("="*30 + "\n")

    # 5. 过滤并保存干净的星表
    # 假设预测值 pred == 1 代表真实的恒星源 (Stars)，0 代表霉斑/划痕/伪源 (Artifacts)
    all_preds = np.array(all_preds)
    original_data = dataset.datt  # 获取原始提取的完整 numpy 矩阵
    
    clean_mask = (all_preds == 1)
    clean_catalog = original_data[clean_mask]
    
    print("\n" + "="*30)
    print("🧹 底片清洗结果 (Cleaning Results)")
    print("="*30)
    print(f"提取总源数 (Raw)      : {len(original_data)}")
    print(f"清洗后源数 (Cleaned)  : {len(clean_catalog)}")
    print(f"剔除伪源数 (Artifacts): {len(original_data) - len(clean_catalog)}")
    print("="*30)
    
    # 将清洗后的数据写回新文件
    # 尝试保留原文件的表头 (Header)
    header_str = ""
    with open(args.input_catalog, "r") as f:
        first_line = f.readline()
        if first_line.startswith("#") or not first_line.split()[0].replace('.', '', 1).isdigit():
            header_str = first_line.strip()
            
    np.savetxt(args.output_catalog, clean_catalog, fmt='%f', header=header_str, comments='')
    print(f"\n[+] 成功！高置信度星表已保存至: {args.output_catalog}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plate-YOLO / Swin Transformer Inference & Cleaning Script")
    
    # 核心 IO 参数
    parser.add_argument('--input_catalog', type=str, required=True, 
                        help='输入的原始 SExtractor 提取星表 (.cat)')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='原始 FITS 底片图像路径 (.fits)')
    parser.add_argument('--output_catalog', type=str, required=True, 
                        help='输出的清洗后高置信度星表路径 (.cat)')
    
    # 模型与系统参数
    parser.add_argument('--weights', type=str, default='./checkpoints/model_best.pth', 
                        help='训练好的模型权重路径')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    
    # 调试与验证
    parser.add_argument('--evaluate', action='store_true', 
                        help='开启此项以计算精确度、召回率和 F1 分数（前提是星表中包含真值标签）')
    
    opt = parser.parse_args()
    main(opt)
