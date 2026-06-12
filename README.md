# 🌟 Classification_Plate

**基于深度学习的中国历史天文数字底片天体测量配准增强** *(Enhancing astrometric registration of Chinese historical Astronomical Digital Plates with deep learning)*

[![Paper](https://img.shields.io/badge/Paper-RAA_2026-blue)](https://doi.org/10.1088/1674-4527/ae5f6a)
[![arXiv](https://img.shields.io/badge/arXiv-2604.04714-b31b1b)](https://arxiv.org/abs/2604.04714)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://xuquanfeng.github.io/AI_Plate/)

本仓库包含了发表于《Research in Astronomy and Astrophysics》(RAA) 2026 的论文 **"Enhancing astrometric registration of Chinese historical Astronomical Digital Plates with deep learning"** 的官方 PyTorch 实现代码。

> **作者**：徐权峰 (Quanfeng Xu), 商正君, 沈世银*, 于涌*, 杨美婷*, 罗浩, 杨静, 唐正宏, 赵建海  
> **单位**：中国科学院上海天文台 (SHAO) / 中国科学院大学

---

## 📖 简介 (Introduction)

自 1900 年以来，中国天文学家收集了大量的夜空天文底片，形成了庞大的历史数据库。这些底片记录了跨越一个多世纪的超新星爆发、小行星轨迹等天象，是长周期时域天文学研究的无价之宝。

为了确定这些百年记录的确切坐标，通常需要先使用 `SExtractor` 提取底片上的源，然后使用现代星表（如 Gaia 星表）进行几何匹配（天体测量配准）。然而，由于早期的存储条件和时间带来的环境退化，许多数字底片布满了霉斑、划痕甚至发生了物理断裂。传统的算法会将霉斑、划痕等噪声与真实的恒星点源混合，导致几何匹配系统崩溃，数千张珍贵的数字底片因此处理失败。

本项目提出了一个**基于 Transformer 的图像分类模型（Swin Transformer）**，利用多尺度特征融合技术，有效地将真实的恒星点源从环境噪声中区分出来。这一 AI 增强流水线极大地提升了退化严重的历史底片的科学价值。

## 🚀 主要成果 (Key Results)

在传统算法无法成功配准的 **1,883** 张受损底片上，我们的模型表现出了对环境退化的极强鲁棒性：

* 成功完成了 **1,353** 张底片的天体测量配准。
* 在严重退化的数据上取得了近 **72%** 的总体成功率。
* 即使在可见霉斑、划痕或轻微脱落的底片（2 级和 3 级底片）上，也保持了稳健的特征提取能力。

## 🛠️ 方法与模型架构 (Methodology)

为了在各类异构数据集的实际应用场景中优化分类性能，我们实施了**两阶段元数据引导的训练策略**（Two-stage metadata-guided training strategy）：

1. **基础模型 (Base Model)**：首先在保存良好的底片上进行训练，建立强大的恒星识别基线能力。
2. **微调模型 (Fine-Tuned Models)**：结合望远镜孔径和不同天文台曝光条件等物理参数，专门针对遭受严重退化的底片进行定制化微调。

这种符合天文物理规律的设计，使得模型在复杂的实际**应用场景 (Application Scenario)** 下具备了极高的可靠性。

## 💻 快速开始 (Quick Start)

### 1. 环境依赖 (Dependencies)
本代码基于 Python 3.9+ 及高版本 PyTorch。
```bash
git clone [https://github.com/xuquanfeng/Classification_Plate.git](https://github.com/xuquanfeng/Classification_Plate.git)
cd Classification_Plate
pip install -r requirements.txt
```

### 2. 数据准备 (Data Preparation)

需要使用 `SExtractor` 提取底片原始切图，建议按如下格式组织目录：

```text
dataset/
├── train/
│   ├── stars/
│   └── artifacts/
└── val/
    ├── stars/
    └── artifacts/

```

### 3. 模型训练 (Training)

训练基础模型（Base Model）：

```bash
python train.py --config configs/base_model.yaml --data_path ./dataset

```

基于元数据引导的微调训练（Fine-tuning）：

```bash
python train.py --config configs/finetune_model.yaml --pretrained ./checkpoints/base.pth

```

### 4. 模型推理与底片清洗 (Inference)

运行分类器处理新的 SExtractor 提取结果，过滤掉霉斑和划痕，并将清洗后的高置信度目录输出以供下游匹配模块使用：

```bash
python inference.py --input_catalog raw.cat --image_dir ./images --output_catalog clean.cat

```

## 📝 引用 (Citation)

如果您在研究中使用了我们的代码、模型或思路，请引用我们的工作：

```bibtex
@article{xu2026,
      title={Enhancing astrometric registration of Chinese historical Astronomical Digital Plates with deep learning}, 
      author={Quanfeng Xu and Zhengjun Shang and Shiyin Shen and Yong Yu and Meiting Yang and Hao Luo and Zhenghong Tang and Jing Yang and Jianhai Zhao},
      journal={Research in Astronomy and Astrophysics},
      url={[http://iopscience.iop.org/article/10.1088/1674-4527/ae5f6a](http://iopscience.iop.org/article/10.1088/1674-4527/ae5f6a)},
      year={2026}
}

```

## ✉️ 协议与联系方式 (Contact)

* 本项目部分代码参考了开源社区，非常感谢！
* 欢迎探讨交流。如有任何技术问题或建议，欢迎在 GitHub 提交 Issue 进行讨论。

```

```
