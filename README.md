# CDM Survey - 认知诊断模型研究项目

[![GitHub Stars](https://img.shields.io/github/stars/joyce99/Survey?style=social)](https://github.com/joyce99/Survey)
[![GitHub Forks](https://img.shields.io/github/forks/joyce99/Survey?style=social)](https://github.com/joyce99/Survey)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](https://github.com/joyce99/Survey)

> 📍 **GitHub 仓库**: https://github.com/joyce99/Survey

## 📖 项目简介

本项目是一个认知诊断模型(Cognitive Diagnosis Model, CDM)的综合研究实现项目，包含了用于综述实验的数据集和多种认知诊断模型的实现代码。项目集成了20+种主流的认知诊断模型，用于分析学生的学习状态和知识掌握情况。支持多个公开数据集，可用于教育数据挖掘、个性化学习推荐等研究场景。

**This project includes the datasets used in the survey experiments and the implementations of various cognitive diagnosis models.**

### 主要特点

- 🎯 **多模型支持**: 集成了20+种认知诊断模型及其变体
- 📊 **多数据集**: 支持ASSISTments、Junyi、MOOC等多个公开数据集
- 🔧 **易于扩展**: 模块化设计，便于添加新模型和数据集
- 📈 **标准评估**: 统一的评估指标(AUC, Accuracy, RMSE)

## 🚀 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+ (可选，用于GPU加速)

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/joyce99/Survey.git
cd Survey
```

#### 2. 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n cdm python=3.8
conda activate cdm

# 或使用 venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

#### 3. 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn pandas tqdm matplotlib seaborn jupyter
```

**核心依赖包：**
- `torch`: 深度学习框架
- `numpy`: 数值计算
- `scikit-learn`: 机器学习工具（用于评估指标）
- `tqdm`: 进度条显示
- `pandas`: 数据处理（可选）
- `matplotlib`, `seaborn`: 可视化（可选）

#### 4. 下载数据集

由于数据集文件较大，需要单独下载：

**百度网盘下载链接**: https://pan.baidu.com/s/1Z-a_BvsgqYQYM9SbvPTfgg?pwd=s997

**提取码**: `s997`

下载后，将数据集文件夹解压到项目的 `data/` 目录下。详细说明请查看 [`data/README.md`](data/README.md)。

## 📦 项目结构

```
CDM_survey/
├── main.py                 # 主程序入口
├── params.py              # 配置参数文件
├── dataloader.py          # 数据加载器
├── model/                 # 模型实现目录
│   ├── IRT.py            # Item Response Theory
│   ├── MIRT.py / myMIRT.py  # Multidimensional IRT
│   ├── DINA.py           # Deterministic Inputs, Noisy "And" gate
│   ├── NCDM.py           # Neural Cognitive Diagnosis Model
│   ├── myRcd.py          # Relation-aware Cognitive Diagnosis
│   ├── KSCD.py           # Knowledge State Cognitive Diagnosis
│   ├── AGCDM.py          # Attention-based Graph CDM
│   ├── KaNCD.py          # Knowledge-aware Neural CD
│   ├── CACD_adapter.py   # Contrastive Affect-aware CD
│   ├── QCCDM_adapter.py  # Q-matrix Causal CDM
│   ├── MF.py             # Multiple-Strategy Fusion
│   ├── *_Affect.py       # 带情感因素的模型变体
│   ├── *_GS.py           # 带猜测和滑动因素的模型变体
│   └── ICD/              # Incremental Cognitive Diagnosis
├── data/                  # 数据集目录
│   ├── assist09/         # ASSISTments 2009 数据集
│   ├── mooper/           # MOOPER 数据集
│   ├── CSEDM-F/          # CSEDM-F 数据集
│   ├── junyi/            # Junyi Academy 数据集
│   ├── math1/            # Math1 数据集
│   ├── NIPS20/           # NIPS 2020 数据集
│   ├── PISA2015/         # PISA 2015 数据集
│   └── MOOCRadar/        # MOOCRadar 数据集
├── RCD/                   # RCD模型的完整实现
├── example/               # 使用示例
└── result/                # 训练结果输出目录（自动创建）
```

## 🎯 使用方法

### 基本使用流程

#### 1. 配置参数

编辑 `params.py` 文件，选择数据集和设置超参数：

```python
# 选择数据集（取消注释对应的数据集）
dataset = 'data/assist09/'
# dataset = 'data/mooper/'
# dataset = 'data/CSEDM-F/'

# 设置超参数
batch_size = 128      # 批次大小
lr = 0.002           # 学习率
epoch = 100          # 训练轮数

# 设置训练和验证数据文件
src = dataset + 'train.json'
tgt = dataset + 'val.json'
```

配置文件说明：
- `dataset`: 数据集路径
- `batch_size`: 训练批次大小，根据显存调整
- `lr`: 学习率，一般设置为0.001-0.01
- `epoch`: 训练轮数
- `src`: 训练数据文件
- `tgt`: 验证数据文件

#### 2. 运行模型

使用命令行运行指定模型：

```bash
# 运行 NCDM 模型（默认）
python main.py --model NCDM

# 运行 IRT 模型
python main.py --model IRT

# 运行 DINA 模型
python main.py --model DINA

# 运行 RCD 模型
python main.py --model RCD

# 运行 KaNCD 模型
python main.py --model KaNCD
```

#### 3. 查看结果

训练完成后，结果会保存在 `result/` 目录下：

```
result/
├── NCDM.txt       # 模型训练结果
├── IRT.txt
├── DINA.txt
└── ...
```

结果文件包含：
- 最佳轮数 (Best Epoch)
- 准确率 (Accuracy)
- AUC 值
- 训练时间（部分模型）

## 🤖 支持的模型

### 基础模型

| 模型名称 | 说明 | 命令行参数 |
|---------|------|-----------|
| IRT | Item Response Theory，项目反应理论 | `--model IRT` |
| MIRT | Multidimensional IRT，多维项目反应理论 | `--model MIRT` |
| DINA | Deterministic Inputs, Noisy "And" gate | `--model DINA` |
| NCDM | Neural Cognitive Diagnosis Model | `--model NCDM` |
| RCD | Relation-aware Cognitive Diagnosis | `--model RCD` |
| KSCD | Knowledge State Cognitive Diagnosis | `--model KSCD` |

### 高级模型

| 模型名称 | 说明 | 命令行参数 |
|---------|------|-----------|
| AGCDM | Attention-based Graph CDM | `--model AGCDM` |
| KaNCD | Knowledge-aware Neural CD | `--model KaNCD` |
| CACD | Contrastive Affect-aware CD | `--model CACD` |
| QCCDM | Q-matrix Causal CDM | `--model QCCDM` |
| MF | Multiple-Strategy Fusion | `--model MF` |

### 模型变体

| 模型名称 | 说明 | 命令行参数 |
|---------|------|-----------|
| NCDM_GS | NCDM + 猜测与滑动因素 | `--model NCDM_GS` |
| RCD_GS | RCD + 猜测与滑动因素 | `--model RCD_GS` |
| KSCD_GS | KSCD + 猜测与滑动因素 | `--model KSCD_GS` |
| IRT_Affect | IRT + 情感因素 | `--model IRT_Affect` |
| MIRT_Affect | MIRT + 情感因素 | `--model MIRT_Affect` |
| DINA_Affect | DINA + 情感因素 | `--model DINA_Affect` |
| RCD_Affect | RCD + 情感因素 | `--model RCD_Affect` |
| MF_Affect | MF + 情感因素 | `--model MF_Affect` |

## 📊 数据格式

### 数据集目录结构

每个数据集目录应包含以下文件：

```
data/your_dataset/
├── config.txt          # 配置文件：学生数、题目数、知识点数
├── train.json         # 训练数据
└── val.json           # 验证数据
```

### config.txt 格式

```
# Number of Students, Number of Exercises, Number of Knowledge Concepts
4163, 17746, 123
```

第一行为注释，第二行为三个整数：学生数、题目数、知识点数

### JSON 数据格式

训练和验证数据采用 JSON 格式，每条记录包含：

```json
[
    {
        "user_id": 1,
        "exer_id": 100,
        "knowledge_code": [3, 5, 12],
        "score": 1
    },
    {
        "user_id": 2,
        "exer_id": 101,
        "knowledge_code": [1, 8],
        "score": 0
    }
]
```

字段说明：
- `user_id`: 学生ID（从1开始）
- `exer_id`: 习题ID（从1开始）
- `knowledge_code`: 习题涉及的知识点ID列表（从1开始）
- `score`: 答题结果，1表示正确，0表示错误

### 使用自己的数据集

1. 在 `data/` 目录下创建新文件夹
2. 准备 `config.txt` 和数据文件
3. 在 `params.py` 中设置数据集路径
4. 运行模型

## 🔧 高级使用

### 修改模型参数

不同模型支持不同的初始化参数，可以在 `main.py` 对应的模型函数中修改：

```python
def NCDM_main():
    # 可以在这里修改模型的初始化参数
    cdm = NCDM.NCDM(params.kn, params.en, params.un)
    e, auc, acc = cdm.train(
        train_data=src, 
        test_data=tgt, 
        epoch=params.epoch,  # 可以修改训练轮数
        device=device, 
        lr=params.lr  # 可以修改学习率
    )
```

### IRT 模型变体选择

IRT 模型支持 1-PL、2-PL、3-PL 三种参数模型。修改方法：

编辑 `model/IRT.py` 文件的 `irf` 函数（第12行），注释/取消注释对应的 return 语句：

```python
def irf(self, theta, a, b, c, D=1.702):
    # 3-PL 模型
    # return c + (1 - c) / (1 + F.exp(-D * a * (theta - b)))
    
    # 2-PL 模型（当前使用）
    return 1 / (1 + F.exp(-D * a * (theta - b)))
    
    # 1-PL 模型
    # return 1 / (1 + F.exp(-D * (theta - b)))
```

### GPU 设置

在 `main.py` 中修改设备设置：

```python
# 使用 CPU
device = 'cpu'

# 使用第一块 GPU
device = 'cuda:0'

# 使用第二块 GPU
device = 'cuda:1'
```

### 批量实验

创建 shell 脚本批量运行多个模型：

```bash
# run_all.sh
#!/bin/bash
for model in IRT MIRT DINA NCDM RCD KSCD
do
    echo "Running $model..."
    python main.py --model $model
done
```

Windows PowerShell:
```powershell
# run_all.ps1
$models = @("IRT", "MIRT", "DINA", "NCDM", "RCD", "KSCD")
foreach ($model in $models) {
    Write-Host "Running $model..."
    python main.py --model $model
}
```

## 📈 评估指标

所有模型统一使用以下评估指标：

- **AUC (Area Under Curve)**: ROC曲线下的面积，范围[0,1]，越大越好
- **Accuracy**: 预测准确率，范围[0,1]，越大越好
- **RMSE (Root Mean Square Error)**: 均方根误差，部分模型输出，越小越好

## 🐛 常见问题

### 1. CUDA out of memory

**问题**: 显存不足

**解决方案**:
- 减小 `params.py` 中的 `batch_size`
- 使用CPU训练（设置 `device='cpu'`）
- 使用更小的数据集

### 2. 找不到数据文件

**问题**: `FileNotFoundError: [Errno 2] No such file or directory`

**解决方案**:
- 检查 `params.py` 中的 `dataset` 路径是否正确
- 确保数据集目录包含 `config.txt`、`train.json`、`val.json`
- 检查文件路径的斜杠方向（Windows使用 `\` 或 `/`）

### 3. 模型训练不收敛

**问题**: AUC/Accuracy 始终很低

**解决方案**:
- 增加训练轮数 `epoch`
- 调整学习率 `lr`（增大或减小）
- 检查数据集格式是否正确
- 尝试不同的模型

### 4. 某些模型报错

**问题**: 特定模型运行时出错

**解决方案**:
- 检查该模型的特殊依赖是否已安装
- 查看模型源代码中的注释说明
- 尝试其他基础模型确认环境配置正确

## 📚 数据集说明

### 📥 数据集下载

**重要**: 由于数据集文件较大，需要从百度网盘下载。

- **下载链接**: https://pan.baidu.com/s/1Z-a_BvsgqYQYM9SbvPTfgg?pwd=s997
- **提取码**: `s997`
- **详细说明**: 查看 [`data/README.md`](data/README.md)

项目包含多个公开数据集：

### ASSISTments 2009 (`assist09`)
- 学生数: 4,163
- 题目数: 17,746  
- 知识点数: 123
- 来源: 在线数学辅导系统

### MOOPER (`mooper`)
- MOOC 平台编程练习数据
- 包含编程题目的学生答题记录

### Junyi Academy (`junyi`)
- 来自台湾均一教育平台
- 包含多学科学习记录

### 其他数据集
- `CSEDM-F`: 计算机科学教育数据
- `math1`: 数学学习数据
- `NIPS20`: NIPS 2020 教育数据挑战
- `PISA2015`: PISA 2015 评估数据
- `MOOCRadar`: MOOC 雷达数据集

## 🔄 更新日志

本项目持续更新中，我们会逐步完善：

- ✅ 多种认知诊断模型实现
- ✅ 标准化数据集格式
- ✅ 统一的训练和评估接口
- 🔄 更多模型的支持
- 🔄 更详细的实验结果
- 🔄 模型可视化工具

## 🤝 贡献

欢迎贡献新的模型实现、数据集支持或bug修复！

### 如何贡献

1. Fork 本仓库到你的 GitHub 账号
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

### 添加新模型

1. 在 `model/` 目录下创建新的模型文件
2. 实现模型类，包含 `__init__` 和 `train` 方法
3. 在 `main.py` 中添加模型函数
4. 在命令行参数中注册模型
5. 提交 Pull Request 并描述你的模型

### 添加新数据集

1. 在 `data/` 目录下创建数据集文件夹
2. 准备符合格式的 `config.txt`、`train.json`、`val.json`
3. 在 `params.py` 中添加数据集路径选项
4. 更新 README 中的数据集列表

## 📄 许可证

本项目仅用于学术研究目的。

## 📧 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 提交 [GitHub Issue](https://github.com/joyce99/Survey/issues)
- 发送 [Pull Request](https://github.com/joyce99/Survey/pulls)
- 访问项目主页: https://github.com/joyce99/Survey

## 🙏 致谢

感谢以下开源项目和数据集提供者：

- PyTorch 深度学习框架
- ASSISTments 数据集
- Junyi Academy
- 各个认知诊断模型的原作者

## 📖 参考文献

如果您在研究中使用了本项目，请引用相关的模型论文以及本项目：

```bibtex example
@misc{cdm_survey_2025,
  title={ Survey of Cognitive Diagnosis in intelligent education: Theory, Methods and Experiments},
  author={Yuhong Zhang and Shengyu Xu},
  year={2025},
  howpublished={\url{https://github.com/joyce99/Survey}},
  note={GitHub repository}
}
```

## 🌟 Star History

如果这个项目对你有帮助，请给我们一个 ⭐ Star！

[![Star History Chart](https://api.star-history.com/svg?repos=joyce99/Survey&type=Date)](https://star-history.com/#joyce99/Survey&Date)

---

**最后更新**: 2025

**项目维护**: [@joyce99](https://github.com/joyce99), [@xushengyu1](https://github.com/xushengyu1)

**开发环境**: Python 3.8, PyTorch 1.13, CUDA 11.8

**项目主页**: https://github.com/joyce99/Survey


