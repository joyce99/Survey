# 数据集下载说明 | Dataset Download Instructions

[中文](#中文说明) | [English](#english-instructions)

---

## 中文说明

### 📦 数据集获取

由于数据集文件较大，我们已将完整数据集上传至百度网盘。

**下载链接**: https://pan.baidu.com/s/1Z-a_BvsgqYQYM9SbvPTfgg?pwd=s997

**提取码**: `s997`

### 📥 下载步骤

1. 点击上述链接访问百度网盘
2. 输入提取码: `s997`
3. 下载完整的数据集文件夹
4. 将下载的数据集文件夹解压到本项目的 `data/` 目录下

### 📂 正确的目录结构

下载并解压后，你的项目目录结构应该如下：

```
CDM_survey/
├── data/
│   ├── README.md              # 本文件
│   ├── assist09/              # ASSISTments 2009 数据集
│   │   ├── config.txt
│   │   ├── train.json
│   │   ├── val.json
│   │   └── ...
│   ├── mooper/                # MOOPER 数据集
│   │   ├── config.txt
│   │   ├── train.json
│   │   ├── val.json
│   │   └── ...
│   ├── CSEDM-F/               # CSEDM-F 数据集
│   ├── junyi/                 # Junyi Academy 数据集
│   ├── math1/                 # Math1 数据集
│   ├── NIPS20/                # NIPS 2020 数据集
│   ├── PISA2015/              # PISA 2015 数据集
│   ├── MOOCRadar/             # MOOCRadar 数据集
│   ├── MOOCRadar-middle/      # MOOCRadar-middle 数据集
│   └── Q-free/                # Q-free 数据集
├── main.py
├── params.py
└── ...
```

### ✅ 验证安装

下载完成后，可以通过以下方式验证数据集是否正确放置：

**Windows (PowerShell):**
```powershell
# 检查 assist09 数据集
Test-Path data\assist09\config.txt
Test-Path data\assist09\train.json
Test-Path data\assist09\val.json
```

**Linux/Mac:**
```bash
# 检查 assist09 数据集
ls -la data/assist09/config.txt
ls -la data/assist09/train.json
ls -la data/assist09/val.json
```

如果上述命令都返回 `True` 或显示文件信息，说明数据集放置正确。

### 📊 数据集说明

本项目包含以下数据集：

| 数据集 | 学生数 | 题目数 | 知识点数 | 说明 |
|--------|--------|--------|---------|------|
| ASSISTments 2009 | 4,163 | 17,746 | 123 | 在线数学辅导系统数据 |
| MOOPER | - | - | - | MOOC 编程练习数据 |
| CSEDM-F | - | - | - | 计算机科学教育数据 |
| Junyi Academy | - | - | - | 台湾均一教育平台数据 |
| Math1 | - | - | - | 数学学习数据 |
| NIPS 2020 | - | - | - | NIPS 2020 教育挑战赛数据 |
| PISA 2015 | - | - | - | PISA 2015 评估数据 |
| MOOCRadar | - | - | - | MOOCRadar 数据集 |

### ⚠️ 注意事项

1. **数据集大小**: 完整数据集可能超过 1GB，请确保有足够的存储空间
2. **数据格式**: 所有数据集已经过预处理，格式统一为 JSON
3. **学术使用**: 这些数据集仅供学术研究使用，请遵守相应的使用协议
4. **引用来源**: 使用数据集时，请引用原始数据集的论文

### 🔗 相关链接

- 项目主页: https://github.com/joyce99/Survey
- 问题反馈: https://github.com/joyce99/Survey/issues

---

## English Instructions

### 📦 Dataset Download

Due to the large size of the dataset files, we have uploaded the complete datasets to Baidu Netdisk.

**Download Link**: https://pan.baidu.com/s/1Z-a_BvsgqYQYM9SbvPTfgg?pwd=s997

**Access Code**: `s997`

### 📥 Download Steps

1. Click the link above to access Baidu Netdisk
2. Enter the access code: `s997`
3. Download the complete dataset folders
4. Extract the downloaded datasets to the `data/` directory of this project

### 📂 Correct Directory Structure

After downloading and extracting, your project directory structure should look like:

```
CDM_survey/
├── data/
│   ├── README.md              # This file
│   ├── assist09/              # ASSISTments 2009 Dataset
│   │   ├── config.txt
│   │   ├── train.json
│   │   ├── val.json
│   │   └── ...
│   ├── mooper/                # MOOPER Dataset
│   ├── CSEDM-F/               # CSEDM-F Dataset
│   ├── junyi/                 # Junyi Academy Dataset
│   ├── math1/                 # Math1 Dataset
│   ├── NIPS20/                # NIPS 2020 Dataset
│   ├── PISA2015/              # PISA 2015 Dataset
│   ├── MOOCRadar/             # MOOCRadar Dataset
│   ├── MOOCRadar-middle/      # MOOCRadar-middle Dataset
│   └── Q-free/                # Q-free Dataset
├── main.py
├── params.py
└── ...
```

### ✅ Verify Installation

After downloading, you can verify the datasets are correctly placed:

**Windows (PowerShell):**
```powershell
# Check assist09 dataset
Test-Path data\assist09\config.txt
Test-Path data\assist09\train.json
Test-Path data\assist09\val.json
```

**Linux/Mac:**
```bash
# Check assist09 dataset
ls -la data/assist09/config.txt
ls -la data/assist09/train.json
ls -la data/assist09/val.json
```

If all commands return `True` or show file information, the datasets are correctly placed.

### 📊 Dataset Information

This project includes the following datasets:

| Dataset | Students | Exercises | Concepts | Description |
|---------|----------|-----------|----------|-------------|
| ASSISTments 2009 | 4,163 | 17,746 | 123 | Online math tutoring system data |
| MOOPER | - | - | - | MOOC programming exercises |
| CSEDM-F | - | - | - | Computer science education data |
| Junyi Academy | - | - | - | Taiwan Junyi education platform |
| Math1 | - | - | - | Mathematics learning data |
| NIPS 2020 | - | - | - | NIPS 2020 education challenge |
| PISA 2015 | - | - | - | PISA 2015 assessment data |
| MOOCRadar | - | - | - | MOOCRadar dataset |

### ⚠️ Notes

1. **Dataset Size**: The complete datasets may exceed 1GB. Ensure sufficient storage space.
2. **Data Format**: All datasets are preprocessed and formatted as JSON files.
3. **Academic Use**: These datasets are for academic research only. Please comply with usage agreements.
4. **Citation**: When using datasets, please cite the original dataset papers.

### 🔗 Links

- Project Homepage: https://github.com/joyce99/Survey
- Issue Tracker: https://github.com/joyce99/Survey/issues

---

## 💡 常见问题 | FAQ

### Q: 百度网盘下载速度慢怎么办？
A: 建议使用百度网盘客户端下载，或者考虑开通百度网盘会员以获得更快的下载速度。

### Q: Can I use alternative download methods?
A: Currently, Baidu Netdisk is the primary distribution method. If you need alternative download options, please contact the project maintainers.

### Q: 数据集是否需要额外的预处理？
A: 不需要。所有数据集已经预处理为统一格式，可以直接使用。

### Q: Do the datasets need additional preprocessing?
A: No. All datasets are already preprocessed to a unified format and ready to use.

---

**需要帮助？ | Need Help?**

如果在下载或使用数据集时遇到问题，欢迎在 [GitHub Issues](https://github.com/joyce99/Survey/issues) 中提问。

If you encounter any issues downloading or using the datasets, please ask in [GitHub Issues](https://github.com/joyce99/Survey/issues).

