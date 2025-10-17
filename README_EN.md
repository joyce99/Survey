# CDM Survey - Cognitive Diagnosis Model Research Project

[![GitHub Stars](https://img.shields.io/github/stars/joyce99/Survey?style=social)](https://github.com/joyce99/Survey)
[![GitHub Forks](https://img.shields.io/github/forks/joyce99/Survey?style=social)](https://github.com/joyce99/Survey)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](https://github.com/joyce99/Survey)

[中文版](README.md) | English

> 📍 **GitHub Repository**: https://github.com/joyce99/Survey

## 📖 Introduction

This project is a comprehensive research implementation of Cognitive Diagnosis Models (CDMs), including datasets used in survey experiments and implementations of various CDM models. It implements 20+ mainstream cognitive diagnosis models for analyzing student learning states and knowledge mastery. It supports multiple public datasets and can be used for educational data mining and personalized learning recommendation research.

### Key Features

- 🎯 **Multiple Models**: 20+ cognitive diagnosis models and variants
- 📊 **Multiple Datasets**: Support for ASSISTments, Junyi, MOOC, and other public datasets
- 🔧 **Easy to Extend**: Modular design for adding new models and datasets
- 📈 **Standard Evaluation**: Unified metrics (AUC, Accuracy, RMSE)

## 🚀 Quick Start

### Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+ (optional, for GPU acceleration)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/joyce99/Survey.git
cd Survey
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n cdm python=3.8
conda activate cdm

# Or using venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
# Install PyTorch (choose appropriate CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

#### 4. Download Datasets

Due to the large size of dataset files, they need to be downloaded separately:

**Baidu Netdisk Link**: https://pan.baidu.com/s/1Z-a_BvsgqYQYM9SbvPTfgg?pwd=s997

**Access Code**: `s997`

After downloading, extract the dataset folders to the `data/` directory of the project. For detailed instructions, see [`data/README.md`](data/README.md).

## 🎯 Usage

### Basic Workflow

#### 1. Configure Parameters

Edit `params.py` to select dataset and set hyperparameters:

```python
# Select dataset (uncomment the one you want to use)
dataset = 'data/assist09/'
# dataset = 'data/mooper/'

# Set hyperparameters
batch_size = 128      # Batch size
lr = 0.002           # Learning rate
epoch = 100          # Number of epochs

# Set training and validation data files
src = dataset + 'train.json'
tgt = dataset + 'val.json'
```

#### 2. Run a Model

Use command-line to run a specific model:

```bash
# Run NCDM model (default)
python main.py --model NCDM

# Run IRT model
python main.py --model IRT

# Run DINA model
python main.py --model DINA

# Run RCD model
python main.py --model RCD
```

#### 3. View Results

After training, results are saved in the `result/` directory:

```
result/
├── NCDM.txt       # Model training results
├── IRT.txt
├── DINA.txt
└── ...
```

## 🤖 Supported Models

### Basic Models

| Model | Description | Command |
|-------|-------------|---------|
| IRT | Item Response Theory | `--model IRT` |
| MIRT | Multidimensional IRT | `--model MIRT` |
| DINA | Deterministic Inputs, Noisy "And" gate | `--model DINA` |
| NCDM | Neural Cognitive Diagnosis Model | `--model NCDM` |
| RCD | Relation-aware Cognitive Diagnosis | `--model RCD` |
| KSCD | Knowledge State Cognitive Diagnosis | `--model KSCD` |

### Advanced Models

| Model | Description | Command |
|-------|-------------|---------|
| AGCDM | Attention-based Graph CDM | `--model AGCDM` |
| KaNCD | Knowledge-aware Neural CD | `--model KaNCD` |
| CACD | Contrastive Affect-aware CD | `--model CACD` |
| QCCDM | Q-matrix Causal CDM | `--model QCCDM` |
| MF | Multiple-Strategy Fusion | `--model MF` |

### Model Variants

- **_GS variants**: Models with guessing and slipping factors
- **_Affect variants**: Models with affective factors

## 📊 Data Format

### Dataset Directory Structure

Each dataset directory should contain:

```
data/your_dataset/
├── config.txt          # Configuration: #students, #exercises, #concepts
├── train.json         # Training data
└── val.json           # Validation data
```

### config.txt Format

```
# Number of Students, Number of Exercises, Number of Knowledge Concepts
4163, 17746, 123
```

### JSON Data Format

Training and validation data use JSON format:

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

Fields:
- `user_id`: Student ID (starting from 1)
- `exer_id`: Exercise ID (starting from 1)
- `knowledge_code`: List of knowledge concept IDs (starting from 1)
- `score`: Answer result, 1 for correct, 0 for incorrect

## 📈 Evaluation Metrics

All models use the following metrics:

- **AUC**: Area Under ROC Curve, range [0,1], higher is better
- **Accuracy**: Prediction accuracy, range [0,1], higher is better
- **RMSE**: Root Mean Square Error, lower is better (for some models)

## 🐛 Troubleshooting

### CUDA out of memory

**Solution**:
- Reduce `batch_size` in `params.py`
- Use CPU training (set `device='cpu'`)
- Use a smaller dataset

### File not found

**Solution**:
- Check if `dataset` path in `params.py` is correct
- Ensure dataset directory contains required files
- Check file path separators (use `/` or `\\`)

### Model not converging

**Solution**:
- Increase training epochs
- Adjust learning rate
- Verify data format
- Try a different model

## 📚 Included Datasets

### 📥 Dataset Download

**Important**: Due to the large size of dataset files, they need to be downloaded from Baidu Netdisk.

- **Download Link**: https://pan.baidu.com/s/1Z-a_BvsgqYQYM9SbvPTfgg?pwd=s997
- **Access Code**: `s997`
- **Detailed Instructions**: See [`data/README.md`](data/README.md)

This project includes multiple public datasets:

### ASSISTments 2009 (`assist09`)
- Students: 4,163
- Exercises: 17,746
- Knowledge Concepts: 123
- Source: Online math tutoring system

### Other Datasets
- `mooper`: MOOC programming exercises
- `junyi`: Junyi Academy learning records
- `CSEDM-F`: Computer science education data
- `math1`: Mathematics learning data
- `NIPS20`: NIPS 2020 education challenge
- `PISA2015`: PISA 2015 assessment data
- `MOOCRadar`: MOOCRadar dataset

## 🔄 Update Log

This project is continuously updated. We will gradually improve:

- ✅ Multiple cognitive diagnosis model implementations
- ✅ Standardized dataset format
- ✅ Unified training and evaluation interface
- 🔄 Support for more models
- 🔄 More detailed experimental results
- 🔄 Model visualization tools

## 🤝 Contributing

Contributions are welcome! 

### How to Contribute

1. Fork this repository to your GitHub account
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Adding New Models

- Create a new model file in `model/` directory
- Implement model class with `__init__` and `train` methods
- Add model function in `main.py`
- Register the model in command-line arguments
- Submit a Pull Request with model description

### Adding New Datasets

- Create dataset folder in `data/` directory
- Prepare `config.txt`, `train.json`, `val.json` files
- Add dataset path option in `params.py`
- Update dataset list in README

## 📄 License

This project is for academic research purposes only.

## 📧 Contact

For questions or suggestions, please:

- Submit a [GitHub Issue](https://github.com/joyce99/Survey/issues)
- Send a [Pull Request](https://github.com/joyce99/Survey/pulls)
- Visit project homepage: https://github.com/joyce99/Survey

## 🙏 Acknowledgments

Thanks to:

- PyTorch deep learning framework
- ASSISTments dataset
- Junyi Academy
- Original authors of various cognitive diagnosis models

## 📖 Citation

If you use this project in your research, please cite the relevant model papers and this project:

```bibtex
@misc{cdm_survey_2024,
  title={CDM Survey: A Comprehensive Collection of Cognitive Diagnosis Models},
  author={joyce99 and xushengyu1},
  year={2024},
  howpublished={\url{https://github.com/joyce99/Survey}},
  note={GitHub repository}
}
```

## 🌟 Star History

If this project helps you, please give us a ⭐ Star!

[![Star History Chart](https://api.star-history.com/svg?repos=joyce99/Survey&type=Date)](https://star-history.com/#joyce99/Survey&Date)

---

**Last Updated**: 2024

**Project Maintainers**: [@joyce99](https://github.com/joyce99), [@xushengyu1](https://github.com/xushengyu1)

**Development Environment**: Python 3.8, PyTorch 1.13, CUDA 11.8

**Project Homepage**: https://github.com/joyce99/Survey

