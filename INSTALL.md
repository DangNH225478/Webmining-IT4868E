# Installation Guide - Learning Equality Curriculum Recommendations

This guide provides detailed instructions to setup and run the Learning Equality Curriculum Recommendations project.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Project Structure](#project-structure)
3. [Data Setup](#data-setup)
4. [Model Weights Setup](#model-weights-setup)
5. [Python Environment Setup](#python-environment-setup)
6. [Running the Notebooks](#running-the-notebooks)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements
- RAM: 32GB minimum
- Storage: 20GB free space
- GPU: NVIDIA GPU with 16GB+ VRAM (for ensemble models)
- CUDA: 11.0 or higher

### Software
- Python: 3.7 - 3.10
- pip or conda package manager

### Data
The dataset and trained weights can be found at https://drive.google.com/drive/folders/1sGr_ZLYveOP6GBAxsOrzM4h4B60lqgNM?usp=sharing
---

## Project Structure

After setup, your project should have this structure:

```
Web-Mining-Learning-Equality---Curriculum-Recommendations/
├── README.md
├── INSTALL.md
│
├── data/                                    # Competition data (you need to download)
│   ├── learning-equality-curriculum-recommendations/
│   │   ├── content.csv
│   │   ├── topics.csv
│   │   ├── correlations.csv
│   │   └── sample_submission.csv
│   │

│
├── weights/                                 # Model weights (you need to download)
│   ├── paraphrase-multilingual-minilm-l12-v2/     # MiniLM-L12 BertModel configs
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   └── ...
│   │
│   ├── paraphrase-multilingual-minilm-l12-v2-finetuned/  # Finetuned MiniLM weights
│   │   ├── vec_model_minilm_finetuned.pth
│   │   ├── V_content.npz
│   │   └── V_topic_train.npz
│   │

│   ├── bert-base-multilingual-uncased/     # BERT configs
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   └── ...
│   │
│   ├── xlm-roberta-base/                   # XLM-RoBERTa base configs
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── sentencepiece.bpe.model
│   │   └── ...
│   │
│   └── xlm-roberta-large/                  # XLM-RoBERTa large configs
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── sentencepiece.bpe.model
│       └── ...
│
├── output/                                  # Generated predictions (auto-created)
│   └── submission.csv
│
├── src/                                     # Inference notebooks
│   ├── TFIDF_Inference.ipynb
│   └── Ensemble_Inference.ipynb
│
└── models/                                  # Model training notebooks and weights
    ├── mBERT/
    │   ├── Notebooks/
    │   │   ├── mBERT_Full.ipynb
    │   │   └── mBERT_KFolds.ipynb
    │   ├── TrainingCurve/                   # Training loss plots
    │   ├── Config/
    │   └── Weights/
    │
    ├── RoBERTa-Base/
    │   ├── Notebooks/
    │   │   ├── RoBERTa_Full.ipynb
    │   │   └── RoBERTa_KFolds.ipynb
    │   ├── TrainingCurve/
    │   ├── Config/
    │   └── Weights/
    │
    ├── RoBERTa-Large/
    │   ├── Notebooks/
    │   │   ├── RoBERTa-Large-Finetune.ipynb
    │   │   ├── RoBERTa-Large-Finetune_Full.ipynb
    │   │   ├── RoBERTa-Large-Pretrain.ipynb
    │   │   └── RoBERTa-Large-Pretrain_Full.ipynb
    │   ├── TrainingCurve/
    │   ├── Config/
    │   └── Weights/
    │
    ├── mBERTLGB/                            # LightGBM models
    └── RoBERTaLGB/
```

---

## Data Setup

### Step 1: Download Competition Data

You need to obtain the Learning Equality Curriculum Recommendations dataset.

**Required files:**
- `content.csv`
- `topics.csv`
- `correlations.csv`

**Note:** Contact your instructor or check the course materials for dataset access.

### Step 2: Organize Data Files

Create the data directory structure:

```bash
# From project root
mkdir -p data/learning-equality-curriculum-recommendations

```

Move the downloaded files:

```bash
# Move competition files to data folder
mv content.csv data/learning-equality-curriculum-recommendations/
mv topics.csv data/learning-equality-curriculum-recommendations/
mv correlations.csv data/learning-equality-curriculum-recommendations/
```



---

## Model Weights Setup

### Step 1: Create Weights Directory

```bash
mkdir -p weights
```

### Step 2: Download Pre-trained Transformer Models

You need to download the base transformer models from HuggingFace:

**Option A: Using transformers library (Automatic)**

The models will be auto-downloaded when you first run the notebooks. However, for manual download:

```python
from transformers import AutoModel, AutoTokenizer

# MiniLM
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
model.save_pretrained("weights/paraphrase-multilingual-minilm-l12-v2")
tokenizer.save_pretrained("weights/paraphrase-multilingual-minilm-l12-v2")

# BERT multilingual
model = AutoModel.from_pretrained("bert-base-multilingual-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
model.save_pretrained("weights/bert-base-multilingual-uncased")
tokenizer.save_pretrained("weights/bert-base-multilingual-uncased")

# XLM-RoBERTa base
model = AutoModel.from_pretrained("xlm-roberta-base")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model.save_pretrained("weights/xlm-roberta-base")
tokenizer.save_pretrained("weights/xlm-roberta-base")

# XLM-RoBERTa large (only for main track)
model = AutoModel.from_pretrained("xlm-roberta-large")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model.save_pretrained("weights/xlm-roberta-large")
tokenizer.save_pretrained("weights/xlm-roberta-large")
```

**Option B: Manual Download from HuggingFace**

Visit these links and download all files:
- https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- https://huggingface.co/bert-base-multilingual-uncased
- https://huggingface.co/xlm-roberta-base
- https://huggingface.co/xlm-roberta-large

### Step 3: Download Fine-tuned Model Weights

⚠️ **Important**: The fine-tuned weights (`.pth` files and LightGBM models) are not included in this repository.

You need to either:

1. **Train the models yourself** using the notebooks in `models/` folder (e.g., `models/mBERT/Notebooks/`, `models/RoBERTa-Base/Notebooks/`)
2. **Contact the repository maintainer** for pre-trained weights

**Model weights structure:**

```
models/
├── mBERT/
│   └── Weights/                            # Trained mBERT weights
├── RoBERTa-Base/
│   └── Weights/                            # Trained RoBERTa-Base weights
├── RoBERTa-Large/
│   └── Weights/                            # Trained RoBERTa-Large weights
├── mBERTLGB/                               # LightGBM models for mBERT
└── RoBERTaLGB/                             # LightGBM models for RoBERTa```

---

## Python Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# Create conda environment
conda create -n lecr python=3.9 -y
conda activate lecr

# Install PyTorch (adjust based on your CUDA version)
# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Option 2: Using pip + venv

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Create requirements.txt

Create a `requirements.txt` file in the project root:

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
torch>=1.10.0
transformers>=4.20.0
lightgbm>=3.3.0
sparse-dot-topn>=0.3.3
tqdm>=4.62.0
jupyter>=1.0.0
notebook>=6.4.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Notebooks

### Create Output Directory

```bash
mkdir -p output
```

```bash
jupyter notebook src/Ensemble_Inference.ipynb
```

- Uses 3 transformer models (BERT + XLM-RoBERTa base + large)
- F2-Score: ~0.760+
- Memory: ~16GB GPU + 32GB RAM
- Time: 10-20 minutes

### Training New Models

To train models from scratch, use notebooks in the `models/` folder:

```bash
# Example: Train mBERT model
jupyter notebook models/mBERT/Notebooks/mBERT_Full.ipynb

# Example: Train RoBERTa-Base model
jupyter notebook models/RoBERTa-Base/Notebooks/RoBERTa_Full.ipynb

# Example: Train RoBERTa-Large model
jupyter notebook models/RoBERTa-Large/Notebooks/RoBERTa-Large-Finetune.ipynb
```

**Note:** Training curves are automatically saved to the `TrainingCurve/` folder within each model directory.

---

## Troubleshooting

### Issue 1: CUDA out of memory

**Solution:**
- Reduce batch size in notebooks (change `BS = 64` to `BS = 32` or lower)
- Use CPU instead of GPU (change `.cuda()` to `.cpu()`)
- Use efficiency track notebooks instead

### Issue 2: File not found errors

**Solution:**
- Verify all paths are relative and match the structure above
- Check that data files are in correct locations
- Ensure you're running notebooks from project root or using correct relative paths

### Issue 3: Transformer model download issues

**Solution:**
```bash
# Set HuggingFace cache directory
export TRANSFORMERS_CACHE=./weights/huggingface_cache

# Download with retry
pip install huggingface-hub
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### Issue 5: LightGBM performance warnings

**Solution:**
```bash
# Install OpenMP version for better performance
conda install lightgbm -c conda-forge
```

### Issue 6: Notebook kernel crashes

**Solution:**
- Increase system memory/swap
- Process data in smaller chunks
- Use efficiency notebooks instead of ensemble

---

## Performance Benchmarks

| Solution | F2-Score | Training Time | Inference Time | GPU Required | Memory |
|----------|----------|---------------|----------------|--------------|--------|
| lecr-efficiency-nobert | 0.530 | N/A (pre-trained) | <1 min | No | 4GB |
| lecr-efficiency-minilm | 0.718 | ~2 hours | 2-5 min | No | 8GB |
| lecr-ensemble-v03 | 0.760+ | ~8 hours | 10-20 min | Yes (16GB) | 32GB |

---

## Additional Resources

- **Course**: IT4868E Web Mining

---

## License

This project is for educational purposes as part of IT4868E Web Mining course.

---

## Citation

```
Learning Equality - Curriculum Recommendations
Local Implementation for IT4868E Web Mining Course
```

