# Learning Equality - Curriculum Recommendations
**Local Project - Web Mining IT4868E**

## Overview
This project implements a curriculum recommendation system using machine learning techniques including:
- Transformer models (BERT, XLM-RoBERTa, MiniLM)
- TF-IDF similarity matching
- LightGBM ensemble models
- Neural network embeddings

## Project Structure
- `src/` - Inference notebooks (ready to run)
- `models/` - Model training notebooks
- `data/` - Dataset files
- `weights/` - Trained model weights
- `output/` - Generated predictions

## Quick Start
See `INSTALL.md` for detailed setup instructions.

### Run Inference (No Training Required)
```bash
# Efficiency track (fast, no GPU)
jupyter notebook src/TFIDF_Inference.ipynb

# Main track (best performance, GPU recommended)
jupyter notebook src/Ensemble_Inference.ipynb
```
### Run GUI app
- cd into the main directory
- Run `python app.py`   

## Documentation
- `INSTALL.md` - Complete installation guide

## Requirements
- Python 3.7-3.10
- PyTorch
- Transformers
- LightGBM
- See `requirements.txt` for full dependencies

