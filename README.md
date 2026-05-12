# DSVS_MiSonGyny2025 🎵

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0+-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/transformers/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-f7931e?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-13adc7?logo=dvc&logoColor=white)](https://dvc.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-f37726?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

DSVS participation in **MiSonGyny at IberLEF 2025**: Detecting Misogyny in Spanish Song Lyrics

</div>

---

## 📋 Overview

This project addresses the detection and classification of misogynistic language in Spanish song lyrics as part of the **MiSonGyny shared task** at IberLEF 2025. The approach uses transformer-based models with Multiple Instance Learning (MIL) to handle the hierarchical nature of songs (songs → verses → sentences).

**Competition Details**: [MiSonGyny @ IberLEF 2025](https://sites.google.com/view/misongyny)

---

## 🎯 Tasks

### Task 1: Misogyny Detection (Binary Classification)
**Objective**: Detect whether a Spanish song contains misogynistic content

- **Labels**: 
  - `0` - Non-Misogynistic (NM)
  - `1` - Misogynistic (M)

- **Description**: Identify hateful, contemptuous, or stereotypical language directed at women in song lyrics

---

### Task 2: Misogyny Type Classification (Multilabel)
**Objective**: Classify which types of misogynistic discourse are present in a song

- **Categories**:
  - **S (Sexualization)**: Sexual content, innuendo, descriptions of sexual acts
  - **V (Violence)**: Physical/verbal aggression, threats, violent language
  - **H (Hate)**: Offensive/discriminatory language, contempt toward women

- **Note**: A song can belong to multiple categories simultaneously

---

## 🔬 Solution Approach

### Architecture Overview

The solution employs a **Multiple Instance Learning (MIL)** framework to handle the hierarchical structure of songs:

```
Song (Bag)
  ├── Verse 1 → BERT Encoding → Instance 1
  ├── Verse 2 → BERT Encoding → Instance 2
  └── Verse N → BERT Encoding → Instance N
       ↓
  Pooling Layer (Max/Attention)
       ↓
  Classification Head
       ↓
  Song-level Label
```

### Key Components

1. **Text Encoder**: BERT-base-cased transformer model
   - Processes each verse independently
   - Outputs contextualized embeddings

2. **Pooling Strategies**:
   - **Max Pooling**: Selects the maximum activation across instances
   - **Attention Pooling**: Learns weighted aggregation of instances

3. **Classification Head**:
   - Dropout layer for regularization
   - Linear layer for label prediction

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | BERT-base-cased |
| Optimizer | Adam (lr=1e-5, weight_decay=0.01) |
| Scheduler | Cosine Annealing (1.5 cycles) |
| Epochs | 5 |
| Batch Size | 5 |
| Gradient Accumulation | 2 steps |
| Max Sequence Length | 128 tokens |
| Loss Function (Task 1) | Cross-Entropy Loss |
| Loss Function (Task 2) | Focal Loss |

### Data Processing

- **Spanish-specific cleaning**: Normalization of accents, contractions, and special characters
- **Verse segmentation**: Songs split into verses for instance-level processing
- **Tokenization**: BERT tokenizer with max_length=128

---

## 📂 Project Structure

```
DSVS_MiSonGyny2025/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── config.env.example        # Configuration template
│
├── scripts/                  # Core implementation
│   ├── __init__.py
│   ├── classifiers.py        # TextClassifier and MILClassifier models
│   ├── dataset_utils.py      # Dataset loading and preprocessing
│   ├── learner.py            # Training and evaluation logic
│   ├── train.py              # Main training script
│   ├── handle_results.py     # Results processing and storage
│   ├── utils.py              # Utility functions
│   ├── task1_config.py       # Task 1 hyperparameters
│   └── task2_config.py       # Task 2 hyperparameters
│
├── notebooks/                # Jupyter notebooks for exploration
│   ├── subtask1_dataset.ipynb
│   ├── subtask1.ipynb
│   └── subtask2.ipynb
│
├── datasets/                 # Data management
│   ├── datasets.yaml         # Dataset registry
│   ├── task1_mil_v*.dvc      # DVC-tracked dataset versions (Task 1)
│   └── task2_mil_v*.dvc      # DVC-tracked dataset versions (Task 2)
│
└── tests/                    # Unit tests
    └── test_trainclassifier.py
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core programming language |
| **PyTorch** | Deep learning framework and neural network implementation |
| **Transformers (HF)** | Pre-trained BERT models and tokenizers |
| **Datasets (HF)** | Efficient data loading and preprocessing |
| **scikit-learn** | Metrics computation (accuracy, precision, recall, F1) |
| **NumPy** | Numerical operations |
| **Pandas** | Data manipulation and analysis |
| **tqdm** | Progress bar visualization |
| **DVC** | Data version control and experiment tracking |
| **Jupyter** | Interactive exploration and experimentation |

---

## 📊 Experiments & Results

The project includes multiple dataset versions (v0-v6) with different preprocessing strategies:

- **v0-v4**: Various cleaning approaches
- **v5**: Optimized preprocessing with LoRA-based fine-tuning variant
- **v6**: Final version with best performance

Each version includes:
- Training and test splits
- Performance metrics: Accuracy, Precision, Recall, F1-score (macro)
- MLflow/DagsHub tracking for reproducibility

---

## 🚀 Usage

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Other dependencies (see requirements in scripts)

### Training

```bash
cd scripts/
python train.py
```

### Evaluation

Metrics are automatically computed during training and logged via MLflow.

---

## 📝 References

- **Competition**: https://sites.google.com/view/misongyny
- **IberLEF 2025**: Iberian Language Evaluation Forum
- **Related Work**: Multiple Instance Learning for weakly-supervised classification

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Contributing

This is a competition submission for IberLEF 2025. For contributions or questions, please reach out to the project maintainers.

## Citation

To cite this work, please use the following BibTeX entry:

```bibtex
@article{damian2025dsvs,
  title={DSVS at MiSonGyny 2025: Multiple Instance Learning for Misogyny Speech Detection in Song Lyrics},
  author={Dami{\'a}n-Sandoval, Sergio and V{\'a}zquez-Santana, David},
  year={2025}
}
```

**Paper**: https://ceur-ws.org/Vol-4098/MiSonGyny2025_paper8.pdf