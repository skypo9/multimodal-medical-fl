# Multi-Modal Medical Federated Learning

A federated learning framework for cross-modal medical image classification using PyTorch and Flower FL, supporting both FedAvg and FedBN aggregation strategies.

## Overview

This repository implements federated learning for cross-modal medical imaging scenarios. The framework enables knowledge sharing between different medical imaging modalities (dermoscopy and chest X-ray) while preserving domain-specific characteristics through advanced aggregation strategies.

### Key Features

- Cross-Modal Learning: Handles multiple medical imaging modalities in federated learning
- Dual Aggregation Strategies: Implements both FedAvg and FedBN for comparison
- Automated Dataset Management: Kaggle dataset downloading and organization
- Comprehensive Evaluation: Performance metrics with F1-score, accuracy, and loss tracking
- Robust Validation: Dataset integrity checking and class distribution analysis

## Quick Start

### Prerequisites

```bash
Python 3.8+
PyTorch >= 1.12.0
Flower (flwr) >= 1.0.0
scikit-learn >= 1.0.0
matplotlib >= 3.5.0
kaggle >= 1.5.12
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-medical-fl.git
cd multimodal-medical-fl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle API (optional):
```bash
# Download kaggle.json from your Kaggle account settings
# Place it in ~/.kaggle/ directory
```

### Usage

#### Multi-Modal FL with FedAvg (default)
```bash
python multimodal_fl_simulation.py --clients 2 --rounds 7 --sample_fraction 0.5
```

#### Multi-Modal FL with FedBN
```bash
python multimodal_fl_simulation.py --clients 2 --rounds 7 --sample_fraction 0.5 --use_fedbn
```

## Supported Datasets

The framework uses two cross-modal datasets:

| Dataset | Modality | Task | Classes |
|---------|----------|------|---------|
| Skin Cancer | Dermoscopy | Lesion Classification | benign/malignant |
| Pneumonia X-ray | Chest Radiography | Disease Detection | normal/pneumonia |

### Cross-Modal Challenge

- **Client 1**: Trains on dermoscopy images (skin lesion classification)
- **Client 2**: Trains on chest X-ray images (pneumonia detection)
- **Challenge**: Different imaging modalities with distinct feature distributions
- **Goal**: Share knowledge across modalities while preserving domain characteristics

## Architecture

### Model
- MultiModalMedicalCNN: Custom CNN with batch normalization and dropout
- Adaptive Global Pooling: Handles variable input sizes
- Binary Classification: Optimized for medical binary tasks

### Federated Strategies

#### FedAvg (Federated Averaging)
- Averages all model parameters across clients
- Best for: IID data distributions
- Simple and well-established baseline

#### FedBN (Federated Batch Normalization)
- Shares convolutional weights, keeps batch norm parameters local
- Best for: Non-IID and cross-modal scenarios
- Preserves domain-specific feature distributions

## Performance Results

### Cross-Modal Results (Skin Cancer + Pneumonia)

| Strategy | Skin Cancer Acc | Pneumonia Acc | Average | Improvement |
|----------|----------------|---------------|---------|-------------|
| FedAvg | 55.0% | 45.5% | 50.25% | Baseline |
| FedBN | 71.0% | 62.5% | 66.75% | +16.5% |

## Configuration

### Command Line Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--clients` | Number of federated clients | 2 |
| `--rounds` | Number of federated rounds | 5 |
| `--sample_fraction` | Fraction of data to sample | 0.3 |
| `--use_fedbn` | Use FedBN instead of FedAvg | False |
| `--lr` | Learning rate | 0.001 |
| `--batch_size` | Training batch size | 32 |

## Repository Structure

```
multimodal-medical-fl/
├── README.md
├── requirements.txt
├── LICENSE
├── multimodal_fl_simulation.py
├── .gitignore
├── datasets/
│   ├── skin_cancer/
│   └── pneumonia_xray/
└── results/
```

## Research Applications

- Multi-Hospital Collaboration: Share knowledge without sharing patient data
- Cross-Modality Learning: Leverage expertise from different imaging specialties
- Privacy-Preserving ML: Medical AI without centralizing patient data
- Strategy Comparison: Benchmark different federated learning approaches

## Development

### Adding New Datasets

```python
datasets_info = {
    "skin_cancer": ["benign", "malignant"],
    "pneumonia_xray": ["normal", "pneumonia"],
    "new_dataset": ["class1", "class2"],  # Add here
}
```

## References

- **FedAvg**: McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017)
- **FedBN**: Li et al. "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization" (ICLR 2021)
- **Medical FL**: Sheller et al. "Federated Learning for Multi-Center Imaging" (Nature Machine Intelligence 2020)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
