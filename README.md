
# Contaminated Multivariate Time-Series Anomaly Detection with Spatio-Temporal Graph Conditional Diffusion Models

This repository contains the official implementation of the paper:

**Contaminated Multivariate Time-Series Anomaly Detection with Spatio-Temporal Graph Conditional Diffusion Models**  
Thi Kieu Khanh Ho, Narges Armanfard  
Published at the 41st Conference on Uncertainty in Artificial Intelligence (UAI 2025)

[Paper Link (coming soon)]

## 📌 Overview
We introduce a novel and practical end-to-end unsupervised method for anomaly detection in multivariate time-series, where training data may be contaminated with anomalies. Our model leverages a Spatio-Temporal Graph Conditional Diffusion Model (ST-GCDM) to robustly model complex dependencies in time-series data, enabling effective detection even under real-world data corruption.

<p align="center"><img src="./model.png" alt="Model Architecture" width="600"/></p>

## 🔍 Key Features
- ✅ Handles contaminated training data with unknown anomaly labels.  
- 🌐 Exploits spatial and temporal dependencies using a spatio-temporal graph.  
- 🌫️ Employs a diffusion-based generative model for conditional sequence modeling.  
- 🔍 Fully unsupervised: does not require clean or labeled training data.  

## 🧠 Methodology
Our approach models normal behavior using conditional diffusion and graph-based encoding of spatial and temporal relations. During inference, it uses reconstruction error and diffusion likelihoods for anomaly scoring.

## 📁 Repository Structure
```
├── data/                   # Datasets or data processing scripts
├── models/                 # Core model implementation (ST-GCDM)
├── utils/                  # Utility functions
├── train.py                # Training pipeline
├── evaluate.py             # Evaluation and anomaly scoring
├── config.yaml             # Experiment configurations
└── README.md
```

## 🚀 Getting Started

### Requirements
- Python 3.8+
- PyTorch >= 1.11
- DGL or PyTorch Geometric (for graph processing)
- Other dependencies: see `requirements.txt`

### Installation
```bash
git clone https://github.com/your_username/st-gcdm-anomaly-detection.git
cd st-gcdm-anomaly-detection
pip install -r requirements.txt
```

### Running the Model

**Training**
```bash
python train.py --config config.yaml
```

**Evaluation**
```bash
python evaluate.py --config config.yaml
```

## 📊 Datasets
We test our method on several real-world multivariate time-series datasets with injected or naturally occurring anomalies. See the `data/` folder for download and preprocessing instructions.

## 📈 Results
Our method achieves state-of-the-art performance on multiple benchmark datasets under varying levels of contamination.

| Dataset | F1 Score | AUC-ROC |
|---------|----------|---------|
| SWaT    | 0.91     | 0.96    |
| SMD     | 0.89     | 0.94    |
| WADI    | 0.88     | 0.93    |

*(Full results available in the paper)*

## 📚 Citation
If you use this code or find our work helpful, please cite:
```bibtex
@article{ho2023multivariate,
  title={Contaminated Multivariate Time-Series Anomaly Detection with Spatio-Temporal Graph Conditional Diffusion Models},
  author={Ho, Thi Kieu Khanh and Armanfard, Narges},
  journal={The 41st Conference on Uncertainty in Artificial Intelligence (UAI 2025)},
  year={2025}
}
```

## 🧩 Contact
For questions or collaborations, feel free to reach out:

- Thi Kieu Khanh Ho - [email or webpage]
- Narges Armanfard - [email or webpage]
