# Hybrid Real-Time Digital Image Tampering Detection & Localization

## Overview

This repository contains the official implementation of our research work on **real-time detection and localization of digital image tampering** using a hybrid multi-model framework. The system integrates **Vision Transformers, EfficientNet, metadata-based learning, and conditional segmentation** to achieve robust multi-manipulation detection while maintaining real-time performance.

The framework is designed to bridge the gap between **forensic accuracy and deployable real-time systems**, supporting multiple manipulation types including:

- Copy–Move Forgery  
- Image Splicing  
- Inpainting / Content Removal  
- Deepfake / GAN-generated Images  
- JPEG / Seam-Carving Manipulations  
- Metadata / EXIF Tampering  

---

## Key Contributions

- Hybrid multi-model architecture combining **CNN, Transformer, and Metadata-based detection**
- **Confidence-weighted ensemble fusion** for improved robustness
- **Conditional localization** that activates segmentation only for high-confidence tampered images
- Real-time optimized inference suitable for constrained GPU environments
- Reproducible pipeline using public forensic datasets
- Deployment-aware evaluation including latency and robustness

---

## Architecture

The system consists of five main stages:

1. **Adaptive Preprocessing**
2. **Parallel Multi-Model Detection**
   - Vision Transformer (ViT)
   - EfficientNet-B0
   - LightGBM (metadata classifier)
3. **Confidence-Weighted Fusion**
4. **Conditional Localization (YOLOv8-Seg)**
5. **Interpretable Output Generation**

---

## Datasets Used

All datasets used are publicly available.

| Dataset | Purpose |
|--------|---------|
| CASIA v2 | Copy–Move & Splicing detection |
| Inpainting Localization Dataset | Pixel-level localization |
| Deepfake & Real Images | GAN / Face manipulation detection |
| Seam-Carving / JPEG Forgery Dataset | Compression & seam-carving detection |
| EXIF Metadata Dataset | Metadata tampering detection |

Links:

- CASIA v2: https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset
- Inpainting Localization: https://www.kaggle.com/datasets/duyminhle/inpainting-localization-eval-dataset
- Deepfake Dataset: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
- JPEG / Seam-Carving: https://www.kaggle.com/datasets/liuqzsc/seam-carving-jpeg-image-forgery-dataset
- EXIF Metadata: https://www.kaggle.com/datasets/mdappelmahmudpranto/meta-and-exif-data-original-and-manipulated-images

---

## Experimental Environment

Experiments were conducted using:

- GPU: NVIDIA Tesla T4 (Google Colab Free Tier)
- VRAM: 16 GB
- RAM: ~12 GB
- Python: 3.10
- CUDA: 12.x
- Frameworks: PyTorch / TensorFlow (GPU-enabled)

---

## Performance Summary

| Task | Accuracy | F1 Score |
|------|---------|----------|
| Deepfake Detection | 88% | 0.96 |
| JPEG / Seam-Carving | 90% | 0.96 |
| Copy-Move / Splicing | 78% | 0.85 |
| Metadata Tampering | 93% | 0.96 |
| Inpainting | 92% | 0.92 |

Average inference latency: **~316 ms per image (~11.4 FPS)**

---

## Repository Structure

Hybrid-Tamper-Detection/
│
├── notebooks/
│ └── hybrid_tamper_detection.ipynb
│
├── models/
│ ├── vit/
│ ├── efficientnet/
│ ├── lightgbm/
│ └── yolov8/
│
├── configs/
│ ├── training.yaml
│ ├── inference.yaml
│ └── calibration.yaml
│
├── data/
│ └── (dataset loading scripts)
│
├── utils/
│ ├── preprocessing.py
│ ├── fusion.py
│ ├── metadata.py
│ └── evaluation.py
│
├── results/
│ ├── metrics
│ ├── plots
│ └── logs
│
└── README.md


---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/hybrid-tamper-detection.git
cd hybrid-tamper-detection
Install dependencies:

pip install -r requirements.txt
Usage
Run Notebook (Recommended)
Open the main notebook:

notebooks/hybrid_tamper_detection.ipynb
Run all cells sequentially.

Inference (Script Mode)
python inference.py --image path_to_image
Conditional Localization Logic
Localization is triggered only when:

Ensemble Confidence >= Localization Threshold
This reduces average computational cost while preserving localization accuracy.

Reproducibility
The repository includes:

Dataset preprocessing scripts

Model training configurations

Calibration and fusion logic

Evaluation metrics

Experimental logs

Set seeds for deterministic results:

seed = 42
Limitations
Localization may degrade under heavy compression or extreme downsampling

Performance may reduce when manipulation artifacts are removed by diffusion-based editing

Designed primarily for image-level forensics (video extension planned)

Future Work
Real-time video forgery detection

Edge-device optimized deployment

Diffusion-based forgery detection

Continual learning & adaptive fusion

Citation
If you use this work, please cite:

@article{hybridtamper2026,
  title={Hybrid Framework for Real-Time Detection and Localization of Digital Image Tampering},
  author={Kamble, Vitthal B. and Uke, Nilesh J.},
  journal={Submitted to The Visual Computer},
  year={2026}
}
License
This project is released under the MIT License.

Contact
Vitthal B. Kamble
Research Scholar, VIIT Pune
Email: vitthalk13@gmail.com

Acknowledgements
We thank the open-source community, dataset contributors, and reviewers for valuable feedback and support.


---