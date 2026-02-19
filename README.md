# Hybrid Real-Time Digital Image Tampering Detection & Localization

## Overview

This repository contains the implementation of a hybrid, real-time framework for detecting and localizing digital image tampering. The system combines Vision Transformers, EfficientNet, metadata-based learning, and conditional segmentation to provide robust multi-manipulation detection while remaining suitable for real-time deployment.

Key manipulation types supported:

- Copy–Move Forgery
- Image Splicing
- Inpainting / Content Removal
- Deepfake / GAN-generated Images
- JPEG / Seam-Carving Manipulations
- Metadata / EXIF Tampering

## Features

- Hybrid multi-model architecture (CNN + Transformer + metadata classifier)
- Confidence-weighted ensemble fusion
- Conditional localization: segmentation runs only for high-confidence tampered images
- Real-time optimized inference for constrained GPUs
- Reproducible training and evaluation pipeline

## Architecture

The pipeline is organized in five stages:

1. Adaptive preprocessing
2. Parallel multi-model detection
   - Vision Transformer (ViT)
   - EfficientNet-B0
   - LightGBM (metadata classifier)
3. Confidence-weighted fusion
4. Conditional localization (YOLOv8-Seg)
5. Interpretable output generation

## Datasets

Public datasets are used for training and evaluation. Example datasets referenced in experiments:

- CASIA v2 — copy–move & splicing
- Inpainting localization dataset — pixel-level inpainting evaluation
- Deepfake datasets — GAN / face manipulation detection
- JPEG / seam-carving dataset — compression & seam-carving forgeries
- EXIF metadata dataset — metadata tampering

Links used in experiments (example sources):

- CASIA v2: https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset
- Inpainting Localization: https://www.kaggle.com/datasets/duyminhle/inpainting-localization-eval-dataset
- Deepfake Dataset: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
- JPEG / Seam-Carving: https://www.kaggle.com/datasets/liuqzsc/seam-carving-jpeg-image-forgery-dataset
- EXIF Metadata: https://www.kaggle.com/datasets/mdappelmahmudpranto/meta-and-exif-data-original-and-manipulated-images

## Performance (Summary)

| Task | Accuracy | F1 Score |
|------|---------:|---------:|
| Deepfake Detection | 88% | 0.96 |
| JPEG / Seam-Carving | 90% | 0.96 |
| Copy-Move / Splicing | 78% | 0.85 |
| Metadata Tampering | 93% | 0.96 |
| Inpainting | 92% | 0.92 |

Average inference latency reported: ~316 ms per image (~11.4 FPS).

## Repository Structure

Top-level layout (representative):

```
Hybrid-Tamper-Detection/
├── notebooks/
│   └── hybrid_tamper_detection.ipynb
├── models/
│   ├── vit/
│   ├── efficientnet/
│   ├── lightgbm/
│   └── yolov8/
├── configs/
├── data/
├── utils/
├── results/
└── README.md
```

## Installation

Prerequisites: Python 3.8+ (3.10 recommended), GPU with CUDA support for model acceleration.

Install dependencies (example):

```bash
pip install -r requirements.txt
```

## Usage

Run the main notebook for exploration and experiments:

```
notebooks/hybrid_tamper_detection.ipynb
```

Run inference (script mode):

```bash
python inference.py --image path_to_image
```

Conditional localization

Localization is triggered only when the ensemble confidence exceeds a configurable localization threshold. This conserves computation by only running segmentation for likely-tampered images.

## Configuration

Configuration files live in `configs/` (e.g. `training.yaml`, `inference.yaml`, `calibration.yaml`). Set seeds and deterministic settings for reproducibility (example):

```python
seed = 42
```

## Reproducibility

- Dataset preprocessing scripts are included in `data/`.
- Model training configurations and calibration logic are available in `configs/`.
- Evaluation metrics and logging are stored under `results/`.

## Limitations

- Localization quality can degrade under heavy compression or extreme downsampling.
- Performance may reduce if manipulation artifacts are removed by diffusion-based editing.
- Current focus is image-level forensics; video support is planned.

## Future Work

- Real-time video forgery detection
- Edge-device optimized deployment
- Diffusion-based forgery detection
- Continual learning & adaptive fusion

## Citation

If you use this work, please cite:

```
@article{hybridtamper2026,
  title={Hybrid Framework for Real-Time Detection and Localization of Digital Image Tampering},
  author={Kamble, Vitthal B. and Uke, Nilesh J.},
  journal={Submitted to The Visual Computer},
  year={2026}
}
```

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Contact

Vitthal B. Kamble — Research Scholar, VIIT Pune — vitthalk13@gmail.com

## Acknowledgements

Thanks to the open-source community, dataset contributors, and reviewers for their support.
