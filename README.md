# Automatically Discovering and Naming What a Vision Model Doesn't Know

> An Explainable Multimodal AI Pipeline for Automatic Failure Discovery, Interpretation, and Targeted Model Improvement using CLIP.

---

## Overview

Modern deep vision models achieve remarkable accuracy, yet they remain difficult to debug when they fail. Traditional error analysis requires manually inspecting hundreds or thousands of misclassified images to identify common patterns—a tedious and subjective process.

This project aims to automate that entire workflow.

Instead of treating CLIP purely as an image retrieval model, we leverage its shared image-text embedding space as a diagnostic tool for discovering, interpreting, and correcting systematic failure modes of image classifiers.

The project investigates whether automatically generated natural-language descriptions of model failures can be used to improve downstream model performance through targeted retraining.

---

## Motivation

Consider an image classifier trained on birds.

Suppose the model consistently misclassifies images that contain:

- overcast skies
- backlit birds
- birds partially hidden by water
- extreme camera angles

Normally, identifying these patterns requires manual inspection.

Our goal is to automatically discover these hidden failure modes and describe them in human-readable language such as:

> "Overcast backlit birds standing in shallow water."

The discovered descriptions are then used to retrieve similar images and retrain the model on its own weaknesses.

---

# Project Pipeline

```

Training Dataset
│
▼
Train Baseline Vision Classifier
│
▼
Collect Misclassified Images
│
▼
Extract CLIP Image Embeddings
│
▼
Cluster Similar Failure Cases
│
▼
Generate Natural Language Failure Descriptions
│
▼
Retrieve Similar Images using Text Queries
│
▼
Fine-tune Second Model
│
▼
Evaluate Improvement on Held-out Test Set

```

---

# Objectives

- Train a baseline image classification model.
- Identify all validation misclassifications.
- Represent failure samples using CLIP embeddings.
- Discover clusters of semantically similar errors.
- Automatically generate human-readable names for each failure cluster.
- Retrieve additional training data matching discovered failure modes.
- Fine-tune a second classifier using only retrieved slices.
- Evaluate whether targeted retraining improves generalization.

---

# Features

## Automated Failure Discovery

Detect recurring visual failure patterns without manually labeling errors.

---

## Explainable AI

Convert opaque model failures into interpretable natural-language descriptions.

Example:

```

Failure Cluster #3

↓

"Low-light side-profile birds with water reflections"

```

---

## CLIP-based Semantic Clustering

Use CLIP's joint image-text embedding space to group visually similar mistakes.

---

## Targeted Dataset Expansion

Retrieve additional training samples corresponding to discovered failure modes instead of blindly increasing dataset size.

---

## Closed Feedback Loop

```

Model
↓

Mistakes
↓

Diagnosis
↓

Targeted Data
↓

Improved Model

```

---

# Datasets

Possible datasets include:

- DeepFashion
- iNaturalist 2021
- Waterbirds Benchmark
- LAION-Fashion Subset

Dataset selection will depend on experimental feasibility and computational resources.

---

# Technology Stack

## Languages

- Python

## Deep Learning

- PyTorch
- TorchVision

## Vision-Language Models

- OpenAI CLIP
- HuggingFace Transformers

## Machine Learning

- scikit-learn
- NumPy
- Pandas

## Visualization

- Matplotlib
- Seaborn

---

# Project Structure

```

project/

│

├── data/

├── notebooks/

├── models/

│ ├── baseline/

│ ├── clip/

│ └── retrained/

│

├── clustering/

├── captioning/

├── retrieval/

├── evaluation/

├── utils/

├── results/

└── README.md

```

---

# Methodology

## Phase 1 — Baseline Model

- Train an image classifier.
- Evaluate validation accuracy.
- Store all incorrect predictions.

---

## Phase 2 — Error Representation

- Encode failed images using CLIP.
- Project failures into the shared embedding space.

---

## Phase 3 — Failure Discovery

- Cluster failure embeddings.
- Discover visually coherent failure groups.

---

## Phase 4 — Automatic Naming

Generate interpretable descriptions for each discovered cluster using contrastive captioning.

Example:

```

Cluster 2

↓

"Dark indoor images with reflective surfaces"

```

---

## Phase 5 — Targeted Improvement

Use generated descriptions as retrieval queries against large image collections.

Examples:

- LAION
- DeepFashion subsets

Collect new images matching discovered weaknesses.

---

## Phase 6 — Retraining

Train a second classifier using the retrieved image slices.

---

## Phase 7 — Evaluation

Compare:

- Baseline Model
- Improved Model

using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# Expected Outcomes

- Automated vision model debugging.
- Explainable failure discovery.
- Human-readable error interpretation.
- Data-centric model improvement.
- Improved robustness against discovered failure modes.

---

# References

### CLIP

Radford et al.

Learning Transferable Visual Models From Natural Language Supervision

NeurIPS 2021

https://arxiv.org/abs/2103.00020

---

### DrML

Zhang et al.

Diagnosing and Rectifying Vision Models using Language

ICLR 2023

https://arxiv.org/abs/2302.04269

---

### Error Discovery

Wang et al.

Error Discovery by Clustering Influence Embeddings

NeurIPS 2023

https://arxiv.org/abs/2312.04712

---

# Current Status

- [x] Literature Review
- [x] Project Planning
- [ ] Dataset Preparation
- [ ] Baseline Model
- [ ] CLIP Embedding Extraction
- [ ] Failure Clustering
- [ ] Automatic Caption Generation
- [ ] Data Retrieval
- [ ] Model Retraining
- [ ] Final Evaluation

---

# Team

**Point of Contact**

Vaibhav Pratap Singh

Indian Institute of Technology Guwahati

Team Members

- Vaibhav Pratap Singh (PoC)
- Akshit Arora
- Siddhant Patil
- Medha Rama Murthy

---

# Future Scope

Potential extensions include:

- Vision Transformers (ViT)
- Segment Anything Model (SAM)
- BLIP / BLIP-2
- Active Learning
- Continual Learning
- Open-vocabulary Failure Analysis
- Multimodal Explainability Benchmarks

---

## License

This project is being developed as part of the **Coding Club IIT Guwahati Summer Projects 2026**.
