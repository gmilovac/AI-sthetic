# AI-sthetic

## Project Name: **_AI-sthetic_**

## Project Details

AI-sthetic is a machine learning project aimed at training a **Multi-Head GAN** to generate images conditioned on **realism** and **aesthetic quality** using the AVA (Aesthetic Visual Analysis) dataset.  
The project currently implements a complete TensorFlow **data ingestion pipeline**, enabling standardized preprocessing, caching, and dataset filtering for efficient GAN training.

This repository serves as the foundation for the upcoming **Generator** and **Discriminator** architectures, which will operate on the fully normalized and batched dataset produced by the data pipeline.

In its current state, the project:

- Loads and preprocesses the full 255k-image AVA dataset (or any available subset)
- Classifies each image as aesthetic / non-aesthetic using the AVA mean score
- Normalizes and formats image batches for GAN consumption
- Implements TensorFlow caching for fast repeatability
- Ensures robustness against missing images or malformed metadata

The core system is designed to scale smoothly into full GAN training once the model components are added.

---

## Project Description

The goal of AI-sthetic is to explore aesthetic-conditioned image generation by leveraging the rich human-labeled AVA dataset.  
Using a two-head architecture (realism + aesthetic score), the model will learn to generate visually compelling images that also score highly based on aesthetic criteria.

The project emphasizes:

- **Data fidelity**: strict preprocessing standards matching GAN requirements
- **Training stability**: robust dataset construction and caching
- **Extensibility**: the data pipeline is built so future GAN variants can plug directly into it
- **Aesthetic control**: enabling targeted image generation conditioned on aesthetic preference

Users will eventually be able to sample images from the generator with specific aesthetic targets or realism constraints.

---

## Design Choices

### **Data Pipeline Architecture**

The TensorFlow `tf.data` pipeline forms the backbone of the training process.

Key decisions include:

- **256×256 resizing** – balances detail with computational feasibility
- **[-1, 1] pixel normalization** – matches `tanh`-based generator outputs
- **Batch format** structured as: (image_tensor, (realism_label, aesthetic_label))

- **File existence filtering** ensures missing images do not break training
- **Float-casting** all labels prevents TensorFlow graph-type errors
- **Sequential caching** stores preprocessed tensors to disk (`./tf_cache`) for major speedups on subsequent training runs

These choices ensure the dataset is consistent, efficient, and paired to GAN behavior.

---

### **Aesthetic Labeling Strategy**

Because AVA images include a distribution of scores, we compute the **mean aesthetic score** and apply a simple binary threshold:

- **Mean ≥ 5.5 → aesthetic (1.0)**
- **Mean < 5.5 → non-aesthetic (0.0)**

This threshold aligns with aesthetic-generation literature and offers balanced label distribution.

---

### **File Structure & Dataset Handling**

A consistent folder hierarchy is required to avoid file I/O errors and allow predictable caching:

```
AI-sthetic/
├── data_pipeline.py
├── data/
│ ├── ground_truth_dataset.csv or AVA.txt
│ └── images/
│ └── \*.jpg
└── tf_cache/
```

All image paths are validated before inclusion, preventing TensorFlow crashes from missing files and ensuring dataset stability.

---

## Current Progress

- ✔️ Completed full data ingestion and preprocessing pipeline
- ✔️ Implemented caching for fast repeated training
- ✔️ Added robust metadata parsing with automatic missing-file filtering
- ✔️ Validated output shapes, normalization, and label formatting
- ✔️ Confirmed compatibility with intended GAN model structure

**Next Steps:**

- Implement Generator & Discriminator architectures
- Add Multi-Head loss functions (realism + aesthetic)
- Build training loop and evaluation scripts
- Create visualization utilities for generated images

---

## How to Run

**Verify the data pipeline:**

```
python data_pipeline.py
```

Expected output includes:

- number of images successfully loaded
- missing-file warnings (if any)
- batch shapes (32, 256, 256, 3)
- pixel range of [-1, 1]
- aesthetic label distribution in the batch

If caching runs successfully, subsequent runs will load much faster.

---

## Team Members & Contributions

**Gordan Milovac** (_gmilovac_): Data pipeline design, preprocessing logic, project structure

**Marcus Winter** (_mwinter02_):

**Jeffrey Mu** (_jeffreymu1_):

**Arib Syed** (_Arib-S_):

---

## Tests

Current testing includes:

- Manual verification of image normalization and shape consistency
- File existence checks against metadata entries
- Stress-testing the cache by repeated pipeline execution
- Ensuring type consistency across all TensorFlow operations

Additional unit tests will be added during model development.

---

## Errors / Bugs

- No active pipeline-breaking bugs reported
- Known dataset issue: AVA dataset may contain missing images, which are automatically filtered
