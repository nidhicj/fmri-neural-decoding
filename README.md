# Decoding Neural Brain Activity for Image Classification

Predicting **what category of image a person is looking at** from their fMRI brain activity, using an LSTM and a 3D-CNN on the BOLD5000 dataset.

*Group lab project, Machine Learning & Computer Vision Lab for Human-Computer Interaction (Fachpraktikum, WS 2020/21).*

## Overview

Earlier work tried to *reconstruct* the image a participant saw from their fMRI scan. The reconstructions tend to be blurry or show the wrong object. This project asks a simpler, easier-to-check question instead: **given the brain activity, which class of stimulus was shown?**

Two deep-learning approaches were built and compared:

| Model | Input | Framework |
|---|---|---|
| **LSTM** | Time series of voxel activity from 10 visual-cortex regions of interest (ROIs) | TensorFlow / Keras |
| **3D-CNN** | Volumetric fMRI frames (5 time steps × 91 × 109 × 91) | PyTorch |

## Approach

**Data: [BOLD5000](https://figshare.com/articles/dataset/BOLD5000/6459449/).** Four participants (CSI1–CSI4) viewed images from ImageNet, COCO and SUN.

**Labelling** (`lstm/labelImages/`). Each stimulus file name is matched by regex to its source dataset, then mapped to a super-category. The final 5 classes are *animal, artifact, scene, person, household*. Class distributions are plotted in `lstm/*Dist.png`.

**Preprocessing** (`utils/`)
- `nuisance_signal_regression.py` regresses out nuisance signals (motion parameters, CSF, white matter) with FSL. Global signal is deliberately kept.
- `multiply_mask.py` and `choose_salient_voxels_from_roi.py` apply ROI masks and explore voxel selection by correlation with the labels.
- ROIs used: higher-level visual areas (L/R PPA, RSC, OPA) and early visual cortex (L/R EarlyVis). ROI vectors are zero-padded to a common length so that all participants can be pooled.

**LSTM** (`lstm/lstm_classifier.ipynb`)
- Architecture: LSTM (with input and recurrent dropout) → Dense(64, ReLU) → Dense(24, ReLU) → Softmax. See `lstm/model_architecture.png`.
- Data from all 4 participants is pooled and shuffled, then split into 13,641 training and 3,411 held-out samples.
- Hyperparameters (learning rate, both dropout rates, batch size, hidden units) come from a randomised search. Training uses Adam with early stopping.

**3D-CNN** (`cnn/`)
- Three `Conv3d` layers with tanh activations, then a 1×1×1 convolutional classifier (`cnn/models/model.py`).
- Trained with a configurable CLI script (`cnn/main.py`). The run config is saved under `cnn/model_outputs/`.
- `cnn/fmri_tsne.ipynb` explores t-SNE embeddings of the activity and K-means clustering of them.

## Results

LSTM on the 3,411-sample held-out split, 5 classes. Numbers come from the confusion matrix saved in `lstm/lstm_classifier.ipynb`:

- **Overall accuracy ≈ 46.9%**, against a majority-class baseline of ≈ 31.4%.
- Best recognised classes: *animal* (67% recall) and *scene* (51%). *artifact* and *household* are most often confused with each other.

No final test metrics for the 3D-CNN are saved in the repository.

## Tech stack

Python 3.8 · TensorFlow 2.2 / Keras · PyTorch 1.7 · scikit-learn · nibabel / nilearn · h5py · FSL · Jupyter

## How to run

```bash
# Conda environment (pinned, linux-64)
conda create --name fmri --file requirements.txt
conda activate fmri
```

- **LSTM**: open and run `lstm/lstm_classifier.ipynb`. It prepares the ROI data and trains with the tuned hyperparameters.
- **3D-CNN**:
  ```bash
  cd cnn
  python3 main.py --epochs 100 -b 2 --early_stopping 40 --num_workers 6 --optimizer Adam --lr 0.01 --weight_decay 1e-4
  ```

> The notebooks and scripts point to the lab cluster's BOLD5000 copy. Download the dataset from the link above and update the data paths before running.

## Repository structure

```
lstm/   LSTM notebook, labelling scripts, class-distribution and architecture figures
cnn/    3D-CNN model, data loaders, loss, training script, HDF5 creation, t-SNE notebook
utils/  fMRI preprocessing: nuisance regression, ROI masking, voxel selection
```

## References

1. BOLD5000 dataset: https://figshare.com/articles/dataset/BOLD5000/6459449/
2. The LSTM implementation was adapted from [arashjamalian/fmriNet](https://github.com/arashjamalian/fmriNet).
