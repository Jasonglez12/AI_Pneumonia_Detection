# Pneumonia Detection with Transfer Learning (CECS 456)

Pneumonia detector built on the Kaggle Chest X-Ray dataset using two transfer learning models: ResNet50 and VGG16. Trains, evaluates, and compares both models; produces plots and saved weights for submission.

## Project Structure
- `data/chest_xray/` – dataset with `train/`, `val/`, `test/` and `NORMAL` / `PNEUMONIA` subfolders
- `1_data_exploration.ipynb` – dataset counts and sample visualizations
- `2_model_resnet50.ipynb` – ResNet50 training/evaluation, saves `models/resnet50_final.h5`
- `3_model_vgg16.ipynb` – VGG16 training/evaluation, saves `models/vgg16_final.h5`
- `4_model_comparison.ipynb` – evaluates both models on test set and compares metrics
- `models/` – saved model weights
- `results/` – figures, histories, and metrics CSV/JSON outputs

## Setup
1) Install Python 3.10+ and required packages (already provided per instructions). If needed:
   ```
   pip install tensorflow pandas numpy matplotlib seaborn scikit-learn pillow
   ```
2) Download the Kaggle Chest X-Ray Pneumonia dataset and place it at `data/chest_xray/` with the provided train/val/test structure.

## Usage
Run notebooks from the project root in order:
1. `1_data_exploration.ipynb` – verifies dataset, shows distributions, saves plots to `results/`.
2. `2_model_resnet50.ipynb` – trains ResNet50 (ImageNet weights, frozen base), augments data, trains 10 epochs, evaluates test set, saves curves, confusion matrix, metrics CSV/JSON, and `models/resnet50_final.h5`.
3. `3_model_vgg16.ipynb` – same pipeline for VGG16, outputs `models/vgg16_final.h5` plus plots/metrics.
4. `4_model_comparison.ipynb` – loads both saved models, evaluates on test set, creates comparison tables and plots in `results/`.

## Key Training Settings
- Input size: 224×224, RGB, `class_mode='binary'`
- Augmentation: rotation, shifts, shear, zoom, horizontal flip
- Optimizer: Adam, lr = 1e-4; loss: binary crossentropy
- Head: GlobalAveragePooling → Dense(256, relu) → Dropout(0.5) → Dense(1, sigmoid)
- Callbacks: ModelCheckpoint (best val_loss) and EarlyStopping (patience=3)
- Epochs: 10; Batch size: 32

## Results (fill after training)
| Model   | Accuracy | Precision | Recall | F1-score |
|---------|----------|-----------|--------|----------|
| ResNet50|          |           |        |          |
| VGG16   |          |           |        |          |

## Notes
- All plots saved at 300 DPI in `results/`.
- GPU (e.g., RTX 3070 Ti) recommended; expected training time ~30–60 minutes per model.

