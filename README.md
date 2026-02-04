# DINO-v3-for-Mitral-Valve-Segmentation
## DATASET:

- DATASET A: 39 high-resolution medical videos from patients with moderate to severe mitral valve disease. Each video includes an annotated window of interest and 3 annotated frames.
19/39 videos were used for training and 20/39 for testing.
Resolution: 800 × 600.

- DATASET B: 46 low-resolution videos (128 × 128). The dataset includes both healthy and pathological anatomy. Each video contains an annotated window of interest and 3 annotated frames.

## PROPOSED APPROACH

First, the videos are cleaned and processed using CLAHE, image inpainting, and a denoising model. The frames are then converted into tensors, padded, and resized to a fixed dimension of 128 × 128.

Since only three annotated frames are provided per video, it is not feasible to train a robust model using the original annotations alone. To address this limitation, high-quality pseudo-labels are generated using DINO. The three annotated frames are used to extract video features, and the labels are propagated throughout the video, helping to address challenges related to temporal dynamics and limited labeled data.

Next, part of the propagated labels is removed using PCA followed by GMM-based anomaly detection to eliminate low-quality or inconsistent annotations. Using this refined dataset, a U-Net model is trained to segment the input images.

Finally, before submission, a simple post-processing pipeline is applied to resize the segmentation outputs back to their original dimensions and interpolate the probability masks.
# HOW TO USE THE REPO
## Preprocessing
To save time during training, you can preprocess all videos offline using `preprocess_dataset.py`.
```bash
python task2_aml/preprocess_dataset.py --path /path/to/data --save_path /path/to/save/preprocessed
```

## Training
To train the model, use `main.py`. You can utilize the preprocessed data and optionally filter frames.

**Example Command:**
```bash
!python task2_aml/main.py \
    --path /content/drive/MyDrive/mv_data \
    --save_path task2_aml/models \
    --preprocessed_dir /content/drive/MyDrive/mv_data/preprocessed \
    --filter_frames /content/drive/MyDrive/mv_data/kept_frames.pt
```

### Arguments
- `--path`: Path to the dataset directory.
- `--save_path`: Directory to save trained models.
- `--preprocessed_dir`: Directory containing preprocessed `.pt` files (from `preprocess_dataset.py`).
- `--filter_frames`: Path to a `.pt` file specifying which frames to keep (e.g., after anomaly detection).

**Note:** To train on **all frames** (disabling filtering), set `--filter_frames false`.
