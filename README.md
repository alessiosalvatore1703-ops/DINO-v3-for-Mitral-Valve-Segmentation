# Task 2 Project

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
