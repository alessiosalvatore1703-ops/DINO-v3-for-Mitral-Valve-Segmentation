import utils
import numpy as np
import pandas as pd
import dataset
import gc
from torch.utils.data import DataLoader
import wandb
import torch
from unet import UNet
from unet_se import UNetSE
from drunet import UNetRes
from vit_seg import ViTSeg, vit_tiny_seg, vit_small_seg, vit_base_seg
from runet import RUNet
from losses import DiceLoss, PowerJaccardLoss
import argparse
import os
from torchmetrics import JaccardIndex
import torchvision.transforms.v2 as T
import torch.nn.functional as F
import elasticdeform.torch as etorch
from transforms import ElasticDeformation
from collections import defaultdict
from training import train_model

def load_pseudo_labels(train_videos, test_videos, data_path, filter_frames_path):
    # load pseudo labels from data/pseudo_labels/patient_id.npz
    pseudo_labels = {}
    for f in os.listdir(os.path.join(data_path, 'pseudo_labels')):
        if f.endswith('.npz'):
            # unpack npz file
            key = f.split('.')[0]
            npz = np.load(os.path.join(data_path, 'pseudo_labels', f))
            pseudo_labels[key] = (npz['masks'], npz['conf'])
    
    # load pseudo labels into videos
    for video in train_videos:
        video.load_pseudo_label(pseudo_labels)

    # only use frames kept after anomaly detection
    if filter_frames_path and filter_frames_path != "false":
        kept_frames = torch.load(filter_frames_path)
        for video in train_videos + test_videos:
            if video.name not in kept_frames:
                print(f"Video {video.name} not in kept frames")
                continue
            print(f"Video {video.name} kept frames {len(kept_frames[video.name])}")
            video.frames = kept_frames[video.name]

def load_preprocessed_data_and_cleanup(train_videos, test_videos, preprocessed_dir, rnmf_dir):
    # Load preprocessed data if path provided
    if preprocessed_dir:
        print(f"Loading preprocessed data from {preprocessed_dir}...")
        for video in train_videos + test_videos:
            pt_path = os.path.join(preprocessed_dir, f"{video.name}.pt")
            if os.path.exists(pt_path):
                try: 
                    video.load_preprocessed_data(pt_path)
                    # Load RNMF data
                    video.load_rnmf_data(rnmf_dir)
                except Exception as e:
                    print(f"Failed to load {pt_path}: {e}")
            else:
                print(f"Warning: Preprocessed file for {video.name} not found at {pt_path}")
                assert False

    # we dont care about train videos before preprocessing
    for video in train_videos:
        video.videos = None
        video.masks = None
        video.pseudo_label = None

if __name__ == "__main__":        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="data", help="Path to data directory")
    parser.add_argument("-s", "--save_path", type=str, default="models", help="Path to save models")
    parser.add_argument("-l", "--log_interval", type=int, default=10, help="Log interval for training")
    parser.add_argument("-pd", "--preprocessed_dir", type=str, default="data/preprocessed", help="Path to directory containing preprocessed .pt files")
    parser.add_argument("-ff", "--filter_frames", type=str, default="data/kept_frames.pt", help="Only use frames kept after anomaly detection")
    parser.add_argument("--rnmf_dir", type=str, default="data/rnmf_processed", help="Path to rnmf processed data")
    parser.add_argument("--train_bbox", action="store_true", help="Train bbox model")
    parser.add_argument("--bbox_model_dir", type=str, default="models/bbox", help="Path to save bbox model")
    parser.add_argument("--precomputed_bbox_dir", type=str, default=None, 
                        help="Path to precomputed bbox files from generate_bbox.py (skips bbox model inference)")
    args = parser.parse_args()

    config = {
        "epochs": 5,
        "batch_size": 64,
        "lr": 1e-3,
        "patience": 4,
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,
        "log_interval": args.log_interval,
        "save_path": args.save_path,
        "data_path": args.path,
        "preprocessed_dir": args.preprocessed_dir,
        "rnmf_dir": args.rnmf_dir,
        "scheduler_mode": "min",
    }
    
    wandb.init(project="eth-aml-2025-project-task-2", config=config)

    # load data
    train_data = utils.load_zipped_pickle(os.path.join(args.path, 'train.pkl'))
    test_data = utils.load_zipped_pickle(os.path.join(args.path, 'test.pkl'))
    
    # preprocess data
    train_videos = utils.preprocess_train_data(train_data)
    del train_data
    test_videos = utils.preprocess_test_data(test_data)
    del test_data

    # load pseudo labels
    load_pseudo_labels(train_videos, test_videos, args.path, args.filter_frames)

    # Load preprocessed data if path provided
    load_preprocessed_data_and_cleanup(train_videos, test_videos, args.preprocessed_dir, args.rnmf_dir)


    amateur_videos = []
    expert_videos = []
    for video in train_videos:
        if video.label_type == "amateur":
            amateur_videos.append(video)
        elif video.label_type == "expert":
            expert_videos.append(video)

    # Split experts: 15 for training, 5 for validation
    expert_train = expert_videos[:15]
    expert_val = expert_videos[15:20]
    
    print(f"Amateur videos: {len(amateur_videos)}")
    print(f"Expert videos for training: {len(expert_train)}")
    print(f"Expert videos for validation: {len(expert_val)}")

    transforms = T.Compose([
        T.RandomAffine(
            degrees=(-15, 15),
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=(-15, 15),
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        # T.RandomApply([T.ElasticTransform(alpha=5.0)], p=0.1),
        # T.RandomApply([ElasticDeformation(sigma=2, points=3)], p=0.1),
        T.RandomPerspective(),
    ])

    # transforms = None
    # amateur_dataset = amateur_dataset[:1]
    # expert_dataset = expert_dataset[:1]

    # Bbox training
    if args.train_bbox:
        print("Training bbox model...")

        bbox_transforms = T.Compose([
            T.RandomAffine(
                degrees=(-15, 15),
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=(-15, 15),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomPerspective(),
        ])

        # Training: all amateurs + 15 experts, Validation: 5 experts
        bbox_train_videos = amateur_videos + expert_train
        bbox_val_videos = expert_val
        
        bbox_train_dataset = dataset.FramesDataset(bbox_train_videos, target="box", transforms=bbox_transforms, box_dilation=5)
        bbox_val_dataset = dataset.FramesDataset(bbox_val_videos, target="box", transforms=None, box_dilation=5)

        print(f"Bbox train dataset size: {len(bbox_train_dataset)} (amateurs + 15 experts)")
        print(f"Bbox val dataset size: {len(bbox_val_dataset)} (5 experts)")

        bbox_train_loader = DataLoader(bbox_train_dataset, batch_size=config["batch_size"], shuffle=True)
        bbox_val_loader = DataLoader(bbox_val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # Initialize UNet
        # Input: 1 channel (gray frame), Output: 1 channel (binary box mask)
        bbox_model = UNet(n_channels=[1, 64, 128, 256, 512], n_classes=1)
        
        bbox_optimizer = torch.optim.AdamW(bbox_model.parameters(), lr=config["lr"])
        bbox_criterion = PowerJaccardLoss() 
        bbox_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            bbox_optimizer, mode=config["scheduler_mode"], factor=config["scheduler_factor"], patience=config["scheduler_patience"]
        )
        bbox_metric = JaccardIndex(task="binary").to(device)
        
        if not os.path.exists(args.bbox_model_dir):
            os.makedirs(args.bbox_model_dir)
            
        print("Starting Bbox Model Training")
        train_model(
            bbox_model,
            bbox_train_loader,
            bbox_val_loader,
            epochs=config["epochs"],
            optimizer=bbox_optimizer,
            loss_fn=bbox_criterion,
            metric_fn=bbox_metric,
            device=device,
            patience=config["patience"],
            save_path=args.bbox_model_dir,
            log_interval=config["log_interval"],
            scheduler=bbox_scheduler,
        )
        
        # Cleanup
        del bbox_optimizer, bbox_train_loader, bbox_val_loader, bbox_train_dataset, bbox_val_dataset
        gc.collect()
        torch.cuda.empty_cache()
        print("Bbox model training finished.")
        
        # Load best bbox model for inference
        bbox_model.load_state_dict(torch.load(os.path.join(args.bbox_model_dir, "model.pth")))
        
    else:
        # Load bbox model if not training
        print("Loading bbox model from disk...")
        bbox_model = UNet(n_channels=[1, 64, 128, 256, 512], n_classes=1)
        try:
            bbox_model.load_state_dict(torch.load(os.path.join(args.bbox_model_dir, "model.pth")))
            print(f"Bbox model loaded from {args.bbox_model_dir}")
        except Exception as e:
            print(f"Failed to load bbox model: {e}")
            bbox_model = None

    # Filter frames with bbox - either from precomputed files or using bbox model
    def apply_bbox_mask_to_video(video, global_bbox_mask, device):
        """Apply a bbox mask to filter video and rnmf data"""
        vid, label, box = video.processed_data
        vid = vid.to(device)
        
        rnmf = video.rnmf_data.to(device)
        if rnmf.ndim == 3:
            rnmf = rnmf.unsqueeze(1)
        
        global_bbox_mask = global_bbox_mask.to(device)
        
        BATCH_SIZE = 64
        filtered_frames = []
        filtered_rnmf_frames = []
        
        for i in range(0, vid.shape[0], BATCH_SIZE):
            batch = vid[i:min(i+BATCH_SIZE, vid.shape[0])]
            filtered_batch = batch * global_bbox_mask
            filtered_frames.append(filtered_batch.cpu())
            
            batch_rnmf = rnmf[i:min(i+BATCH_SIZE, rnmf.shape[0])]
            filtered_batch_rnmf = batch_rnmf * global_bbox_mask
            filtered_rnmf_frames.append(filtered_batch_rnmf.cpu())
        
        filtered_vid = torch.cat(filtered_frames, dim=0)
        video.processed_data = (filtered_vid, label, box)
        
        filtered_rnmf = torch.cat(filtered_rnmf_frames, dim=0)
        video.rnmf_data = filtered_rnmf

    # Option 1: Use precomputed bboxes from generate_bbox.py
    if args.precomputed_bbox_dir is not None:
        print(f"Loading precomputed bboxes from {args.precomputed_bbox_dir}...")
        
        for video in train_videos + test_videos:
            bbox_file = os.path.join(args.precomputed_bbox_dir, f"{video.name}_bbox.pt")
            if os.path.exists(bbox_file):
                bbox_data = torch.load(bbox_file)
                global_bbox_mask = bbox_data['global_bbox_mask']
                video.pred_bbox = bbox_data['pred_bbox']
                
                apply_bbox_mask_to_video(video, global_bbox_mask, device)
                print(f"  Loaded bbox for {video.name}: {video.pred_bbox}")
            else:
                print(f"  WARNING: No precomputed bbox for {video.name}, skipping filtering")
        
        print("Precomputed bbox loading complete.")
        gc.collect()
        torch.cuda.empty_cache()

    # Option 2: Compute bboxes using bbox model
    elif bbox_model is not None:
        print("Filtering frames with bbox model...")
        bbox_model.to(device)
        bbox_model.eval()
                
        with torch.no_grad():
            for video in train_videos + test_videos:
                # video.processed_data is (vid, label, box)
                # vid: (T, C, H, W)
                assert video.processed_data is not None, f"Video {video.name} missing processed data"
                assert video.rnmf_data is not None, f"Video {video.name} missing RNMF data"

                vid, label, box = video.processed_data
                vid = vid.to(device)
                
                # Predict mask for all frames                     
                BATCH_SIZE = 64
                 
                # First pass: find the union of all bboxes in the video
                accumulated_mask = torch.zeros((1, vid.shape[2], vid.shape[3]), device=device)
                 
                for i in range(0, vid.shape[0], BATCH_SIZE):
                     batch = vid[i:min(i+BATCH_SIZE, vid.shape[0])]
                     preds = bbox_model.predict(batch, threshold=0.5) # (B, 1, H, W)
                     
                     batch_max_mask = torch.max(preds, dim=0)[0] # (1, H, W)
                     accumulated_mask = torch.maximum(accumulated_mask, batch_max_mask)
                 
                # Compute global bounding box from accumulated mask
                if accumulated_mask.sum() > 0:
                     coords = torch.nonzero(accumulated_mask[0]) # (N, 2)
                     y_min, y_max = coords[:, 0].min(), coords[:, 0].max()
                     x_min, x_max = coords[:, 1].min(), coords[:, 1].max()
                     
                     global_bbox_mask = torch.zeros_like(accumulated_mask)
                     global_bbox_mask[:, y_min:y_max+1, x_min:x_max+1] = 1.0

                     # Store bbox in original coordinates
                     H_orig, W_orig = video.video.shape[:2]
                     S = vid.shape[2]
                     scale = max(H_orig, W_orig) / S
                     
                     # (x_min, y_min, x_max, y_max)
                     video.pred_bbox = (
                         int(x_min.item() * scale),
                         int(y_min.item() * scale),
                         int(x_max.item() * scale),
                         int(y_max.item() * scale)
                     )
                     # Clip to original image dimensions
                     video.pred_bbox = (
                         max(0, min(W_orig, video.pred_bbox[0])),
                         max(0, min(H_orig, video.pred_bbox[1])),
                         max(0, min(W_orig, video.pred_bbox[2])),
                         max(0, min(H_orig, video.pred_bbox[3]))
                     )
                else:
                     global_bbox_mask = torch.zeros_like(accumulated_mask)
                     video.pred_bbox = (0, 0, 0, 0)
                
                # Apply the mask
                apply_bbox_mask_to_video(video, global_bbox_mask, device)

        print("Bbox filtering complete.")
        gc.collect()
        torch.cuda.empty_cache()


    
    # Training: all amateurs + 15 experts, Validation: 5 experts
    train_videos_combined = amateur_videos + expert_train
    val_videos_combined = expert_val
    
    train_mask_dataset = dataset.Frames3Dataset(train_videos_combined, target="mask", denoiser_model=None, transforms=transforms)
    val_mask_dataset = dataset.Frames3Dataset(val_videos_combined, target="mask", denoiser_model=None, transforms=None)  # No transforms for validation

    train_mask_dataloader = DataLoader(train_mask_dataset, batch_size=config["batch_size"], shuffle=True)
    val_mask_dataloader = DataLoader(val_mask_dataset, batch_size=config["batch_size"], shuffle=False)

    print(f"Train dataset size: {len(train_mask_dataset)} (amateurs + 15 experts)")
    print(f"Val dataset size: {len(val_mask_dataset)} (5 experts)")

    label_model = UNetSE(n_channels=6, n_classes=3)

    label_model.to(device)
    label_model.train()

    label_optimizer = torch.optim.AdamW(label_model.parameters(), lr=config["lr"])
    label_criterion = PowerJaccardLoss()
    label_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(label_optimizer, mode=config["scheduler_mode"], factor=config["scheduler_factor"], patience=config["scheduler_patience"])
    label_metric = JaccardIndex(task="binary").to(device)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train_model(
        label_model,
        train_mask_dataloader,
        val_mask_dataloader,
        epochs=config["epochs"],
        optimizer=label_optimizer,
        loss_fn=label_criterion,
        metric_fn=label_metric,
        device=device,
        patience=config["patience"],
        save_path=config["save_path"],
        log_interval=config["log_interval"],
        scheduler=label_scheduler,
    )

    # Free up memory
    del train_videos
    del amateur_videos
    del expert_videos
    del expert_train
    del expert_val
    del train_videos_combined
    del val_videos_combined
    del train_mask_dataset
    del val_mask_dataset
    del train_mask_dataloader
    del val_mask_dataloader
    gc.collect()
    torch.cuda.empty_cache()

    # load best model
    label_model.load_state_dict(torch.load(os.path.join(args.save_path, "model.pth")))

    # predict labels
    label_model.eval()
    
    if os.path.exists("submission.csv"):
        os.remove("submission.csv")

    if not os.path.exists("predictions"):
        os.makedirs("predictions")

    for i, video in enumerate(test_videos):
        # preprocess
        assert video.processed_data is not None, f"Video {video.name} missing processed data"
        assert video.rnmf_data is not None, f"Video {video.name} missing RNMF data"

        vid, _, _ = video.processed_data
        vid = vid.to(device)
        rnmf_vid = video.rnmf_data.to(device)
            
        
        # predict
        votes = torch.zeros_like(vid)
        counts = torch.zeros_like(vid)

        if rnmf_vid.ndim == 3:
            rnmf_vid = rnmf_vid.unsqueeze(1)

        BATCH_SIZE = 128
        T_vid = vid.shape[0]
        
        with torch.no_grad():
            for i in range(0, T_vid, BATCH_SIZE):
                # Vectorized batch construction
                end_idx = min(i + BATCH_SIZE, T_vid)
                batch_indices = torch.arange(i, end_idx, device=device)
                
                # Calculate neighboring indices with clamping
                idx_prev = torch.clamp(batch_indices - 1, min=0)
                idx_curr = batch_indices
                idx_next = torch.clamp(batch_indices + 1, max=T_vid - 1)
                
                # Gather frames: (B, C, H, W) -> (B, 1, H, W)
                frames_prev = vid[idx_prev]
                frames_curr = vid[idx_curr]
                frames_next = vid[idx_next]
                
                rnmf_prev = rnmf_vid[idx_prev]
                rnmf_curr = rnmf_vid[idx_curr]
                rnmf_next = rnmf_vid[idx_next]
                
                # Concatenate along channel dimension (dim=1) to get (B, 3, H, W)
                X_frames = torch.cat([frames_prev, frames_curr, frames_next], dim=1)
                X_rnmf = torch.cat([rnmf_prev, rnmf_curr, rnmf_next], dim=1)
                
                # Combined: (B, 6, H, W)
                batch = torch.cat([X_frames, X_rnmf], dim=1)

                # Filter batch with bbox_model
                if bbox_model is not None:
                    bbox_model.eval()
                    # frames_curr is (B, 1, H, W)
                    bbox_preds = bbox_model.predict(frames_curr, threshold=0.5) # (B, 1, H, W)
                    batch = batch * bbox_preds
                else:
                    assert False

                
                # Predict
                curr_preds = label_model.predict(batch, threshold=0.6) # (B, 3, H, W)
                
                # Accumulate votes using index_put_ (faster than loop)
                # preds channel 0 -> idx_prev
                # preds channel 1 -> idx_curr
                # preds channel 2 -> idx_next
                
                preds_0 = curr_preds[:, 0].unsqueeze(1) # (B, 1, H, W)
                preds_1 = curr_preds[:, 1].unsqueeze(1)
                preds_2 = curr_preds[:, 2].unsqueeze(1)
                
                votes.index_put_((idx_prev,), preds_0, accumulate=True)
                votes.index_put_((idx_curr,), preds_1, accumulate=True)
                votes.index_put_((idx_next,), preds_2, accumulate=True)
                
                # Accumulate counts
                ones = torch.ones_like(preds_0)
                counts.index_put_((idx_prev,), ones, accumulate=True)
                counts.index_put_((idx_curr,), ones, accumulate=True)
                counts.index_put_((idx_next,), ones, accumulate=True)
            
            # Majority voting: >= 2 votes out of 3
            pred = (votes >= 2).float()

        # postprocess (+ convert to boolean mask)
        pred_label = video.postprocess(pred) # (H, W, T)

        # save prediction
        video.label = pred_label

        # Store result tuple
        # (prediction tensor (postprocessed) and global bbox permuted correctly)
        # video.pred_bbox is (x_min, y_min, x_max, y_max)
        torch.save((torch.from_numpy(pred_label), getattr(video, 'pred_bbox', (0, 0, 0, 0))), f"predictions/{video.name}.pt")
        print(f"Saved prediction to predictions/{video.name}.pt")
        
        
        print(f"Video {video.name} predicted shape: {pred_label.shape}")

        if i == 0:
            video.save_gif(path="plots")
        
        utils.append_to_submission(video, "submission.csv")

        # cleanup
        video.label = None
        video.processed_data = None
        video.rnmf_data = None
        if hasattr(video, 'video'):
            video.video = None
        gc.collect()

    # alert 

    # alert 
    wandb.alert(
        title="Run ended",
        text="Run ended successfully",
    )
    wandb.finish()