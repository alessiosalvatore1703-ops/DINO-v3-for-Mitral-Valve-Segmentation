"""
Script to generate accumulated bounding boxes for test videos using a pretrained bbox model.
Saves the bbox data and generates GIFs for visualization.
"""

import utils
import numpy as np
import torch
import argparse
import os
import gc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
from unet import UNet


def generate_bbox_for_video(video, bbox_model, device, batch_size=64, shrink_margin=0.1):
    """
    Generate accumulated bounding box for a single video.
    
    Args:
        video: Video object with processed_data
        bbox_model: Pretrained bbox UNet model
        device: torch device
        batch_size: Batch size for inference
        shrink_margin: Fraction to shrink the bbox (0.1 = shrink by 10% on each side)
    
    Returns:
        global_bbox_mask: (1, H, W) tensor with the accumulated bbox mask
        pred_bbox: (x_min, y_min, x_max, y_max) tuple in original coordinates
    """
    assert video.processed_data is not None, f"Video {video.name} missing processed data"
    
    vid, label, box = video.processed_data
    vid = vid.to(device)
    
    # First pass: find the union of all bboxes in the video
    accumulated_mask = torch.zeros((1, vid.shape[2], vid.shape[3]), device=device)
    
    for i in range(0, vid.shape[0], batch_size):
        batch = vid[i:min(i+batch_size, vid.shape[0])]
        preds = bbox_model.predict(batch, threshold=0.5)  # (B, 1, H, W)
        
        batch_max_mask = torch.max(preds, dim=0)[0]  # (1, H, W)
        accumulated_mask = torch.maximum(accumulated_mask, batch_max_mask)
    
    # Compute global bounding box from accumulated mask
    if accumulated_mask.sum() > 0:
        coords = torch.nonzero(accumulated_mask[0])  # (N, 2)
        y_min, y_max = coords[:, 0].min(), coords[:, 0].max()
        x_min, x_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Apply shrink margin to make bbox tighter
        if shrink_margin > 0:
            width = x_max - x_min
            height = y_max - y_min
            shrink_x = int(width * shrink_margin)
            shrink_y = int(height * shrink_margin)
            x_min = x_min + shrink_x
            x_max = x_max - shrink_x
            y_min = y_min + shrink_y
            y_max = y_max - shrink_y
            # Ensure valid bbox
            if x_max <= x_min:
                x_max = x_min + 1
            if y_max <= y_min:
                y_max = y_min + 1
        
        global_bbox_mask = torch.zeros_like(accumulated_mask)
        global_bbox_mask[:, y_min:y_max+1, x_min:x_max+1] = 1.0
        
        # Store bbox in original coordinates
        H_orig, W_orig = video.video.shape[:2]
        S = vid.shape[2]
        scale = max(H_orig, W_orig) / S
        
        # (x_min, y_min, x_max, y_max)
        pred_bbox = (
            int(x_min.item() * scale),
            int(y_min.item() * scale),
            int(x_max.item() * scale),
            int(y_max.item() * scale)
        )
        # Clip to original image dimensions
        pred_bbox = (
            max(0, min(W_orig, pred_bbox[0])),
            max(0, min(H_orig, pred_bbox[1])),
            max(0, min(W_orig, pred_bbox[2])),
            max(0, min(H_orig, pred_bbox[3]))
        )
        
        # Also compute bbox in preprocessed coordinates
        pred_bbox_preprocessed = (
            int(x_min.item()),
            int(y_min.item()),
            int(x_max.item()),
            int(y_max.item())
        )
    else:
        global_bbox_mask = torch.zeros_like(accumulated_mask)
        pred_bbox = (0, 0, 0, 0)
        pred_bbox_preprocessed = (0, 0, 0, 0)
    
    return global_bbox_mask.cpu(), pred_bbox, pred_bbox_preprocessed, accumulated_mask.cpu()


def save_bbox_gif(video, pred_bbox_preprocessed, accumulated_mask, global_bbox_mask, output_path, fps=10):
    """
    Save a GIF visualization of the preprocessed video with the bounding box overlay.
    
    Args:
        video: Video object with processed_data (preprocessed images)
        pred_bbox_preprocessed: (x_min, y_min, x_max, y_max) in preprocessed coordinates
        accumulated_mask: The accumulated mask from bbox predictions
        global_bbox_mask: The global bbox mask (rectangle)
        output_path: Path to save the GIF
        fps: Frames per second
    """
    images = []
    x_min, y_min, x_max, y_max = pred_bbox_preprocessed
    
    # Get preprocessed video: (T, C, H, W)
    vid, _, _ = video.processed_data
    T_frames = vid.shape[0]
    
    for t in range(T_frames):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Left: Preprocessed frame with bbox
        ax1 = axes[0]
        # vid is (T, C, H, W), get frame t and squeeze channel dim
        frame = vid[t, 0].numpy()  # (H, W)
        ax1.imshow(frame, cmap='gray')
        ax1.set_title(f"{video.name} - Frame {t}")
        
        # Draw bounding box
        if x_max > x_min and y_max > y_min:
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax1.add_patch(rect)
        ax1.axis('off')
        
        # Middle: Accumulated mask (heatmap of all predictions)
        ax2 = axes[1]
        ax2.imshow(accumulated_mask[0].numpy(), cmap='hot')
        ax2.set_title("Accumulated Predictions")
        ax2.axis('off')
        
        # Right: Frame with bbox mask applied
        ax3 = axes[2]
        masked_frame = frame * global_bbox_mask[0].numpy()
        ax3.imshow(masked_frame, cmap='gray')
        ax3.set_title("Masked Frame")
        ax3.axis('off')
        
        # Convert figure to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, :3]  # Remove alpha channel
        images.append(image)
        plt.close(fig)
    
    # Save GIF
    imageio.mimsave(output_path, images, fps=fps)
    print(f"Saved GIF to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate bounding boxes for test videos")
    parser.add_argument("-p", "--path", type=str, default="data", help="Path to data directory")
    parser.add_argument("-pd", "--preprocessed_dir", type=str, default="data/preprocessed", 
                        help="Path to preprocessed .pt files")
    parser.add_argument("--bbox_model_path", type=str, default="models/bbox/model.pth",
                        help="Path to pretrained bbox model weights")
    parser.add_argument("-o", "--output_dir", type=str, default="bbox_output",
                        help="Output directory for bbox data and GIFs")
    parser.add_argument("--gif_dir", type=str, default="bbox_gifs",
                        help="Directory to save GIF visualizations")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--shrink_margin", type=float, default=0.0, 
                        help="Fraction to shrink bbox (0.0 = no shrinking, 0.15 = shrink by 15%% on each side)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    test_data = utils.load_zipped_pickle(os.path.join(args.path, 'test.pkl'))
    test_videos = utils.preprocess_test_data(test_data)
    del test_data
    print(f"Loaded {len(test_videos)} test videos")
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {args.preprocessed_dir}...")
    for video in test_videos:
        pt_path = os.path.join(args.preprocessed_dir, f"{video.name}.pt")
        if os.path.exists(pt_path):
            try:
                video.load_preprocessed_data(pt_path)
            except Exception as e:
                print(f"Failed to load {pt_path}: {e}")
                continue
        else:
            print(f"Warning: Preprocessed file for {video.name} not found at {pt_path}")
    
    # Load bbox model
    print(f"Loading bbox model from {args.bbox_model_path}...")
    bbox_model = UNet(n_channels=[1, 64, 128, 256, 512], n_classes=1)
    bbox_model.load_state_dict(torch.load(args.bbox_model_path, map_location=device))
    bbox_model.to(device)
    bbox_model.eval()
    print("Bbox model loaded successfully")
    
    # Process each test video
    bbox_results = {}
    
    with torch.no_grad():
        for i, video in enumerate(test_videos):
            print(f"\nProcessing video {i+1}/{len(test_videos)}: {video.name}")
            
            if video.processed_data is None:
                print(f"  Skipping - no processed data")
                continue
            
            # Generate bbox
            global_bbox_mask, pred_bbox, pred_bbox_preprocessed, accumulated_mask = generate_bbox_for_video(
                video, bbox_model, device, args.batch_size, args.shrink_margin
            )
            
            print(f"  BBox (original coords): {pred_bbox}")
            print(f"  BBox (preprocessed coords): {pred_bbox_preprocessed}")
            
            # Store results
            video.pred_bbox = pred_bbox
            bbox_results[video.name] = {
                'pred_bbox': pred_bbox,
                'pred_bbox_preprocessed': pred_bbox_preprocessed,
                'global_bbox_mask': global_bbox_mask,
                'accumulated_mask': accumulated_mask,
            }
            
            # Save bbox data for this video
            bbox_save_path = os.path.join(args.output_dir, f"{video.name}_bbox.pt")
            torch.save({
                'pred_bbox': pred_bbox,
                'pred_bbox_preprocessed': pred_bbox_preprocessed,
                'global_bbox_mask': global_bbox_mask,
                'accumulated_mask': accumulated_mask,
            }, bbox_save_path)
            print(f"  Saved bbox data to {bbox_save_path}")
            
            # Save GIF visualization (using preprocessed images)
            gif_path = os.path.join(args.gif_dir, f"{video.name}_bbox.gif")
            save_bbox_gif(video, pred_bbox_preprocessed, accumulated_mask, global_bbox_mask, gif_path)
            
            # Cleanup to save memory
            gc.collect()
            torch.cuda.empty_cache()
    
    # Save summary of all bboxes
    summary_path = os.path.join(args.output_dir, "all_bboxes.pt")
    summary_data = {name: {'pred_bbox': data['pred_bbox'], 'pred_bbox_preprocessed': data['pred_bbox_preprocessed']} 
                    for name, data in bbox_results.items()}
    torch.save(summary_data, summary_path)
    print(f"\nSaved bbox summary to {summary_path}")
    
    print(f"\nDone! Processed {len(bbox_results)} videos")
    print(f"BBox data saved to: {args.output_dir}/")
    print(f"GIFs saved to: {args.gif_dir}/")


if __name__ == "__main__":
    main()

