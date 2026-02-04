import argparse
import os
import torch
import utils
from drunet import UNetRes
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

SIZE = (112, 112)

def main():
    parser = argparse.ArgumentParser(description="Preprocess and store videos")
    parser.add_argument("--path", type=str, default="data", help="Path to data directory containing pickles")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save preprocessed tensors")
    parser.add_argument("--denoiser_path", type=str, default=None, help="Path to denoiser model weights")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print(f"Using device: {device}")

    # Load Denoiser if provided
    denoiser_model = None
    if args.denoiser_path and os.path.exists(args.denoiser_path):
        print(f"Loading denoiser from {args.denoiser_path}")
        n_channels = 1
        denoiser_model = UNetRes(in_nc=n_channels + 1,
                    out_nc=n_channels,
                    nc=[64, 128, 256, 512],
                    nb=4,
                    act_mode='R',
                    downsample_mode="strideconv",
                    upsample_mode="convtranspose"
                    )
        denoiser_model.load_state_dict(torch.load(args.denoiser_path, map_location=device), strict=True)
        denoiser_model.to(device)
        denoiser_model.eval()
    elif args.denoiser_path:
        print(f"Warning: Denoiser path {args.denoiser_path} provided but not found.")

    # Load Data
    print("Loading data...")
    train_data = utils.load_zipped_pickle(os.path.join(args.path, 'train.pkl'))
    test_data = utils.load_zipped_pickle(os.path.join(args.path, 'test.pkl'))
    
    train_videos = utils.preprocess_train_data(train_data)
    test_videos = utils.preprocess_test_data(test_data)
    
    all_videos = train_videos + test_videos
    print(f"Found {len(all_videos)} videos in total.")

    # load pseudo labels from data/pseudo_labels/patient_id.npz
    pseudo_labels = {}
    for f in os.listdir(os.path.join(args.path, 'pseudo_labels')):
        if f.endswith('.npz'):
            # unpack npz file
            key = f.split('.')[0]
            npz = np.load(os.path.join(args.path, 'pseudo_labels', f))
            pseudo_labels[key] = (npz['masks'], npz['conf'])
    
    # load pseudo labels into videos
    for video in train_videos:
        video.load_pseudo_label(pseudo_labels)

    print("Preprocessing videos...")
    for video in tqdm(all_videos):
        try:
            vid, label, box = video.preprocess_v2(size=SIZE, device=device, denoiser_model=denoiser_model)
            
            vid = vid.cpu()
            if label is not None:
                label = label.cpu()
            if box is not None:
                box = box.cpu()

            save_file = os.path.join(args.save_path, f"{video.name}.pt")
            torch.save((vid, label, box), save_file)
            
        except Exception as e:
            print(f"Error processing {video.name}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
