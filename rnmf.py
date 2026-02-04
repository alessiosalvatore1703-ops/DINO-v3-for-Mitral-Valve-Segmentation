
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from drunet import UNetRes
from tqdm import tqdm
import numpy as np
from torch.nn.functional import normalize

device = "cuda" if torch.cuda.is_available() else "cpu"

SIZE = (112, 112)

def rnmf(data, rank, regval, tol=1e-6, max_iter=10000):
    """https://github.com/neel-dey/robust-nmf/tree/master"""
    device = data.device

    shape = data.shape
    data = data.view(shape[0], -1)

    basis = torch.rand(data.shape[0], rank).to(device)
    coeff = torch.rand(rank, data.shape[1]).to(device)
    outlier = torch.rand(data.shape).to(device)

    approx = basis @ coeff + outlier + tol
    err = torch.zeros(max_iter + 1).to(device)
    obj = torch.zeros(max_iter + 1).to(device)

    # Gaussian noise assumption, beta=2
    error = lambda a, b: 0.5 * (torch.norm(a - b, p="fro") ** 2)
    objective = lambda e, r, o: e[-1] + r * torch.sum(torch.sqrt(torch.sum(o**2, dim=0)))

    err[0] = error(data, approx)
    obj[0] = objective(err, regval, outlier)

    iter = 0
    for iter in range(max_iter):
        outlier *= data / (approx + regval * normalize(outlier, p=2, dim=0, eps=tol))
        approx = basis @ coeff + outlier + tol

        coeff *= (basis.t() @ data) / (basis.t() @ approx)
        approx = basis @ coeff + outlier + tol

        basis *= (data @ coeff.t()) / (approx @ coeff.t())
        approx = basis @ coeff + outlier + tol

        err[iter + 1] = error(data, approx)
        obj[iter + 1] = objective(err, regval, outlier)

        if torch.abs((obj[iter] - obj[iter + 1]) / obj[iter]) <= tol:
            break

        if iter == (max_iter - 1):
            print("rnmf reached max_iter")

    outlier = outlier.view(shape)

    return basis, coeff, outlier, err[:iter], obj[:iter]

def main():
    parser = argparse.ArgumentParser(description="Preprocess and store videos with RNMF (Outlier Extraction)")
    parser.add_argument("--path", type=str, required=True, help="Path to directory containing preprocessed .pt files")
    parser.add_argument("--save_path", type=str, default="data/rnmf_processed", help="Path to save preprocessed tensors")
    parser.add_argument("--test_first", action="store_true", help="Only process the first video for testing")
    parser.add_argument("--rank", type=int, default=2, help="Rank for RNMF background")
    parser.add_argument("--lam", type=float, default=0.1, help="Lambda sparsity parameter for RNMF")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print(f"Using device: {device}")

    # Load Data from preprocessed directory
    print(f"Loading preprocessed files from {args.path}...")
    files = [f for f in os.listdir(args.path) if f.endswith('.pt')]
    files.sort()
    
    if not files:
        print(f"No .pt files found in {args.path}")
        return

    print(f"Found {len(files)} files.")

    files_to_process = files
    if args.test_first:
        print("TEST MODE: Processing only the first video.")
        files_to_process = files[:1]

    for filename in tqdm(files_to_process):
        try:
            filepath = os.path.join(args.path, filename)
            # Load (vid, label, box)
            # vid: (T, C, H, W)
            vid, label, box = torch.load(filepath, map_location=device)
            
            # Ensure vid is on device
            vid = vid.to(device)

            # 2. Apply RNMF to extract outliers (movement)
            print(f"Running RNMF on {filename}...")
            # Unpack the tuple result from rnmf
            _, _, movement, _, _ = rnmf(vid, rank=args.rank, regval=args.lam, max_iter=100)
            
            # movement is on device
            movement = movement.cpu()
            if label is not None:
                label = label.cpu()
            if box is not None:
                box = box.cpu()
            
            # 3. Save
            # Filename is typically "VideoName.pt"
            video_name = os.path.splitext(filename)[0]
            save_file = os.path.join(args.save_path, f"{video_name}.pt")
            
            torch.save((movement, label, box), save_file)
            print(f"Saved RNMF result to {save_file}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    print("Done.")

if __name__ == "__main__":
    main()