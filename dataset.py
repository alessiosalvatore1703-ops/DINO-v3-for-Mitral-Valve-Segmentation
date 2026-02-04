import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from video import Video

try:
    from torchvision.tv_tensors import Image, Mask
except:
    from torchvision.datapoints import Image, Mask

device = "cuda" if torch.cuda.is_available() else "cpu"

class FramesDataset(Dataset):
    '''
    Simple dataset for video frames for baseline supervised approach
    '''

    def __init__(
        self,
        videos: list[Video],
        target: str,
        denoiser_model=None,
        transforms = None,
        size=(112, 112),
        box_dilation=None):


        self.transforms = transforms
        self.target = target

        # only keep frames that are labeled
        self.X = []
        self.Y = []
        for video in videos:
            if video.processed_data is not None:
                # Use preprocessed data
                # (vid, label, box)
                x, y_label, y_box = video.processed_data

                # x is (T, C, H, W)
                # y_label is (T, C, H, W)
                # y_box is (C, H, W)
                for idx in video.frames:
                    self.X.append(x[idx, ...])
                    if target == "box":
                         # box is the same for all frames
                        if box_dilation:
                            # (C, H, W) -> (1, C, H, W)
                            y_b = y_box.unsqueeze(0)
                            # kernel size = 2*dilation + 1 to keep center
                            k = 2 * box_dilation + 1
                            y_b = F.max_pool2d(y_b, kernel_size=k, stride=1, padding=box_dilation)
                            self.Y.append(y_b.squeeze(0))
                        else:
                            self.Y.append(y_box)
                    elif target == "mask":
                        self.Y.append(y_label[idx, ...])

            else:
                # Fallback to on-the-fly preprocessing
                x, y_label, y_box = video.preprocess_base(size=size, device=device)
                for idx in video.frames:
                    self.X.append(x[idx, ...])
                    if target == "box":
                        if box_dilation:
                            # (C, H, W) -> (1, C, H, W)
                            y_b = y_box.unsqueeze(0)
                            # kernel size = 2*dilation + 1 to keep center
                            k = 2 * box_dilation + 1
                            y_b = F.max_pool2d(y_b, kernel_size=k, stride=1, padding=box_dilation)
                            self.Y.append(y_b.squeeze(0))
                        else:
                            self.Y.append(y_box) # box is the same for all frames
                    elif target == "mask":
                        self.Y.append(y_label[idx, ...])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X, Y = self.X[idx], self.Y[idx]
        if self.transforms is not None:
            return self.transforms(Image(X), Mask(Y))
        return X, Y


class Frames3Dataset(Dataset):
    '''
    Dataset for video frames serving 3 consecutive frames (t-1, t, t+1)
    to provide temporal context. Edge cases are handled by replicating the boundary frame.
    '''

    def __init__(
        self,
        videos: list[Video],
        target: str,
        denoiser_model=None,
        transforms = None,
        size=(112, 112)
        ):

        self.transforms = transforms
        self.target = target

        # Store full video data to access neighbors
        self.videos_data = [] 
        # Store (video_idx, frame_idx) for each sample
        self.samples = []

        for i, video in enumerate(videos):
            if video.processed_data is not None:
                # Use preprocessed data
                # (vid, label, box)
                x, y_label, y_box = video.processed_data
            else:
                # Fallback to on-the-fly preprocessing
                x, y_label, y_box = video.preprocess_base(size=size, device=device)

            # Store the full tensors. 
            # x is (T, C, H, W), y_label is (T, C, H, W), y_box is (C, H, W)
            # Check for RNMF data
            rnmf = video.rnmf_data if video.rnmf_data is not None else None
            
            self.videos_data.append((x, y_label, y_box, rnmf))

            for idx in video.frames:
                self.samples.append((i, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_idx, frame_idx = self.samples[idx]
        x_vid, y_vid_label, y_vid_box, rnmf_vid = self.videos_data[video_idx]
        
        # Get 3 frames: t-1, t, t+1
        frames = []
        rnmf_frames = []
        T_dim = x_vid.shape[0]
        
        for t in [frame_idx - 1, frame_idx, frame_idx + 1]:
            # Handle edge cases by replicating boundary frames
            if t < 0: 
                t = 0
            elif t >= T_dim: 
                t = T_dim - 1
            
            frames.append(x_vid[t])
            if rnmf_vid is not None:
                rnmf_frames.append(rnmf_vid[t])
            
        # Concatenate along channel dimension
        # Each frame is (C, H, W), usually C=1. Result is (3*C, H, W)
        X_frames = torch.cat(frames, dim=0)

        if rnmf_vid is not None:
            # Result (3*C + 3*C, H, W) -> (6, H, W) if C=1
            X_rnmf = torch.cat(rnmf_frames, dim=0)
            X = torch.cat([X_frames, X_rnmf], dim=0)
        else:
            X = X_frames

        if self.target == "box":
            # box is the same for all frames
            Y = y_vid_box
        elif self.target == "mask":
            # Retrieve 3 label frames: t-1, t, t+1
            label_frames = []
            for t in [frame_idx - 1, frame_idx, frame_idx + 1]:
                 # Handle edge cases by replicating boundary frames
                if t < 0: 
                    t = 0
                elif t >= T_dim: 
                    t = T_dim - 1
                label_frames.append(y_vid_label[t])
            
            # Concatenate along channel dimension (3, H, W)
            Y = torch.cat(label_frames, dim=0)
        
        if self.transforms is not None:
             return self.transforms(Image(X), Mask(Y))
        return X, Y