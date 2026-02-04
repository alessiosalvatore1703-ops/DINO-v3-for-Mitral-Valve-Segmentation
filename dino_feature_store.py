import os
import numpy as np
import torch

class DinoFeatureStore:
    def __init__(self, per_video_dir):
        self.per_video_dir = per_video_dir
        self.cache = {}  # name -> torch.FloatTensor (T,C)

    def _load(self, name):
        if name in self.cache:
            return self.cache[name]
        path = os.path.join(self.per_video_dir, f"{name}.npy")
        E = np.load(path)  # (T,C)
        E = torch.from_numpy(E).float()
        self.cache[name] = E
        return E

    def get_window(self, name, t, window):
        E = self._load(name)  # (T,C)
        T = E.shape[0]
        k = window // 2
        idxs = [min(max(t + dt, 0), T - 1) for dt in range(-k, k + 1)]
        return E[idxs]  # (window, C)
    
    def all_windows(self, name, window):
        E = self._load(name)  # (T,C)
        T = E.shape[0]
        return torch.stack([self.get_window(name, t, window) for t in range(T)], dim=0)  # (T,L,C)

