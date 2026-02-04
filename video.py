import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import imageio
from skimage import filters
import cv2

class Video:
    '''
    Video object for video segmentation task.
    Stores video, box, label, frames and label type and provide basic methods to access them.
    '''
    def __init__(
        self,
        name: str = None,
        video: np.ndarray = None,
        box: np.ndarray = None,
        label: np.ndarray = None,
        frames: list = None,
        label_type: str = None,
    ):
        self.name = name
        self.video = video

        self.box = box
        self.label = label
        self.frames = frames
        self.label_type = label_type
        self.total_frames = video.shape[2]      # better not use this... it's not updated when frames are modified
        self.processed_data = None
        self.rnmf_data = None

        # print(f"Video {self.name} loaded: X shape {self.video.shape}, Y_box shape {self.box.shape if self.box is not None else None}, Y_label shape {self.label.shape if self.label is not None else None}")

    def __repr__(self):
        return f"Video(name={self.name}, video={self.video.shape}, box={self.box.shape if self.box is not None else None}, label={self.label.shape if self.label is not None else None}, labeled frames={self.frames if self.frames is not None else None}, total frames={self.total_frames}, label_type={self.label_type if self.label_type is not None else None}, processed={self.processed_data is not None}, rnmf={self.rnmf_data is not None})"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        '''
        Returns idx-th frame of the video as a tuple (video, box, label)
        Box is always returned as the box of the whole video
        Label is an empty mask for non-labeled frames
        '''
        if idx >= self.total_frames:
            raise IndexError(f"Frame {idx} not in video {self.name}")
        
        raise NotImplementedError("WIP")

    def is_labeled(self, idx):
        return idx in self.frames

    def save_gif(self, path='plots/', from_processed=False):
        '''
        Save the video as a gif with bounding boxes and labels
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        
        images = []
        
        # Determine source
        use_processed = from_processed and self.processed_data is not None
        if from_processed and self.processed_data is None:
             print(f"Warning: Processed data not found for {self.name}, using original.")

        if use_processed:
            vid_t, label_t, box_t = self.processed_data
            # vid_t: (T, C, H, W)
            # label_t: (T, 1, H, W) or None
            # box_t: (1, H, W) or None
            num_frames = vid_t.shape[0]
        else:
            num_frames = self.total_frames
        
        # Iterate over all frames
        for i in range(num_frames):
            
            if use_processed:
                 # Video
                 v = vid_t[i].cpu() # (C, H, W)
                 if v.shape[0] == 3:
                      img = v.permute(1, 2, 0).numpy()
                      # Normalize for viz
                      img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                 else:
                      img = v.squeeze(0).numpy()
                 
                 # Mask
                 mask = label_t[i, 0].cpu().numpy() if label_t is not None else None
                 
                 # Box
                 # box_t is (1, H, W)
                 box_mask = box_t[0].cpu().numpy() if box_t is not None else None
                 
            else:
                img = self.video[:, :, i]
                mask = self.label[:, :, i] if self.label is not None else None
                box_mask = self.box if self.box is not None else None # box is static 2D
            
            # plot bounding box on img
            fig, ax = plt.subplots(1)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"{self.name} - {i}")

            # Plot box if it exists
            if box_mask is not None:
                # Ensure box is 2D
                if box_mask.ndim == 2:
                    ys, xs = np.where(box_mask)
                    if len(xs) > 0 and len(ys) > 0:
                        x_min = np.min(xs)
                        x_max = np.max(xs)
                        y_min = np.min(ys)
                        y_max = np.max(ys)
                        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
            
            # Plot label if it exists
            if mask is not None:
                # Check if mask is not empty (optional, but good for visualization)
                if np.any(mask):
                     ax.imshow(mask, alpha=0.5, cmap='jet')

            # save plot to image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:, :, :3]
            images.append(image)
            plt.close(fig)
        
        # Save the gif
        name = self.name
        if from_processed:
            name += "_processed"
        if not name.endswith('.gif'):
            name += '.gif'
        imageio.mimsave(os.path.join(path, name), images, fps=10)

    def is_test_video(self):
        '''
        Returns True if the video is a test video, False otherwise, by checking if all execept name and video field are None
        '''
        return self.box is None and self.label is None and self.frames is None and self.label_type is None

    def load_pseudo_label(self, pseudo_labels):
        if self.name not in pseudo_labels:
            raise ValueError(f"Pseudo label for video {self.name} not found")
        
        # sanity check shape
        if pseudo_labels[self.name][0].shape != self.video.shape:
            raise ValueError(f"Pseudo label shape {pseudo_labels[self.name][0].shape} does not match video shape {self.video.shape}")

        self.label = pseudo_labels[self.name][0]
        self.frames = list(range(self.total_frames))

        print(f"Pseudo label loaded for video {self.name}, video shape {self.video.shape}, label shape {self.label.shape}")

    def load_preprocessed_data(self, path):
        '''
        Load preprocessed tensors from path
        '''
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessed file {path} not found")
        
        # (vid, label, box)
        self.processed_data = torch.load(path)
        print(f"Loaded processed data for {self.name} from {path}")

    def load_rnmf_data(self, path):
        '''
        Load RNMF data from path.
        Expected path: directory where {video.name}.pt exists.
        '''
        filepath = os.path.join(path, f"{self.name}.pt")
        if not os.path.exists(filepath):
             # Try standard path if not found, or default
             filepath = os.path.join("data/rnmf_processed", f"{self.name}.pt")
             if not os.path.exists(filepath):
                 print(f"Warning: RNMF file for {self.name} not found in {path} or default.")
                 return

        # (movement, label, box)
        movement, _, _ = torch.load(filepath)
        self.rnmf_data = movement
        print(f"Loaded RNMF data for {self.name} from {filepath}")

    def preprocess_base(self, size, device="cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Baseline preprocessing, scales, pad, resize img, label and box, return tuple (video, label, box) tensors

        Preprocess video (H, W, T) -> (T, C=1, size[0], size[1])
        Preprocess label (H, W, T) -> (T, C=1, size[0], size[1])
        Preprocess box (H, W) -> (C=1, size[0], size[1])

        # TODO: normalization? histogram equalization?
        '''

        label = None
        box = None

        H, W = self.video.shape[:2]

        vid = torch.from_numpy(self.video).permute(2, 0, 1)

        if vid.ndim == 3:
            vid = vid.unsqueeze(1)

        # scale
        vid = vid.float() / 255.0

        # pad to square
        pad_w = max(0, H - W)
        pad_h = max(0, W - H)
        vid = F.pad(vid, [0, pad_w, 0, pad_h])

        # resize
        vid = F.interpolate(vid, size=size, mode='bilinear', align_corners=False)
        vid = vid.to(device)

        if self.label is not None:
            label = torch.from_numpy(self.label).permute(2, 0, 1).float()
            
            if label.ndim == 3:
                label = label.unsqueeze(1)
            
            # pad
            label = F.pad(label, [0, pad_w, 0, pad_h])
            
            # resize
            label = F.interpolate(label, size=size, mode='nearest')
            label = label.to(device)


        if self.box is not None:
            # box is 2d boolean mask

            box = torch.from_numpy(self.box).unsqueeze(0).unsqueeze(0).float()

            # pad
            box = F.pad(box, [0, pad_w, 0, pad_h])
            
            # resize
            box = F.interpolate(box, size=size, mode='nearest')

            box = box.squeeze(0)
            box = box.to(device)

        return vid, label, box

    def preprocess_v2(self, size, device="cpu", denoiser_model=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Preprocessing inspired by https://github.com/luke-ck/ultrasound-segmentation/blob/d0596ff9c06d91d77a452a226ee5718353fbd6bb/src/data_manager.py#L20

        Preprocess video (H, W, T) -> (T, C=1, size[0], size[1])
        Preprocess label (H, W, T) -> (T, C=1, size[0], size[1])
        Preprocess box (H, W) -> (C=1, size[0], size[1])
        '''

        label = None
        box = None

        H, W = self.video.shape[:2]

        # CLAHE the video frame by frame
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
        
        for i in range(self.video.shape[2]):
            self.video[:, :, i] = clahe.apply(self.video[:, :, i])
            self.video[:, :, i] = self._heal_image(self.video[:, :, i])

        vid = torch.from_numpy(self.video).permute(2, 0, 1)

        if vid.ndim == 3:
            vid = vid.unsqueeze(1)

        # scale
        vid = vid.float() / 255.0

        # denoise
        if denoiser_model is not None:
            # noise level from: https://github.com/luke-ck/ultrasound-segmentation/blob/d0596ff9c06d91d77a452a226ee5718353fbd6bb/src/data_manager.py#L17
            sigma = 5.0 / 255.0 
            
            # Check model device
            model_device = next(denoiser_model.parameters()).device
            
            with torch.no_grad():
                for i in range(vid.shape[0]):
                    # (1, H, W)
                    img = vid[i].to(model_device).unsqueeze(0) # (1, 1, H, W)
                    
                    # Create noise map
                    noise_map = torch.FloatTensor(img.shape).fill_(sigma).to(model_device)
                    
                    # Concat
                    inp = torch.cat((img, noise_map), dim=1) # (1, 2, H, W)
                    
                    # Forward
                    out = denoiser_model(inp) # (1, 1, H, W)
                    
                    # Assign back (make sure to move back if needed, though here vid is likely CPU initially)
                    vid[i] = out.squeeze(0).to(vid.device)

        # pad to square
        pad_w = max(0, H - W)
        pad_h = max(0, W - H)
        vid = F.pad(vid, [0, pad_w, 0, pad_h])

        # resize
        vid = F.interpolate(vid, size=size, mode='bilinear', align_corners=False)
        vid = vid.to(device)

        if self.label is not None:

            processed_labels = []

            # frame by frame
            for i in range(self.label.shape[2]):
                l = self.label[:, :, i].astype(np.float32)
                l = cv2.GaussianBlur(l, (3, 3), 2)

                try:
                    thresh = filters.threshold_otsu(l)
                    l = (l > thresh).astype(np.float32)
                except ValueError:
                    assert False, "Thresholding failed"

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                l = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
                processed_labels.append(l)

            label = np.stack(processed_labels, axis=2)
            label = torch.from_numpy(label).permute(2, 0, 1).float()
            
            if label.ndim == 3:
                label = label.unsqueeze(1)
            
            # pad
            label = F.pad(label, [0, pad_w, 0, pad_h])
            
            # resize
            label = F.interpolate(label, size=size, mode='nearest')
            label = label.to(device)

        if self.box is not None:
            # box is 2d boolean mask

            box = torch.from_numpy(self.box).unsqueeze(0).unsqueeze(0).float()

            # pad
            box = F.pad(box, [0, pad_w, 0, pad_h])
            
            # resize
            box = F.interpolate(box, size=size, mode='nearest')

            box = box.squeeze(0)
            box = box.to(device)

        return vid, label, box

    def _heal_image(self, img: np.ndarray):
        '''
            Image inpainting to fill in small holes in the image
            https://github.com/luke-ck/ultrasound-segmentation/blob/master/src/utils.py#L57
        '''
        assert len(img.shape) == 2, "Image must be grayscale"
        height, width = img.shape
        # Threshold the image to create a binary image
        ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # Find the contours in the binary image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create a black image to draw the contours on
        contour_img = np.zeros((height, width, 3), np.uint8)
        # Define the minimum patch size
        min_patch_size = 30

        # Loop over the contours and fill in the patches with area less than min_patch_size
        for contour in contours:
            # Calculate the area of the contour
            contour_area = cv2.contourArea(contour)

            # Check if the contour area is less than the minimum patch size
            if contour_area < min_patch_size:
                cv2.drawContours(contour_img, [contour], 0, (0, 0, 255), 2)
                # Create a mask with the same shape as the image
                mask = np.zeros((height, width), np.uint8)

                # Draw the contour on the mask
                cv2.drawContours(mask, [contour], 0, 255, -1)

                # Apply the healing tool to the mask using the cv2.inpaint function
                img = cv2.inpaint(img, mask, 2, cv2.INPAINT_TELEA)
        return img

    def postprocess(self, pred: torch.Tensor) -> np.ndarray:
        '''
        Reverse preprocessing: resize back to original square size, then crop padding.
        Input: pred (T, C, size, size)
        Output: (H, W, T) numpy array (boolean mask)
        '''
        H, W = self.video.shape[:2]
        max_dim = max(H, W)
        
        # Resize back to square padded shape
        pred = F.interpolate(pred, size=(max_dim, max_dim), mode='nearest')
        
        # Crop to original H, W (ie: unpad)
        pred = pred[:, :, :H, :W]
        
        # Convert to numpy and shape (H, W, T)
        pred = pred.squeeze(1).permute(1, 2, 0).cpu().numpy()
               
        # Convert to boolean mask with as type np.bool_
        pred = pred.astype(np.bool_)

        return pred
        

    