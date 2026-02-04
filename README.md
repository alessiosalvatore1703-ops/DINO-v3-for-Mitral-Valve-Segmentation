# DINO-v3-for-Mitral-Valve-Segmentation
## TRAINING DATASET:

- DATASET A: 39 high-resolution medical videos from patients with moderate to severe mitral valve disease. Each video includes an annotated window of interest and 3 annotated frames.
19/39 videos were used for training and 20/39 for testing.
Resolution: 800 × 600.

- DATASET B: 46 low-resolution videos (128 × 128). The dataset includes both healthy and pathological anatomy. Each video contains an annotated window of interest and 3 annotated frames.

## PROPOSED APPROACH

First, the videos are cleaned and processed using CLAHE, image inpainting, and a denoising model. The frames are then converted into tensors, padded, and resized to a fixed dimension of 128 × 128.

Since only three annotated frames are provided per video, it is not feasible to train a robust model using the original annotations alone. To address this limitation, high-quality pseudo-labels are generated using DINO. The three annotated frames are used to extract video features, and the labels are propagated throughout the video, helping to address challenges related to temporal dynamics and limited labeled data.

Next, part of the propagated labels is removed using PCA followed by GMM-based anomaly detection to eliminate low-quality or inconsistent annotations. Using this refined dataset, a U-Net model is trained to segment the input images.

Finally, before submission, a simple post-processing pipeline is applied to resize the segmentation outputs back to their original dimensions and interpolate the probability masks.
