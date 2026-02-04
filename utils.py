import pickle
import gzip
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
from video import Video
import torch
from scipy import ndimage


def viz_gifs():
    train_data = utils.load_zipped_pickle('data/train.pkl')
    test_data = utils.load_zipped_pickle('data/test.pkl')

    videos = utils.preprocess_train_data(train_data)
    test_names, test_videos = utils.preprocess_test_data(test_data)

    # save some gifs for visualization
    for video in videos[:3]:
        video.save_gif()

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
    
def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)
        
def preprocess_train_data(data):
    '''
    Preprocess training data for video segmentation task.
    '''
    videos = []
    for item in data:
        video = Video(
            name=item['name'],
            video=item['video'],
            box=item['box'],
            label=item['label'],
            frames=item['frames'],
            label_type=item['dataset']
        )
        # print(video)
        videos.append(video)
    return videos

def preprocess_test_data(data):
    '''
    Preprocess test data for video segmentation task.
    '''
    videos = []
    for item in data:
        video_obj = Video(
            name=item['name'],
            video=item['video'],
            frames=None,
            box=None,
            label=None,
            label_type=None
        )
        # print(video_obj)
        videos.append(video_obj)
    return videos




def get_sequences(arr):
    ''' Dont change me '''
    first_indices, last_indices, lengths = [], [], []
    n, i = len(arr), 0
    arr = [0] + list(arr) + [0]
    for index, value in enumerate(arr[:-1]):
        if arr[index+1]-arr[index] == 1:
            first_indices.append(index)
        if arr[index+1]-arr[index] == -1:
            last_indices.append(index)
    lengths = list(np.array(last_indices)-np.array(first_indices))
    return first_indices, lengths

def remove_small_components(mask, min_size=50):
    """
    Remove small isolated connected components from a binary mask.
    
    Args:
        mask: Binary mask of shape (H, W) or (H, W, T)
        min_size: Minimum number of pixels for a component to be kept
    
    Returns:
        Cleaned mask with small components removed
    """
    if mask.ndim == 2:
        # Single frame
        labeled, num_features = ndimage.label(mask)
        cleaned = np.zeros_like(mask)
        for i in range(1, num_features + 1):
            component = (labeled == i)
            if component.sum() >= min_size:
                cleaned[component] = 1
        return cleaned
    elif mask.ndim == 3:
        # Video (H, W, T) - process each frame
        cleaned = np.zeros_like(mask)
        for t in range(mask.shape[2]):
            cleaned[:, :, t] = remove_small_components(mask[:, :, t], min_size)
        return cleaned
    else:
        raise ValueError(f"Expected 2D or 3D mask, got shape {mask.shape}")


def keep_largest_component(mask):
    """
    Keep only the largest connected component in each frame.
    Useful when you know there's only one mitral valve.
    
    Args:
        mask: Binary mask of shape (H, W) or (H, W, T)
    
    Returns:
        Mask with only the largest component per frame
    """
    if mask.ndim == 2:
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return mask
        # Find largest component
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        return (labeled == largest_label).astype(mask.dtype)
    elif mask.ndim == 3:
        cleaned = np.zeros_like(mask)
        for t in range(mask.shape[2]):
            cleaned[:, :, t] = keep_largest_component(mask[:, :, t])
        return cleaned
    else:
        raise ValueError(f"Expected 2D or 3D mask, got shape {mask.shape}")


def apply_morphological_cleanup(mask, close_size=3, open_size=2):
    """
    Apply morphological operations to clean up the mask.
    - Closing: fills small holes
    - Opening: removes small protrusions
    
    Args:
        mask: Binary mask of shape (H, W) or (H, W, T)
        close_size: Size of structuring element for closing
        open_size: Size of structuring element for opening
    
    Returns:
        Cleaned mask
    """
    from scipy.ndimage import binary_closing, binary_opening
    
    close_struct = np.ones((close_size, close_size))
    open_struct = np.ones((open_size, open_size))
    
    if mask.ndim == 2:
        cleaned = binary_closing(mask, structure=close_struct)
        cleaned = binary_opening(cleaned, structure=open_struct)
        return cleaned.astype(mask.dtype)
    elif mask.ndim == 3:
        cleaned = np.zeros_like(mask)
        for t in range(mask.shape[2]):
            cleaned[:, :, t] = apply_morphological_cleanup(mask[:, :, t], close_size, open_size)
        return cleaned
    else:
        raise ValueError(f"Expected 2D or 3D mask, got shape {mask.shape}")


def postprocess_predictions(mask, min_component_size=50, keep_largest=True, morphological=True):
    """
    Full post-processing pipeline for predictions.
    
    Args:
        mask: Binary mask of shape (H, W, T)
        min_component_size: Remove components smaller than this
        keep_largest: If True, only keep the largest component per frame
        morphological: If True, apply morphological cleanup
    
    Returns:
        Cleaned mask
    """
    cleaned = mask.copy()
    
    # 1. Morphological cleanup (fill holes, remove noise)
    if morphological:
        cleaned = apply_morphological_cleanup(cleaned)
    
    # 2. Remove small components OR keep only largest
    if keep_largest:
        cleaned = keep_largest_component(cleaned)
    else:
        cleaned = remove_small_components(cleaned, min_size=min_component_size)
    
    return cleaned.astype(bool)


def append_to_submission(video, save_path):
    """
    Appends submission data for a single video to the csv file.
    video: Video object with 'label' attribute populated (H, W, T) boolean mask
    save_path: path to save the csv file
    
    Format:
    - id: name_i where name is video name and i is a unique identifier
    - value: [flattenedIdx, len] as a Python list that gets converted to string by pandas
    """
    if video.label is None:
        print(f"Warning: Video {video.name} has no label. Skipping.")
        return
        
    ids = []
    values = []
        
    # video.label is (H, W, T) boolean mask
    # Flatten the ENTIRE 3D mask (not per-frame)
    flattened_mask = video.label.astype(np.int8).flatten()
    
    # Get RLE sequences from the flattened 3D mask
    first_indices, lengths = get_sequences(flattened_mask)
    
    # warn if no sequences found
    if len(first_indices) == 0:
        print(f"WARNING!!! Video {video.name} has no sequences! Skipping!")
        
        # add filler row
        ids.append(f"{video.name}_0")
        values.append("[0, 1]")
    else:
        # Create one row per sequence
        for i, (idx, length) in enumerate(zip(first_indices, lengths)):
            ids.append(f"{video.name}_{i}")
            values.append([int(idx), int(length)])
    
    # Create DataFrame with value as list (pandas will convert to string representation)
    df = pd.DataFrame({"id": ids, "value": values})
    
    # Check if file exists to determine if header is needed
    header = not os.path.exists(save_path)
    
    df.to_csv(save_path, mode='a', header=header, index=False)