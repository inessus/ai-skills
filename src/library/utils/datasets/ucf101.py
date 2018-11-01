from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode


class UCF101(Dataset):
    """UCF101 Landmarks dataset."""

    def __init__(self, info_list, root_dir, transform=None):
        """
        Args:
            info_list (string): Path to the info list file with annotations.
            root_dir (string): Directory with all the video frames.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    # get (16,240,320,3)
    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        video_label = self.landmarks_frame.iloc[idx, 1]
        video_x = self.get_single_video_x(video_path)
        sample = {'video_x': video_x, 'video_label': video_label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_video_x(self, video_path):
        slash_rows = video_path.split('.')
        dir_name = slash_rows[0]
        video_jpgs_path = os.path.join(self.root_dir, dir_name)
        # get the random 16 frame
        data = pd.read_csv(os.path.join(video_jpgs_path, 'n_frames'), delimiter=' ', header=None)
        frame_count = data[0][0]
        video_x = np.zeros((16, 240, 320, 3))

        image_start = random.randint(1, frame_count - 17)
        image_id = image_start
        for i in range(16):
            s = "%05d" % image_id
            image_name = 'image_' + s + '.jpg'
            image_path = os.path.join(video_jpgs_path, image_name)
            tmp_image = io.imread(image_path)
            video_x[i, :, :, :] = tmp_image
            image_id += 1
        return video_x

