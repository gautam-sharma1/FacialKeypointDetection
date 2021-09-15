import os

import cv2 as cv2
import pandas as pd
from torch.utils.data import Dataset

"""
Used to load in the dataset and return a dictionary of {image : keypoints}
Each image has 68 keypoints (x,y) 
"""
class FacialKeypointsDataset(Dataset):
    def __init__(self, csv_file, dataset_location, transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            dataset_location (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.dataset_location = dataset_location
        self.transforms = transforms

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.dataset_location,
                                  self.key_pts_frame.iloc[idx, 0])

        image = cv2.imread(image_name)

        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy()
        key_pts = key_pts.astype('float').reshape(-1, 2)  # changes 1x136 to 68x2
        sample = {'image': image, 'keypoints': key_pts}

        if self.transforms:
            sample = self.transforms(sample)

        return sample


