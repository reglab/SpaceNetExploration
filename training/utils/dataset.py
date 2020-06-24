import os
# Ignore warnings
import warnings

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


warnings.filterwarnings('ignore')

class SpaceNetDataset(Dataset):
    """Class representing a SpaceNet dataset, such as a training set."""

    def __init__(self, root_dir, splits=['trainval', 'test'], transform=None):
        """
        Args:
            root_dir (string): Directory containing folder annotations and .txt files with the
            train/val/test splits
            splits: ['trainval', 'test'] - the SpaceNet utilities code would create these two
                splits while converting the labels from polygons to mask annotations. The two
                splits are created after chipping larger images into the required input size with
                some overlaps. Thus to have splits that do not have overlapping areas, we manually
                split the images (not chips) into train/val/test using utils/split_train_val_test.py,
                followed by using the SpaceNet utilities to annotate each folder, and combine the
                trainval and test splits it creates inside each folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.xml_list = []

        data_files = []
        for split in splits:
            with open(os.path.join(root_dir, split + '.txt')) as f:
                data_files.extend(f.read().splitlines())

        for line in data_files:
            line = line.split(' ')

            image_name = line[0].split('/')[-1]
            xml_name = line[1].split('/')[-1]

            self.image_list.append(image_name)
            self.xml_list.append(xml_name)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'RGB-PanSharpen', self.image_list[idx])
        target_path = os.path.join(self.root_dir, 'annotations', self.image_list[idx].replace('.tif', 'segcls.tif'))

        image = np.array(Image.open(img_path)).astype(np.int)
        target = np.array(Image.open(target_path)).astype(np.int)
        
        # print('BEFORE converting target to building interior and border')
        # print('Printing unique values of target: {}'.format(np.unique(target)))

        target[target == 100] = 1  # building interior
        target[target == 255] = 2  # border
        target[target == 99] = 0 # Self Added by Vincent (Not sure what 99 is?)
        target[target == 256] = 0 # Self Added by Vincent (Not sure what 256 is?)

        # print('POST converting target to building interior and border')
        # print('Printing unique values of target: {}'.format(np.unique(target)))

        sample = {'image': image, 'target': target, 'image_name': self.image_list[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
