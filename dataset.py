from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image


def _list_files(directory):
    return sorted([
        file_name for file_name in os.listdir(directory)
        if not file_name.endswith(':Zone.Identifier')
    ])


def _collect_image_mask_pairs(image_dir, mask_dir):
    image_files = _list_files(image_dir)
    mask_map = {os.path.splitext(file_name)[0]: file_name for file_name in _list_files(mask_dir)}

    pairs = []
    missing_masks = []
    for image_file in image_files:
        stem = os.path.splitext(image_file)[0]
        mask_file = mask_map.get(stem)
        if mask_file is None:
            missing_masks.append(image_file)
            continue
        pairs.append([
            os.path.join(image_dir, image_file),
            os.path.join(mask_dir, mask_file),
        ])

    if missing_masks:
        raise FileNotFoundError(f'Missing masks for {image_dir}: {missing_masks[:5]}')
    if len(pairs) != len(image_files):
        raise ValueError(f'Image/mask pair count mismatch: {image_dir} vs {mask_dir}')

    return pairs


class Branch1_datasets(Dataset):
    def __init__(self, path_Data, config, train=True, test=False):
        super(Branch1_datasets, self)
        if train:
            split = 'train'
            self.transformer = config.train_transformer
        elif test:
            split = 'test'
            self.transformer = config.test_transformer
        else:
            split = 'val'
            self.transformer = config.test_transformer

        image_dir = path_Data + f'{split}/images/'
        mask_dir = path_Data + f'{split}/masks/'
        self.data = _collect_image_mask_pairs(image_dir, mask_dir)

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)


class Branch2_datasets(Dataset):
    def __init__(self, path_Data, config, train=True, test=False, return_image_name=False):
        super(Branch2_datasets, self)

        self.return_image_name = return_image_name or test

        if train:
            split = 'train'
            self.transformer = config.train_transformer
        elif test:
            split = 'test'
            self.transformer = config.test_transformer
        else:
            split = 'val'
            self.transformer = config.test_transformer

        image_dir = path_Data + f'{split}/images/'
        mask_dir = path_Data + f'{split}/masks/'
        feature_dir = path_Data + f'{split}/feature/'

        image_files = _list_files(image_dir)
        mask_map = {os.path.splitext(file_name)[0]: file_name for file_name in _list_files(mask_dir)}
        feature_map = {os.path.splitext(file_name)[0]: file_name for file_name in _list_files(feature_dir)}

        self.data = []
        missing_masks = []
        missing_features = []
        for image_file in image_files:
            stem = os.path.splitext(image_file)[0]
            mask_file = mask_map.get(stem)
            feature_file = feature_map.get(stem)

            if mask_file is None:
                missing_masks.append(image_file)
                continue
            if feature_file is None:
                missing_features.append(image_file)
                continue

            img_path = image_dir + image_file
            msk_path = mask_dir + mask_file
            feature_path = feature_dir + feature_file
            self.data.append((img_path, msk_path, feature_path, image_file))

        if missing_masks:
            raise FileNotFoundError(f'Missing masks for split {split}: {missing_masks[:5]}')
        if missing_features:
            raise FileNotFoundError(f'Missing feature files for split {split}: {missing_features[:5]}')

    def __getitem__(self, indx):
        img_path, msk_path, feature_path, image_file = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        feature = torch.load(feature_path, map_location='cpu')

        if self.transformer is not None:
            img, msk, feature = self.transformer((img, msk, feature))
        if self.return_image_name:
            return img, msk, feature, image_file
        return img, msk, feature

    def __len__(self):
        return len(self.data)

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

