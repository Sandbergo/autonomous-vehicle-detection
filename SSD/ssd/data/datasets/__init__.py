from torch.utils.data import ConcatDataset
import numpy as np
from ssd.config.path_catlog import DatasetCatalog
from .waymo import WaymoDataset
from .tdt4265 import TDT4265Dataset

_DATASETS = {
    'WaymoDataset': WaymoDataset,
    'TDT4265Dataset': TDT4265Dataset
}


def build_dataset(base_path: str, dataset_list, transform=None, target_transform=None, is_train=True):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(base_path, dataset_name)
        args = data['args']
        factory = _DATASETS[data['factory']]
        args['transform'] = transform
        args['target_transform'] = target_transform
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    #get_data_stats(dataset)
    return [dataset]


def get_data_stats(dataset):
    images = [np.array(image[0].reshape((image[0].shape[0], -1))).T for image in dataset]
    images = np.concatenate(images[:100], 1)
    print(hey)
    mean = images.mean(1)
    std = images.std(1)
    
    print(f'mean = {mean}, std = {std}')
    exit(0)
    
    
    pixel_mean = np.zeros(3)
    pixel_std = np.zeros(3)
    k = 1
    print(dataset)
    for image in dataset:
        image = image[0]
        image = np.array(image)
        pixels = image.reshape((image.shape[0], -1)).T
        for pixel in pixels:
            diff = pixel - pixel_mean
            pixel_mean += diff / k
            pixel_std += diff * (pixel - pixel_mean)
            k += 1
            if k>10000:
                break

    pixel_std = np.sqrt(pixel_std / (k - 2))
    
    print(f'mean = {pixel_mean}, std = {pixel_std}')
    exit(0)
    return
