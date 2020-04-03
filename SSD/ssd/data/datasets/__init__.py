from torch.utils.data import ConcatDataset
import tqdm
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
    print('but')
    print(dataset[0][0].shape)
    images = np.array([np.array(image[0].reshape((3, -1))).T for image in dataset])
    images = images.reshape(-1, 3)
    print(images.shape)
    print('heeeeeey')
    #print(images)
    mean = images.mean(0)
    std = images.std(0)
    
    print(f'mean = {mean}, std = {std}')
    exit(0)
    
    return