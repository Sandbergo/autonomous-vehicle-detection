
import matplotlib.pyplot as plt
import sys
import torch
import numpy as np
import pathlib
import matplotlib
path = pathlib.Path()
# Insert all modules a folder above
sys.path.insert(0, str(path.absolute().parent))
from ssd.config.defaults import cfg
from ssd.modeling.box_head.prior_box import PriorBox
from ssd.utils.box_utils import convert_locations_to_boxes
config_path = "../configs/train_waymo.yaml"
cfg.merge_from_file(config_path)
fig, ax = plt.subplots()

#cfg.MODEL.CENTER_VARIANCE = [0.15, 0.1]
#cfg.MODEL.SIZE_VARIANCE = [0.2, 0.15]
#cfg.MODEL.PRIORS.FEATURE_MAPS = [ [40, 30], [20, 15], [10, 8], [5, 4], [3, 2], [1, 1] ]

#cfg.INPUT.IMAGE_SIZE = [300, 300]
cfg.INPUT.IMAGE_SIZE = [320, 240]
#cfg.MODEL.PRIORS.FEATURE_MAPS = [ [38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]]
cfg.MODEL.PRIORS.FEATURE_MAPS = [ [40, 30], [20, 15], [10, 8], [5, 4], [3, 2], [1, 1] ]
cfg.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
cfg.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location
prior_box = PriorBox(cfg)

priors = prior_box()
print("Prior box shape:", priors.shape)
print("First prior example:", priors[5])


locations = torch.zeros_like(priors)[None]
priors_as_location = convert_locations_to_boxes(locations, priors, cfg.MODEL.CENTER_VARIANCE, cfg.MODEL.SIZE_VARIANCE)[0]

layer_to_visualize = 5
aspect_ratio_indeces = [0, 1, 2, 3]

plt.ylim([-100, 240 + 100])
plt.xlim([-100, 320 + 100])


def get_num_boxes_in_fmap(idx):
    boxes_per_location = cfg.MODEL.PRIORS.BOXES_PER_LOCATION[idx]
    feature_map_size = cfg.MODEL.PRIORS.FEATURE_MAPS[idx]
    return int(boxes_per_location*np.prod(feature_map_size))

offset = sum([get_num_boxes_in_fmap(prev_layer) for prev_layer in range(layer_to_visualize)])
boxes_per_location = cfg.MODEL.PRIORS.BOXES_PER_LOCATION[layer_to_visualize]
indeces_to_visualize = []
colors = []
available_colors = ["r", "g", "b", "y", "m", "b", "w"]

for idx in range(offset, offset + get_num_boxes_in_fmap(layer_to_visualize)):
    for aspect_ratio_idx in aspect_ratio_indeces:
        if idx % boxes_per_location == aspect_ratio_idx:
            indeces_to_visualize.append(idx)
            colors.append(available_colors[aspect_ratio_idx])

ax.add_artist(plt.Rectangle([0,0], 320, 240))


def plot_bbox(ax, box, color):
    cx, cy, w, h = box
    print(cx, cy, w, h)
    cx *= 256
    cy *= 256
    w *= 256
    h *= 256
    print(cx, cy, w, h)
    #print(cx, cy)
    #x1, y1 = cx + w/2, cy + h/2
    #x0, y0 = cx - w/2, cy - h/2
    ax.add_artist(matplotlib.patches.Ellipse([cx, cy], w, h, alpha=.1, color=color))
    plt.plot(cx, cy, f'o{color}')

for i, idx in enumerate(indeces_to_visualize):
    prior = priors_as_location[idx]
    color = colors[i]
    plot_bbox(ax, prior, color) 

plt.savefig('test.png')