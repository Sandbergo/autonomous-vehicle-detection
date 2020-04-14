# :camera: How to change aspect ratio NOT WORKING YET
1. Change in SSD/ssd/modelling/backbone/modelname.py forward-function:
```
feature_map_size_list
out_features
```
2. Change in SSD/configs/train_modelname.yaml the following values. Example with new added values marked with asterisks
```
MODEL:
    BACKBONE:
        OUT_CHANNELS: [**256**, 256, 512, 256, 256, 128, 128]
    PRIORS:
        FEATURE_MAPS: [**75**, 38, 19, 10, 5, 3, 1]
        STRIDES: [**4**, 8, 16, 32, 64, 100, 300]
        MIN_SIZES: [**15**, 30, 60, 111, 162, 213, 264]
        MAX_SIZES: [**30**, 60, 111, 162, 213, 264, 315]
        ASPECT_RATIOS: [**[2, 3]**, [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
        # When has 1 aspect ratio, every location has 4 boxes, 2 ratio 6 boxes.
        # #boxes = 2 + #ratio * 2
        BOXES_PER_LOCATION: [**6**, 6, 6, 6, 6, 4, 4]  # number of boxes per feature map location
```
