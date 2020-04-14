# :camera: How to change aspect ratio NOT WORKING YET
1. Change in SSD/ssd/modelling/backbone/modelname.py forward-function:
```
feature_map_size_list = [
     torch.Size([self.output_channels[0], self.output_feature_size[0][0], self.output_feature_size[0][1]]),
     torch.Size([self.output_channels[1], self.output_feature_size[1][0], self.output_feature_size[1][1]]),
     torch.Size([self.output_channels[2], self.output_feature_size[2][0], self.output_feature_size[2][1]]),
     torch.Size([self.output_channels[3], self.output_feature_size[3][0], self.output_feature_size[3][1]]),
     torch.Size([self.output_channels[4], self.output_feature_size[4][0], self.output_feature_size[4][1]]),
     torch.Size([self.output_channels[5], self.output_feature_size[5][0], self.output_feature_size[5][1]])]

```
2. Change in SSD/configs/train_modelname.yaml the following values. Example with new added values marked with asterisks
```
    PRIORS:
        FEATURE_MAPS: [ 38, 19, 10, 5, 3, 1]
        FEATURE_MAPS: [ [40, 30], [20, 15], [10, 8], [5, 4], [3, 2], [1, 1]]
    INPUT:
        IMAGE_SIZE: [320, 240]
```
3. Change in transforms.py
```
image = cv2.resize(image, (self.size[1], self.size[0])) # correct order??
```
4. Change in prior_box.py NOT VERIFIED
```

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, l in enumerate(self.feature_maps[0]):
            for f in self.feature_maps[1]:
                scale_x = self.image_size[0] / self.strides[k]
                scale_y = self.image_size[1] / self.strides[k]
                for i, j in product(range(f), range(l)):#, repeat=2):
                    # unit center x,y
                    cx = (j + 0.5) / scale_x
                    cy = (i + 0.5) / scale_y

                    # small sized square box
                    size = self.min_sizes[k]
                    h = size / self.image_size[0]
                    w = size / self.image_size[1]
                    priors.append([cx, cy, w, h])

                    # big sized square box
                    size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                    h = size / self.image_size[0]
                    w = size / self.image_size[1]
                    priors.append([cx, cy, w, h])

                    # change h/w ratio of the small sized box
                    size = self.min_sizes[k]
                    h = size / self.image_size[0]
                    w = size / self.image_size[1]
                    for ratio in self.aspect_ratios[k]:
                        ratio = sqrt(ratio)
                        priors.append([cx, cy, w * ratio, h / ratio])
                        priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
```
5. Change in defaults.py 
```
cfg.INPUT.IMAGE_SIZE = [320, 240]
```
