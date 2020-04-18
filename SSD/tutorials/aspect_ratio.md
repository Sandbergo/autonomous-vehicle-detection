# :camera: How to change aspect ratio NOT WORKING YET
1. Change in SSD/ssd/modelling/backbone/modelname.py forward-function:
```
out_ch = self.output_channels
out_feat = self.output_feature_size
feature_map_size_list = [
      torch.Size([out_ch[0], out_feat[0][1], out_feat[0][0]]),
      torch.Size([out_ch[1], out_feat[1][1], out_feat[1][0]]),
      torch.Size([out_ch[2], out_feat[2][1], out_feat[2][0]]),
      torch.Size([out_ch[3], out_feat[3][1], out_feat[3][0]]),
      torch.Size([out_ch[4], out_feat[4][1], out_feat[4][0]]),
      torch.Size([out_ch[5], out_feat[5][1], out_feat[5][0]])]

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
image = cv2.resize(image, (self.size[0], self.size[1])) # correct order??
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
        for k, f_lst in enumerate(self.feature_maps):
            scale_x = self.image_size[0] / self.strides[k]
            scale_y = self.image_size[1] / self.strides[k]

            for i, j in product(range(f_lst[0]), range(f_lst[1])):

                cx = (i + 0.5) / scale_x
                cy = (j + 0.5) / scale_y 

                # small sized square box
                size = self.min_sizes[k]
                w = size / self.image_size[0]
                h = size / self.image_size[1]
                priors.append([cx, cy, w, h])

                # big sized square box
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                w = size / self.image_size[0]
                h = size / self.image_size[1]
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size = self.min_sizes[k]
                w = size / self.image_size[0]
                h = size / self.image_size[1]
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
6. change in inference.py
```
class PostProcessor:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.INPUT.IMAGE_SIZE[0]
        self.height = cfg.INPUT.IMAGE_SIZE[1]
```
