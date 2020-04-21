import torch


class BasicModel(torch.nn.Module):
    """
    This is a improved basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        kernel_size= 3

        # improved model
        self.f1 = torch.nn.Sequential(torch.nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size = 2, stride=2),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(kernel_size = 2, stride=2),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=output_channels[0],
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2
            ),
            torch.nn.LeakyReLU(output_channels[0]),
            torch.nn.BatchNorm2d(output_channels[0]),
            torch.nn.Conv2d(
                in_channels=output_channels[0],
                out_channels=output_channels[0],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
        )

        self.f2 = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(output_channels[0]),
            torch.nn.Conv2d(
                in_channels=output_channels[0],
                out_channels=256,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=output_channels[1],
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(output_channels[1]),
            torch.nn.Conv2d(
                in_channels=output_channels[1],
                out_channels=output_channels[1],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
        )

        self.f3 = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(output_channels[1]),
            torch.nn.Conv2d(
                in_channels=output_channels[1],
                out_channels=256,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=output_channels[2],
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(output_channels[2]),
            torch.nn.Conv2d(
                in_channels=output_channels[2],
                out_channels=output_channels[2],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
        )
        self.f4 = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(output_channels[2]),
            torch.nn.Conv2d(
                in_channels=output_channels[2],
                out_channels=256,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=output_channels[3],
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(output_channels[3]),
            torch.nn.Conv2d(
                in_channels=output_channels[3],
                out_channels=output_channels[3],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
        )
        self.f5 = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(output_channels[3]),
            torch.nn.Conv2d(
                in_channels=output_channels[3],
                out_channels=256,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=output_channels[4],
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(output_channels[4]),
            torch.nn.Conv2d(
                in_channels=output_channels[4],
                out_channels=output_channels[4],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
        )
        self.f6 = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(output_channels[4]),
            torch.nn.Conv2d(
                in_channels=output_channels[4],
                out_channels=256,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=output_channels[5],
                kernel_size=kernel_size,
                stride=2,
                padding=0
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(output_channels[5]),
            torch.nn.Conv2d(
                in_channels=output_channels[5],
                out_channels=output_channels[5],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
        )


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """

        feature_map_size_list = [torch.Size([256, 38, 38]),
                                 torch.Size([512, 19, 19]),
                                 torch.Size([256, 10, 10]),
                                 torch.Size([256, 5, 5]),
                                 torch.Size([128, 3, 3]),
                                 torch.Size([128, 1, 1])]

        x1 = self.f1(x)
        x2 = self.f2(x1)
        x3 = self.f3(x2)
        x4 = self.f4(x3)
        x5 = self.f5(x4)
        x6 = self.f6(x5)

        out_features = [x1, x2, x3, x4, x5, x6]

        for idx, feature in enumerate(out_features):
            out_channel = self.output_channels[idx]
            expected_shape = feature_map_size_list[idx]
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"

        return tuple(out_features)
