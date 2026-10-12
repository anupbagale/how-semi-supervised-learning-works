"""
A small CNN (about 15k parameters). Small on purpose: it trains on a laptop CPU
in minutes, and with only a handful of labels a big network would just memorise.
It first halves the resolution (96 -> 48): lesions are 6-11 pixels across, so they
stay visible, and training is about 4x faster.
"""
import torch
import torch.nn as nn

PIXEL_MEAN, PIXEL_STD = 0.25, 0.26


def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU())


class SmallCNN(nn.Module):
    def __init__(self, widths=(8, 16, 32, 32)):
        super().__init__()
        w = widths
        self.features = nn.Sequential(
            nn.AvgPool2d(2),                        # 96 -> 48 (downsample once)
            block(1, w[0]), nn.MaxPool2d(2),        # 48 -> 24
            block(w[0], w[1]), nn.MaxPool2d(2),     # 24 -> 12
            block(w[1], w[2]), nn.MaxPool2d(2),     # 12 -> 6
            block(w[2], w[3]),                      # 6 x 6 feature maps
        )
        self.head = nn.Linear(2 * w[3], 2)

    def feature_maps(self, img):
        return self.features(((img - PIXEL_MEAN) / PIXEL_STD)[:, None])

    def forward(self, img):                         # img: (B, 96, 96) in [0, 1]
        f = self.feature_maps(img)
        # max-pool: "is there a lesion-like spot ANYWHERE?"; mean-pool: overall context
        pooled = torch.cat([f.amax((2, 3)), f.mean((2, 3))], 1)
        return self.head(pooled)
