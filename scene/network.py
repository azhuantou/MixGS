import torch
import torch.nn as nn
from hashencoder.hashgrid import HashEncoder


class GSEncoder(nn.Module):
    def __init__(
            self,
            canonical_num_levels=16,
            canonical_level_dim=2,
            canonical_base_resolution=16,
            canonical_desired_resolution=2048,
            canonical_log2_hashmap_size=19,
            decoder_num_levels=32,
            decoder_level_dim=2,
            bound=256.0,
    ):
        super(GSEncoder, self).__init__()
        self.out_dim = canonical_num_levels * canonical_level_dim + decoder_num_levels * decoder_level_dim * 3
        self.canonical_num_levels = canonical_num_levels
        self.canonical_level_dim = canonical_level_dim
        self.bound = bound

        self.xyz_encoding = HashEncoder(
            input_dim=3,
            num_levels=canonical_num_levels,
            level_dim=canonical_level_dim, 
            per_level_scale=2,
            base_resolution=canonical_base_resolution, 
            log2_hashmap_size=canonical_log2_hashmap_size,
            desired_resolution=canonical_desired_resolution,
        )

    def forward(self, coords, pose):

        return self.xyz_encoding(coords, size=self.bound), pose.unsqueeze(0).repeat(coords.size()[0], 1)


class GSDecoder(nn.Module):
    def __init__(
            self,
            spatial_in_dim,
            mlp_in_dim,
            depth=1,
            width=256,
    ):
        super(GSDecoder, self).__init__()
        self.depth = depth
        self.width = width

        self.spatial_mlp = nn.Sequential(
            nn.Linear(spatial_in_dim, width),
        )

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, width),
            nn.ReLU(),
        )

        mlp = []
        for _ in range(depth):
            mlp.append(nn.Linear(width, width))
            mlp.append(nn.ReLU())
        self.tiny_mlp = nn.Sequential(*mlp)

        self.gaussian_color = nn.Linear(width, 3)
        self.gaussian_rotation = nn.Linear(width, 4)
        self.gaussian_scaling = nn.Linear(width, 3)
        self.gaussian_opacity = nn.Linear(width, 1)

    def forward(self, spatial_h, pose_input, scale_input, rotate_input):
        spatial_h = self.spatial_mlp(spatial_h)
        cat_feat = torch.cat([pose_input, scale_input, rotate_input], dim=1)

        cat_feat = self.mlp(cat_feat)
        h = spatial_h * (2 * torch.sigmoid(cat_feat) - 1)
        h = self.tiny_mlp(h)

        color = self.gaussian_color(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)
        opacity = torch.sigmoid(self.gaussian_opacity(h))

        return color, rotation, scaling, opacity
