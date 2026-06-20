import torch
import torch.nn as nn

STATE_DIM = 13
ACTION_DIM = 2


class AdjustedBackbone(nn.Module):

    def __init__(self, output_dim=64, print_full: bool = False):
        super().__init__()
        self.output_dim = output_dim
        self.print_full = print_full
        self.lidar_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.heatmap_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=4, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.heatmap_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 32 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32 // 16, 32, kernel_size=1),
            nn.Sigmoid(),
        )
        self.combined_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        lidar_channel = x[:, 0:1, :, :]
        heatmap_channel = x[:, 1:2, :, :]
        lidar_features = self.lidar_branch(lidar_channel)
        heatmap_features = self.heatmap_branch(heatmap_channel)
        heatmap_attention_weights = self.heatmap_attention(heatmap_features)
        heatmap_features = heatmap_features * heatmap_attention_weights
        combined_features = torch.cat([lidar_features, heatmap_features], dim=1)
        final_features = self.combined_conv(combined_features)
        flat_features = self.flatten(final_features)
        return self.fc(flat_features)


class Decoder(nn.Module):

    def __init__(
        self, backbone_output_dim=64, state_dim=STATE_DIM, hidden_dim=128, output_dim=48
    ):
        super().__init__()
        total_input_dim = backbone_output_dim + state_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


# Dual-stream encoder for LiDAR proximity and PSTO heatmap channels.
class FullModel(nn.Module):

    def __init__(
        self,
        backbone_output_dim=64,
        state_dim=STATE_DIM,
        decoder_output_dim=48,
        print_full: bool = False,
    ):
        super().__init__()
        self.conv_backbone = AdjustedBackbone(
            output_dim=backbone_output_dim, print_full=print_full
        )
        self.decoder = Decoder(
            backbone_output_dim=backbone_output_dim,
            state_dim=state_dim,
            output_dim=decoder_output_dim,
        )

    def forward(self, fused_image, state):
        original_shape = fused_image.shape[:-3]
        if len(fused_image.shape) > 4:
            img_batch = fused_image.reshape(-1, *fused_image.shape[-3:])
            state_batch = state.reshape(-1, state.shape[-1])
        else:
            img_batch = fused_image
            state_batch = state
        shift_amount = img_batch.shape[-1] // 2
        img_batch_centered = torch.roll(img_batch, shifts=shift_amount, dims=-1)
        img_batch_centered = torch.flip(img_batch_centered, dims=[-1])
        img_batch_centered = torch.flip(img_batch_centered, dims=[-2])
        img_features = self.conv_backbone(img_batch_centered)
        combined_features = torch.cat([state_batch, img_features], dim=-1)
        output = self.decoder(combined_features)
        if len(original_shape) > 0:
            output = output.view(*original_shape, -1)
        return output
