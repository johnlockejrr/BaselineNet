import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),  # Use inplace ReLU to save memory
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

        # Add spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_att)
        
        return x * psi * spatial_att

class UNet(nn.Module):
    def __init__(self, 
                 in_channels: int = 1,
                 out_channels: int = 3,
                 base_channels: int = 64,
                 depth: int = 4):
        """
        U-Net model with attention gates for baseline detection.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (start points, end points, baseline)
            base_channels: Number of base channels in the first layer
            depth: Depth of the U-Net
        """
        super().__init__()
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        
        # First layer
        self.encoder.append(DoubleConv(in_channels, base_channels))
        
        # Subsequent layers
        for i in range(depth - 1):
            self.encoder.append(DoubleConv(base_channels * (2 ** i), 
                                        base_channels * (2 ** (i + 1))))
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * (2 ** (depth - 1)),
                                   base_channels * (2 ** depth))
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.attention = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            self.decoder.append(nn.ConvTranspose2d(base_channels * (2 ** (i + 1)),
                                                 base_channels * (2 ** i),
                                                 kernel_size=2,
                                                 stride=2))
            self.attention.append(AttentionGate(base_channels * (2 ** i),
                                             base_channels * (2 ** i),
                                             base_channels * (2 ** i) // 2))
            self.decoder.append(DoubleConv(base_channels * (2 ** (i + 1)),
                                        base_channels * (2 ** i)))
        
        # Output layer
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        encoder_outputs = []
        for encoder in self.encoder:
            x = encoder(x)
            encoder_outputs.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip = encoder_outputs.pop()
            skip = self.attention[i//2](x, skip)
            x = torch.cat([x, skip], dim=1)
            x = self.decoder[i+1](x)
        
        # Output
        return self.final(x)

class ContinuousLineModule(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Add horizontal and vertical attention
        self.horizontal_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.vertical_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply horizontal attention
        h_att = self.horizontal_attention(x)
        x = x * h_att
        
        # Apply vertical attention
        v_att = self.vertical_attention(x)
        x = x * v_att
        
        # Apply convolutions
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.relu(x + residual)
        
        return x

class BaselineDetectionModel(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        
        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        
        # First encoder block
        self.enc_blocks.append(DoubleConv(in_channels, base_channels))
        
        # Remaining encoder blocks
        for i in range(depth - 1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            self.enc_blocks.append(DoubleConv(in_ch, out_ch))
        
        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.continuous_modules = nn.ModuleList()
        
        # Decoder blocks with attention and continuous line detection
        for i in range(depth - 1, 0, -1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i - 1))
            self.dec_blocks.append(DoubleConv(in_ch + out_ch, out_ch))
            self.attention_gates.append(AttentionGate(in_ch, out_ch, out_ch))
            self.continuous_modules.append(ContinuousLineModule(out_ch))
        
        # Final convolution
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
        
        # Gradient checkpointing flag
        self.use_checkpointing_enabled = False
    
    def use_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_checkpointing_enabled = True
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        enc_features = []
        for i, enc_block in enumerate(self.enc_blocks):
            if self.use_checkpointing_enabled and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    enc_block, x,
                    use_reentrant=False
                )
            else:
                x = enc_block(x)
                
            if i < len(self.enc_blocks) - 1:
                enc_features.append(x)
                x = self.pool(x)
        
        # Decoder
        for i, (dec_block, att_gate, cont_module) in enumerate(zip(self.dec_blocks, self.attention_gates, self.continuous_modules)):
            # Upsample
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Attention gate
            skip = enc_features[-(i + 1)]
            if self.use_checkpointing_enabled and self.training:
                skip = torch.utils.checkpoint.checkpoint(
                    att_gate, x, skip,
                    use_reentrant=False
                )
            else:
                skip = att_gate(x, skip)
            
            # Concatenate and decode
            x = torch.cat([x, skip], dim=1)
            if self.use_checkpointing_enabled and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    dec_block, x,
                    use_reentrant=False
                )
            else:
                x = dec_block(x)
            
            # Apply continuous line detection
            x = cont_module(x)
        
        # Final convolution
        x = self.final_conv(x)
        return x

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make prediction and return start points, end points, and baseline probabilities.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (start_points, end_points, baseline) probabilities
        """
        output = self.forward(x)
        start_points = torch.sigmoid(output[:, 0])
        end_points = torch.sigmoid(output[:, 1])
        baseline = torch.sigmoid(output[:, 2])
        return start_points, end_points, baseline 