"""
File containing an adapted version of GSf implementation by Swathikiran Sudhakaran, https://github.com/swathikirans/GSF
"""

#Standard imports
import torch
from torch import nn

class _GSF(nn.Module):
    def __init__(self, fPlane, num_segments=8, gsf_ch_ratio=100):
        super(_GSF, self).__init__()

        fPlane_temp = int(fPlane * gsf_ch_ratio / 100)
        if fPlane_temp % 2 != 0:
            fPlane_temp += 1
        self.fPlane = fPlane_temp
        self.conv3D = nn.Conv3d(self.fPlane, 2, (3, 3, 3), stride=1,
                                padding=(1, 1, 1), groups=2)
        self.tanh = nn.Tanh()

        self.num_segments = num_segments
        self.bn = nn.BatchNorm3d(num_features=self.fPlane)
        self.relu = nn.ReLU()
        self.channel_conv1 = nn.Conv2d(2, 1, (3, 3), padding=(3//2, 3//2))
        self.channel_conv2 = nn.Conv2d(2, 1, (3, 3), padding=(3//2, 3//2))
        self.sigmoid = nn.Sigmoid()

    def lshift_zeroPad(self, x):
        out = torch.roll(x, shifts=-1, dims=2)
        out[:, :, -1] = 0
        return out

    def rshift_zeroPad(self, x):
        out = torch.roll(x, shifts=1, dims=2)
        out[:, :, 0] = 0
        return out

    def forward(self, x_full):
        x = x_full[:, :self.fPlane, :, :]
        batchSize = x.size(0) // self.num_segments
        shape = x.size(1), x.size(2), x.size(3)
        x = x.reshape(batchSize, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous()
        x_bn = self.bn(x)
        x_bn_relu = self.relu(x_bn)
        gate = self.tanh(self.conv3D(x_bn_relu))
        gate_group1 = gate[:, 0].unsqueeze(1)
        gate_group2 = gate[:, 1].unsqueeze(1)

        x_group1 = x[:, :self.fPlane // 2]
        x_group2 = x[:, self.fPlane // 2:]

        y_group1 = gate_group1 * x_group1
        y_group2 = gate_group2 * x_group2

        r_group1 = x_group1 - y_group1
        r_group2 = x_group2 - y_group2 # BxCxNxWxH

        y_group1 = self.lshift_zeroPad(y_group1)
        y_group2 = self.rshift_zeroPad(y_group2)

        r_1 = torch.mean(r_group1, dim=-1, keepdim=False)
        r_1 = torch.mean(r_1, dim=-1, keepdim=False).unsqueeze(3)
        r_2 = torch.mean(r_group2, dim=-1, keepdim=False)
        r_2 = torch.mean(r_2, dim=-1, keepdim=False).unsqueeze(3)


        y_1 = torch.mean(y_group1, dim=-1, keepdim=False)
        y_1 = torch.mean(y_1, dim=-1, keepdim=False).unsqueeze(3)
        y_2 = torch.mean(y_group2, dim=-1, keepdim=False)
        y_2 = torch.mean(y_2, dim=-1, keepdim=False).unsqueeze(3) # BxCxN

        y_r_1 = torch.cat([y_1, r_1], dim=3).permute(0, 3, 1, 2)
        y_r_2 = torch.cat([y_2, r_2], dim=3).permute(0, 3, 1, 2) # Bx2xCxN

        y_1_weights = self.sigmoid(self.channel_conv1(y_r_1)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        r_1_weights = 1 - y_1_weights
        y_2_weights = self.sigmoid(self.channel_conv2(y_r_2)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        r_2_weights = 1 - y_2_weights

        y_group1 = y_group1*y_1_weights + r_group1*r_1_weights
        y_group2 = y_group2*y_2_weights + r_group2*r_2_weights

        y_group1 = y_group1.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4,
                                                                                                        5)
        y_group2 = y_group2.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4,
                                                                                                        5)

        y = torch.cat((y_group1.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:]),
                       y_group2.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:])), dim=1)
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(batchSize * self.num_segments, *shape) # [B*T, C, H, W]

        y = torch.cat([y, x_full[:, self.fPlane:, :, :]], dim=1)

        return y
    


class HGSF(nn.Module):
    def __init__(self, fPlane, num_segments=8, gsf_ch_ratio=100, dilations=[1, 2, 3]):
        """
        Initializes the HGSF (Hierarchical Gated Shift Fusion) module.

        Args:
            fPlane (int): The number of feature planes in the input tensor.
            num_segments (int, optional): Number of temporal segments. Default is 8.
            gsf_ch_ratio (int, optional): Ratio to adjust the number of channels for GSF. Default is 100.
            dilations (list of int, optional): List of dilation rates for the convolutional layers. Default is [1, 2, 3].

        Expected Input Shape:
            x (torch.Tensor): Input tensor of shape [B*T, C, H, W], where:
                B - Batch size
                T - Number of temporal segments
                C - Number of channels
                H - Height of the feature map
                W - Width of the feature map

        Output Shape:
            torch.Tensor: Output tensor of shape [B*T, C, H, W].
        """
        super(HGSF, self).__init__()

        fPlane_temp = int(fPlane * gsf_ch_ratio / 100)
        if fPlane_temp % 2 != 0:
            fPlane_temp += 1
        self.fPlane = fPlane_temp
        self.num_segments = num_segments
        self.dilations = dilations

        # Multiple conv3D for multiple dilation levels
        self.gates = nn.ModuleList()
        self.conv1s = nn.ModuleList()
        self.conv2s = nn.ModuleList()
        for d in dilations:
            padding = (d, 1, 1)  # maintain shape
            self.gates.append(
                nn.Conv3d(self.fPlane, 2, (3, 3, 3), stride=1, padding=padding, groups=2, dilation=(d, 1, 1))
            )
            self.conv1s.append(
                nn.Conv2d(2, 1, (3, 3), padding=(3 // 2, 3 // 2), dilation=(1, 1))
            )
            self.conv2s.append(
                nn.Conv2d(2, 1, (3, 3), padding=(3 // 2, 3 // 2), dilation=(1, 1))
            )

        self.tanh = nn.Tanh()

        self.bn = nn.BatchNorm3d(num_features=self.fPlane)
        self.relu = nn.ReLU()

       
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def lshift_zeroPad(self, x, shift):
        out = torch.roll(x, shifts=-shift, dims=2)
        out[:, :, -shift:] = 0
        return out

    def rshift_zeroPad(self, x, shift):
        out = torch.roll(x, shifts=shift, dims=2)
        out[:, :, :shift] = 0
        return out
    
    def _fuse(self, x_full, conv3D, conv2D1, conv2D2, shift_amount):
        x = x_full[:, :self.fPlane, :, :]
        batchSize = x.size(0) // self.num_segments
        shape = x.size(1), x.size(2), x.size(3)
        x = x.reshape(batchSize, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous()
        x_bn = self.bn(x)
        x_bn_relu = self.relu(x_bn)
        gate = self.tanh(conv3D(x_bn_relu))
        gate_group1 = gate[:, 0].unsqueeze(1)
        gate_group2 = gate[:, 1].unsqueeze(1)

        x_group1 = x[:, :self.fPlane // 2]
        x_group2 = x[:, self.fPlane // 2:]

        y_group1 = gate_group1 * x_group1
        y_group2 = gate_group2 * x_group2

        r_group1 = x_group1 - y_group1
        r_group2 = x_group2 - y_group2 # BxCxNxWxH

        y_group1 = self.lshift_zeroPad(y_group1, shift_amount)
        y_group2 = self.rshift_zeroPad(y_group2, shift_amount)

        r_1 = torch.mean(r_group1, dim=-1, keepdim=False)
        r_1 = torch.mean(r_1, dim=-1, keepdim=False).unsqueeze(3)
        r_2 = torch.mean(r_group2, dim=-1, keepdim=False)
        r_2 = torch.mean(r_2, dim=-1, keepdim=False).unsqueeze(3)


        y_1 = torch.mean(y_group1, dim=-1, keepdim=False)
        y_1 = torch.mean(y_1, dim=-1, keepdim=False).unsqueeze(3)
        y_2 = torch.mean(y_group2, dim=-1, keepdim=False)
        y_2 = torch.mean(y_2, dim=-1, keepdim=False).unsqueeze(3) # BxCxN

        y_r_1 = torch.cat([y_1, r_1], dim=3).permute(0, 3, 1, 2)
        y_r_2 = torch.cat([y_2, r_2], dim=3).permute(0, 3, 1, 2) # Bx2xCxN

        y_1_weights = self.sigmoid(conv2D1(y_r_1)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        r_1_weights = 1 - y_1_weights
        y_2_weights = self.sigmoid(conv2D2(y_r_2)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        r_2_weights = 1 - y_2_weights

        y_group1 = y_group1*y_1_weights + r_group1*r_1_weights
        y_group2 = y_group2*y_2_weights + r_group2*r_2_weights

        y_group1 = y_group1.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4,
                                                                                                        5)
        y_group2 = y_group2.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4,
                                                                                                        5)

        y = torch.cat((y_group1.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:]),
                       y_group2.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:])), dim=1)
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(batchSize * self.num_segments, *shape)
        y = torch.cat([y, x_full[:, self.fPlane:, :, :]], dim=1)

        return y

    
    def forward(self, x):
        fused_outputs = []
        for gate_conv, conv2D1, conv2D2, dilation in zip(self.gates, self.conv1s, self.conv2s, self.dilations):
            fused = self._fuse(x, gate_conv, conv2D1, conv2D2, shift_amount=dilation) # [B*T, C, H, W]
            fused_outputs.append(fused)
        
        fused_outputs = torch.stack(fused_outputs, dim=0)  # [num_dilations, B*T, C, H, W]

        y = torch.sum(fused_outputs, dim=0)  # [B*T, C, H, W]

        return y

