# Adapted PyTorch GSM implementation by Swathikiran Sudhakaran, https://github.com/swathikirans/GSM
# Original license for GSM
"""
BSD 2-Clause License for GSM

Copyright (c) 2019, FBK
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
from torch import nn
from torch.cuda import FloatTensor as ftens


class _GSM(nn.Module):
    def __init__(self, fPlane, num_segments=3):
        super(_GSM, self).__init__()

        self.conv3D = nn.Conv3d(fPlane, 2, (3, 3, 3), stride=1,
                                padding=(1, 1, 1), groups=2)
        nn.init.constant_(self.conv3D.weight, 0)
        nn.init.constant_(self.conv3D.bias, 0)
        self.tanh = nn.Tanh()
        self.fPlane = fPlane
        self.num_segments = num_segments
        self.bn = nn.BatchNorm3d(num_features=fPlane)
        self.relu = nn.ReLU()

    def lshift_zeroPad(self, x):
        return torch.cat((x[:,:,1:], ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0)), dim=2)

    def rshift_zeroPad(self, x):
        return torch.cat((ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0), x[:,:,:-1]), dim=2)

    def forward(self, x):
        # x: [B*T, C, H, W]
        batchSize = x.size(0) // self.num_segments 
        shape = x.size(1), x.size(2), x.size(3)
        assert  shape[0] == self.fPlane
        x = x.view(batchSize, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        x_bn = self.bn(x)
        x_bn_relu = self.relu(x_bn)
        gate = self.tanh(self.conv3D(x_bn_relu)) # [B, 2, T, H, W]
        gate_group1 = gate[:, 0].unsqueeze(1) # [B, 1, T, H, W]
        gate_group2 = gate[:, 1].unsqueeze(1) # [B, 1, T, H, W]
        x_group1 = x[:, :self.fPlane // 2] # get first half of channels in shape [B, C/2, T, H, W]
        x_group2 = x[:, self.fPlane // 2:] # get second half of channels
        y_group1 = gate_group1 * x_group1 # use weight on first half of channels
        y_group2 = gate_group2 * x_group2 # use weight on second half of channels

        r_group1 = x_group1 - y_group1 # compute residual for first half
        r_group2 = x_group2 - y_group2 # compute residual for second half

        y_group1 = self.lshift_zeroPad(y_group1) + r_group1 # shift left and add residual
        y_group2 = self.rshift_zeroPad(y_group2) + r_group2 # shift right and add residual

        y_group1 = y_group1.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5) # reshape back 
        y_group2 = y_group2.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)

        y = torch.cat((y_group1.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:]),
                       y_group2.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:])), dim=1)

        return y.permute(0, 2, 1, 3, 4).contiguous().view(batchSize*self.num_segments, *shape)



class TokenGSM(nn.Module):
    def __init__(self, channels, shift_ratio=0.25):
        super().__init__()
        self.channels = channels
        self.shift_size = int(channels * shift_ratio)

        # Gating mechanism (optional)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, T, N, C]
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, N, T, C]

        out = torch.zeros_like(x)

        # Split channels
        c1 = x[:, :, :-1, :self.shift_size]     # shift forward
        c2 = x[:, :, 1:, self.shift_size:2*self.shift_size]  # shift backward
        c_rest = x[:, :, :, 2*self.shift_size:] # stay

        out[:, :, 1:, :self.shift_size] = c1
        out[:, :, :-1, self.shift_size:2*self.shift_size] = c2
        out[:, :, :, 2*self.shift_size:] = c_rest

        # Optional: gated fusion
        g = self.gate(x.permute(0, 2, 1, 3))  # back to [B, T, N, C]
        out = g * out.permute(0, 2, 1, 3) + (1 - g) * x.permute(0, 2, 1, 3)

        return out
    


class HGSM(nn.Module):
    def __init__(self, fPlane, num_segments=3, dilations=[1, 2]):
        """
        fPlane: number of input channels
        num_segments: number of frames
        dilations: list of dilation values, e.g., [1, 2, 4]
        """
        super(HGSM, self).__init__()
        self.fPlane = fPlane
        self.num_segments = num_segments
        self.dilations = dilations

        self.gates = nn.ModuleList()
        for d in dilations:
            # Each dilation has its own conv3d
            padding = (d, 1, 1)  # maintain size
            self.gates.append(
                nn.Conv3d(fPlane, 2, (3, 3, 3), stride=1, padding=padding, groups=2, dilation=(d, 1, 1))
            )

        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm3d(fPlane)
        self.relu = nn.ReLU()

        # Optional: learnable weights to fuse different dilation outputs
        self.alpha = nn.Parameter(torch.ones(len(dilations)))
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
        assert shift > 0
        out = torch.roll(x, shifts=-shift, dims=2)
        out[:, :, -shift:] = 0
        return out

    def rshift_zeroPad(self, x, shift):
        assert shift > 0
        out = torch.roll(x, shifts=shift, dims=2)
        out[:, :, :shift] = 0
        return out

    def _fuse(self, x, gate_conv, shift_amount):
        # x: [B, C, T, H, W]
        batchSize = x.size(0)
        shape = x.size(1), x.size(3), x.size(4)  # [C, H, W]

        gate = self.tanh(gate_conv(x))  # [B, 2, T, H, W]
        gate_group1 = gate[:, 0].unsqueeze(1)
        gate_group2 = gate[:, 1].unsqueeze(1)

        x_group1 = x[:, :self.fPlane // 2] # shape [B, C/2, T, H, W]
        x_group2 = x[:, self.fPlane // 2:] # shape [B, C/2, T, H, W]

        y_group1 = gate_group1 * x_group1 # shape [B, C/2, T, H, W] gate_group means weights which controls how much features to keep 
        y_group2 = gate_group2 * x_group2 # shape [B, C/2, T, H, W]

        r_group1 = x_group1 - y_group1 # compute residual for first half
        r_group2 = x_group2 - y_group2 # compute residual for second half

        y_group1 = self.lshift_zeroPad(y_group1, shift_amount) + r_group1 # shift the first half left and add residual (shift left means add current frame to the previous frame for all temporal positions)
        y_group2 = self.rshift_zeroPad(y_group2, shift_amount) + r_group2

        # Reshape back
        y_group1 = y_group1.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)
        y_group2 = y_group2.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)

        y = torch.cat((y_group1.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:]),
                       y_group2.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:])), dim=1)

        # [B, C, T, H, W]
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(batchSize*self.num_segments, *shape)
        return y

    def forward(self, x):
        # x: [B*T, C, H, W]
        batchSize = x.size(0) // self.num_segments
        shape = x.size(1), x.size(2), x.size(3)
        assert shape[0] == self.fPlane

        x = x.view(batchSize, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        x_bn = self.bn(x)
        x_bn_relu = self.relu(x_bn)

        fused_outputs = []
        for gate_conv, dilation in zip(self.gates, self.dilations):
            fused = self._fuse(x_bn_relu, gate_conv, shift_amount=dilation)
            fused_outputs.append(fused)

        fused_outputs = torch.stack(fused_outputs, dim=0)  # [num_dilations, B*T, C, H, W]

        # Weighted sum
        y = (self.alpha.view(-1, 1, 1, 1, 1) * fused_outputs).sum(dim=0)

        return y
    


class AGSM(nn.Module):
    def __init__(self, fPlane, num_segments=3, num_heads=4):
        super(AGSM, self).__init__()

        self.conv3D = nn.Conv3d(fPlane, 2, (3, 3, 3), stride=1,
                                padding=(1, 1, 1), groups=2)
        nn.init.constant_(self.conv3D.weight, 0)
        nn.init.constant_(self.conv3D.bias, 0)
        self.tanh = nn.Tanh()
        self.fPlane = fPlane
        self.num_segments = num_segments
        self.num_heads = num_heads
        self.bn = nn.BatchNorm3d(num_features=fPlane)
        self.relu = nn.ReLU()

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=fPlane, out_channels=num_heads, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Sigmoid()
        )


    def lshift_zeroPad(self, x):
        return torch.cat((x[:,:,1:], ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0)), dim=2)

    def rshift_zeroPad(self, x):
        return torch.cat((ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0), x[:,:,:-1]), dim=2)

    def forward(self, x):
        # x: [B*T, C, H, W]
        batchSize = x.size(0) // self.num_segments 
        shape = x.size(1), x.size(2), x.size(3)
        assert  shape[0] == self.fPlane

        # Apply spatial attention
        x_split = torch.chunk(x, self.num_heads, dim=1)  # shape: List of [B*T, C_head, H, W]
        spatial_attn = self.spatial_attention(x) # [B*T, num_heads, H, W]
        # Apply each head
        attn_applied = [
            x_split[i] * spatial_attn[:, i:i+1]
            for i in range(self.num_heads)
        ]# Apply each head
        x = torch.cat(attn_applied, dim=1) # [B*T, C, H, W]

        x = x.view(batchSize, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        x_bn = self.bn(x)
        x_bn_relu = self.relu(x_bn)
        gate = self.tanh(self.conv3D(x_bn_relu)) # [B, 2, T, H, W]
        gate_group1 = gate[:, 0].unsqueeze(1) # [B, 1, T, H, W]
        gate_group2 = gate[:, 1].unsqueeze(1) # [B, 1, T, H, W]
        x_group1 = x[:, :self.fPlane // 2] # get first half of channels
        x_group2 = x[:, self.fPlane // 2:] # get second half of channels
        y_group1 = gate_group1 * x_group1 # use weight on first half of channels
        y_group2 = gate_group2 * x_group2 # use weight on second half of channels

        r_group1 = x_group1 - y_group1 # compute residual for first half
        r_group2 = x_group2 - y_group2 # compute residual for second half

        y_group1 = self.lshift_zeroPad(y_group1) + r_group1 # shift left and add residual
        y_group2 = self.rshift_zeroPad(y_group2) + r_group2 # shift right and add residual

        y_group1 = y_group1.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)
        y_group2 = y_group2.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)

        y = torch.cat((y_group1.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:]),
                       y_group2.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:])), dim=1)

        return y.permute(0, 2, 1, 3, 4).contiguous().view(batchSize*self.num_segments, *shape)







class HAGSM(nn.Module):
    def __init__(self, fPlane, num_segments=3, dilations=[1, 2], num_heads=4):
        """
        fPlane: number of input channels
        num_segments: number of frames
        dilations: list of dilation values, e.g., [1, 2, 4]
        """
        super(HAGSM, self).__init__()
        self.fPlane = fPlane
        self.num_segments = num_segments
        self.dilations = dilations
        self.num_heads = num_heads

        if self.num_heads > 0:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(in_channels=fPlane, out_channels=num_heads, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Sigmoid()
            )

        self.gates = nn.ModuleList()
        for d in dilations:
            # Each dilation has its own conv3d
            padding = (d, 1, 1)  # maintain size
            self.gates.append(
                nn.Conv3d(fPlane, 2, (3, 3, 3), stride=1, padding=padding, groups=2, dilation=(d, 1, 1))
            )

        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm3d(fPlane)
        self.relu = nn.ReLU()

        # Optional: learnable weights to fuse different dilation outputs
        self.alpha = nn.Parameter(torch.ones(len(dilations)))
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
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def lshift_zeroPad(self, x, shift):
        assert shift > 0
        out = torch.roll(x, shifts=-shift, dims=2)
        out[:, :, -shift:] = 0
        return out

    def rshift_zeroPad(self, x, shift):
        assert shift > 0
        out = torch.roll(x, shifts=shift, dims=2)
        out[:, :, :shift] = 0
        return out

    def _fuse(self, x, gate_conv, shift_amount):
        # x: [B, C, T, H, W]
        batchSize = x.size(0)
        shape = x.size(1), x.size(3), x.size(4)  # [C, H, W]

        gate = self.tanh(gate_conv(x))  # [B, 2, T, H, W]
        gate_group1 = gate[:, 0].unsqueeze(1)
        gate_group2 = gate[:, 1].unsqueeze(1)

        x_group1 = x[:, :self.fPlane // 2] # shape [B, C/2, T, H, W]
        x_group2 = x[:, self.fPlane // 2:] # shape [B, C/2, T, H, W]

        y_group1 = gate_group1 * x_group1 # shape [B, C/2, T, H, W] gate_group means weights which controls how much features to keep 
        y_group2 = gate_group2 * x_group2 # shape [B, C/2, T, H, W]

        r_group1 = x_group1 - y_group1 # compute residual for first half
        r_group2 = x_group2 - y_group2 # compute residual for second half

        y_group1 = self.lshift_zeroPad(y_group1, shift_amount) + r_group1 # shift the first half left and add residual (shift left means add current frame to the previous frame for all temporal positions)
        y_group2 = self.rshift_zeroPad(y_group2, shift_amount) + r_group2

        # Reshape back
        y_group1 = y_group1.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)
        y_group2 = y_group2.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)

        y = torch.cat((y_group1.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:]),
                       y_group2.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:])), dim=1)

        # [B, C, T, H, W]
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(batchSize*self.num_segments, *shape)
        return y

    def forward(self, x):
        # x: [B*T, C, H, W]
        batchSize = x.size(0) // self.num_segments
        shape = x.size(1), x.size(2), x.size(3)
        assert shape[0] == self.fPlane

        if self.num_heads > 0:
            # Apply spatial attention
            x_split = torch.chunk(x, self.num_heads, dim=1)  # shape: List of [B*T, C_head, H, W]
            spatial_attn = self.spatial_attention(x) # [B*T, num_heads, H, W]
            # Apply each head
            attn_applied = [
                x_split[i] * spatial_attn[:, i:i+1]
                for i in range(self.num_heads)
            ]# Apply each head
            x = torch.cat(attn_applied, dim=1) # [B*T, C, H, W]
        
        
        x = x.view(batchSize, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        x_bn = self.bn(x)
        x_bn_relu = self.relu(x_bn)

        fused_outputs = []
        for gate_conv, dilation in zip(self.gates, self.dilations):
            fused = self._fuse(x_bn_relu, gate_conv, shift_amount=dilation)
            fused_outputs.append(fused)

        fused_outputs = torch.stack(fused_outputs, dim=0)  # [num_dilations, B*T, C, H, W]

        # Weighted sum
        y = (self.alpha.view(-1, 1, 1, 1, 1) * fused_outputs).sum(dim=0)

        return y