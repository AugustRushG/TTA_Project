import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerGSM(nn.Module):
    def __init__(self, num_frames, feature_dim=768):
        """
        Transformer-based GSM model for video action recognition.
        """
        super(TransformerGSM, self).__init__()
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.fc = nn.Linear(feature_dim, 2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def lshift_cls_token_zeroPad(self, x, shift):
        # sfhit cls token to left and pad zeros at the end
        # x in shape (B, T, 1, C)
        out = torch.roll(x, shifts=-shift, dims=1)
        out[:, -shift:, :, :] = 0
        return out

    def rshift_zeroPad(self, x, shift):
        # shift cls token to right and pad zeros at the beginning
        # x in shape (B, T, 1, C)
        out = torch.roll(x, shifts=shift, dims=1)
        out[:, :shift, :, :] = 0
        return out 

    def forward(self, x):
        """
        x: input tensor of shape (batch_size * temporal frames, num_patches+1, embed_dim)
        """
        # Implement the forward pass for the Transformer-based 
        B_T, N, C = x.shape  # Batch size times temporal frames, number of patches + 1, embedding dimension
        B = B_T // self.num_frames  # Actual batch size
        T = self.num_frames
        x = x.view(B, T, N, C)  # Reshape to (B, T, N, C)

        cls_tokens = x[:, :, 0:1, :]  # Extract CLS tokens (B, T, 1, C)
        patch_tokens = x[:, :, 1:, :]  # Extract patch tokens (B, T, N-1, C)

        # split cls tokens into two groups 
        cls_tokens_group1 = cls_tokens[:,:,:, C//2:]  # (B, T, 1, C/2)
        cls_tokens_group2 = cls_tokens[:,:,:, :C//2]  # (B, T, 1, C/2)

        cls_relu_norm = self.relu(self.layer_norm(cls_tokens))  # (B, T, 1, C)
        gate = self.tanh(self.fc(cls_relu_norm))  # (B, T, 1, 2)

        gate_group1 = gate[..., 0:1]  # [B, T, 1, 1]
        gate_group2 = gate[..., 1:2]  # [B, T, 1, 1]
  
        y_cls_tokens_group1 = gate_group1 * cls_tokens_group1  # (B, T, 1, C/2)
        y_cls_tokens_group2 = gate_group2 * cls_tokens_group2  # (B, T, 1, C/2)

        r_group1 = cls_tokens_group1 - y_cls_tokens_group1 # (B, T, 1, C/2)
        r_group2 = cls_tokens_group2 - y_cls_tokens_group2 # (B, T, 1, C/2)

        # shift them for past and future
        past_shifted = self.lshift_cls_token_zeroPad(y_cls_tokens_group1, shift=1)  # (B, T, 1, C/2)
        future_shifted = self.rshift_zeroPad(y_cls_tokens_group2, shift=1)  # (B, T, 1, C/2)

        # residual connection to ensure stability
        cls_tokens_group1 = r_group1 + past_shifted # (B, T, 1, C/2)
        cls_tokens_group2 = r_group2 + future_shifted # (B, T, 1, C/2)

        cls_tokens = torch.cat((cls_tokens_group2, cls_tokens_group1), dim=-1)  # (B, T, 1, C)
        x = torch.cat((cls_tokens, patch_tokens), dim=2)  # (B, T, N, C)
        x = x.view(B_T, N, C)  # Reshape back to (B*T, N, C)

        return x
    

class TransformerGSM_V2(nn.Module):
    def __init__(self, num_frames, feature_dim=768):
        """
        Transformer-based GSM model for video action recognition.
        """
        super(TransformerGSM_V2, self).__init__()
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.fc = nn.Linear(feature_dim, 2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def lshift(self, x, shift):
        # sfhit features with duplicate padding
        # x in shape (B, T, N, C)
        out = torch.roll(x, shifts=-shift, dims=1)
        last_feature = out[:, -shift-1, :, :].unsqueeze(1) # (B, 1, N, C)
        out[:, -shift:, :, :] = last_feature.expand(-1, shift, -1, -1)
        return out

    def rshift(self, x, shift):
        # shift features with duplicate padding
        # x in shape (B, T, N, C)
        out = torch.roll(x, shifts=shift, dims=1)
        first_feature = out[:, shift, :, :].unsqueeze(1) # (B, 1, N, C)
        out[:, :shift, :, :] = first_feature.expand(-1, shift, -1, -1)
        return out 

    def forward(self, x):
        """
        x: input tensor of shape (batch_size * temporal frames, num_patches+1, embed_dim)
        """
        # Implement the forward pass for the Transformer-based 
        B_T, N, C = x.shape  # Batch size times temporal frames, number of patches + 1, embedding dimension
        B = B_T // self.num_frames  # Actual batch size
        T = self.num_frames
        x = x.view(B, T, N, C)  # Reshape to (B, T, N, C)

        x_group1 = x[:,:,:, C//2:]  # (B, T, N, C/2)
        x_group2 = x[:,:,:, :C//2]  # (B, T, N, C/2)

        x_relu_norm = self.relu(self.layer_norm(x))  # (B, T, N, C)
        gate = self.tanh(self.fc(x_relu_norm))  # (B, T, N, 2)
        gate_group1 = gate[..., 0:1]  # [B, T, N, 1]
        gate_group2 = gate[..., 1:2]  # [B, T, N, 1]

        y_x_group1 = gate_group1 * x_group1  # (B, T, N, C/2)
        y_x_group2 = gate_group2 * x_group2  # (B, T, N, C/2)

        r_group1 = x_group1 - y_x_group1 # (B, T, N, C/2)
        r_group2 = x_group2 - y_x_group2 # (B, T, N, C/2)

        past_shift = self.lshift(y_x_group1, shift=1)  # (B, T, N, C/2)
        future_shift = self.rshift(y_x_group2, shift=1)  # (B, T, N, C/2)

        # residual connection to ensure stability
        x_group1 = r_group1 + past_shift # (B, T, N, C/2)
        x_group2 = r_group2 + future_shift # (B, T, N, C/2)
        x = torch.cat((x_group2, x_group1), dim=-1)  # (B, T, N, C)
        x = x.view(B_T, N, C)  # Reshape back to (B*T, N, C)

        return x


class AdpativeTransformerGSM(nn.Module):
    def __init__(self, num_frames):
        """
        Transformer-based GSM model for video action recognition.
        """
        super(AdpativeTransformerGSM, self).__init__()
        self.num_frames = num_frames
        self.past_shift_raw = nn.Parameter(torch.tensor(1.0))
        self.future_shift_raw = nn.Parameter(torch.tensor(1.0))

    def lshift_cls_token_zeroPad(self, x, shift):
        # x: [B, T, 1, C]
        B, T, _, C = x.shape
        x = x.squeeze(2)     # [B, T, C]

        # create time indices
        idx = torch.arange(T, device=x.device)     # [T] so this will be like [0, 1, 2, ..., T-1]

        # shift to past: t -> t + shift
        src_idx = idx + shift                      # [T] so if shift is 1, this will be [1, 2, 3, ..., T]
        src_idx = torch.clamp(src_idx, 0, T - 1)   # [T] clamp to valid range [0, T-1]

        # expand indices to batch & feature dims
        src_idx = src_idx.unsqueeze(0).expand(B, -1)         # [B, T]
        src_idx = src_idx.unsqueeze(-1).expand(-1, -1, C)    # [B, T, C]

        # gather
        out = torch.gather(x, dim=1, index=src_idx)          # [B, T, C] This gathers the shifted tokens works like indiexing 

        return out.unsqueeze(2)                              # [B, T, 1, C]

    def rshift_cls_token_zeroPad(self, x, shift):
        B, T, _, C = x.shape
        x = x.squeeze(2)

        idx = torch.arange(T, device=x.device)
        src_idx = idx - shift
        src_idx = torch.clamp(src_idx, 0, T - 1)

        src_idx = src_idx.unsqueeze(0).expand(B, -1)
        src_idx = src_idx.unsqueeze(-1).expand(-1, -1, C)

        out = torch.gather(x, 1, src_idx)
        return out.unsqueeze(2)

    def forward(self, x):
        """
        x: input tensor of shape (batch_size * temporal frames, num_patches+1, embed_dim)
        """
        # Implement the forward pass for the Transformer-based 
        B_T, N, C = x.shape  # Batch size times temporal frames, number of patches + 1, embedding dimension
        B = B_T // self.num_frames  # Actual batch size
        T = self.num_frames
        x = x.view(B, T, N, C)  # Reshape to (B, T, N, C)
        cls_tokens = x[:, :, 0:1, :]  # Extract CLS tokens (B, T, 1, C)
        patch_tokens = x[:, :, 1:, :]  # Extract patch tokens (B, T, N-1, C)

        # split cls tokens into two groups 
        cls_tokens_group1 = cls_tokens[:,:,:, C//2:]  # (B, T, 1, C/2)
        cls_tokens_group2 = cls_tokens[:,:,:, :C//2]  # (B, T, 1, C/2)

        # shift them for past and future
        past_cont = F.softplus(self.past_shift_raw)  
        past_round = torch.round(past_cont)
        past_STE = past_round + (past_cont - past_cont.detach())  # STE
        past_shift = past_STE.long()   # 0-D tensor, NOT .item()

        future_cont = F.softplus(self.future_shift_raw)  
        future_round = torch.round(future_cont)
        future_STE = future_round + (future_cont - future_cont.detach())  # STE
        future_shift = future_STE.long()   # 0-D tensor, NOT .item

        past_shifted = self.lshift_cls_token_zeroPad(cls_tokens_group1, shift=past_shift)  # (B, T, 1, C/2)
        future_shifted = self.rshift_cls_token_zeroPad(cls_tokens_group2, shift=future_shift)  # (B, T, 1, C/2)

        # residual connection to ensure stability
        cls_tokens_group1 = cls_tokens_group1 + past_shifted
        cls_tokens_group2 = cls_tokens_group2 + future_shifted

        cls_tokens = torch.cat((cls_tokens_group2, cls_tokens_group1), dim=-1)  # (B, T, 1, C)
        x = torch.cat((cls_tokens, patch_tokens), dim=2)  # (B, T, N, C)
        x = x.view(B_T, N, C)  # Reshape back to (B*T, N, C)

        return x



class MultiScaleTransformerGSM(nn.Module):
    def __init__(self, num_frames, feature_dim=768, shift_distances=[1,2,3]):
        super().__init__()
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        self.shift_distances = shift_distances
        
        self.layer_norm = nn.LayerNorm(feature_dim)

        # Depthwise temporal kernels
        self.temporal_gates = nn.ModuleList([
            nn.Conv1d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=2*d + 1,
                padding=d,
                groups=feature_dim
            )
            for d in shift_distances
        ])

        # --- IMPROVEMENT 1: Element-wise Gating ---
        # Instead of 3 scalars, we output 'feature_dim' gates.
        # This allows the model to turn off specific channels individually.
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), 
            nn.Sigmoid() 
        )

        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.relu = nn.ReLU()

        self.fusion_weight = nn.Parameter(torch.ones(len(shift_distances)))

    # ... [Keep your lshift and rshift functions exactly as they are] ...
    def lshift(self, x, shift):
        B, T, N, C = x.shape
        if shift == 0: return x
        shift = min(shift, T)
        out = torch.zeros_like(x)
        out[:, :-shift] = x[:, shift:]
        return out

    def rshift(self, x, shift):
        B, T, N, C = x.shape
        if shift == 0: return x
        shift = min(shift, T)
        out = torch.zeros_like(x)
        out[:, shift:] = x[:, :-shift]
        return out

    def shift(self, x, gate_conv, shift_distance):
        B_T, N, C = x.shape
        B = B_T // self.num_frames
        T = self.num_frames

        x = x.view(B, T, N, C)

        # 1. Split Channels (Past / Future / Static)
        part_dim = C // 3
        # Handle remainder if C is not perfectly divisible by 3
        static_dim = C - (2 * part_dim)
        
        x_past, x_future, x_static = torch.split(x, [part_dim, part_dim, static_dim], dim=-1)

        # 2. Compute Gates
        # Normalize for stable gating calculation
        x_norm = self.relu(self.layer_norm(x))
        x_1d = x_norm.permute(0,2,3,1).reshape(B*N, C, T)
        x_gate_feat = gate_conv(x_1d).reshape(B, N, C, T).permute(0,3,1,2)
        
        # Output shape: (B, T, N, C) -> Full element-wise gating
        gate = self.gate_mlp(x_gate_feat) 

        # Split gates to match the features
        g_past, g_future, g_static = torch.split(gate, [part_dim, part_dim, static_dim], dim=-1)

        # 3. Apply Gates
        y_past   = g_past * x_past
        y_future = g_future * x_future
        y_static = g_static * x_static

        # 4. Shift
        # Apply shift only to the specific gated branches
        out_past   = self.lshift(y_past, shift_distance)
        out_future = self.rshift(y_future, shift_distance)
        out_static = y_static

        # 5. Concatenate
        # --- IMPROVEMENT 2: Consistent Ordering ---
        # Keep [Past, Future, Static] order
        out = torch.cat([out_past, out_future, out_static], dim=-1)

        return out.reshape(B_T, N, C)

    def forward(self, x):
        # ... [Your existing forward logic is good] ...
        outs = []
        for d, conv in zip(self.shift_distances, self.temporal_gates):
            outs.append(self.shift(x, conv, d))

        outs = torch.stack(outs, dim=0)
        w = torch.softmax(self.fusion_weight, dim=0)[:,None,None,None]
        fused_result = (w * outs).sum(dim=0)
        
        fused_result = self.proj(fused_result)
        out = x + fused_result 
        return out


if __name__ == "__main__":
    num_dim = 768
    model = MultiScaleTransformerGSM(num_frames=100, feature_dim=num_dim)
    input_tensor = torch.randn(8*100, 197, num_dim)  # Example input tensor
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Should be (8*100, 197, 768)