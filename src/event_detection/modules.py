import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SingleStageTCN
from .impl.asformer import MyTransformer

class FCPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(x.reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1)


class GRUPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self._fc_out = FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class TCNPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_stages=1, num_layers=5):
        super().__init__()

        self._tcn = SingleStageTCN(
            feat_dim, 256, num_classes, num_layers, True)
        self._stages = None
        if num_stages > 1:
            self._stages = nn.ModuleList([SingleStageTCN(
                num_classes, 256, num_classes, num_layers, True)
                for _ in range(num_stages - 1)])

    def forward(self, x):
        x = self._tcn(x)
        if self._stages is None:
            return x
        else:
            outputs = [x]
            for stage in self._stages:
                x = stage(F.softmax(x, dim=2))
                outputs.append(x)
            return torch.stack(outputs, dim=0)


class ASFormerPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_decoders=3, num_layers=5):
        super().__init__()

        r1, r2 = 2, 2
        num_f_maps = 64
        self._net = MyTransformer(
            num_decoders, num_layers, r1, r2, num_f_maps, feat_dim,
            num_classes, channel_masking_rate=0.3)

    def forward(self, x):
        B, T, D = x.shape
        return self._net(
            x.permute(0, 2, 1), torch.ones((B, 1, T), device=x.device)
        ).permute(0, 1, 3, 2)
    





class EDSGPMIXERLayers(nn.Module):
    def __init__(self, feat_dim, clip_len, num_layers=1, ks=3, k=2, k_factor = 2, concat = True):
        super().__init__()
        self.num_layers = num_layers
        self.tot_layers = num_layers * 2 + 1
        self._sgp = nn.ModuleList(SGPBlock(feat_dim, kernel_size=ks, k=k, init_conv_vars=0.1) for _ in range(self.tot_layers))
        self._pooling = nn.ModuleList(nn.AdaptiveMaxPool1d(output_size = math.ceil(clip_len / (k_factor**(i+1)))) for i in range(num_layers))
        #self._upsample = nn.ModuleList(nn.Upsample(size = math.ceil(clip_len / (k_factor**i)), mode = 'linear', align_corners = True) for i in range(num_layers))
        self._sgpMixer = nn.ModuleList(SGPMixer(feat_dim, kernel_size=ks, k=k, init_conv_vars=0.1, 
                                        t_size = math.ceil(clip_len / (k_factor**i)), concat=concat) for i in range(num_layers))

    def forward(self, x):
        store_x = [] #Store the intermediate outputs
        #Downsample
        x = x.permute(0, 2, 1)
        for i in range(self.num_layers):
            x = self._sgp[i](x)
            store_x.append(x)
            x = self._pooling[i](x)
        
        #Intermediate
        x = self._sgp[self.num_layers](x)

        #Upsample
        for i in range(self.num_layers):
            x = self._sgpMixer[- (i + 1)](x = x, z = store_x[- (i + 1)])
            x = self._sgp[self.num_layers + i + 1](x)
        x = x.permute(0, 2, 1)

        return x
    
class SGPBlock(nn.Module):

    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            init_conv_vars=0.1,  # init gaussian variance for the weight
            mode='normal'
    ):
        super().__init__()
        # must use odd sized kernel
        # assert (kernel_size % 2 == 1) and (kernel_size > 1)
        # padding = kernel_size // 2

        self.kernel_size = kernel_size

        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)

        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )


        self.act = act_layer()
        self.sigm = nn.Sigmoid()
        self.reset_params(init_conv_vars=init_conv_vars)

        self.mode = mode

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def forward(self, x):
        # X shape: B, C, T
        B, C, T = x.shape

        out = self.ln(x)
        psi = self.psi(out)
        fc = self.fc(out)
        convw = self.convw(out)
        convkw = self.convkw(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        if self.mode == 'normal':
            out = fc * phi + (convw + convkw) * psi + out #fc * phi instant level / (convw + convkw) * psi window level
        elif self.mode == 'sigm1':
            out = fc * phi + self.sigm(convw + convkw) * psi + out
        elif self.mode == 'sigm2':
            out = fc * self.sigm(phi) + self.sigm(convw + convkw) * psi + out
        elif self.mode == 'sigm3':
            out = self.sigm(fc) * phi + (convw + convkw) * self.sigm(psi) + out
        #out = fc * phi + out #only instant level
        #out = (convw + convkw) * psi + out #only window level
        #out = fc * phi + self.sigm(convw + convkw) * psi + out # sigmoid down branch window-level
        #out = fc * self.sigm(phi) + self.sigm(convw + convkw) * psi + out # sigmoid down branch window-level + up branch instant-level
        #out = self.sigm(fc) * phi + (convw + convkw) * self.sigm(psi) + out # sigmoid up branch window-level + down branch instant-level


        out = x + out
        # FFN
        out = out + self.mlp(self.gn(out))

        return out
    
class SGPMixer(nn.Module):

    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            init_conv_vars=0.1,  # init gaussian variance for the weight
            t_size = 0,
            concat = True
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.concat = concat

        if n_out is None:
            n_out = n_embd

        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi1 = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.psi2 = nn.Conv1d(n_embd, n_embd, kernel_size = kernel_size, stride = 1, padding = kernel_size // 2, groups = n_embd)
        self.convw1 = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw1 = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
        self.convw2 = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw2 = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)

        self.fc1 = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.global_fc1 = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        self.fc2 = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.global_fc2 = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        
        self.upsample = nn.Upsample(size = t_size, mode = 'linear', align_corners = True)

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        if self.concat:
            self.concat_fc = nn.Conv1d(n_embd * 6, n_embd, 1, groups = group)

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.psi2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc2.weight, 0, init_conv_vars)

        torch.nn.init.constant_(self.psi1.bias, 0)
        torch.nn.init.constant_(self.psi2.bias, 0)
        torch.nn.init.constant_(self.convw1.bias, 0)
        torch.nn.init.constant_(self.convkw1.bias, 0)
        torch.nn.init.constant_(self.convw2.bias, 0)
        torch.nn.init.constant_(self.convkw2.bias, 0)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.constant_(self.fc2.bias, 0)
        torch.nn.init.constant_(self.global_fc1.bias, 0)
        torch.nn.init.constant_(self.global_fc2.bias, 0)

        if self.concat:
            torch.nn.init.normal_(self.concat_fc.weight, 0, init_conv_vars)
            torch.nn.init.constant_(self.concat_fc.bias, 0)

    def forward(self, x, z):
        # X shape: B, C, T
        B, C, T = x.shape
        z = self.ln1(z)
        x = self.ln2(x)
        x = self.upsample(x)
        #x = self.ln2(x) # modified to have upsample inside sgp-mixer module (which seems more elegant)
        psi1 = self.psi1(z)
        psi2 = self.psi2(x)
        convw1 = self.convw1(z)
        convkw1 = self.convkw1(z)
        convw2 = self.convw2(x)
        convkw2 = self.convkw2(x)
        #Instant level branches
        fc1 = self.fc1(z)
        fc2 = self.fc2(x)
        phi1 = torch.relu(self.global_fc1(z.mean(dim=-1, keepdim=True)))
        phi2 = torch.relu(self.global_fc2(x.mean(dim=-1, keepdim=True)))

        out1 = (convw1 + convkw1) * psi1
        out2 = (convw2 + convkw2) * psi2
        out3 = fc1 * phi1
        out4 = fc2 * phi2

        if self.concat:
            out = torch.cat((out1, out2, out3, out4, z, x), dim = 1)
            out = self.act(self.concat_fc(out))

        else:
            out = out1 + out2 + out3 + out4 + z + x

        #out = z + out
        # FFN
        out = out + self.mlp(self.gn(out))

        return out

class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out
    

class FCLayers(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(self.dropout(x).reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1)
    
class FC2Layers(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc1 = FCLayers(feat_dim, num_classes[0])
        self._fc2 = FCLayers(feat_dim, num_classes[1])

    def forward(self, x):
        x = torch.cat([self._fc1(x), self._fc2(x)], dim = 2)
        return x
    

class ObjectFusion(nn.Module):
    def __init__(self, 
                 env_dim, obj_dim, 
                 hidden_dim,
                 num_encoders, heads,
                 max_obj, dropout=0.1):
        super().__init__()

        self._env_dim = env_dim
        self._obj_dim = obj_dim
        self.hidden_dim = hidden_dim
        self._max_obj = max_obj

        self._env_linear = nn.Linear(env_dim, hidden_dim)
        self._obj_linear = nn.Linear(obj_dim, hidden_dim)
        self._env_norm = Norm(env_dim)
        self._obj_norm = Norm(obj_dim)
        self._obj_env_norm = Norm(hidden_dim)

        self._obj_fuser = Encoder(
            hidden_dim, 
            num_encoders, 
            heads, 
            use_pe=False,
            dropout=dropout)
        self._env_obj_fuser = Encoder(
            hidden_dim,
            num_encoders,
            heads,
            use_pe=False,
            dropout=dropout)

    def fuse_obj(self, env_feat, obj_feat, obj_mask):
        assert(env_feat.size()[-1] == obj_feat.size()[-1]), \
            'Hidden dimension of environment and object must be the same'
        
        batch_size, frames, max_obj, hidden_dim = obj_feat.size()

        # Broadcast all env feat to obj feat
        obj_env_feat = torch.unsqueeze(env_feat, 2) + obj_feat  
        obj_fused_feat = torch.zeros(batch_size, frames, hidden_dim).cuda()
        if (max_obj == 0):
            return obj_fused_feat
        
        # Step each frame
        for begin in range(0, frames):
            end = begin + 1

            # Single frame: batch x max_obj x dim
            # Single mask: batch x max_obj
            frame_oe_feat = obj_env_feat[:, begin:end].contiguous().view(-1, max_obj, hidden_dim)
            mask = obj_mask[:, begin:end].contiguous().view(-1, max_obj)

            # Hard-attention mask
            l2_norm = torch.norm(frame_oe_feat, dim=-1)     # score: batch x max_obj
            l2_norm_softmax = masked_softmax(l2_norm, mask) # softmax: batch x max_obj

            # Adaptive threshold = 1/nObject
            # Output: batch
            esp = 1e-8
            adaptive_thresh = torch.clamp(1. / (esp + torch.sum(mask, dim=-1, keepdim=True)), 0., 1.)

            # Create mask for hard-attn
            # Output: batch x max_obj
            hard_attn_mask = l2_norm_softmax >= adaptive_thresh

            # Choose batch to keep, just keep the batch has more than 1 object
            # Output: batch
            keep_mask = (torch.sum(hard_attn_mask, dim=-1) > 0)
            keep_idx = torch.masked_select(
                torch.arange(hard_attn_mask.size(0)).cuda(),
                keep_mask
            )

            # Get object feature
            # Output: max_obj x batch x hidden_dim
            fuser_input = obj_feat[:, begin:end].contiguous().view(-1, max_obj, hidden_dim)

            if (len(keep_idx)>0):
                fuser_input = fuser_input[keep_idx]         # batch x max_obj x hidden_dim
                hard_attn_mask = hard_attn_mask[keep_idx]   # batch x max_obj

                # Pass to encoder
                # Output: batch x max_obj x hidden_dim
                fuser_output = self._obj_fuser(fuser_input, key_padding_mask=~hard_attn_mask)

                # Normalize result over objects
                fuser_output = torch.sum(fuser_output, dim=1) / torch.sum(hard_attn_mask, dim=-1, keepdim=True)

                padded_output = torch.zeros(batch_size, hidden_dim).cuda()
                padded_output[keep_idx] = fuser_output
                obj_fused_feat[:, begin:end] = padded_output.view(batch_size, -1, hidden_dim)
        
        return obj_fused_feat

    # Fuse object feature to environment feature
    # env feature size: batch x frames x env_dim
    # obj feature size: batch x frames x max_obj x obj_dim
    # obj mask size: batch x frames x max_obj
    # project feature size: batch x frames x hidden_dim
    def forward(self, env_feat, obj_feat, obj_mask):
        env_feat = self._env_norm(env_feat) 
        env_feat = self._env_linear(env_feat)
        
        cnt_nan = torch.sum(torch.isnan(env_feat)).item()
        assert cnt_nan == 0, 'Env feat contains nan'

        obj_feat = self._obj_norm(obj_feat) 
        obj_feat = self._obj_linear(obj_feat)
        
        cnt_nan = torch.sum(torch.isnan(obj_feat)).item()
        assert cnt_nan == 0, 'Obj feat contains nan'

        # Fuse object
        # Output: batch x frames x hidden_dim

        obj_fused_feat = self.fuse_obj(env_feat, obj_feat, obj_mask)
        
        cnt_nan = torch.sum(torch.isnan(obj_fused_feat)).item()
        assert cnt_nan == 0, 'Obj fused feat contains nan'
        
        # Fuse environment end fused object feature
        stacked_feat = torch.stack([env_feat, obj_fused_feat], dim=2)
        batch_size, frames, hidden_dim = env_feat.size()        # Not stacked feature
        project_feat = torch.zeros(batch_size, frames, hidden_dim).cuda()

        for begin in range(0, frames):
            end = begin+1

            fuser_input = stacked_feat[:, begin:end].contiguous()       # batch x (hidden*2)
            fuser_input = fuser_input.view(-1, 2, hidden_dim)           # batch x 2 x hidden

            fuser_output = self._env_obj_fuser(fuser_input)             # batch x 2 x hidden
            
            cnt_nan = torch.sum(torch.isnan(fuser_output)).item()
            assert cnt_nan == 0, 'Obj env fused feat contains nan'
        
            fuser_output = torch.mean(fuser_output, dim=1)              # batch x hidden

            project_feat[:, begin:end] = fuser_output.view(batch_size, -1, hidden_dim)


        cnt_nan = torch.sum(torch.isnan(project_feat)).item()
        assert cnt_nan == 0, 'Projected feat contains nan'

        return project_feat



def step(optimizer, scaler, loss, lr_scheduler=None, backward_only=False):
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    if not backward_only:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        if lr_scheduler is not None:
            lr_scheduler.step()
        optimizer.zero_grad()

def process_prediction(pred, predD):
    pred = torch.softmax(pred, axis=2)
    aux_pred = torch.zeros_like(pred)
    for b in range(pred.shape[0]):
        for t in range(pred.shape[1]):
            displ = predD[b, t].round().int()

            aux_pred[b, max(0, min(pred.shape[1]-1, t - displ))] = torch.maximum(aux_pred[b, max(0, min(pred.shape[1]-1, t - displ))], pred[b, t])
    return aux_pred

def process_double_head(pred, predD, num_classes = 1):

    pred1 = torch.softmax(pred[:, :, :num_classes], axis=2) #preds 1st head
    aux_pred = torch.zeros_like(pred1)

    for b in range(pred1.shape[0]):
        for t in range(pred1.shape[1]):
            displ = predD[b, t].round().int()
            aux_pred[b, max(0, min(pred1.shape[1]-1, t - displ))] = torch.maximum(aux_pred[b, max(0, min(pred1.shape[1]-1, t - displ))], pred1[b, t]) #maximum aggregation

    return aux_pred

def process_labels(label, labelD, num_classes = 18):

    label_aux = torch.zeros((label.shape[0], label.shape[1], num_classes))
    label_aux[:, :, 0] = 1 #Background class
    events = label.nonzero()
    for i in range(events.shape[0]):
        if ((events[i, 1] - int(labelD[events[i, 0], events[i, 1]])) < label.shape[1]) & ((events[i, 1] - int(labelD[events[i, 0], events[i, 1]])) >= 0):
            label_aux[events[i, 0], events[i, 1] - int(labelD[events[i, 0], events[i, 1]]), label[events[i, 0], events[i, 1]]] = 1
            label_aux[events[i, 0], events[i, 1] - int(labelD[events[i, 0], events[i, 1]]), 0] = 0
    
    return label_aux