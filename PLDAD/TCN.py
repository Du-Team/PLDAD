import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
class CausalConv1d(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1, groups=1, bias=True):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out

class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()

        self.dw = CausalConv1d(channels, kernel_size, dilation=dilation, groups=channels, bias=False)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  
        residual = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual

class TCN(nn.Module):
    def __init__(self, fea_size, dim, seq_len, device, patch_size=16, patch_stride=8, stem_ratio=6, downsample_ratio=2,
                 ffn_ratio=2,
                 num_blocks=[1, 1], large_size=[31, 29],
                 small_size=[5, 5],
                 small_kernel_merged=False, backbone_dropout=0.1, use_multi_scale=True, revin=True, affine=True,
                 subtract_last=False):
        super(TCN, self).__init__()

        self.seq_len = seq_len
        self.device = device
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(fea_size, affine=affine, subtract_last=subtract_last)

        self.num_layers = max(1, sum(num_blocks))
        kernel = large_size[0] if isinstance(large_size, (list, tuple)) else int(large_size)
        kernel = max(2, int(kernel))

        channels = fea_size  
        layers = []
        for i in range(self.num_layers):
            dilation = 2 ** i
            layers.append(TemporalBlock(channels, kernel_size=kernel, dilation=dilation, dropout=backbone_dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):  
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)

        x = x.permute(0, 2, 1).contiguous()  
        x = self.network(x)
        x = x.permute(0, 2, 1).contiguous()  

        if self.revin:
            x = self.revin_layer(x.permute(0, 2, 1), 'denorm').permute(0, 2, 1)
        return x

    def structural_reparam(self):
        return None
