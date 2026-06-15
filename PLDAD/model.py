import time
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from utils import *
from PLDAD.earlyStopping import EarlyStopping
from types import SimpleNamespace
from PLDAD.GAT import *
from PLDAD.TCN import *
from PLDAD.diffusion import *

default_device = get_default_device()

def resolve_torch_device(opt=None):
    if opt is None or not hasattr(opt, 'device'):
        return default_device

    dev = opt.device
    if isinstance(dev, torch.device):
        return dev

    return torch.device(dev)

class Attention_layer(nn.Module):
    def __init__(self, n_feature, num_heads, hid_dim, dropout=0.1):
        super(Attention_layer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_feature,
            nhead=num_heads,
            dim_feedforward=hid_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, inputs, attn_mask=None):

        output = self.encoder(inputs, mask=attn_mask)
        return output, None

class FunDiffFourierEmbedding(nn.Module):

    def __init__(self, embed_dim, scale=16.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale

        self.register_buffer('kernel', torch.randn(1, embed_dim // 2) * scale)

    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(-1)

        proj = x @ self.kernel

        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

class FunDiffCrossAttnBlock(nn.Module):

    def __init__(self, dim, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()

        self.norm_q = nn.LayerNorm(dim, eps=1e-5)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-5)

        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

        self.norm_mlp = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x_q, x_kv):

        q = self.norm_q(x_q)
        k = v = self.norm_kv(x_kv)

        attn_out, _ = self.attn(q, k, v)
        x = x_q + attn_out

        x = x + self.mlp(self.norm_mlp(x))
        return x

class ContinuousDecoder(nn.Module):

    def __init__(self, latent_dim, output_dim, hidden_dim=128, depth=2, num_heads=4):
        super().__init__()

        self.coord_embed = FunDiffFourierEmbedding(hidden_dim)

        self.z_proj = nn.Linear(latent_dim, hidden_dim)

        self.layers = nn.ModuleList([
            FunDiffCrossAttnBlock(hidden_dim, num_heads=num_heads)
            for _ in range(depth)
        ])

        self.norm_final = nn.LayerNorm(hidden_dim, eps=1e-5)

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z, coords):

        x = self.coord_embed(coords)  

        if z.dim() == 2:
            z = z.unsqueeze(1)  
        keys = self.z_proj(z)  

        for layer in self.layers:

            x = layer(x_q=x, x_kv=keys)

        x = self.norm_final(x)
        return self.final_mlp(x)

class SCAD(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = resolve_torch_device(opt)
        self.num_levels = 1

        self.mask_ratio = getattr(opt, 'mask_ratio', 0.5)      
        self.jitter_scale = getattr(opt, 'jitter_scale', 0.8)  

        self.ver_module = GAT(num_nodes=opt.dim, seq_len=opt.window_size, num_levels=self.num_levels,
                              device=self.device)

        self.hor_module = TCN(fea_size=opt.dim, dim=opt.window_size, seq_len=opt.window_size,
                              device=self.device, patch_size=4, patch_stride=2, 
                              num_blocks=[3], large_size=[3], small_size=[3], 
                              revin=True, affine=True, subtract_last=False)

        self.ver_relu = nn.ReLU()
        self.hor_relu = nn.ReLU()
        self.ver_atte = Attention_layer(n_feature=opt.window_size, num_heads=2, hid_dim=opt.window_size,
                                        dropout=opt.drop_out)
        self.hor_atte = Attention_layer(n_feature=opt.window_size, num_heads=2, hid_dim=opt.window_size,
                                        dropout=opt.drop_out)
        self.final_out_channels = opt.window_size
        self.features_len = opt.dim
        self.project_channels = opt.project_channels

        self.fusion_attention = Attention_layer(
            n_feature=opt.dim * 2,
            num_heads=2,
            hid_dim=opt.dim * 8,
            dropout=opt.drop_out
        )
        self.fusion_linear = nn.Linear(
            opt.window_size * opt.dim * 2,
            self.project_channels
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.project_channels, self.project_channels // 2),
            nn.BatchNorm1d(self.project_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.project_channels // 2, self.project_channels),
        )

        self.cond_proj = nn.Linear(self.project_channels, self.final_out_channels * self.features_len)

        self.dropout = nn.Dropout(opt.drop_out)
        self.sigmoid = nn.Sigmoid()

        self.continuous_decoder = ContinuousDecoder(
            latent_dim=self.project_channels,
            output_dim=opt.dim,
            hidden_dim=128,
            depth=2,
            num_heads=4
        )
        self.train_stage = getattr(opt, 'train_stage', 'jft')
        self.diffusion_loss_type = getattr(opt, 'diffusion_loss_type', 'x0')
        self.jft_diff_weight = float(getattr(opt, 'jft_diff_weight', 1.0))

        self.criterion = nn.MSELoss()

        diff_cfg = SimpleNamespace(
            train=SimpleNamespace(
                device=self.device,
                epochs=opt.epochs,        
                patience=opt.patience,    
                lr=opt.lr_jft
            ),
            diff=SimpleNamespace(
                timesteps=opt.diff_timesteps,
                beta_schedule=opt.diff_beta_schedule,
                beta_start=opt.diff_beta_start,
                beta_end=opt.diff_beta_end,
                noise_type=opt.diff_noise_type,
                n_copies_to_test=opt.n_copies_to_test
            ),
            data=SimpleNamespace(
                pred_len=1,
                feature_dim=self.project_channels,
                test_batch_size=opt.test_batch_size,
                seq_len=1
            ),
            model=SimpleNamespace(
                hidden_dim=max(16, opt.diff_hidden_dim),
                n_heads=opt.diff_n_heads,
                attn_dropout=opt.diff_attn_dropout,
                mlp_ratio=opt.diff_mlp_ratio,
                n_depth=opt.diff_n_depth,
                n_emb=opt.diff_n_emb
            ),
            use_cond=False,  
        )

        self.diffusion = nonlinear_conditional_ddpm(diff_cfg)

    def _encode_to_z(self, x):
        batch_size = x.size(0)
        vertical_outputs = self.ver_module(x)
        vertical_outputs = self.ver_relu(vertical_outputs)
        vertical_outputs = vertical_outputs.transpose(1, 2)
        vertical_outputs, _ = self.ver_atte(vertical_outputs)
        vertical_outputs = vertical_outputs.transpose(1, 2)

        horizontal_outputs = self.hor_module(x)
        horizontal_outputs = self.hor_relu(horizontal_outputs)
        horizontal_outputs = horizontal_outputs.transpose(1, 2)
        horizontal_outputs, _ = self.hor_atte(horizontal_outputs)
        horizontal_outputs = horizontal_outputs.transpose(1, 2)

        fused_input = torch.cat([vertical_outputs, horizontal_outputs], dim=2)  
        fused_output, _ = self.fusion_attention(fused_input)  
        fused_output_flat = fused_output.reshape(batch_size, -1)
        hid_var = self.fusion_linear(fused_output_flat)
        hid_var = hid_var.unsqueeze(1)
        enc_vec = self.projection_head(hid_var.squeeze(1))
        return enc_vec

    def _decode_from_z(self, z, win_size, fea_size):

        batch_size = z.size(0)
        coords = torch.linspace(0, 1, steps=win_size, device=self.device)
        coords = coords.view(1, win_size, 1).expand(batch_size, -1, -1)

        if self.training:

            step_size = 1.0 / (win_size - 1) if win_size > 1 else 1.0
            jitter = (torch.rand_like(coords) - 0.5) * step_size * self.jitter_scale
            coords = coords + jitter
            coords = torch.clamp(coords, 0.0, 1.0)  

        x_recon = self.continuous_decoder(z, coords)
        self._last_coords = coords  
        return x_recon

    def training_step(self, x):
        batch_size = x.size(0)
        win_size = x.size(1)
        fea_size = x.size(2)

        if self.train_stage == 'mfae':

            mask = torch.rand(batch_size, win_size, fea_size, device=self.device) > self.mask_ratio

            x_masked = x * mask.float()

            z = self._encode_to_z(x_masked)

            x_recon = self._decode_from_z(z, win_size, fea_size)

            loss_rec = F.mse_loss(x_recon, x)

            return loss_rec

        elif self.train_stage == 'lpl':
            with torch.no_grad():
                z = self._encode_to_z(x).detach()  
            z_seq = z.unsqueeze(1)

            n = z_seq.size(0)
            t = torch.randint(low=1, high=self.diffusion.num_timesteps, size=(n // 2 + 1,), device=self.device)
            t = torch.cat([t, self.diffusion.num_timesteps - t], dim=0)[:n]
            y_t, actual_noise = self.diffusion.q_sample(z_seq, None, t)
            pred_y0 = self.diffusion(None, y_t, t, None)

            if self.diffusion_loss_type == 'noise':
                sqrt_alpha_bar_t = extract(self.diffusion.alphas_bar_sqrt, t, y_t)
                sqrt_one_minus_alpha_bar_t = extract(self.diffusion.one_minus_alphas_bar_sqrt, t, y_t)
                pred_noise = (y_t - sqrt_alpha_bar_t * pred_y0) / (sqrt_one_minus_alpha_bar_t + 1e-8)
                loss_diff = F.mse_loss(pred_noise, actual_noise)
            else:  
                loss_diff = F.mse_loss(pred_y0, z_seq)

            return loss_diff

        else:  

            z = self._encode_to_z(x)
            z_seq = z.unsqueeze(1)

            n = z_seq.size(0)
            t = torch.randint(low=1, high=self.diffusion.num_timesteps, size=(n // 2 + 1,), device=self.device)
            t = torch.cat([t, self.diffusion.num_timesteps - t], dim=0)[:n]
            y_t, actual_noise = self.diffusion.q_sample(z_seq, None, t)

            pred_y0 = self.diffusion(None, y_t, t, None)

            if self.diffusion_loss_type == 'noise':
                sqrt_alpha_bar_t = extract(self.diffusion.alphas_bar_sqrt, t, y_t)
                sqrt_one_minus_alpha_bar_t = extract(self.diffusion.one_minus_alphas_bar_sqrt, t, y_t)
                pred_noise = (y_t - sqrt_alpha_bar_t * pred_y0) / (sqrt_one_minus_alpha_bar_t + 1e-8)
                loss_diff = F.mse_loss(pred_noise, actual_noise)
            else:  
                loss_diff = F.mse_loss(pred_y0, z_seq)

            z_hat = pred_y0.squeeze(1) 
            x_recon = self._decode_from_z(z_hat, win_size, fea_size)
            loss_rec = F.mse_loss(x_recon, x)

            total_loss = loss_rec + self.jft_diff_weight * loss_diff

            return total_loss

    def validation_step(self, x):
        batch_size = x.size(0)
        win_size = x.size(1)
        fea_size = x.size(2)

        if self.train_stage == 'mfae':

            z = self._encode_to_z(x)
            x_recon = self._decode_from_z(z, win_size, fea_size)

            score = torch.mean(torch.mean((x - x_recon) ** 2, axis=2), axis=1)
            loss = torch.mean(score)
            return loss, score

        elif self.train_stage == 'lpl':

            z = self._encode_to_z(x)
            z_seq = z.unsqueeze(1)

            n = z_seq.size(0)
            t = torch.randint(low=1, high=self.diffusion.num_timesteps, size=(n // 2 + 1,), device=self.device)
            t = torch.cat([t, self.diffusion.num_timesteps - t], dim=0)[:n]
            y_t, actual_noise = self.diffusion.q_sample(z_seq, None, t)
            pred_y0 = self.diffusion(None, y_t, t, None)

            if self.diffusion_loss_type == 'noise':
                sqrt_alpha_bar_t = extract(self.diffusion.alphas_bar_sqrt, t, y_t)
                sqrt_one_minus_alpha_bar_t = extract(self.diffusion.one_minus_alphas_bar_sqrt, t, y_t)
                pred_noise = (y_t - sqrt_alpha_bar_t * pred_y0) / (sqrt_one_minus_alpha_bar_t + 1e-8)
                loss = F.mse_loss(pred_noise, actual_noise)
                score = torch.mean(torch.mean((pred_noise - actual_noise) ** 2, dim=2), dim=1) 
            else:  
                loss = F.mse_loss(pred_y0, z_seq)
                score = torch.mean(torch.mean((pred_y0 - z_seq) ** 2, dim=2), dim=1)

            return loss, score

        else:  

            z = self._encode_to_z(x)
            z_seq = z.unsqueeze(1)

            n = z_seq.size(0)
            t = torch.randint(low=1, high=self.diffusion.num_timesteps, size=(n // 2 + 1,), device=self.device)
            t = torch.cat([t, self.diffusion.num_timesteps - t], dim=0)[:n]
            y_t, actual_noise = self.diffusion.q_sample(z_seq, None, t)

            pred_y0 = self.diffusion(None, y_t, t, None)
            z_hat = pred_y0.squeeze(1)

            x_recon = self._decode_from_z(z_hat, win_size, fea_size)

            score = torch.mean(torch.mean((x - x_recon) ** 2, axis=2), axis=1)
            loss = torch.mean(score)

            return loss, score

    def validation_epoch_end(self, outputs):
        batch_losses = [x for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return epoch_loss

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss: {:.4f}".format(epoch, result))

    def test_step(self, x):

        win_size = x.size(1)
        fea_size = x.size(2)

        z = self._encode_to_z(x)

        if self.train_stage == 'mfae':
            x_recon = self._decode_from_z(z, win_size, fea_size)
        else:

            z_seq = z.unsqueeze(1)
            n = z_seq.size(0)

            t = torch.randint(low=1, high=self.diffusion.num_timesteps, size=(n // 2 + 1,), device=self.device)
            t = torch.cat([t, self.diffusion.num_timesteps - t], dim=0)[:n]

            y_t, _ = self.diffusion.q_sample(z_seq, None, t)

            pred_y0 = self.diffusion(None, y_t, t, None)
            z_hat = pred_y0.squeeze(1)

            x_recon = self._decode_from_z(z_hat, win_size, fea_size)

        score = torch.mean(torch.mean((x - x_recon) ** 2, axis=2), axis=1)
        return score

def evaluate(model, val_loader, opt):
    outputs = []
    run_device = resolve_torch_device(opt)
    use_amp = run_device.type == 'cuda' and hasattr(torch.cuda, 'amp')
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device=run_device, dtype=torch.float32, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output, score = model.validation_step(data)
            outputs.append(output)

    return model.validation_epoch_end(outputs), score

def training(opt, model, train_loader, val_loader, model_path, opt_func=torch.optim.Adam):
    history = []
    run_device = resolve_torch_device(opt)

    train_stage = getattr(model, 'train_stage', 'jft')
    if train_stage == 'mfae':
        learning_rate = opt.lr_mfae
    elif train_stage == 'lpl':
        learning_rate = opt.lr_lpl
    else:  
        learning_rate = opt.lr_jft

    print(f"[V13] Stage: {train_stage}, Learning Rate: {learning_rate} (from options)")
    if train_stage == 'jft':
        print(f"[V15] JFT diffusion weight: {model.jft_diff_weight}")

    optimizer = opt_func(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    early_stopping = EarlyStopping(opt, patience=opt.patience, verbose=False, model_path=model_path)
    use_amp = run_device.type == 'cuda' and hasattr(torch.cuda, 'amp')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print('启用混合精度训练(AMP)以加速训练')

    cur_epoch = 0

    train_steps = len(train_loader)

    start_time = time.time()

    for epoch in range(opt.epochs):
        model.train()
        cur_epoch += 1

        epoch_start_time = time.time()
        epoch_train_time = []
        epoch_iter = 0

        model._batch_counter = 0
        for i, data in enumerate(train_loader):
            if data.size(0) == 1:
                continue

            optimizer.zero_grad(set_to_none=True)

            epoch_iter += 1
            data = data.to(device=run_device, dtype=torch.float32, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    loss = model.training_step(data)
            else:
                loss = model.training_step(data)

            if not torch.isfinite(loss):
                print(f"警告: Epoch {epoch}, Batch {i}: 检测到非有限损失值，跳过此批次")
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        epoch_train_time.append(time.time() - epoch_start_time)

        val_error, score = evaluate(model, val_loader, opt)
        model.epoch_end(epoch, val_error)
        early_stopping(val_error, model)

        print('epoch', epoch)

        if early_stopping.early_stop:
            print('train finished with early stopping')
            break

        history.append(val_error)

    if not early_stopping.early_stop:
        print('train finished with total epochs')
        torch.save(model.state_dict(), model_path)

    total_train_time = time.time() - start_time

    return total_train_time, history, np.mean(epoch_train_time)

def testing(model, test_loader, opt):
    results = []
    pred_time = []
    run_device = resolve_torch_device(opt)
    use_amp = run_device.type == 'cuda' and hasattr(torch.cuda, 'amp')

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            start_time = time.time()

            batch = batch.to(device=run_device, dtype=torch.float32, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                score = model.test_step(batch)

            results.append(score)

            pred_time.append(time.time() - start_time)

    return results, np.mean(pred_time)
