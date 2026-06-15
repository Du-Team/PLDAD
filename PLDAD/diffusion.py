import time
import torch.nn.functional as F
import os
import torch.nn as nn
import torch
import random
import numpy as np
import math
from random import random
from utils import *
from types import SimpleNamespace
from math import log, sqrt

class nonlinear_conditional_ddpm(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.device = config.train.device
        self.num_timesteps = config.diff.timesteps

        betas = make_beta_schedule(schedule=config.diff.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=config.diff.beta_start, end=config.diff.beta_end)

        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.to('cpu').cumprod(dim=0).to(self.device)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diff.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  

        self.config = config
        self.pred_len = config.data.pred_len
        self.noise_type = config.diff.noise_type

        self.diffusion_model = denoiser(config)
        if config.use_cond:
            self.condition_model = condition(config)

        if self.noise_type == "t_phi":
            print("Using T_phi")

            self.t_phi = T_phi(config)
            self.time_emb = StepEmbedding(config.data.feature_dim, freq_dim=256)

    def mu_t_phi(self, t, batch_y, cond_info=None):

        batch_y = batch_y
        out = self.t_phi(batch_y, self.time_emb(t).unsqueeze(1))

        return out

    def q_sample(self, batch_y, condition_info, t):

        sqrt_alpha_bar_t = extract(self.alphas_bar_sqrt, t, batch_y)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, batch_y)

        if self.noise_type == "t_phi":
            batch_y_trans = self.mu_t_phi(t=t, batch_y=batch_y)
            noise = torch.randn_like(batch_y)
            y_t = sqrt_alpha_bar_t * batch_y_trans + sqrt_one_minus_alpha_bar_t * noise

        else:
            noise = torch.randn_like(batch_y)
            y_t = sqrt_alpha_bar_t * batch_y + sqrt_one_minus_alpha_bar_t * noise

        if self.config.use_cond:
            y_t = y_t + (1 - sqrt_alpha_bar_t) * condition_info

        return y_t, noise

    def get_gammas(self, t, y_t):

        alpha_t = extract(self.alphas, t, y_t).squeeze(1).squeeze(1)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, y_t).squeeze(1).squeeze(1)
        sqrt_one_minus_alpha_bar_t_m_1 = extract(self.one_minus_alphas_bar_sqrt, t - 1, y_t).squeeze(1).squeeze(1)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()

        gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
        gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())
        gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
            sqrt_one_minus_alpha_bar_t.square())

        beta_t_hat = ((sqrt_one_minus_alpha_bar_t_m_1.square()) / (sqrt_one_minus_alpha_bar_t.square())) * (1 - alpha_t)

        return gamma_0.unsqueeze(1).unsqueeze(2), gamma_1.unsqueeze(1).unsqueeze(2), gamma_2.unsqueeze(1).unsqueeze(
            2), sqrt_alpha_bar_t.unsqueeze(1).unsqueeze(2), beta_t_hat.unsqueeze(1).unsqueeze(2)

    def get_prior(self, batch_y, cond_info=None):

        T = torch.tensor([self.num_timesteps - 1]).repeat(batch_y.shape[0]).to(self.device)
        batch_y_mean = self.mu_t_phi(t=T, batch_y=batch_y)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, T, batch_y)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()

        u = sqrt_alpha_bar_t * batch_y_mean

        if self.config.use_cond:
            u = u - (sqrt_alpha_bar_t) * cond_info

        return (1 / 2) * (torch.mean((u) ** 2, dim=(1, 2)))

    def p_sample_loop(self, batch_y, x):

        t = torch.tensor([self.num_timesteps - 1]).repeat(batch_y.shape[0]).to(self.device)
        z = torch.randn_like(batch_y)

        if self.config.use_cond:
            cond_info = self.condition_model(x)
            y_t = cond_info + z
        else:
            y_t = z
            cond_info = None

        for t in reversed(range(1, self.num_timesteps)):
            y_t, cond_info = self.p_sample(x, y_t, t, cond_info)

        z = self.p_sample_t_1to0(x, y_t, cond_info)
        return z

    def p_sample(self, x, y_t, t, cond_info=None):

        t = torch.tensor([t]).to(self.device)

        alpha_t = extract(self.alphas, t, y_t)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, y_t)
        sqrt_one_minus_alpha_bar_t_m_1 = extract(self.one_minus_alphas_bar_sqrt, t - 1, y_t)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()

        gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
        gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())
        gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
            sqrt_one_minus_alpha_bar_t.square())

        if self.config.use_cond:
            y_0_reparam = self.forward(x, y_t, t, cond_info).to(self.device).detach()
        else:
            y_0_reparam = self.forward(x, y_t, t).to(self.device).detach()

        if self.noise_type == "t_phi":
            z = torch.randn_like(y_0_reparam)
            t1 = ((gamma_1 * sqrt_alpha_bar_t) + gamma_0) * (self.mu_t_phi(batch_y=y_0_reparam, t=t - 1))
            t2 = (gamma_1 * sqrt_alpha_bar_t) * (self.mu_t_phi(batch_y=y_0_reparam, t=t))

            y_t_m_1_hat = (gamma_1 * y_t) - (t2 - t1)

        else:
            z = torch.randn_like(y_t)
            y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y_t

        if self.config.use_cond:
            y_t_m_1_hat = y_t_m_1_hat + gamma_2 * cond_info

        beta_t_hat = (sqrt_one_minus_alpha_bar_t_m_1.square()) / (sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)
        y_t_m_1 = y_t_m_1_hat.to(self.device) + beta_t_hat.sqrt().to(self.device) * z.to(self.device)

        if self.config.use_cond and x is not None:
            cond_info = self.condition_model(x)

        return y_t_m_1, cond_info

    def p_sample_t_1to0(self, x, y_t, cond_info):

        t = torch.tensor([0]).to(self.device)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, y_t)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()

        if self.config.use_cond:
            y_0_reparam = self.forward(x, y_t, t, cond_info).to(self.device).detach()
            y_0_reparam = y_0_reparam
        else:
            y_0_reparam = self.forward(x, y_t, t).to(self.device).detach()

        y_t_m_1 = y_0_reparam.to(self.device)

        return y_t_m_1

    def forward(self, x, y_t, t, cond_info=None):

        dec_out = self.diffusion_model(x, y_t, t, cond_info)

        return dec_out

    def p_sample_loop_with_cond(self, shape, cond_info):

        B = shape[0]
        t_tensor = torch.tensor([self.num_timesteps - 1]).repeat(B).to(self.device)
        z = torch.randn(shape, device=self.device)

        if self.config.use_cond:
            y_t = cond_info + z
        else:
            y_t = z

        for t in reversed(range(1, self.num_timesteps)):
            y_t, _ = self.p_sample(x=None, y_t=y_t, t=t, cond_info=cond_info)

        y_0 = self.p_sample_t_1to0(x=None, y_t=y_t, cond_info=cond_info)
        return y_0

class Decoder(nn.Module):

    def __init__(self, hidden_dim, d_model, pred_len, n_emb):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.mlp = nn.Sequential(
            DataEmbedding(d_model, d_model, n_emb - 1),
            nn.Linear(d_model, pred_len)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * d_model, bias=True)
        )

    def forward(self, x, k):

        shift, scale = self.adaLN_modulation(k).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.mlp(x)
        return x

class condition(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dec = nn.Linear(config.data.seq_len, config.data.pred_len)

    def forward(self, x):
        out = self.dec(x.permute(0, 2, 1)).permute(0, 2, 1)

        return out

class StepEmbedding(nn.Module):
    def __init__(self, hidden_dim, freq_dim=256):
        super(StepEmbedding, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def sinusoidal_embedding(k, freq_dim, max_period=1000):
        half_dim = freq_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(device=k.device)
        k_freqs = k[:, None].float() * freqs[None]
        k_emb = torch.cat([torch.cos(k_freqs), torch.sin(k_freqs)], dim=-1)
        return k_emb

    def forward(self, k):
        k_emb = self.sinusoidal_embedding(k, self.freq_dim)
        k_emb = self.mlp(k_emb)
        return k_emb

class MLPResidual(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super(MLPResidual, self).__init__()
        self.lin_emb = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid(),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout)
        )
        self.lin_res = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x_emb = self.lin_emb(x)
        x_res = self.lin_res(x)
        x_out = self.norm(x_emb + x_res)
        return x_out

class DataEmbedding(nn.Module):

    def __init__(self, in_dim, out_dim, n_emb):
        super(DataEmbedding, self).__init__()
        self.feat_embedding = [MLPResidual(in_dim, out_dim)]
        if n_emb > 1:
            for i in range(n_emb - 1):
                self.feat_embedding.append(MLPResidual(out_dim, out_dim))
        self.feat_embedding = nn.Sequential(*self.feat_embedding)

    def forward(self, x):
        return self.feat_embedding(x)

class T_phi(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(config.data.feature_dim, config.data.feature_dim))
        self.b1 = nn.Parameter(torch.empty(config.data.feature_dim))

        self.w2 = nn.Parameter(torch.empty(config.data.pred_len, config.data.pred_len))
        self.b2 = nn.Parameter(torch.empty(config.data.pred_len))

        self.act = nn.Tanh()

        self.init_weights(self.w1, self.b1)

    def init_weights(self, weight, bias):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

    def forward(self, x, t_emb):
        out = x + t_emb

        out = (out.permute(0, 2, 1) @ self.w2.T) + self.b2
        out = out.permute(0, 2, 1)

        out = (out @ self.w1.T) + self.b1
        out = self.act(out)

        return out

class FullAttention(nn.Module):

    def __init__(self, d_model, n_heads, attn_dropout, d_key=None, d_value=None):
        super(FullAttention, self).__init__()
        d_key = d_key or (d_model // n_heads)
        d_value = d_value or (d_model // n_heads)
        self.n_heads = n_heads

        self.WQ = nn.Linear(d_model, d_key * n_heads, bias=False)
        self.WK = nn.Linear(d_model, d_key * n_heads, bias=False)
        self.WV = nn.Linear(d_model, d_value * n_heads, bias=False)
        self.WO = nn.Linear(d_value * n_heads, d_model)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value):
        B, l_query, d_query = query.shape
        _, l_key, _ = key.shape

        Q = self.WQ(query).view(B, l_query, self.n_heads, -1)
        K = self.WK(key).view(B, l_key, self.n_heads, -1)
        V = self.WV(value).view(B, l_key, self.n_heads, -1)

        scale = 1. / sqrt(Q.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", Q, K)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        agg = torch.einsum("bhls,bshd->blhd", A, V).contiguous()
        O = self.WO(agg.view(B, l_query, -1))
        return O

class AttnMLP(nn.Module):

    def __init__(
            self,
            in_dim,
            hidden_dim=None,
            out_dim=None,
            norm_layer=None,
            bias=True,
            drop=0.
    ):
        super(AttnMLP, self).__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias)
        self.act = nn.Sigmoid()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_dim) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class DiTBlock(nn.Module):

    def __init__(self, hidden_dim, d_model, n_heads, attn_dropout, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = FullAttention(d_model=d_model, n_heads=n_heads, attn_dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = AttnMLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, drop=0.1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * d_model, bias=True)
        )

    def forward(self, x, c):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_mod, x_mod, x_mod)
        x_mod = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod)
        return x

class denoiser(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.y_embedder = DataEmbedding(config.data.pred_len, config.model.hidden_dim, config.model.n_emb)
        self.k_embedder = StepEmbedding(config.model.hidden_dim, freq_dim=256)

        d_model = config.model.hidden_dim * 2
        self.blocks = nn.ModuleList([
            DiTBlock(config.model.hidden_dim, d_model, config.model.n_heads, config.model.attn_dropout,
                     config.model.mlp_ratio)
            for _ in range(config.model.n_depth)])
        self.decoder = Decoder(config.model.hidden_dim, d_model, config.data.pred_len, config.model.n_emb)
        self.act = nn.Identity()
        self.initialize_weights()
        self.config = config
        if config.use_cond:
            self.cond_embedder = DataEmbedding(config.data.pred_len, config.model.hidden_dim, config.model.n_emb)

    def initialize_weights(self):

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.decoder.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.adaLN_modulation[-1].bias, 0)

    def forward(self, x, y, k, cond_info):

        y = self.y_embedder(y.permute(0, 2, 1))

        c = self.k_embedder(k)

        if self.config.use_cond:
            cond_info = self.cond_embedder(cond_info.permute(0, 2, 1))
            h = torch.cat([y, cond_info], dim=-1)
        else:
            zeros = torch.zeros_like(y)  
            h = torch.cat([y, zeros], dim=-1)

        for block in self.blocks:
            h = block(h, c)

        out = self.decoder(h, c).permute(0, 2, 1)
        out = self.act(out)

        return out

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(num_timesteps)])
        if schedule == "cosine_reverse":
            betas = betas.flip(0)
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    return betas
