import config
import torch
import math
import torch.nn as nn
from helpers import DWT

# Group Normalization groups for a given number of channels
def gn_groups(num_channels, ch_per_group=16):
    g = max(1, num_channels // ch_per_group) 
    while num_channels % g != 0:
        g -= 1
    return g
 
# Time variable embedding
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t): # t: (B,) - one time value per sample in the batch

        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1: 
            emb = nn.functional.pad(emb, (0,1))
        return emb # (B, 128)

# Cosmological Parameters Embedding
class MLPEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim), # (B,5) -> (B,128)
            nn.SiLU(),                       # Nonlinearity
            nn.Linear(embed_dim, embed_dim)) # (B,128) -> (B,128)
        
    def forward(self, x): # x: (B, 5) - normalized cosmological parameters
        return self.mlp(x) # (B, 128)

# Adaptive Group Normalization
class AdaGN(nn.Module):
    def __init__(self, out_c, cond_dim):
        super().__init__()

        self.norm = nn.GroupNorm(gn_groups(out_c), out_c if out_c > 0 else 1, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * out_c)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond_emb):  # x: (B, out_c, D, H, W) - feature map, cond_emb: (B, 256): time + param conditioning
        normed = self.norm(x) # (B, out_c, D, H, W)
        
        # Projects conditioning to scale and shift parameters
        gamma, beta = self.proj(cond_emb).chunk(2, dim=1) # gamma, beta: (B, out_c) 

        # Reshapes for broadcasting over 3 spatial dimensions
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]

        # Applies adaptive scaling and shifting
        return (1 + gamma) * normed + beta
    
# Unconditioned 3D conv block
class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c, K=3, S=1, P=1, circular = False): # conv that preserves spatial dimensions
        super().__init__()
        if P!=0 and circular == True:
            self.conv = nn.Conv3d(in_c, out_c, K, S, P, bias=False, padding_mode='circular')
        else:
            self.conv = nn.Conv3d(in_c, out_c, K, S, P, bias=False)        
        self.norm = nn.GroupNorm(gn_groups(out_c), out_c if out_c > 0 else 1) 
        self.act = nn.SiLU()
    def forward(self, x): return self.act(self.norm(self.conv(x)))

# Conditioned 3D conv block
class ConvBlock3D_cond(nn.Module):
    def __init__(self, in_c, out_c, cond_dim, K=3, S=1, P=1, circular = False):
        super().__init__()
        if P!=0 and circular == True:
            self.conv = nn.Conv3d(in_c, out_c, K, S, P, bias=False, padding_mode='circular')
        else:
            self.conv = nn.Conv3d(in_c, out_c, K, S, P, bias=False)
        self.norm = AdaGN(out_c, cond_dim) # conditioning-dependent normalization
        self.act = nn.SiLU()
    def forward(self, x, cond_emb): return self.act(self.norm(self.conv(x), cond_emb))

# DWT Convolutional Block
class DWTConvBlock(nn.Module):
    def __init__(self, in_c, out_c, cond_dim):
        super().__init__()
        self.space_conv = ConvBlock3D(in_c, out_c, circular = False)
        self.low_conv = ConvBlock3D(in_c, out_c, circular = False)
        self.high_conv = ConvBlock3D(in_c * 7, out_c, circular = False)
        self.out_conv = ConvBlock3D_cond(out_c * 3, out_c, cond_dim, K=1, P = 0)
        

    def forward(self,x, cond_emb): # x: (B, channels, size, size, size)
        batches, channels, size, _, _ = x.shape 
        x_space = self.space_conv(x) # (B, out_c, size, size, size)

        x_dwt = DWT(x.view(batches * channels, 1, size, size, size)) # DWT output: (B*channels, 8, size/2, size/2, size/2)
    
        # Merges the 8 subbands into the channel dim
        x_dwt = x_dwt.view(batches, channels * 8, size // 2, size // 2, size // 2) # x_dwt: (B, channels*8, size/2, size/2, size/2) 

        # Selects LLL subband (index 0 in each group of 8)
        low_idx = list(range(0, channels * 8, 8))
        # Selects all 7 high-frequency subbands (indices 1-7 in each group of 8)
        high_idx = [i for i in range(channels * 8) if i%8!=0]

        low_dwt = self.low_conv(x_dwt[:, low_idx, :, :, :]) # (B, out_c, size/2, size/2, size/2)
        high_dwt = self.high_conv(x_dwt[:, high_idx, :, :, :]) # (B, out_c, size/2, size/2, size/2)

        # Trilinear upsamples back to original DWT resolution
        low_up = nn.functional.interpolate(low_dwt,  size=(size, size, size), mode='trilinear', align_corners=False)  # (B, out_c, size, size, size)
        high_up = nn.functional.interpolate(high_dwt, size=(size, size, size), mode='trilinear', align_corners=False) # (B, out_c, size, size, size)

        # Three Paths: General DWT features, smooth low-freq features, high-freq edge features
        fused = torch.cat([x_space, low_up, high_up], dim=1) # (B, out_c*3, size, size, size)
        
        return self.out_conv(fused, cond_emb) # (B, out_c, size, size, size)

# Skip gate (alpha_main: General Attention, alpha_hf: Small scales amplifier)
class DWT_SkipGate(nn.Module):
    def __init__(self, skip_c, gate_c, inter_c):
        super().__init__()
        self.dec_conv = nn.Conv3d(gate_c, inter_c, 1, bias = True)
        self.skip_conv = nn.Conv3d(skip_c, inter_c, 1, bias = False)

        # Groups = skip_c to not mix feature maps high-frequency content
        self.hf_compress = nn.Conv3d(skip_c * 14, skip_c,
                                     kernel_size=1, groups=skip_c, bias=False)
        with torch.no_grad():
            self.hf_compress.weight.zero_()
            #self.hf_compress.weight[:, 7:] = 1.0 / 7 
            self.hf_compress.weight[:, 1::2] = 1.0 / 7  

        self.hf_norm = nn.GroupNorm(gn_groups(skip_c), skip_c)

        self.hf_proj = nn.Conv3d(skip_c, inter_c, 1, bias=False)
        nn.init.zeros_(self.hf_proj.weight)

        self.gate = nn.Sequential(
            nn.Conv3d(inter_c, inter_c, 1, bias=True),
            nn.SiLU(),
            nn.Conv3d(inter_c, skip_c, 1, bias=True),
            nn.Sigmoid()
        )
        self.gate_h = nn.Sequential(
            nn.Conv3d(inter_c, inter_c // 2, 1, bias=True),
            nn.SiLU(),
            nn.Conv3d(inter_c // 2, skip_c, 1, bias=True),
            nn.Sigmoid()
        )
        

        self.norm = nn.GroupNorm(gn_groups(skip_c), skip_c)
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)
        nn.init.zeros_(self.gate_h[-2].weight)
        nn.init.zeros_(self.gate_h[-2].bias)


    def forward(self, x_skip, x_dec): # x_skip: (B, skip_c, S, S, S) - x_dec:  (B, gate_c, S/2, S/2, S/2)
        B, C, S, _, _ = x_skip.shape
        x_skip_reshaped = x_skip.view(B * C, 1, S, S, S) # Reshapes to (B*skip_c, 1, S, S, S) for DWT processing
        dwt_skip = DWT(x_skip_reshaped) # DWT output: (B*skip_c, 8, S/2, S/2, S/2)

        hf = dwt_skip[:, 1:, :, :, :].reshape(B, C*7, S//2, S//2, S//2) # Select high-frequency subbands
        hf = torch.cat([hf, hf.abs()], dim=2).reshape(B, C*14, S//2, S//2, S//2)
        skip_high = self.hf_norm(self.hf_compress(hf))
        #skip_high = self.hf_compress(skip_high) # Compress high-frequency info for C feature maps - (B, C, S/2, S/2, S/2)
        upped_high = nn.functional.interpolate(skip_high, size=(S, S, S), mode='trilinear', align_corners=False) # (B, C, S, S, S)

        # Decoder context for gates
        upped_dec = nn.functional.interpolate(self.dec_conv(x_dec), size=(S, S, S), mode='trilinear', align_corners=False)
        combined = nn.functional.silu(upped_dec + self.skip_conv(x_skip))
        
        alpha_main = self.gate(combined) # (B, skip_c, S, S, S), values in (0,1)

        hf_combined = nn.functional.silu(combined + self.hf_proj(upped_high)) # Combines with high-frequency content
        alpha_hf = self.gate_h(hf_combined) # Provides an additional boost where high-frequency content is detected

        gated = alpha_main * x_skip * (1.0 + alpha_hf)
        return self.norm(gated)
    
# Encoder Level
class DownBlock3D(nn.Module):
    def __init__(self, in_c, out_c, cond_dim, DWT_flag = False):
        super().__init__()
        self.DWT_flag = DWT_flag

        if DWT_flag == True:
            self.d1 = DWTConvBlock(in_c, out_c, cond_dim)
            self.d2 = DWTConvBlock(out_c, out_c, cond_dim)
        else:
            self.c1 = ConvBlock3D_cond(in_c, out_c, cond_dim)
            self.c2 = ConvBlock3D_cond(out_c, out_c, cond_dim)
        
        self.pool = nn.Conv3d(out_c, out_c, kernel_size=2, stride=2,
                              groups=out_c, bias=False)
        
        nn.init.constant_(self.pool.weight, 1/8)

    def forward(self, x, cond_emb = None): 

        if self.DWT_flag == True:
            skip = self.d2(self.d1(x, cond_emb), cond_emb)
        else:
            skip = self.c2(self.c1(x, cond_emb), cond_emb)

        return self.pool(skip), skip

# Decoder Level
class UpBlock3D(nn.Module):
    def __init__(self, in_c, skip_c, out_c, cond_dim, attention = False):
        super().__init__()
        self.attention = attention
        self.upconv = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        #self.upconv = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2)
        if attention == True:
            self.att_gate = DWT_SkipGate(skip_c, out_c, skip_c // 2)

        self.c1 = ConvBlock3D_cond(out_c + skip_c, out_c, cond_dim)
        self.c2 = ConvBlock3D_cond(out_c, out_c, cond_dim)
    def forward(self, x, skip, cond_emb):
        x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

        x = self.upconv(x)

        if self.attention == True:  
            skip = self.att_gate(skip, x)

        x = torch.cat([x, skip], dim=1)
        x = self.c1(x, cond_emb)

        return self.c2(x, cond_emb)

# Squeeze-Excite Bottleneck
class Bottleneck(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        mid = channels // 4

        self.norm = nn.GroupNorm(gn_groups(channels), channels)

        self.fc1 = nn.Linear(channels + cond_dim, mid)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(mid, channels)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, cond_emb): # x shape: (B, 204, 16, 16, 16)
        
        residual = x
        x = self.norm(x)

        gap = x.mean(dim=(2, 3, 4)) # (B, 204) global box descriptor

        descriptor = torch.cat([gap, cond_emb], dim=1) # (B, 460)

        gate = torch.sigmoid(self.fc2(self.act(self.fc1(descriptor))))

        # (B, 204) - global gating vector in (0,1) conditioned on the bottleneck features and the time+param embedding
        x = x * gate[:, :, None, None, None]

        return x + residual
    
class Attention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(64, channels if channels > 0 else 1), channels if channels > 0 else 1)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        batch, ch, d, h, w = x.shape
        residual = x
        x = self.norm(x)
        x = x.view(batch, ch, -1).permute(0, 2, 1)
        x, _ = self.attn(x, x, x)
        x = x.permute(0, 2, 1).view(batch, ch, d, h, w)
        return x + residual

    
class UNET3D(nn.Module):
    """
    Encoder:
      down_blocks.0: DWT  (B, 102, 64, 64, 64)  -> skip0 (B, 102, 64, 64, 64)
      down_blocks.1: conv  (B, 204, 32, 32, 32)  -> skip1 (B, 204, 32, 32, 32)
 
    Bottleneck: SE gate at (B, 204, 16, 16, 16)
 
    Decoder:
      up_blocks.0: plain (B, 102, 16, 16, 16)  + skip1 -> (B, 204, 32, 32, 32)
      up_blocks.1: DWT gate (B, 204, 32, 32, 32)  + skip0 -> (B, 204, 64, 64, 64)
    """
    def __init__(self, in_channels = 1, base_channels = 102, channel_mults = (1, 2),
                 time_embed_dim = 128, param_embed_dim = 128, num_params = config.N_PARAMS):
        super().__init__()
        self.time_embedding = SinusoidalEmbedding(time_embed_dim)
        self.param_embedding = MLPEmbedding(num_params, param_embed_dim)
        cond_embed_dim = time_embed_dim + param_embed_dim
        self.init_conv = nn.Conv3d(in_channels, base_channels, kernel_size = 3, padding = 1) # Keeps the same spatial dimensions, mixes subbands

        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            if mult in [1]:
                self.down_blocks.append(DownBlock3D(current_channels, out_ch, cond_embed_dim, True))
            else:
                self.down_blocks.append(DownBlock3D(current_channels, out_ch, cond_embed_dim, False))
            channels.append(out_ch)
            current_channels = out_ch
        
        self.bottleneck = Bottleneck(current_channels, cond_embed_dim)
        #bottleneck_projection_channels = current_channels * 2
        #self.bottleneck_conv1 = ConvBlock3D(current_channels, bottleneck_projection_channels)
        #self.bottleneck_conv2 = ConvBlock3D(bottleneck_projection_channels, current_channels)
        #self.attention = Attention(bottleneck_projection_channels)
        #self.cond_proj = nn.Linear(cond_embed_dim, bottleneck_projection_channels)
        #nn.init.zeros_(self.cond_proj.weight)
        #nn.init.zeros_(self.cond_proj.bias)

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            in_ch_up = base_channels * mult 
            skip_ch_up = channels[i+1]      
            out_ch_up = base_channels * channel_mults[i-1] if i > 0 else base_channels
            if mult in [1]:
                self.up_blocks.append(UpBlock3D(in_ch_up, skip_ch_up, out_ch_up, cond_embed_dim, True))
            else:
                self.up_blocks.append(UpBlock3D(in_ch_up, skip_ch_up, out_ch_up, cond_embed_dim, False))
            current_channels = out_ch_up

        self.refine = ConvBlock3D_cond(current_channels, current_channels, cond_embed_dim)
        self.high_refine = ConvBlock3D_cond(current_channels, current_channels, cond_embed_dim)


        self.pre_final_norm = nn.GroupNorm(gn_groups(current_channels), current_channels)

        self.final_conv_low = nn.Conv3d(current_channels, 1, kernel_size=1)
        self.final_conv_high = nn.Conv3d(current_channels, 7, kernel_size=3, padding=1)

        nn.init.normal_(self.final_conv_low.weight,  std=1e-4)
        nn.init.zeros_(self.final_conv_low.bias)
        nn.init.normal_(self.final_conv_high.weight, std=1e-4)
        nn.init.zeros_(self.final_conv_high.bias)


    def forward(self, x, t, params): # x_t: (B, 8, 64, 64, 64)
        time_emb = self.time_embedding(t) # t (B,) -> (B, 128): encodes flow position in [0,1] as sinusoidal frequencies
        param_emb = self.param_embedding(params) # params (B,5) -> (B, 128): nonlinear embedding of cosmological parameters
        cond_emb = torch.cat([time_emb, param_emb], dim=1) 
        x = self.init_conv(x) # (B, 102, 64, 64, 64) - initial mixing of DWT subbands to learn cross-subband correlations
        
        # Downsampling path with skip connections
        skips = []
        for block in self.down_blocks: 
            x, skip = block(x, cond_emb)   
            skips.append(skip)

        #x = self.bottleneck_conv1(x)
        #x = x + self.cond_proj(cond_emb)[:, :, None, None, None]
        #x = self.attention(x)
        #x = self.bottleneck_conv2(x)
        x = self.bottleneck(x, cond_emb) # (B, 204, 16, 16, 16) - global context gating conditioned on time and parameters
        
        # Upsampling path with skip connections and attention gates
        for block in self.up_blocks: x = block(x, skips.pop(), cond_emb)

        x = x + self.refine(x, cond_emb) # additional capacity for refinement 

        # Output heads
        x_normed = self.pre_final_norm(x)  
        x_hi = self.high_refine(x_normed, cond_emb) # additional capacity for refinement of high-frequency subbands
        with torch.amp.autocast('cuda', enabled=False):
            pred_low = self.final_conv_low(x_normed.float()) # (B, 1, 64, 64, 64) - LLL subband
            pred_high = self.final_conv_high(x_hi.float()) # (B, 7, 64, 64, 64) - LLH-HHH subbands

        return torch.cat([pred_low, pred_high], dim=1) 
