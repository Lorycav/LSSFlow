import config
import torch
import math
from collections import OrderedDict
import numpy as np
from contextlib import contextmanager

N_SHELL_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
N_SHELL_MAX = 64

# Discrete Haar Wavelet Transform for 3D fields
def DWT(field: torch.Tensor) -> torch.Tensor:
 
    f_low = (field[:, :, 0::2, :, :] + field[:, :, 1::2, :, :]) / 2.0
    f_high = (field[:, :, 0::2, :, :] - field[:, :, 1::2, :, :]) / 2.0

    ll = (f_low[:, :, :, 0::2, :] + f_low[:, :, :, 1::2, :]) / 2.0
    lh = (f_low[:, :, :, 0::2, :] - f_low[:, :, :, 1::2, :]) / 2.0
    hl = (f_high[:, :, :, 0::2, :] + f_high[:, :, :, 1::2, :]) / 2.0
    hh = (f_high[:, :, :, 0::2, :] - f_high[:, :, :, 1::2, :]) / 2.0

    lll = (ll[:, :, :, :, 0::2] + ll[:, :, :, :, 1::2]) / 2.0
    llh = (ll[:, :, :, :, 0::2] - ll[:, :, :, :, 1::2]) / 2.0
    lhl = (lh[:, :, :, :, 0::2] + lh[:, :, :, :, 1::2]) / 2.0
    lhh = (lh[:, :, :, :, 0::2] - lh[:, :, :, :, 1::2]) / 2.0
    hll = (hl[:, :, :, :, 0::2] + hl[:, :, :, :, 1::2]) / 2.0     
    hlh = (hl[:, :, :, :, 0::2] - hl[:, :, :, :, 1::2]) / 2.0
    hhl = (hh[:, :, :, :, 0::2] + hh[:, :, :, :, 1::2]) / 2.0
    hhh = (hh[:, :, :, :, 0::2] - hh[:, :, :, :, 1::2]) / 2.0

    return torch.cat([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], dim = 1)   

# Inverse Discrete Haar Wavelet Transform for 3D fields
def IDWT(input: torch.Tensor) -> torch.Tensor:

    lll, llh, lhl, lhh, hll, hlh, hhl, hhh = [input[:, i:i+1] for i in range(8)]
    down_size = lll.shape[2]
    up_size = down_size * 2
    batches = lll.shape[0]
    device = input.device

    def up_z(low_z, high_z):

        upped = torch.zeros(batches, 1, low_z.shape[2], low_z.shape[3], up_size, device = device, dtype = input.dtype)
        upped[:, :, :, :, 0::2] = low_z + high_z
        upped[:, :, :, :, 1::2] = low_z - high_z
        return upped
    
    def up_y(low_y, high_y):

        upped = torch.zeros(batches, 1, low_y.shape[2], up_size, low_y.shape[4], device = device, dtype = input.dtype)
        upped[:, :, :, 0::2, :] = low_y + high_y
        upped[:, :, :, 1::2, :] = low_y - high_y
        return upped
    
    def up_x(low_x, high_x):

        upped = torch.zeros(batches, 1, up_size, low_x.shape[3], low_x.shape[4], device = device, dtype = input.dtype)
        upped[:, :, 0::2, :, :] = low_x + high_x
        upped[:, :, 1::2, :, :] = low_x - high_x
        return upped
    
    ll = up_z(lll, llh)
    lh = up_z(lhl, lhh)
    hl = up_z(hll, hlh)
    hh = up_z(hhl, hhh)

    l = up_y(ll, lh)
    h = up_y(hl, hh)

    return up_x(l, h)

def pdf_edges(lo=-3.0, hi=3.0, n_bins=240):
    return np.linspace(lo, hi, n_bins + 1)

def pdf_1pt(field, edges, eps=1e-3):
    x = np.asarray(field, dtype=np.float64).ravel()
    frac = float((x < -1.0 + eps).mean())
    x = np.log10(np.clip(x, -1.0 + eps, None) + 1.0)
    h, _ = np.histogram(x, bins=edges, density=True)
    return h, frac

# Asinh-spaced bin edges for the normalized DWT coefficients
def subband_edges(lim=100.0, n_bins=200):
    a_max = np.arcsinh(lim)
    return np.sinh(np.linspace(-a_max, a_max, n_bins + 1))

# Histograms of the 8 DWT subbands
def subband_pdfs(z, edges):
    if z.ndim == 4:
        z = z[None]
    pdfs = np.zeros((z.shape[0], 8, len(edges) - 1))
    for i in range(z.shape[0]):
        for b in range(8):
            pdfs[i, b], _ = np.histogram(z[i, b], bins=edges, density=True)
    return pdfs

# Position of a simulation inside a dataset split
def position_of_sim(dataset, sim_id):
    indices = list(dataset.indices)
    if sim_id not in indices:
        raise ValueError(f"Simulation {sim_id} is not in this split")
    return indices.index(sim_id)

# From normalized DWT coefficients back to the density contrast
def to_density(z, transform_dwt_h, data_transform):
    return data_transform.inverse_torch(IDWT(transform_dwt_h.inverse(z)))

# Member-by-member ratio of the generated power spectra
def paired_ratio(d, d_fid):
    ratio = d["P_gen_all"] / d_fid["P_gen_all"]
    return ratio.mean(0), ratio.std(0, ddof=1) / np.sqrt(len(ratio))

# Integer wavenumbers on the rfft grid, and their norm
def kgrid(dims, device='cpu', dtype=torch.float32):
    kx = torch.fft.fftfreq(dims, d=1.0 / dims, device=device, dtype=dtype)
    kz = torch.fft.rfftfreq(dims, d=1.0 / dims, device=device, dtype=dtype)
    KX, KY, KZ = torch.meshgrid(kx, kx, kz, indexing="ij")
    return KX, KY, KZ, torch.sqrt(KX**2 + KY**2 + KZ**2)

# K mask for noise application
def high_band_mask(n_hr=128, n_lr=64, device='cpu'):
    KX, KY, KZ, k_norm = kgrid(n_hr, device)
    nyq_lr, nyq_hr = n_lr // 2, n_hr // 2
    inside_lr = (KX.abs() <= nyq_lr) & (KY.abs() <= nyq_lr) & (KZ.abs() <= nyq_lr)
    return (~inside_lr) & (k_norm < nyq_hr)

# Noise Application
def high_band_noise(mask, sigma, shape=(128,128,128), device='cpu', dtype=torch.float32):
    eta = torch.randn(shape, device=device, dtype=dtype)
    E   = torch.fft.rfftn(eta) * mask
    n   = torch.fft.irfftn(E, s=shape)
    return n * (sigma / n.std())

# Power Spectrum
def sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x == 0, torch.ones_like(x), torch.sin(x) / x)

def power_spectrum_3d_torch(field_1: torch.Tensor, field_2: torch.Tensor, BoxSize: float, MAS_index: int = 4, clump = 20) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = field_1.device
    try:
        if not (field_1.ndim == 3 and field_1.shape[0] == field_1.shape[1] == field_1.shape[2]):
            raise ValueError("Unexpected shape, field 1 needs to be a cubic 3D tensor")
        if not (field_2.ndim == 3 and field_2.shape[0] == field_2.shape[1] == field_2.shape[2]):
            raise ValueError("Unexpected shape, field 1 needs to be a cubic 3D tensor")
        if not (field_1.shape[0] == field_2.shape[0] and field_1.shape[1] == field_2.shape[1] and field_1.shape[2] == field_2.shape[2]):
            raise ValueError("Unexpected shapes, the two fields must have the same shape")


        dims = field_1.shape[0]
        middle = dims // 2

        # Fundamental frequency
        kF = 2.0 * math.pi / BoxSize

        # Max sampled frequency
        #kmax = int(math.sqrt(3 * middle * middle))
        kmax = middle - 1

        # FFT
        delta_k1 = torch.fft.rfftn(field_1)
        delta_k2 = torch.fft.rfftn(field_2)

        # Ks grid
        idx_128 = torch.arange(dims, device = device)
        idx_65 = torch.arange(dims // 2 + 1, device = device)
        kxx, kyy, kzz = torch.meshgrid(idx_128, idx_128, idx_65, indexing="ij")

        kx_s = torch.where(kxx > middle, kxx - dims, kxx)
        ky_s = torch.where(kyy > middle, kyy - dims, kyy)
        kz_s = torch.where(kzz > middle, kzz - dims, kzz)

        # Avoiding double counts
        kz_planes = (kz_s == 0) | ((kz_s == middle) & (dims % 2 == 0))
        kx_planes = (kx_s == 0) | ((kx_s == middle) & (dims % 2 == 0))
        k_sphere  = torch.sqrt(kx_s.float()**2 + ky_s.float()**2 + kz_s.float()**2)
        k_mask = (~kz_planes | (kx_s >= 0)) & (~(kz_planes & kx_planes) | (ky_s >= 0)) & (k_sphere < middle)  
        #k_mask = (~kz_planes | (kx_s >= 0)) & (~(kz_planes & kx_planes) | (ky_s >= 0))
        kx = kx_s[k_mask]
        ky = ky_s[k_mask]
        kz = kz_s[k_mask]

        k_int = torch.sqrt((kx.float() ** 2) + (ky.float() ** 2) + (kz.float() ** 2))
        k_index = torch.floor(k_int).to(torch.int64)

        Wx = sinc((math.pi / dims) * kx.float())
        Wy = sinc((math.pi / dims) * ky.float())
        Wz = sinc((math.pi / dims) * kz.float())

        # total window = (Wx * Wy * Wz) ** p
        W_total = (Wx * Wy * Wz) ** MAS_index

        # MAS correction factor approximately 1 / W_total. Avoid division by zero.
        eps = 1e-12
        MAS_factor = 1.0 / (W_total + eps)
        max_deconv = clump
        if clump != 0:
            MAS_factor = torch.clamp(MAS_factor, max=max_deconv)

        delta_k_s = delta_k1[k_mask]
        real_1 = delta_k_s.real * MAS_factor
        imag_1 = delta_k_s.imag * MAS_factor
        delta_2_1 = real_1 * real_1 + imag_1 * imag_1

        delta_k_s = delta_k2[k_mask]
        real_2 = delta_k_s.real * MAS_factor        
        imag_2 = delta_k_s.imag * MAS_factor
        delta_2_2 = real_2 * real_2 + imag_2 * imag_2

        delta_2_X = real_1 * real_2 + imag_1 * imag_2

        k = torch.zeros(kmax + 1, device=device)
        Pk0_s1 = torch.zeros(kmax + 1, device=device)
        Pk0_s2 = torch.zeros(kmax + 1, device=device)
        Pk0_sX = torch.zeros(kmax + 1, device=device)
        Nmodes = torch.zeros(kmax + 1, device=device)


        k = k.index_add(0, k_index, k_int)
        Pk0_s1 = Pk0_s1.index_add(0, k_index, delta_2_1)
        Pk0_s2 = Pk0_s2.index_add(0, k_index, delta_2_2)
        Pk0_sX = Pk0_sX.index_add(0, k_index, delta_2_X)
        Nmodes = Nmodes.index_add(0, k_index, torch.ones_like(delta_2_1))[1:]

        k_mean = torch.zeros_like(Nmodes)
        k_mean[Nmodes > 0] = (k[1:][Nmodes > 0] / Nmodes[Nmodes > 0]) * kF

        normalization = (BoxSize / (dims ** 2)) ** 3

        Pk0_1 = torch.zeros_like(Nmodes) 
        Pk0_2 = torch.zeros_like(Nmodes) 
        Pk0_X = torch.zeros_like(Nmodes) 


        Pk0_1[Nmodes > 0] = (Pk0_s1[1:][Nmodes > 0] / Nmodes[Nmodes > 0]) * normalization
        Pk0_2[Nmodes > 0] = (Pk0_s2[1:][Nmodes > 0] / Nmodes[Nmodes > 0]) * normalization
        Pk0_X[Nmodes > 0] = (Pk0_sX[1:][Nmodes > 0] / Nmodes[Nmodes > 0]) * normalization

    except Exception as e:
        print(f"Error computing the power spectrum")
        raise

    return (k_mean, Pk0_1, Pk0_2, Pk0_X)

def compute_pk(field1: torch.Tensor, field2: torch.Tensor, box_size: float = config.BOX_SIZE):
    f1 = torch.as_tensor(field1.squeeze()).float().cpu()
    f2 = torch.as_tensor(field2.squeeze()).float().cpu()
    k, pk_1, pk_2, pk_X = power_spectrum_3d_torch(f1, f2, box_size)

    return k, pk_1, pk_2, pk_X

# Power spectrum, cross-correlation and transfer function of a field against a reference field
def pk_r_T(field, ref, BoxSize, clump=20):
    k, P_f, P_ref, P_x = power_spectrum_3d_torch(field, ref, BoxSize, MAS_index=4, clump=clump)
    r = P_x / torch.sqrt(P_f.clamp_min(1e-30) * P_ref.clamp_min(1e-30))
    T = torch.sqrt(P_f.clamp_min(0) / P_ref.clamp_min(1e-30))
    return k.numpy(), P_f.numpy(), r.numpy(), T.numpy()

# Fourier shells of a field for the bispectrum estimator
class ShellCache:
    def __init__(self, field, BoxSize, dk=3.0, MAS_index=4, clump=20, device="cpu", dtype=torch.float64, maxsize=6):
        field = torch.as_tensor(field).to(device=device, dtype=dtype)
        if not (field.ndim == 3 and field.shape[0] == field.shape[1] == field.shape[2]):
            raise ValueError("Unexpected shape, the field needs to be a cubic 3D tensor")

        self.dims = field.shape[0]
        self.BoxSize = BoxSize
        self.dk = dk
        self.device = device
        self.maxsize = maxsize
        self.kF = 2.0 * math.pi / BoxSize

        # Ks grid
        KX, KY, KZ, self.k_norm = kgrid(self.dims, device, dtype)

        W_total = (sinc(math.pi * KX / self.dims) * sinc(math.pi * KY / self.dims) * sinc(math.pi * KZ / self.dims)) ** MAS_index
        MAS_factor = 1.0 / (W_total + 1e-12)
        if clump != 0:
            MAS_factor = torch.clamp(MAS_factor, max=clump)
        self.delta_k = torch.fft.rfftn(field) * MAS_factor
        self.delta_k[(self.k_norm == 0) | (self.k_norm >= self.dims // 2)] = 0

        self.I_cache = OrderedDict()

    # Shell of width dk around n
    def shell(self, n):
        return ((self.k_norm >= n - 0.5 * self.dk) & (self.k_norm < n + 0.5 * self.dk)).to(self.k_norm.dtype)

    # Field filtered on the shell n
    def I(self, n):
        n = round(n, 3)
        if n in self.I_cache:
            self.I_cache.move_to_end(n)
        else:
            self.I_cache[n] = torch.fft.irfftn(self.delta_k * self.shell(n), s=(self.dims,) * 3)
            if len(self.I_cache) > self.maxsize:
                self.I_cache.popitem(last=False)
        return self.I_cache[n]

    def N(self, n):
        key = (self.dims, self.dk, round(n, 3), str(self.device))
        if key in N_SHELL_CACHE:
            N_SHELL_CACHE.move_to_end(key)
        else:
            N_SHELL_CACHE[key] = torch.fft.irfftn(self.shell(round(n, 3)).to(self.delta_k.dtype), s=(self.dims,) * 3)
            if len(N_SHELL_CACHE) > N_SHELL_MAX:
                N_SHELL_CACHE.popitem(last=False)
        return N_SHELL_CACHE[key]

    # Shell-averaged power spectrum
    def P(self, n):
        return float((self.BoxSize / self.dims ** 2) ** 3 * (self.I(n) ** 2).sum() / (self.N(n) ** 2).sum())

# Bispectrum of a list of triangles (n1, n2, n3), in units of the fundamental frequency
def bispectrum(field=None, BoxSize=None, triangles=(), dk=3.0, MAS_index=4, clump=20, device="cpu", cache=None):
    if cache is None:
        cache = ShellCache(field, BoxSize, dk=dk, MAS_index=MAS_index, clump=clump, device=device)
    dims = cache.dims
    middle = dims // 2
    normalization = (cache.BoxSize ** 2 / dims ** 3) ** 3

    out = {"k1": [], "k2": [], "k3": [], "B": [], "Q": [], "mask": []}
    for n1, n2, n3 in triangles:
        inside = all(n - 0.5 * cache.dk >= 0.5 and n + 0.5 * cache.dk < middle for n in (n1, n2, n3))
        closes = abs(n1 - n2) - 1.5 * cache.dk <= n3 <= n1 + n2 + 1.5 * cache.dk
        if not (inside and closes):
            out["mask"].append(False)
            continue

        den = (cache.N(n1) * cache.N(n2) * cache.N(n3)).sum()
        if den * dims ** 6 < 1:
            out["mask"].append(False)
            continue

        num = (cache.I(n1) * cache.I(n2) * cache.I(n3)).sum()
        B = float(normalization * num / den)
        P1, P2, P3 = cache.P(n1), cache.P(n2), cache.P(n3)

        out["k1"].append(n1 * cache.kF)
        out["k2"].append(n2 * cache.kF)
        out["k3"].append(n3 * cache.kF)
        out["B"].append(B)
        out["Q"].append(B / (P1 * P2 + P2 * P3 + P3 * P1))
        out["mask"].append(True)

    return {key: np.array(values) for key, values in out.items()}

# Equilateral Configuration: n1 = n2 = n3
def equilateral(n_min=4, n_max=60, dk=3.0):
    ns = np.arange(n_min, n_max + 1e-9, dk)
    return [(float(n), float(n), float(n)) for n in ns]

# Angular scan: n1, n2 fixed, n3 varies with angle theta
def angle_scan(n1=16, n2=40, n_theta=17, dk=3.0):
    th = np.linspace(0.0, np.pi, n_theta)
    n3 = np.sqrt(n1 ** 2 + n2 ** 2 + 2.0 * n1 * n2 * np.cos(th))
    n3 = np.clip(n3, dk, None)
    return [(float(n1), float(n2), float(x)) for x in n3], th

# Squeezed Configuration
def squeezed(n_soft=6, n_min=8, n_max=56, dk=3.0):
    ns = np.arange(n_min, n_max + 1e-9, dk)
    return [(float(n), float(n), float(n_soft)) for n in ns]

# Conditional power spectrum
def conditional_pk(members, clump=20):
    mean = members.mean(0)
    P = [power_spectrum_3d_torch(g - mean, g - mean, config.BOX_SIZE, MAS_index=4, clump=clump)[1].numpy() for g in members]
    return np.mean(P, axis=0) * len(members) / (len(members) - 1)

# Bispectra of one field for several configurations
def bispectra_of(field, triangles, dk=3.0, clump=20, device="cpu"):
    cache = ShellCache(field, config.BOX_SIZE, dk=dk, MAS_index=4, clump=clump, device=device, maxsize=8)
    out = {name: bispectrum(triangles=tri, cache=cache) for name, tri in triangles.items()}
    del cache
    torch.cuda.empty_cache()
    return out

@contextmanager
def source_noise(dataset, scale):
    old = dataset.noise_scale
    dataset.noise_scale = float(scale)
    try:
        yield dataset
    finally:
        dataset.noise_scale = old
