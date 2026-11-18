# %% [code]
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from helpers import sinc
import math

# Configuration
BASE_PATH = Path("../../../../PCS/")
HR_FOLDER_TEMPLATE = BASE_PATH / "df_m_{}_128_z=0.npy"
LR_FOLDER_TEMPLATE = BASE_PATH / "df_m_{}_64_r_z=0.npy"
LR_UP_TEMPLATE = BASE_PATH / "fourier3/data/df_m_{}_64_z=0.npy"
HR_UP_TEMPLATE = BASE_PATH / "fourier3/data/df_m_{}_128_z=0.npy"

NUM_SIMULATIONS = 2000
GRID_SIZE_HR = 128
GRID_SIZE_LR = 64

CLAMP = 4.0
POCS_EPS = 1e-7
POCS_MAX_ITER = 100
POCS_MIN_ITER = 3

# Fourier upsampling of the LR field, followed by alternating projections (POCS) to keep delta >= -1 without changing the LR modes
def fourier_upsample(lr, up_size=GRID_SIZE_HR, clamp=CLAMP, eps=POCS_EPS, max_iter=POCS_MAX_ITER, min_iter=POCS_MIN_ITER):
    device = lr.device
    down_size = lr.shape[0]

    # Moves to Fourier Space
    f = torch.fft.fftn(lr)

    # Window correction: from the PCS window of the LR grid to the one of the HR grid
    kk = torch.fft.fftfreq(down_size, d=1.0 / down_size, device=device)
    kx, ky, kz = torch.meshgrid(kk, kk, kk, indexing="ij")
    def W(N):
        return (sinc(math.pi * kx / N) * sinc(math.pi * ky / N) * sinc(math.pi * kz / N)) ** 4
    f = f * torch.clamp(W(up_size) / W(down_size), max=clamp)

    # Zero Padding
    f_shifted = torch.fft.fftshift(f)
    pad_size = (up_size - down_size) // 2 # cells to pad on each side
    f_padded = torch.zeros((up_size, up_size, up_size), dtype=torch.complex64, device=device)
    f_padded[pad_size:pad_size + down_size, pad_size:pad_size + down_size, pad_size:pad_size + down_size] = f_shifted

    # Nyquist Splitting
    nyq = pad_size
    nyq_conj = pad_size + down_size

    half = f_padded[nyq, :, :] / 2
    f_padded[nyq, :, :] = half
    f_padded[nyq_conj, :, :] = half

    half = f_padded[:, nyq, :] / 2
    f_padded[:, nyq, :] = half
    f_padded[:, nyq_conj, :] = half

    half = f_padded[:, :, nyq] / 2
    f_padded[:, :, nyq] = half
    f_padded[:, :, nyq_conj] = half

    scale = (up_size / down_size) ** 3
    mask_shift = torch.zeros((up_size, up_size, up_size), dtype=torch.bool, device=device)
    mask_shift[pad_size:pad_size + down_size + 1, pad_size:pad_size + down_size + 1, pad_size:pad_size + down_size + 1] = True
    mask = torch.fft.ifftshift(mask_shift)
    F_target = torch.fft.ifftshift(f_padded) * scale

    # Back to real space
    x = torch.fft.ifftn(F_target).real

    # Modes inside the HR Nyquist sphere
    kk_up = torch.fft.fftfreq(up_size, d=1.0 / up_size, device=device)
    kx_up, ky_up, kz_up = torch.meshgrid(kk_up, kk_up, kk_up, indexing="ij")
    knorm_up = torch.sqrt(kx_up**2 + ky_up**2 + kz_up**2)
    sel = (knorm_up < up_size // 2) & (knorm_up > 0)

    power_old = None
    change = float('inf')
    it = 0
    while it < max_iter and (it < min_iter or change > eps):
        x = torch.clamp(x, min=-1.0) # Projection on delta >= -1
        X = torch.fft.fftn(x)
        X[mask] = F_target[mask] # Projection on the LR modes
        x = torch.fft.ifftn(X).real

        # Relative change of the modes power
        power_new = torch.abs(X[sel])**2
        if power_old is not None:
            change = float(torch.sqrt(torch.sum((power_new - power_old)**2)) / torch.sqrt(torch.sum(power_new**2)))
        power_old = power_new
        it += 1

    return torch.clamp(x, min=-1.0), it

# Generate LR Data
def upsample_lr_data(hr_path: Path, lr_path: Path, hr_copy_path: Path, lr_up_path: Path, hist):
    if lr_up_path.exists(): return
    try:
        hr_data = np.load(hr_path).astype(np.float32)
        lr_data = np.load(lr_path).astype(np.float32)
        if hr_data.shape != (GRID_SIZE_HR, GRID_SIZE_HR, GRID_SIZE_HR):
             raise ValueError(f"Unexpected LR shape {hr_data.shape} in {hr_path}")
        if lr_data.shape != (GRID_SIZE_LR, GRID_SIZE_LR, GRID_SIZE_LR):
             raise ValueError(f"Unexpected LR shape {lr_data.shape} in {lr_path}")
        lr_up, it = fourier_upsample(torch.from_numpy(lr_data))
        hist.append(it)
        lr_up_path.parent.mkdir(parents=True, exist_ok=True)
        hr_copy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(lr_up_path, lr_up.numpy())
        np.save(hr_copy_path, hr_data)
    except FileNotFoundError: print(f"Warning: LR file not found: {lr_path}")
    except Exception as e: print(f"Error processing {lr_path}: {e}")

def run_preprocessing():
    print("Starting LR data upsampling...")
    hist = []
    for i in tqdm(range(NUM_SIMULATIONS), desc="Upsampling LR data"):
        upsample_lr_data(Path(str(HR_FOLDER_TEMPLATE).format(i)), Path(str(LR_FOLDER_TEMPLATE).format(i)),
                         Path(str(HR_UP_TEMPLATE).format(i)),Path(str(LR_UP_TEMPLATE).format(i)), hist)
    np.save('hist.npy',np.array(hist))
    print("LR data upsampling complete.")


# Main 
if __name__ == "__main__":    
    run_preprocessing()
