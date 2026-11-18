# %% [code]
import config
from config import K_NYQ_LR, K_NYQ_HR
import numpy as np
import math
import sys
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.auto import tqdm
import time
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from stats import compute_precomputed_stats
from helpers import IDWT, power_spectrum_3d_torch, compute_pk, equilateral, angle_scan, squeezed, pdf_edges, pdf_1pt, source_noise, pk_r_T, subband_edges, subband_pdfs, position_of_sim, conditional_pk, bispectra_of, to_density, paired_ratio
from dataset_helpers import load_params, get_dataloaders
from plotting_stats import bispectrum_angle_plot, bispectrum_eq_plot, transfer_plot, cross_corr_plot, band_plot, pdf_plot, bispectrum_band_plot, field_maps_plot, members_plot, conv_plot, evolution_plot, evolution_scalar_plot, cosmology_response_plot, trajectory_maps_plot, subband_pdf_plot, param_title 
import train_helpers as train
import diagnostics as diag
import model as mod
from datetime import timedelta
import traceback

# Configuration
torch.backends.cudnn.benchmark = True

# External Parameters
PARALLEL = os.environ.get('PARALLEL', 'True') in ('1','True') # Wether to use gpus in parallel or not
GLOBAL_AMP = os.environ.get('AMP', 'True') in ('1','True') # To either use mixed precision or not
VAL_FLAG = os.environ.get('VAL_FLAG', 'True') in ('1','True') # Whether to run validation and visualization after training
config.LOG_FLAG = os.environ.get('LOG_FLAG', 'True') in ('1','True') # Whether to use logging transform or normalization for data preprocessing
STATS_FLAG = os.environ.get('STATS_FLAG', 'True') in ('1','True') # Whether to load or compute precomputed stats for normalization/logging
LOCAL_FLAG = os.environ.get('LOCAL_FLAG', 'True') in ('1','True') # Whether to use local paths for data or paths from environment variables
EPOCHS = int(os.environ.get('EPOCHS', '100')) # Number of training epochs
STAT_PATH = os.environ.get('PARAM_PATH') # Path to parameter statistics
BASE_PATH_STR = os.environ.get('BASE_PATH_STR') # Base path for data files
PARAM_FILE_STR = os.environ.get('PARAM_FILE_STR') # Path to parameter file
MAX_SCALE = 65536.0 # Max scale of the scaler
MIN_SCALE = 1.0 # Min scale of the scaler 

if LOCAL_FLAG == True:
    BASE_PATH = Path("../../../../PCS/fourier3/data/")
    PARAM_FILE = Path("../../../../PCS/fourier3/data/latin_hypercube_params.txt")
else:
    BASE_PATH = Path(BASE_PATH_STR)
    PARAM_FILE = Path(PARAM_FILE_STR)

HR_FOLDER_TEMPLATE = BASE_PATH / "df_m_{}_128_z=0.npy"
LR_FOLDER_TEMPLATE = BASE_PATH / "df_m_{}_64_z=0.npy"
NUM_SIMULATIONS = 2000
GRID_SIZE_HR = 128
GRID_SIZE_LR = 64
config.BOX_SIZE = 1000.0
config.N_PARAMS = 5
config.BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '8'))
config.LEARNING_RATE = 3e-4
EMA_DECAY = 0.999
config.WARMUP_STEPS = 2000

BISPEC_CFGS = (("eq",   "Equilateral", "k [h/Mpc]",     True),
               ("sq",   "Squeezed",    "k [h/Mpc]",     True),
               ("ang",  "Angle scan",  r"$\theta/\pi$", False),
               ("ang2", "Angle scan",  r"$\theta/\pi$", False))

COSMO_PARAMS = [(r"$\Omega_m$", 0, 0.1, 0.5),
                (r"$\Omega_b$", 1, 0.03, 0.07),
                (r"$h$",        2, 0.5, 0.9),
                (r"$n_s$",      3, 0.8, 1.2),
                (r"$\sigma_8$", 4, 0.6, 1.0)]

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
NUM_ODE_STEPS = int(os.environ.get('NUM_ODE_STEPS', '8'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("./cfm_checkpoints")
RESULTS_DIR = Path("./cfm_results")
PK_DIR = RESULTS_DIR / "power_spectra"

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
PK_DIR.mkdir(exist_ok=True)

# Training step
def train_epoch(model, loader, optimizer, ema, scheduler, device, rank, parallel_flag, amp_flag, 
                scaler=None, decay_scheduler=None, global_step=0):
    
    model.train() # Train Mode

    amp_bool = bool(amp_flag) and (scaler is not None) # Mixed Precision Flag
    parallel_bool = bool(parallel_flag) and dist.is_initialized() # Parallel Flag
    utilization_samples = [] # GPU memory utilization list

    # Diagnostics
    total_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0
    grad_acc : dict = {}
    grad_count: dict = {}
    n_skipped = 0 # Diagnostics for eventual nan instabilities (AMP scaling)

    # Training Loop
    pbar = tqdm(loader, desc="Training Epoch", leave=False)
    for i,batch in enumerate(pbar):
        if batch is None: continue
        hr, lr, p = batch[0].to(device), batch[1].to(device), batch[2].to(device) # Load batch and move to device (8, 1, 128, 128, 128), (8, 1, 128, 128, 128), (8, num_params)
        optimizer.zero_grad() # Zero gradients before backward pass
        t_val = torch.rand(hr.shape[0], device=device).view(-1, 1, 1, 1, 1) # Random t for each sample in batch, broadcastable to (B, 1, 1, 1, 1)
        x_t = (1 - t_val) * lr + t_val * hr # Interpolation between LR and HR
        target_v = hr - lr # Target velocity field for the model to predict

        with torch.amp.autocast('cuda', enabled = amp_bool):
            predicted_v = model(x_t, t_val.view(-1), p)

        with torch.amp.autocast('cuda', enabled=False):
            loss = train.normed_mse(predicted_v.float(), target_v.float())

        is_loss_finite = bool(torch.isfinite(loss))
            
        if amp_bool:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else: 
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

        # Safety net and sync between the gpus for seldom nans (linked to scaler)
        is_epoch_finite = is_loss_finite and bool(torch.isfinite(grad_norm))

        if parallel_bool:
            finite = torch.tensor([1.0 if is_epoch_finite else 0.0], device=device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN) # Look for a nan among the processes
            finite_epoch = finite.item() > 0 # Nans found
        else:
            finite_epoch = is_epoch_finite
 
        if finite_epoch:
            diag.gradients_diagnostic(model, grad_acc, grad_count)
            if amp_bool:
                scaler.step(optimizer)
            else:
                optimizer.step()
            ema.update(model)
        else:
            n_skipped += 1

        if amp_bool:
            scaler.update()
            with torch.no_grad():
                scaler._scale.clamp_(min=MIN_SCALE, max=MAX_SCALE)   

        if global_step < config.WARMUP_STEPS: # Steps the learning rate scheduler during warmup phase
            scheduler.step()
        elif decay_scheduler is not None: # Steps the decay scheduler after warmup
            decay_scheduler.step()
        global_step += 1

        if finite_epoch:                                 
            total_loss += loss.item()
            total_grad_norm += grad_norm.item()
            num_batches += 1
            
        if i % 20 == 0: # GPU utilization Diagnostics
            free_mem, tot_mem = torch.cuda.mem_get_info(device)
            used_mem = tot_mem-free_mem
            utilization_samples.append((used_mem/tot_mem)*100)

        pbar.set_postfix(loss=loss.item(), grad=f"{grad_norm:.2e}", scale=(f"{scaler.get_scale():.0f}" if amp_bool else "off"))
    
    print(f'Batches Skipped for nans:{n_skipped}')
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0
    avg_grad_norms = {name:grad_acc[name] / grad_count[name] for name in grad_acc}
    if rank == 0:
        print(f"Average GPU Utilization during epoch: {np.mean(utilization_samples):.2f}%")
    return avg_loss, avg_grad_norm, global_step, avg_grad_norms

# Validation Step
def validate_epoch(model, loader, ema, device, amp_flag):
    model.eval() # Evaluation Mode
    amp_bool = bool(amp_flag) # Mixed Precision Flag

    # Diagnostics
    total_loss = 0.0
    num_batches = 0

    # Validation Loop
    pbar = tqdm(loader, desc="Validation Epoch", leave=False)
    with torch.no_grad():
        for batch in pbar:
            if batch is None: continue
            hr, lr, p = batch[0].to(device), batch[1].to(device), batch[2].to(device) # Load batch and move to device
            t_val = torch.rand(hr.shape[0], device=device).view(-1, 1, 1, 1, 1)
            x_t = (1 - t_val) * lr + t_val * hr
            target_v = hr - lr
            with torch.amp.autocast('cuda', enabled = amp_bool):
                with train.weights_context(model, ema):
                    predicted_v = model(x_t, t_val.view(-1), p)
            with torch.amp.autocast('cuda', enabled=False):
                loss = train.normed_mse(predicted_v.float(), target_v.float())

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

# Training 
def run_training(model, train_loader, val_loader, optimizer, ema, device, rank, epochs, ckpt_dir, res_dir, parallel_flag = PARALLEL, amp_flag = GLOBAL_AMP, scaler = None, scheduler = None):
    print(f"Starting training on {device}...")
    
    # Load Previous Losses
    train_loss_path = Path("train_losses.npy")
    val_loss_path = Path("val_losses.npy")
    grad_norm_path = Path("grad_norms.npy")
    group_norms_path = Path("group_grad_norms.npy")
    subband_mse_path  = Path("subband_mse.npy")
    subband_var_path  = Path("subband_var.npy")
    subband_corr_path = Path("subband_corr.npy")
    scaler_path = Path('scalers.npy')

    if train_loss_path.exists() and val_loss_path.exists() and grad_norm_path.exists()  and group_norms_path.exists():
        train_losses = list(np.load(train_loss_path))
        val_losses = list(np.load(val_loss_path))
        grad_norms = list(np.load(grad_norm_path))
        epoch_offset = len(train_losses)
        group_norms_history = list(np.load(group_norms_path, allow_pickle=True))
    else:
        train_losses, val_losses, grad_norms, group_norms_history = [], [], [], []
        epoch_offset = 0

    if subband_mse_path.exists() and subband_var_path.exists() and subband_corr_path.exists() and scaler_path.exists():
        subband_mse_hist  = list(np.load(subband_mse_path))
        subband_var_hist  = list(np.load(subband_var_path))
        subband_corr_hist = list(np.load(subband_corr_path))
        scalers = list(np.load(scaler_path))
    else:
        subband_mse_hist, subband_var_hist, subband_corr_hist, scalers = [], [], [], []

    if val_losses:
        best_val_loss = min(val_losses)
    else:
        best_val_loss = float('inf')

    # Decay Scheduler
    steps_per_epoch = len(train_loader)
    decay_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    global_step = epoch_offset * steps_per_epoch

    # Load old scheduler at restart
    ckpt_path_l = ckpt_dir / "latest_model.pth"
    ckpt_l = torch.load(ckpt_path_l, map_location='cpu') if ckpt_path_l.exists() else {}

    if 'decay_scheduler_state_dict' in ckpt_l:
        try:
            decay_scheduler.load_state_dict(ckpt_l['decay_scheduler_state_dict'])
        except Exception:
            pass
        for pg in optimizer.param_groups:
            pg['lr'] = decay_scheduler.get_last_lr()[0]

    start_time = time.time()
    # List of intermediate checkpoints to save
    saved_epochs = set(list(range(1, 11)) + list(range(15, 51, 5)) + list(range(60, 151, 10)) + list(range(180, epoch_offset + epochs + 1, 20))) # Epochs at which to save checkpoints
    for epoch in range(epochs):
        epoch_start_time = time.time()

        tot_epoch = epoch_offset + epoch + 1
        if parallel_flag and dist.is_initialized():
            train_loader.sampler.set_epoch(tot_epoch)

        # Train
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

        train_loss, grad_norm, global_step, avg_grad_norms = train_epoch(model, train_loader, optimizer, ema, scheduler, device, rank, parallel_flag, amp_flag, 
                                                                         scaler, decay_scheduler, global_step)
        # Gradient diagnostics per group/layer
        group_norms_history.append(diag.log_gradient_norms(tot_epoch, avg_grad_norms))
        # Validation
        val_loss = validate_epoch(model, val_loader, ema, device, amp_flag)
        # Per-subband diagnostics 
        subband_stats = diag.per_subband_diagnostics(model, val_loader, device=device, ema=ema, n_batches=10)
        subband_mse_hist.append(subband_stats['mse'])
        subband_var_hist.append(subband_stats['var'])
        subband_corr_hist.append(subband_stats['corr'])

        print("Current Scaler: {0}".format(scaler.get_scale() if scaler is not None else 'N/A'))
        scalers.append(scaler.get_scale() if scaler is not None else float('nan'))

        # Save losses and diagnostics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        grad_norms.append(grad_norm)

        if rank == 0:
            print(
                f"Epoch {epoch_offset}+{epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e} | "
                f"GradNorm: {grad_norm:.4e} | Time: {time.time()-epoch_start_time:.2f}s"
            )       

            saved = {
                'epoch': tot_epoch,
                'model_state_dict': train.get_module(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'decay_scheduler_state_dict': decay_scheduler.state_dict() if decay_scheduler is not None else None,
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'val_loss': val_loss,
            }

            if (val_loss < best_val_loss):
                for ckpt in ckpt_dir.glob("best_model_epoch_*.pth"):
                    ckpt.unlink()
                best_val_loss = val_loss
                torch.save(saved, ckpt_dir / f"best_model_epoch_{tot_epoch}.pth")
                print(f"Saved best model checkpoint.")
            if tot_epoch in saved_epochs:
                torch.save(saved, ckpt_dir / f"model_epoch_{tot_epoch}.pth")
            torch.save(saved, ckpt_dir / "latest_model.pth")

    print(f"Training finished. Total time: {time.time() - start_time:.2f}s")

    if epochs > 0 and rank == 0: 
        # Save final losses and diagnostics
        np.save("train_losses.npy",np.array(train_losses))
        np.save("val_losses.npy",np.array(val_losses))
        np.save("grad_norms.npy", np.array(grad_norms))
        np.save("scalers.npy", np.array(scalers))
        np.save("group_grad_norms.npy", np.array(group_norms_history, dtype = object))
        np.save("subband_mse.npy", np.array(subband_mse_hist,  dtype=np.float32))
        np.save("subband_var.npy", np.array(subband_var_hist,  dtype=np.float32))
        np.save("subband_corr.npy", np.array(subband_corr_hist, dtype=np.float32))

        # Plots
        epoch_axis = list(range(1, len(train_losses) + 1))

        # Scalers
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(epoch_axis, scalers, label='scaler', color = 'blue')
        ax.set_ylabel('Scalers')
        ax.set_xlabel('Epochs')
        ax.legend(loc='best')
        ax.grid(True)
        ax.set_title('Scalers')
        plt.tight_layout()
        plt.savefig(res_dir / "scalers.png")
        plt.close()

        # Losses and Gradients
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax1.plot(epoch_axis[0:599], train_losses[0:599], label='Training Loss')
        ax1.plot(epoch_axis[0:599], val_losses[0:599], label='Validation Loss')
        ax1.set_ylabel('Loss (MSE)',fontsize=18)
        ax1.legend(fontsize=12,loc='best')
        ax1.grid(True, which='major')
        ax1.grid(True, which='minor')
        ax1.set_yscale('log')
        ax1.tick_params(labelsize=13,which='both')

        ax2.plot(epoch_axis[0:599], grad_norms[0:599], label='total', color='black', linewidth=2, linestyle='--')

        layers = []
        for g in group_norms_history:
            for l in g:
                if l not in layers:
                    layers.append(l)

        colors = plt.cm.tab20(np.linspace(0, 1, len(layers)))

        for layer, color in zip(layers, colors):
            vals = [d.get(layer, float('nan')) for d in group_norms_history]
            ax2.plot(epoch_axis[0:599], vals[0:599], label=layer, color=color, linewidth=1)

        ax2.set_ylabel('Gradient Norm',fontsize=18)
        ax2.set_xlabel('Epoch',fontsize=18)
        ax2.legend(fontsize=12, ncol=2, loc='best')
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.tick_params(labelsize=13)

        plt.suptitle('Training History',fontsize=25)
        plt.tight_layout()
        plt.savefig(res_dir / "training_history.png")
        plt.close()
        print(f"Saved training history plot.")

    return model

# Inference 
@torch.no_grad()
def sample_ode(model, x0, transform_dwt_h, params_cond, num_steps, device, keep_trajectory=True, amp=False, snap_times=None):
    model.eval()
    dt = 1.0 / num_steps
    x_t = x0.to(device).float()
    params_cond = params_cond.to(device)

    if snap_times is not None:
        saved_steps = [round(t * num_steps) for t in snap_times]
    elif keep_trajectory:
        saved_steps = list(range(num_steps + 1))
    else:
        saved_steps = []

    trajectory = [x_t.cpu()] if 0 in saved_steps else []
    for i in tqdm(range(num_steps), desc="ODE Sampling", leave=False):
        t_batch = torch.full((x0.shape[0],), i * dt, device=device)
        with torch.amp.autocast('cuda', enabled=amp):
            v_t = model(x_t, t_batch, params_cond)
        x_t = x_t + v_t.float() * dt # Euler step
        if i + 1 in saved_steps:
            trajectory.append(x_t.cpu())

    x_space = IDWT(transform_dwt_h.inverse(x_t.cpu()))
    return x_space, trajectory

# Ensemble of generated fields for one simulation
@torch.no_grad()
def ensemble_diagnostics(model, dataset, transform_dwt_h, data_transform_h, device, sim_id, n_samples=16, batch=4,
                         num_steps=NUM_ODE_STEPS, do_bispectrum=True, n_bispec=8, dk=3.0, clump=20, results_dir=RESULTS_DIR,
                         tag="", ema=None, use_ema=True, slice_idx=64, do_convergence=True, seed=None, amp=False,
                         do_subbands=True, param_override=None):
    out_dir = results_dir / "ensemble"
    out_dir.mkdir(exist_ok=True, parents=True)
    pos = position_of_sim(dataset, sim_id)
    model.eval().to(device)

    if seed is not None:
        torch.manual_seed(seed)
    items = [dataset[pos] for _ in range(n_samples)]
    with source_noise(dataset, 0.0):
        item_clean = dataset[pos]
    hr = to_density(items[0][0].unsqueeze(0), transform_dwt_h, data_transform_h)[0, 0]
    src_clean = to_density(item_clean[1].unsqueeze(0), transform_dwt_h, data_transform_h)[0, 0]

    gen, src, raw = [], [], []
    with train.weights_context(model, ema, use_ema):
        for s in range(0, n_samples, batch):
            lr = torch.stack([it[1] for it in items[s:s + batch]])
            if param_override is None:
                p = torch.stack([it[2] for it in items[s:s + batch]])
            else: # Same (normalized) parameters for every member
                p = param_override.expand(len(lr), -1)
            g, states = sample_ode(model, lr, transform_dwt_h, p, num_steps, device, keep_trajectory=False, amp=amp, snap_times=(1.0,))
            raw.append(states[-1])
            gen.append(data_transform_h.inverse_torch(g)[:, 0])
            src.append(to_density(lr, transform_dwt_h, data_transform_h)[:, 0])
    gen = torch.cat(gen)
    src = torch.cat(src)

    # Power spectra, cross-correlations and transfer functions against the true HR field
    k, P_true, _, _ = power_spectrum_3d_torch(hr, hr, config.BOX_SIZE, MAS_index=4, clump=clump)
    k, P_true = k.numpy(), P_true.numpy()
    P_gen, r_gen, T_gen, P_src, r_src = [], [], [], [], []
    for a in range(n_samples):
        _, P, r, T = pk_r_T(gen[a], hr, config.BOX_SIZE, clump)
        P_gen.append(P)
        r_gen.append(r)
        T_gen.append(T)
        _, P, r, _ = pk_r_T(src[a], hr, config.BOX_SIZE, clump)
        P_src.append(P)
        r_src.append(r)
    P_gen, r_gen, T_gen = np.array(P_gen), np.array(r_gen), np.array(T_gen)
    P_src, r_src = np.array(P_src), np.array(r_src)

    gen_mean = gen.mean(0)
    _, P_genmean, r_genmean, T_genmean = pk_r_T(gen_mean, hr, config.BOX_SIZE, clump)
    _, P_srcclean, r_srcclean, _ = pk_r_T(src_clean, hr, config.BOX_SIZE, clump)

    # Conditional variance
    P_cond = conditional_pk(gen, clump)
    conv_n, conv_P = [n_samples], [P_cond]
    if do_convergence:
        conv_n = [m for m in (2, 4, 8, 16, 32, 64) if m <= n_samples]
        conv_P = [conditional_pk(gen[:m], clump) for m in conv_n]

    # One-point PDFs
    edges = pdf_edges()
    pdf_true, _ = pdf_1pt(hr.numpy(), edges)
    pdf_genmean, _ = pdf_1pt(gen_mean.numpy(), edges)
    pdf_srcclean, _ = pdf_1pt(src_clean.numpy(), edges)
    pdf_src = np.array([pdf_1pt(s.numpy(), edges)[0] for s in src])
    pdf_gen, frac_neg = [], []
    for g in gen:
        pdf, frac = pdf_1pt(g.numpy(), edges)
        pdf_gen.append(pdf)
        frac_neg.append(frac)
    pdf_gen = np.array(pdf_gen)

    d = dict(sim_id=sim_id, n_samples=n_samples, num_steps=num_steps, noise_scale=dataset.noise_scale,
             params=dataset.params[pos],
             k=k, P_true=P_true, P_cond=P_cond, conv_n=np.array(conv_n), conv_P=np.array(conv_P),
             P_gen_mean=P_gen.mean(0), P_gen_all=P_gen, P_gen_std=P_gen.std(0),
             P_src_mean=P_src.mean(0), P_src_std=P_src.std(0),
             P_genmean=P_genmean, P_srcclean=P_srcclean,
             r_gen_mean=r_gen.mean(0), r_gen_std=r_gen.std(0),
             r_src_mean=r_src.mean(0), r_src_std=r_src.std(0),
             r_genmean=r_genmean, r_srcclean=r_srcclean,
             T_gen_mean=T_gen.mean(0), T_gen_std=T_gen.std(0), T_genmean=T_genmean,
             pdf_edges=edges, pdf_true=pdf_true,
             pdf_gen_mean=pdf_gen.mean(0), pdf_gen_std=pdf_gen.std(0),
             pdf_src_mean=pdf_src.mean(0), pdf_src_std=pdf_src.std(0),
             pdf_genmean=pdf_genmean, pdf_srcclean=pdf_srcclean,
             frac_below_minus_one=np.mean(frac_neg),
             map_hr=hr[slice_idx].numpy(), map_src=src[0, slice_idx].numpy(),
             map_src_clean=src_clean[slice_idx].numpy(), map_genmean=gen_mean[slice_idx].numpy(),
             map_gen0=gen[0, slice_idx].numpy(), map_gen1=gen[1, slice_idx].numpy(),
             map_gen2=gen[min(2, n_samples - 1), slice_idx].numpy(),
             map_std_log=torch.log(gen[:, slice_idx] + 2.0).std(0).numpy())

    # One-point PDFs of the DWT subbands
    if do_subbands:
        edges_dwt = subband_edges()
        z_gen = torch.cat(raw).numpy()
        pdf_gen_dwt = subband_pdfs(z_gen, edges_dwt)
        pdf_src_dwt = subband_pdfs(torch.stack([it[1] for it in items]).numpy(), edges_dwt)
        d.update(band_edges=edges_dwt,
                 band_pdf_true=subband_pdfs(items[0][0].numpy(), edges_dwt)[0],
                 band_pdf_genmean=subband_pdfs(z_gen.mean(0), edges_dwt)[0],
                 band_pdf_srcclean=subband_pdfs(item_clean[1].numpy(), edges_dwt)[0],
                 band_pdf_gen_mean=pdf_gen_dwt.mean(0), band_pdf_gen_std=pdf_gen_dwt.std(0),
                 band_pdf_src_mean=pdf_src_dwt.mean(0), band_pdf_src_std=pdf_src_dwt.std(0))

    # Bispectra
    if do_bispectrum:
        triangles = {"eq": equilateral(n_min=4, n_max=60, dk=dk), "sq": squeezed(n_soft=6, n_min=8, n_max=56, dk=dk)}
        thetas, legs = {}, {}
        for name, (n1, n2) in (("ang", (16, 40)), ("ang2", (16, 28))):
            triangles[name], thetas[name] = angle_scan(n1=n1, n2=n2, n_theta=17, dk=dk)
            legs[name] = (n1, n2)

        n_bis = min(n_bispec, n_samples)
        b_true = bispectra_of(hr, triangles, dk, clump, device)
        b_mean = bispectra_of(gen_mean, triangles, dk, clump, device)
        b_clean = bispectra_of(src_clean, triangles, dk, clump, device)
        b_gen = [bispectra_of(gen[a], triangles, dk, clump, device) for a in range(n_bis)]
        b_src = [bispectra_of(src[a], triangles, dk, clump, device) for a in range(n_bis)]
        # Truth, ensemble mean and clean source, then mean and std over the members and the sources
        kF = 2.0 * math.pi / config.BOX_SIZE
        for name in triangles:
            d[f"{name}_k"] = b_true[name]["k1"]
            d[f"{name}_B_true"] = b_true[name]["B"]
            d[f"{name}_Q_true"] = b_true[name]["Q"]
            d[f"{name}_B_genmean"] = b_mean[name]["B"]
            d[f"{name}_Q_genmean"] = b_mean[name]["Q"]
            d[f"{name}_B_srcclean"] = b_clean[name]["B"]
            d[f"{name}_Q_srcclean"] = b_clean[name]["Q"]
            for key, ens in (("gen", b_gen), ("src", b_src)):
                B = np.array([e[name]["B"] for e in ens])
                Q = np.array([e[name]["Q"] for e in ens])
                d[f"{name}_B_{key}_mean"], d[f"{name}_B_{key}_std"] = B.mean(0), B.std(0)
                d[f"{name}_Q_{key}_mean"], d[f"{name}_Q_{key}_std"] = Q.mean(0), Q.std(0)
            if name in thetas: # Angle scans: angles of the valid triangles, and k1, k2
                d[f"{name}_theta"] = thetas[name][b_true[name]["mask"]]
                d[f"{name}_legs"] = np.array(legs[name]) * kF

    np.savez(out_dir / f"ensemble_sim{sim_id}{tag}.npz", **d)

    low = k < K_NYQ_LR
    band = (k >= K_NYQ_LR) & (k < K_NYQ_HR)
    ratio = P_cond / P_true
    print(f"Ensemble sim {sim_id}{tag}: {n_samples} samples, {num_steps} steps, noise scale {dataset.noise_scale:g}, EMA {use_ema}")
    print(f"    P_cond/P_true: {ratio[low].mean():.4f} below LR Nyquist, {ratio[band].mean():.4f} in band")
    print(f"    r(k) generated: {d['r_gen_mean'][low].mean():.4f} below, {d['r_gen_mean'][band].mean():.4f} in band")
    print(f"    r(k) source: {d['r_src_mean'][low].mean():.4f} below, {d['r_src_mean'][band].mean():.4f} in band")
    print(f"    T(k) in band: {d['T_gen_mean'][band].mean():.4f} per member, {T_genmean[band].mean():.4f} for the ensemble mean")
    print(f"    Fraction of delta < -1 in the generated fields: {np.mean(frac_neg):.3e}")
    return d

def ensemble_curves(d, key):
    return {"True HR": (d[f"{key}_true"], None),
            "Generated": (d[f"{key}_gen_mean"], d[f"{key}_gen_std"]),
            "Ensemble mean": (d[f"{key}_genmean"], None),
            "Noisy source": (d[f"{key}_src_mean"], d[f"{key}_src_std"]),
            "Clean source": (d[f"{key}_srcclean"], None)}

# Plots of the ensemble diagnostics of one simulation
def plot_ensemble(d, results_dir=RESULTS_DIR, tag=""):
    out_dir = results_dir / "ensemble"
    out_dir.mkdir(exist_ok=True, parents=True)
    sim_id, k = d["sim_id"], d["k"]
    sub = f"sim {sim_id}, {d['n_samples']} samples"

    # Power spectrum, cross-correlation and transfer function
    curves = ensemble_curves(d, "P")
    curves["Conditional variance"] = (d["P_cond"], None)
    band_plot(k, curves, out_dir / f"ens_pk_{sim_id}{tag}.png", "k [h/Mpc]", r"$P(k)\ [(\mathrm{Mpc}/h)^3]$",
              f"Power spectrum -- {sub}", ylog=True, ratio_ref="True HR")
    curves = {"Generated": (d["r_gen_mean"], d["r_gen_std"]),
              "Ensemble mean": (d["r_genmean"], None),
              "Noisy source": (d["r_src_mean"], d["r_src_std"]),
              "Clean source": (d["r_srcclean"], None)}
    band_plot(k, curves, out_dir / f"ens_rk_{sim_id}{tag}.png", "k [h/Mpc]", "r(k)",
              f"Cross-correlation with truth -- {sub}", hline=1.0, ylim=(0.0, 1.05))
    curves = {"Generated": (d["T_gen_mean"], d["T_gen_std"]), "Ensemble mean": (d["T_genmean"], None)}
    band_plot(k, curves, out_dir / f"ens_Tk_{sim_id}{tag}.png", "k [h/Mpc]", "T(k)",
              f"Transfer function -- {sub}", hline=1.0, ylim=(0.0, 1.6))

    # One-point PDF
    pdf_plot(d["pdf_edges"], ensemble_curves(d, "pdf"), out_dir / f"ens_pdf_{sim_id}{tag}.png",
             title=f"One-point PDF -- {sub}", ratio_ref="True HR")

    # Bispectra
    for name, label, xlabel, xlog in BISPEC_CFGS:
        if f"{name}_B_true" not in d:
            continue
        if xlog:
            x, legs = d[f"{name}_k"], None
            title = f"{label} bispectrum -- {sub}"
        else:
            x, legs = d[f"{name}_theta"] / np.pi, d[f"{name}_legs"]
            title = (f"{label} bispectrum, $k_1={legs[0]:.3f}$, $k_2={legs[1]:.3f}"
                     r"\,h\,\mathrm{Mpc}^{-1}$ -- " + sub)
        curves_Q = ensemble_curves(d, f"{name}_Q")
        if name in ("eq", "sq"):
            del curves_Q["Clean source"]
        bispectrum_band_plot(x, ensemble_curves(d, f"{name}_B"), curves_Q,
                             out_dir / f"ens_bis_{name}_{sim_id}{tag}.png", title, xlabel=xlabel,
                             ratio_ref="True HR", legs=legs)

    if "band_pdf_true" in d:
        subband_pdf_plot(d, out_dir / f"ens_bands_pdf_{sim_id}{tag}.png")
    field_maps_plot(d, out_dir / f"ens_maps_{sim_id}{tag}.png")
    members_plot(d, out_dir / f"ens_members_{sim_id}{tag}.png")
    if len(d["conv_n"]) > 1:
        conv_plot(d, out_dir / f"ens_conv_{sim_id}{tag}.png")

# Trajectory diagnostics
@torch.no_grad()
def trajectory_diagnostics(model, dataset, transform_dwt_h, data_transform_h, device, sim_id, n_samples=2, num_steps=NUM_ODE_STEPS,
                           snap_times=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0), seed=1234, ema=None, use_ema=True, amp=False,
                           slice_idx=64, clump=20, results_dir=RESULTS_DIR, tag=""):
    out_dir = results_dir / "trajectory"
    out_dir.mkdir(exist_ok=True, parents=True)
    snap_times = sorted(snap_times)
    pos = position_of_sim(dataset, sim_id)
    model.eval().to(device)
    if seed is not None:
        torch.manual_seed(seed)
    items = [dataset[pos] for _ in range(n_samples)]
    hr = to_density(items[0][0].unsqueeze(0), transform_dwt_h, data_transform_h)[0, 0]
    std_true = items[0][0].std(dim=(1, 2, 3)).numpy()

    lr = torch.stack([it[1] for it in items])
    p = torch.stack([it[2] for it in items])
    with train.weights_context(model, ema, use_ema):
        _, states = sample_ode(model, lr, transform_dwt_h, p, num_steps, device, keep_trajectory=False, amp=amp, snap_times=snap_times)

    maps, P_t, r_t, std_t = [], [], [], []
    for z in states:
        fields = to_density(z, transform_dwt_h, data_transform_h)[:, 0]
        P, r = [], []
        for f in fields:
            k, P_f, r_f, _ = pk_r_T(f, hr, config.BOX_SIZE, clump)
            P.append(P_f)
            r.append(r_f)
        maps.append(fields[:, slice_idx].numpy())
        P_t.append(P)
        r_t.append(r)
        std_t.append(z.std(dim=(2, 3, 4)).numpy())
    _, P_true, _, _ = power_spectrum_3d_torch(hr, hr, config.BOX_SIZE, MAS_index=4, clump=clump)

    d = dict(sim_id=sim_id, n_samples=n_samples, params=dataset.params[pos], t=np.array(snap_times), k=k,
             P_true=P_true.numpy(), map_hr=hr[slice_idx].numpy(), band_std_true=std_true,
             maps=np.array(maps), P_t=np.array(P_t), r_t=np.array(r_t), band_std_t=np.array(std_t))
    np.savez(out_dir / f"trajectory_sim{sim_id}{tag}.npz", **d)
    trajectory_maps_plot(d, out_dir / f"traj_maps_{sim_id}{tag}.png")

    band = (k >= K_NYQ_LR) & (k < K_NYQ_HR)
    T = np.sqrt(d["P_t"].mean(1) / d["P_true"])
    r_mean = d["r_t"].mean(1)
    std_ratio = d["band_std_t"].mean(1) / std_true
    print(f"Trajectory sim {sim_id}{tag}")
    for j, t in enumerate(snap_times):
        print(f"    t={t:4.2f}  <T>={T[j][band].mean():.3f}  <r>={r_mean[j][band].mean():.3f}"
              f"  sigma/sigma_true LLL {std_ratio[j, 0]:.3f}  HHH {std_ratio[j, 7]:.3f}")
    return d

def run_results(net, dataset, transform_dwt_h, data_transform_h, device, ema=None, sim_ids=(663,), n_samples=16,
                sweep_samples=4, batch=4, amp=False, results_dir=RESULTS_DIR, ensembles=None, noise_scales=(0.0, 0.5, 1.0, 2.0)):

    if ensembles is None:
        ensembles = {}
        for sim_id in sim_ids:
            try:
                d = ensemble_diagnostics(net, dataset, transform_dwt_h, data_transform_h, device, sim_id, n_samples=n_samples,
                                         batch=batch, ema=ema, amp=amp, results_dir=results_dir)
            except ValueError as e:
                print(f"Skipping sim {sim_id}: {e}")
                continue
            plot_ensemble(d, results_dir)
            ensembles[sim_id] = d

    for sim_id, d in ensembles.items():
        p = dataset.params[position_of_sim(dataset, sim_id)]
        band = (d["k"] >= K_NYQ_LR) & (d["k"] < K_NYQ_HR)
        print(f"Sim {sim_id}: s8={p[4]:.3f}, Om={p[0]:.3f}, S8={p[4] * np.sqrt(p[0] / 0.3):.3f}, "
              f"P_cond/P_true={d['P_cond'][band].mean() / d['P_true'][band].mean():.4f}, r={d['r_gen_mean'][band].mean():.4f}")

    for scale in noise_scales:
        with source_noise(dataset, scale):
            ensemble_diagnostics(net, dataset, transform_dwt_h, data_transform_h, device, sim_ids[0],
                                 n_samples=2 if scale == 0 else sweep_samples, batch=batch, n_bispec=2,
                                 do_convergence=False, ema=ema, amp=amp, results_dir=results_dir, tag=f"_ns{scale:g}")
    return ensembles

def strip_prefixes(key):
    for prefix in ("module._orig_mod.", "module.", "_orig_mod."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key

def load_ckpt_weights(model, path, use_ema=True, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    net = train.get_module(model)
    net.load_state_dict({strip_prefixes(k): v for k, v in ckpt['model_state_dict'].items()}, strict=False)
    if use_ema:
        shadow = ckpt['ema_state_dict']['shadow']
        with torch.no_grad():
            for name, param in net.named_parameters():
                if name in shadow:
                    param.copy_(shadow[name])
    return ckpt['epoch']
 
# Ensemble diagnostics of one simulation at several checkpoints
@torch.no_grad()
def run_checkpoint_evolution(net, dataset, transform_dwt_h, data_transform_h, device, ckpt_dir=CHECKPOINT_DIR, results_dir=RESULTS_DIR,
                             sim_id=663, n_samples=4, num_steps=NUM_ODE_STEPS, seed=1234, use_ema=True, max_ckpts=14,
                             do_bispectrum=True, n_bispec=2, amp=False):
    out_dir = results_dir / "evolution"
    out_dir.mkdir(exist_ok=True, parents=True)

    paths = sorted(ckpt_dir.glob("model_epoch_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
    latest = ckpt_dir / "latest_model.pth"
    latest_epoch = torch.load(latest, map_location="cpu")["epoch"]
    paths = [p for p in paths if int(p.stem.split("_")[-1]) != latest_epoch]
    if len(paths) > max_ckpts - 1:
        idx = np.unique(np.geomspace(1, len(paths), max_ckpts - 1).astype(int) - 1)
        paths = [paths[i] for i in idx]
    paths.append(latest)

    epochs, ds = [], []
    for path in paths:
        epoch = load_ckpt_weights(net, path, use_ema=use_ema, device=device)
        print(f"Checkpoint evolution: epoch {epoch} from {path.name}")
        d = ensemble_diagnostics(net, dataset, transform_dwt_h, data_transform_h, device, sim_id, n_samples=n_samples,
                                 batch=min(4, n_samples), num_steps=num_steps, do_bispectrum=do_bispectrum, n_bispec=n_bispec,
                                 do_convergence=False, seed=seed, use_ema=False,
                                 results_dir=results_dir, tag=f"_ep{epoch}", amp=amp)
        epochs.append(epoch)
        ds.append(d)
 
    k, d0 = ds[0]["k"], ds[0]
    band = (k >= K_NYQ_LR) & (k < K_NYQ_HR)
    low = k < K_NYQ_LR

    evolution_plot(epochs, k, [d["T_gen_mean"] for d in ds], out_dir / "evo_Tk.png", "k [h/Mpc]", "T(k)",
                   f"Transfer function over training -- sim {sim_id}", hline=1.0, ylim=(0.0, 1.6))
    evolution_plot(epochs, k, [d["r_gen_mean"] for d in ds], out_dir / "evo_rk.png", "k [h/Mpc]", "r(k)",
                   f"Cross-correlation over training -- sim {sim_id}", hline=1.0, ylim=(0.0, 1.05))
    evolution_plot(epochs, k, [d["P_gen_mean"] / d0["P_true"] for d in ds], out_dir / "evo_pk_ratio.png", "k [h/Mpc]",
                   r"$P_{\rm gen}/P_{\rm true}$", f"Power spectrum over training -- sim {sim_id}", hline=1.0, ylim=(0.0, 1.8))
    evolution_plot(epochs, k, [d["P_cond"] / d0["P_true"] for d in ds], out_dir / "evo_pcond.png", "k [h/Mpc]",
                   r"$P_{\rm cond}/P_{\rm true}$", f"Conditional variance over training -- sim {sim_id}", hline=1.0, ylim=(-0.05, 1.35))

    centres = 0.5 * (d0["pdf_edges"][:-1] + d0["pdf_edges"][1:])
    pdf_true = np.where(d0["pdf_true"] > 1e-12, d0["pdf_true"], np.nan)
    evolution_plot(epochs, centres, [d["pdf_gen_mean"] / pdf_true for d in ds], out_dir / "evo_pdf.png", r"$\log_{10}(1+\delta)$",
                   r"$p_{\rm gen}/p_{\rm true}$", f"One-point PDF over training -- sim {sim_id}", hline=1.0, xlog=False, ylim=(0.0, 2.0))
 
    bis = {}
    for name, label, xlabel, xlog in BISPEC_CFGS:
        if f"{name}_B_true" not in d0:
            continue
        B_true = np.where(np.abs(d0[f"{name}_B_true"]) > 1e-30, d0[f"{name}_B_true"], np.nan)
        if xlog:
            x, extra = d0[f"{name}_k"], ""
            sel = (x >= K_NYQ_LR) & (x < K_NYQ_HR)
            key = rf"$\langle B/B_{{\rm true}}\rangle$ {label.lower()}"
        else:
            k1, k2 = d0[f"{name}_legs"]
            x, extra = d0[f"{name}_theta"] / np.pi, f", $k_1={k1:.3f}$, $k_2={k2:.3f}$"
            sel = np.ones(len(x), dtype=bool)
            key = rf"$\langle B/B_{{\rm true}}\rangle$ angle, $k_2={k2:.3f}$"
        evolution_plot(epochs, x, [d[f"{name}_B_gen_mean"] / B_true for d in ds], out_dir / f"evo_bis_{name}.png", xlabel,
                       r"$B/B_{\rm true}$", f"{label} bispectrum over training{extra} -- sim {sim_id}",
                       hline=1.0, xlog=xlog, ylim=(0.0, 2.0))
        bis[key] = [np.nanmean(d[f"{name}_B_gen_mean"][sel] / B_true[sel]) for d in ds]
 
    series = {r"$\langle T\rangle$ in band": [d["T_gen_mean"][band].mean() for d in ds],
              r"$\langle T\rangle$ ensemble mean": [d["T_genmean"][band].mean() for d in ds],
              r"$\langle r\rangle$ in band": [d["r_gen_mean"][band].mean() for d in ds],
              r"$\langle P_{\rm cond}/P_{\rm true}\rangle$": [d["P_cond"][band].mean() / d0["P_true"][band].mean() for d in ds],
              r"$\langle r\rangle$ below $k_{\rm Nyq}^{\rm LR}$": [d["r_gen_mean"][low].mean() for d in ds]}
    series.update(bis)
    loss = None
    if Path("val_losses.npy").exists():
        val_losses = np.load("val_losses.npy")
        loss = [val_losses[e - 1] if e <= len(val_losses) else np.nan for e in epochs]

    evolution_scalar_plot(epochs, series, out_dir / "evo_scalar.png", "Summary statistics against training epoch",
                          loss=loss, hline=1.0)
    if bis:
        evolution_scalar_plot(epochs, bis, out_dir / "evo_scalar_bispec.png", "Bispectrum against training epoch",
                              loss=loss, hline=1.0, ncol=1, ylabel=r"$\langle B/B_{\rm true}\rangle$")
    np.save(out_dir / "evolution_summary.npy", dict(epochs=epochs, series=series, loss=loss), allow_pickle=True)

# Evaluation & Visualization
def evaluate_and_visualize(model, loader, data_transform_h, data_transform_l, transform_dwt_h, device, num_examples=5,
                           num_ode_steps=NUM_ODE_STEPS, results_dir=RESULTS_DIR):    
    model.eval().to(device)
    examples_saved = 0
    all_mse = []
    vis_dir = results_dir / "visualizations"
    anim_dir = results_dir / "animations"
    vis_dir.mkdir(exist_ok=True)
    anim_dir.mkdir(exist_ok=True)
    print("Starting evaluation and visualization...")
    pk_ratio = []

    for batch in tqdm(loader, desc="Evaluating / Visualizing"):
        if examples_saved >= num_examples or batch is None: break
        hr_batch_norm, lr_batch_norm, params_batch_norm = batch   
        generated_hr_norm, trajectory = sample_ode(model, lr_batch_norm, transform_dwt_h, params_batch_norm, num_ode_steps, device)

        generated_hr = data_transform_h.inverse_torch(generated_hr_norm)
        hr_true = to_density(hr_batch_norm, transform_dwt_h, data_transform_h)
        lr_orig = to_density(lr_batch_norm, transform_dwt_h, data_transform_l)


        all_mse.extend(nn.functional.mse_loss(generated_hr, hr_true, reduction='none').mean(dim=(1,2,3,4)).numpy())

        for i in range(min(num_examples - examples_saved, hr_batch_norm.shape[0])):
            idx_file = examples_saved
            sl = GRID_SIZE_HR // 2
            lr_s, gen_s, true_s = lr_orig[i,0,sl].numpy(), generated_hr[i,0,sl].numpy(), hr_true[i,0,sl].numpy()
            
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            vmin,vmax = min(lr_s.min(),gen_s.min(),true_s.min()), max(lr_s.max(),gen_s.max(),true_s.max())
            titles = [f'Input LR (Blocky {GRID_SIZE_LR}^3)', 'Generated HR', 'True HR', 'Absolute Difference', 'Relative Difference']
            data_slices = [lr_s, gen_s, true_s, np.abs(gen_s - true_s), np.abs(gen_s - true_s)/(np.abs(true_s)+1e-6)]
            cmaps = ['viridis']*3 + ['magma']*2
            for ax, data, title, cmap_i in zip(axes, data_slices, titles, cmaps):
                im_opts = {'origin':'lower', 'cmap':cmap_i}
                if 'Difference' not in title: im_opts.update({'vmin':vmin, 'vmax':vmax})
                if 'Relative' in title: im_opts.update({'vmax':2.0})
                im = ax.imshow(data, **im_opts)
                ax.set_title(title)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f'SR Comparison - Example {idx_file} (Central Slice)', fontsize=16)
            plt.tight_layout(rect=[0,0.03,1,0.95])
            plt.savefig(vis_dir / f"comparison_{idx_file}.png")
            plt.close(fig)

            # Power spectra
            k, Pgen, Ptrue, Pcross = compute_pk(generated_hr[i, 0], hr_true[i, 0])
            np.save(vis_dir / 'k.npy', k.numpy())
            np.save(vis_dir / f'Pgen_{idx_file}.npy', Pgen.numpy())
            np.save(vis_dir / f'Ptrue_{idx_file}.npy', Ptrue.numpy())
            np.save(vis_dir / f'Pcross_{idx_file}.npy', Pcross.numpy())
            pk_ratio.append(Pgen.numpy() / Ptrue.numpy())
            
            fig_pk, ax_pk = plt.subplots(figsize=(6,5))
            ax_pk.loglog(k,Ptrue,label="True HR",lw=2)
            ax_pk.loglog(k,Pgen,label="Generated HR",lw=2)
            ax_pk.loglog(k,Pcross,label="Cross",ls="--")
            axins = inset_axes(ax_pk,width="45%",height="45%",loc="lower left")
            axins.semilogx(k,Pgen/(Ptrue+1e-9))
            axins.set_ylim(0.5,1.5)
            axins.axhline(1,c="gray",lw=0.8)
            axins.set_title("Gen/True",fontsize=8)
            ax_pk.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
            ax_pk.set_ylabel(r"$P(k)$")
            ax_pk.set_title(f"P(k) – ex {idx_file}")
            ax_pk.legend()
            ax_pk.grid(True,which="both",lw=0.3)
            fig_pk.tight_layout()
            fig_pk.savefig(PK_DIR / f"pk_comparison_{idx_file}.png",dpi=150)
            plt.close(fig_pk)

            # Transfer function and cross-correlation
            T = torch.sqrt(Pgen / Ptrue).numpy()
            r_k = (Pcross / torch.sqrt(Pgen * Ptrue)).numpy()
            transfer_plot(k, {"Generated": T}, vis_dir / f'transfer_{idx_file}.png', title="Transfer function of the generated field")
            cross_corr_plot(k, {"Generated": r_k}, vis_dir / f'cross_corr_{idx_file}.png', title="Cross-correlation of the generated field with HR")
            
            # Bispectra of the generated and target fields
            tri_ang, theta = angle_scan(n1=16, n2=40, n_theta=17, dk=3.0)
            triangles = {"eq": equilateral(n_min=4, n_max=60, dk=3.0), "sq": squeezed(n_soft=6, n_min=8, n_max=56, dk=3.0), "ang": tri_ang}
            b_gen = bispectra_of(generated_hr[i, 0], triangles, device=device)
            b_true = bispectra_of(hr_true[i, 0], triangles, device=device)
            bispectrum_eq_plot({"Generated": b_gen["eq"], "Target": b_true["eq"]}, vis_dir / f'bispectrum_equilateral_{idx_file}.png',
                               "Equilateral bispectrum, generated and target", ref_label="Target")
            bispectrum_eq_plot({"Generated": b_gen["sq"], "Target": b_true["sq"]}, vis_dir / f'bispectrum_squeezed_{idx_file}.png',
                               "Squeezed bispectrum, generated and target", ref_label="Target")
            kF = 2.0 * math.pi / config.BOX_SIZE
            bispectrum_angle_plot({"Generated": b_gen["ang"], "Target": b_true["ang"]}, theta,
                                  vis_dir / f'bispectrum_angle_{idx_file}.png', k1=16 * kF, k2=40 * kF, ref_label="Target")

            # Animation of the flow for the first examples
            if idx_file < 5:
                print(f"Generating animation for example {idx_file}...")
                frames = [to_density(frame[i:i+1], transform_dwt_h, data_transform_h)[0, 0, sl].numpy() for frame in trajectory]
                fig_anim, ax_anim = plt.subplots(figsize=(6,6))
                ims = []
                vmin_a, vmax_a = min(f.min() for f in frames ), max(f.max() for f in frames )
                for step, frame_data in enumerate(frames ):
                    im = ax_anim.imshow(frame_data, origin='lower', cmap='viridis', vmin=vmin_a, vmax=vmax_a, animated=True)
                    title_obj = ax_anim.text(0.5,1.01,f'Flow: Step {step}/{num_ode_steps}', ha='center',va='bottom',transform=ax_anim.transAxes)
                    ax_anim.set_xticks([])
                    ax_anim.set_yticks([])
                    ims.append([im, title_obj])
                ani = animation.ArtistAnimation(fig_anim, ims, interval=50, blit=False, repeat_delay=1000)
                ani.save(anim_dir / f"flow_animation_example_{idx_file}.gif", writer=animation.PillowWriter(fps=15))
                plt.close(fig_anim)
                print(f"Saved flow animation example {idx_file}.")
            examples_saved += 1
            if examples_saved >= num_examples: break
        del trajectory, generated_hr_norm, generated_hr, hr_true, lr_orig
        torch.cuda.empty_cache()

    np.save(vis_dir / "pk_ratio.npy", np.mean(pk_ratio, axis=0))
    print(f"\nEvaluation Complete. Mean MSE over test examples: {np.mean(all_mse):.4e}")

# Response of the generated power spectrum varying each cosmological parameter at fixed source
@torch.no_grad()
def run_cosmology_response(net, dataset, transform_dwt_h, data_transform_h, param_transform, device, sim_id=663, ema=None,
                           results_dir=RESULTS_DIR, n_samples=4, n_values=3, seed=1234, num_steps=NUM_ODE_STEPS, use_ema=True,
                           params=COSMO_PARAMS, amp=False):
    out_dir = results_dir / "cosmology"
    out_dir.mkdir(exist_ok=True, parents=True)
    pos = position_of_sim(dataset, sim_id)
    base = dataset.params[pos].copy() # True parameters of the simulation
    print(f"Cosmology response: sim {sim_id}, {n_samples} source realisations, {param_title(base)}")

    def ensemble_with(values, tag):
        override = param_transform(torch.from_numpy(values))
        return ensemble_diagnostics(net, dataset, transform_dwt_h, data_transform_h, device, sim_id, n_samples=n_samples,
                                    batch=min(4, n_samples), num_steps=num_steps, do_bispectrum=False, do_convergence=False,
                                    seed=seed, ema=ema, use_ema=use_ema, amp=amp, results_dir=results_dir, tag=tag,
                                    param_override=override)

    d_fid = ensemble_with(base, "_cosmo_fid")
    k = d_fid["k"]
    band = (k >= K_NYQ_LR) & (k < K_NYQ_HR)
    low = k < K_NYQ_LR

    res = {}
    for label, j, p_lo, p_hi in params:

        grid = [v for v in np.linspace(p_lo, p_hi, n_values) if not np.isclose(v, base[j], atol=1e-3)]
        values = np.sort(np.append(grid, base[j]))
        ratios, errs, band_avg, band_err, low_avg = [], [], [], [], []
        for v in values:
            if v == base[j]:
                d = d_fid
            else:
                varied = base.copy()
                varied[j] = v
                d = ensemble_with(varied, f"_cosmo_p{j}_{v:.4g}")
            ratio, err = paired_ratio(d, d_fid)
            ratios.append(ratio)
            errs.append(err)
            band_avg.append(ratio[band].mean())
            band_err.append(err[band].mean())
            low_avg.append(ratio[low].mean())
        res[label] = dict(values=values, fid=base[j], lo=p_lo, hi=p_hi, ratios=ratios, ratios_err=errs,
                          band=band_avg, band_err=band_err, band_lo=low_avg)

        span = max(band_avg) - min(band_avg)
        print(f"Cosmology response {label}: span {span:.4f} in band ({span / np.mean(band_err):.1f} sigma), "
              f"{max(low_avg) - min(low_avg):.4f} below LR Nyquist")

    cosmology_response_plot(k, res, out_dir / f"cosmology_response_{sim_id}.png",
                            title=f"Cosmology response at fixed source ({param_title(base)})")
    np.save(out_dir / f"cosmology_response_{sim_id}.npy", dict(k=k, base=base, n_samples=n_samples, res=res), allow_pickle=True)
    return res
 
 
# Function to build the model and training objects
def model_building(device, parallel_flag, amp_flag, rank = 0):
    model = mod.UNET3D(in_channels = 8, base_channels = 102, channel_mults = (1, 2), num_params = config.N_PARAMS).to(device)
    model = torch.compile(model)
    if parallel_flag  and dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    ema = train.EMA(model, max_decay = EMA_DECAY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: train.step_control(step, config.WARMUP_STEPS))
    if amp_flag == True:
        scaler = torch.amp.GradScaler('cuda', init_scale=512, growth_interval=1000)
    else:
        scaler = None

    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")

    return model, ema, optimizer, scheduler, scaler

# Function to load old checkpoints
def load_checkpoint(model, optimizer, scheduler, ema, scaler, device, ckpt_dir):
    path = ckpt_dir / "latest_model.pth"
    if not path.exists():
        print(f"No checkpoint at {path}. Starting fresh.")
        return
    
    print(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location = device)

    stripped = {strip_prefixes(k): v for k, v in ckpt['model_state_dict'].items()}
    train.get_module(model).load_state_dict(stripped, strict=False)

    try:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    except (ValueError, KeyError) as e:
        print(f"Optimizer restore skipped: {e}")

    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'ema_state_dict' in ckpt:
        ema.load_state_dict(ckpt['ema_state_dict'])
    if scaler is not None and ckpt.get('scaler_state_dict') is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])

# Function to run the training in parallel between the two gpus
def parallel_worker(rank, world_size, args, amp_flag = GLOBAL_AMP):

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29500", rank=rank, world_size=world_size, timeout=timedelta(hours=4))
    device = torch.device(f"cuda:{rank}")

    (train_loader, val_loader, test_loader, data_transform_h, data_transform_l, transform_dwt_h, transform_dwt_l, param_transform) = get_dataloaders(
        args['params_all'], args["hr_template"], args["lr_template"], args["batch_size"], True, args['precomp_stats'])

    model, ema, optimizer, scheduler, scaler = model_building(device, parallel_flag=True, amp_flag=amp_flag, rank=rank)
    load_checkpoint(model, optimizer, scheduler, ema, scaler, device, CHECKPOINT_DIR)
    print(f"Scaler: {scaler.get_scale() if scaler is not None else 'N/A'}")
    model = run_training(model, train_loader, val_loader, optimizer, ema, device, rank, EPOCHS, CHECKPOINT_DIR, RESULTS_DIR,
                         parallel_flag=True, amp_flag=amp_flag, scaler=scaler, scheduler=scheduler)

    if VAL_FLAG:
        net = train.get_module(model)
        test_set = test_loader.dataset
        sims = (1044, 166, 1528, 315, 748, 1795, 1017, 1484, 95)

        summaries = []
        for sim_id in sims[rank::world_size]:
            try:
                d = ensemble_diagnostics(net, test_set, transform_dwt_h, data_transform_h, device, sim_id, ema=ema)
                plot_ensemble(d)
                summaries.append({key: d[key] for key in ("sim_id", "k", "P_true", "P_cond", "r_gen_mean")})
            except Exception:
                traceback.print_exc()

        gathered = [None] * world_size
        dist.all_gather_object(gathered, summaries)
        ensembles = {}
        for rank_summaries in gathered:
            for s in rank_summaries:
                ensembles[s["sim_id"]] = s

        try:
            if rank == 0:
                run_results(net, test_set, transform_dwt_h, data_transform_h, device, ema=ema, sim_ids=sims, ensembles=ensembles)

                trajectory_diagnostics(net, test_set, transform_dwt_h, data_transform_h, device, sims[0], ema=ema)

                run_checkpoint_evolution(net, test_set, transform_dwt_h, data_transform_h, device, sim_id=sims[0])
            elif rank == 1:
                run_cosmology_response(net, test_set, transform_dwt_h, data_transform_h, param_transform, device, sim_id=sims[0], ema=ema)
        except Exception:
            traceback.print_exc()

    dist.barrier()
    dist.destroy_process_group()

# Main 
if __name__ == "__main__":

    try: all_params = load_params(PARAM_FILE, NUM_SIMULATIONS, config.N_PARAMS)
    except Exception: sys.exit(1)

    if STATS_FLAG == True:
        precomputed_stats = compute_precomputed_stats(
            all_params, HR_FOLDER_TEMPLATE, LR_FOLDER_TEMPLATE, TRAIN_SPLIT, VAL_SPLIT
        )
    else:
        precomputed_stats = np.load(STAT_PATH, allow_pickle=True)
    
    num_gpus = torch.cuda.device_count()
    
    if (PARALLEL == True) and (num_gpus > 1):
        print(f"Spawning {num_gpus} DDP processes")

        args = {
            'params_all': all_params,
            'batch_size': config.BATCH_SIZE,
            'hr_template': HR_FOLDER_TEMPLATE,
            'lr_template': LR_FOLDER_TEMPLATE,
            'precomp_stats': precomputed_stats,
        }
        mp.spawn(parallel_worker, nprocs=num_gpus, args=(num_gpus, args), join=True)

    else:

        train_loader, val_loader, test_loader, data_transform_h, data_transform_l, transform_dwt_h, transform_dwt_l, param_transform = get_dataloaders(
            all_params, HR_FOLDER_TEMPLATE, LR_FOLDER_TEMPLATE, config.BATCH_SIZE, False, precomputed_stats)

        model, ema, optimizer, scheduler, scaler = model_building(DEVICE, parallel_flag=False, amp_flag=GLOBAL_AMP, rank=0)
        load_checkpoint(model, optimizer, scheduler, ema, scaler, DEVICE, CHECKPOINT_DIR)
        print(f"Scaler: {scaler.get_scale() if scaler is not None else 'N/A'}")
        
        model = run_training(model, train_loader, val_loader, optimizer, ema, DEVICE, 0, EPOCHS, CHECKPOINT_DIR, RESULTS_DIR, parallel_flag=False, amp_flag = GLOBAL_AMP, scaler=scaler, scheduler=scheduler)

        if VAL_FLAG:
            evaluate_and_visualize(train.get_module(model), test_loader, data_transform_h, data_transform_l, transform_dwt_h, DEVICE, num_examples=24, num_ode_steps=NUM_ODE_STEPS, results_dir=RESULTS_DIR)