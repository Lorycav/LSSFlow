import torch
import numpy as np
from collections import defaultdict
from train_helpers import get_module, weights_context

# Gradients Accumulator
def gradients_diagnostic(model, acc: dict, counts: dict):
    mod = get_module(model)
    for name, param in mod.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            acc[name] = acc.get(name, 0.0) + norm
            counts[name] = counts.get(name, 0) + 1
 
# Function to compute per-subband diagnostics
def per_subband_diagnostics(model, loader, ema, device, n_batches=10):
    model.eval()
    sum_mse = torch.zeros(8)
    sum_var_r = torch.zeros(8)
    sum_corr = torch.zeros(8)
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches: break
            if batch is None: continue
            hr, lr, p = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            t_val = torch.rand(hr.shape[0], device=device).view(-1, 1, 1, 1, 1)
            x_t = (1 - t_val) * lr + t_val * hr
            target_v = hr - lr

            with weights_context(model, ema):
                pred_v = model(x_t, t_val.view(-1), p)

            p_c = pred_v.flatten(2)
            t_c = target_v.flatten(2)
            sum_mse += (p_c - t_c).pow(2).mean(dim=(0, 2)).cpu()
            sum_var_r += (p_c.var(dim=2) / (t_c.var(dim=2) + 1e-8)).mean(dim=0).cpu()
            p_n = p_c - p_c.mean(dim=2, keepdim=True)
            t_n = t_c - t_c.mean(dim=2, keepdim=True)
            corr = (p_n * t_n).sum(dim=2) / (p_n.norm(dim=2) * t_n.norm(dim=2) + 1e-8)
            sum_corr += corr.mean(dim=0).cpu()
            count += 1

    names = ['lll', 'llh', 'lhl', 'lhh', 'hll', 'hlh', 'hhl', 'hhh']
    print("\nSub-band diagnostics")
    print(f"{'Band':<6} {'MSE':>10} {'VarRatio':>10} {'Corr':>8}")
    for c in range(8):
        print(f"{names[c]:<6} {sum_mse[c]/count:>10.4e}"
              f"{sum_var_r[c]/count:>10.4f}"
              f"{sum_corr[c]/count:>8.4f}")

    return {'mse': (sum_mse / count).numpy(), 'var': (sum_var_r / count).numpy(), 'corr': (sum_corr / count).numpy()}

# Function to log the mean gradient norm of each block of the network
def log_gradient_norms(epoch, norms: dict):
    groups = defaultdict(list)
    for name, norm in norms.items():
        parts = name.split('.')
        if parts[0] in ('down_blocks', 'up_blocks'):
            groups[f"{parts[0]}.{parts[1]}"].append(norm)
        else:
            groups[parts[0]].append(norm)

    print(f"\nGradient norms epoch {epoch}")
    mean_norms = {}
    for key, vals in groups.items():
        mean_norms[key] = float(np.mean(vals))
        print(f"{key:<20}: mean={mean_norms[key]:.2e}  max={np.max(vals):.2e}")
    return mean_norms
