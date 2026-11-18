import numpy as np
import os
import config
import math
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pickle
from dataset_helpers import Logging, Normalize, QuijoteSuperresDataset
from helpers import DWT, IDWT, power_spectrum_3d_torch, equilateral, angle_scan, squeezed, bispectrum, high_band_mask, pk_r_T
from plotting_stats import hist_bands_plotter, hist_bands_plotter_dwt, input_fields, plot_dwt, ps_plots_input, dwt_contr
from plotting_stats import cross_corr_plot, transfer_plot
from plotting_stats import bispectrum_eq_plot, bispectrum_angle_plot


# Computes statistics over the dataset for Logging
def parallel_stats(idx, data):
    hr, lr, p = data[idx]
    hr_np = hr.squeeze().numpy()
    lr_np = lr.squeeze().numpy()

    return{
        "sum_h": hr_np.sum(),
        "sum_l": lr_np.sum(),
        'sq_sum_h': (hr_np ** 2).sum(),
        'sq_sum_l': (lr_np ** 2).sum(),
        "count_h": hr_np.size,
        "count_l": lr_np.size,
        "max_h": hr_np.max(),
        "max_l": lr_np.max(),
        'params': p.numpy()
    }

# Function to compute plotting histograms and DWT Normalization
def parallel_hists(idx, data, bins_l, bins_h, bins_log_h, bins_log_l, bins_dwt_h, bins_dwt_l, hi_mask):
    hr, lr, _ = data[idx]
    log = Logging()
    hr_log = log(hr)
    lr_log = log(lr)
    hr_dwt = DWT(hr_log.unsqueeze(0)).squeeze(0)
    lr_dwt = DWT(lr_log.unsqueeze(0)).squeeze(0)
    hr_np = hr.squeeze().numpy()
    lr_np = lr.squeeze().numpy()
  
    hr_hist, _ = np.histogram(hr_np, bins=bins_h, density = True)
    lr_hist, _ = np.histogram(lr_np, bins=bins_l, density = True)
    hr_log_hist, _ = np.histogram(hr_log.squeeze().numpy(), bins=bins_log_h, density = True)
    lr_log_hist, _ = np.histogram(lr_log.squeeze().numpy(), bins=bins_log_l, density = True)
    hr_dwt_hist = [r[0] for r in [np.histogram(hr_dwt[i].squeeze().numpy(), bins=bins_dwt_h[i]) for i in range(hr_dwt.shape[0])]]
    lr_dwt_hist = [r[0] for r in [np.histogram(lr_dwt[i].squeeze().numpy(), bins=bins_dwt_l[i]) for i in range(lr_dwt.shape[0])]]
    return{
        'hr_hist': hr_hist,
        'lr_hist': lr_hist,
        'hr_log_hist': hr_log_hist,
        'lr_log_hist': lr_log_hist,
        'hr_dwt_hist': hr_dwt_hist,
        'lr_dwt_hist': lr_dwt_hist,
        'dwt_sum_h': hr_dwt.sum(axis=(1, 2, 3)),
        'dwt_sum_l': lr_dwt.sum(axis=(1, 2, 3)),
        'dwt_sq_sum_h': (hr_dwt ** 2).sum(axis=(1, 2, 3)),
        'dwt_sq_sum_l': (lr_dwt ** 2).sum(axis=(1, 2, 3)),
        'sigma_hi_sq': float(torch.fft.irfftn(torch.fft.rfftn(hr_log[0]) * hi_mask, s=(128,)*3).pow(2).mean())
    }

# Function to compute DWT Distributions Histograms
def parallel_hists_dwt(idx, data, bins_dwt_h, bins_dwt_l, mean_dwt_h, std_dwt_h, mean_dwt_l, std_dwt_l):
    hr, lr, _ = data[idx]
    log = Logging()
    hr_log = log(hr)
    lr_log = log(lr)
    hr_dwt = DWT(hr_log.unsqueeze(0)).squeeze(0)
    lr_dwt = DWT(lr_log.unsqueeze(0)).squeeze(0)
    norm_h = Normalize(mean_dwt_h, std_dwt_h)
    norm_l = Normalize(mean_dwt_l, std_dwt_l)

    hr_dwt_norm = norm_h(hr_dwt)
    lr_dwt_norm = norm_l(lr_dwt)    

    hr_dwt_hist = [r[0] for r in [np.histogram(hr_dwt_norm[i].squeeze().numpy(), bins=bins_dwt_h[i]) for i in range(hr_dwt_norm.shape[0])]]
    lr_dwt_hist = [r[0] for r in [np.histogram(lr_dwt_norm[i].squeeze().numpy(), bins=bins_dwt_l[i]) for i in range(lr_dwt_norm.shape[0])]]
    
    return{
        'hr_dwt_hist': hr_dwt_hist,
        'lr_dwt_hist': lr_dwt_hist,
    }

# Function to compute the dwt bins
def dwt_bins(cube_dwt, n_bins=257, phi=99.9):
    edges = []
    for i in range(cube_dwt.shape[0]):
        x = cube_dwt[i].flatten().numpy()
        if i == 0:
            lo, hi = np.percentile(x, 100 - phi), np.percentile(x, phi)
        else:
            r = np.percentile(np.abs(x), phi)
            lo, hi = -r, r
        edges.append(np.linspace(lo, hi, n_bins + 1))
    return edges

def spectra_vs_reference(fields, ref, BoxSize, clump=20):
    out = {}
    ref_t = torch.as_tensor(ref, dtype=torch.float32)
    for label, fld in fields.items():
        k, P, r, T = pk_r_T(torch.as_tensor(fld, dtype=torch.float32), ref_t, BoxSize, clump)
        out[label] = dict(P=P, r=r, T=T)
    _, p_ref, _, _ = power_spectrum_3d_torch(ref_t, ref_t, BoxSize, MAS_index=4, clump=clump)
    return k, p_ref.numpy(), out
 
def compute_precomputed_stats(all_params, hr_template, lr_template, train_split, val_split, num_workers=4):
    load_from_file = True
    if load_from_file and os.path.exists('param_stats'):
        with open('param_stats', 'rb') as file:
            stats_params = pickle.load(file)
            
        normalize_h = Normalize(stats_params['mean_dwt_hr'], stats_params['std_dwt_hr'])
        normalize_l = Normalize(stats_params['mean_dwt_lr'], stats_params['std_dwt_lr'])
        logging = Logging()

        bins = np.load('bins.npy')
        bins_h_dwt_norm = np.load('bins_h_norm.npy')
        bins_l_dwt_norm = np.load('bins_l_norm.npy')
        bins_h_dwt = np.load('bins_h.npy')
        bins_l_dwt = np.load('bins_l.npy')
        bins_log = np.load('bins_log.npy')
        means = np.load('means.npy', allow_pickle=True).item()
        means_log = np.load('means_log.npy', allow_pickle=True).item()
        means_dwt_norm = np.load('mean_norm.npy', allow_pickle=True).item()
        means_dwt = np.load('mean_dwt.npy', allow_pickle=True).item()
        bands = np.load('bands.npy', allow_pickle=True).item()
        bands_dwt_norm = np.load('bands_norm.npy', allow_pickle=True).item()
        bands_dwt = np.load('bands_dwt.npy', allow_pickle=True).item()
        bands_log = np.load('bands_log.npy', allow_pickle=True).item()
    
    else:
    
        num_simulations = len(all_params)
        indices = list(range(num_simulations))
        np.random.seed(42)
        np.random.shuffle(indices)
        n_train = int(num_simulations * train_split)
        n_val = int(num_simulations * val_split)
        train_indices, val_indices, test_indices = indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]
        train_set = set(train_indices)
        temp_dataset = QuijoteSuperresDataset(list(range(num_simulations)), all_params, hr_template, lr_template, dwt_flag = False)
        print("Calculating preprocessing statistics...")
        results = Parallel(n_jobs=num_workers)(delayed(parallel_stats)(i, temp_dataset) 
                                               for i in tqdm(range(2000), desc="Calculating Parameters"))

        res_train = [r for i, r in enumerate(results) if i in train_set]

        # HR computations
        summ_h = sum(r['sum_h'] for r in res_train) # sum of all values across sims
        sq_sum_h = sum(r['sq_sum_h'] for r in res_train) # sum of all squaredvalues across sims
        count_h = sum(r['count_h'] for r in res_train)
        mean_array_h = np.array([r['sum_h'] for r in results]) / np.array([r['count_h'] for r in results])
        std_array_h = np.sqrt(np.array([r['sq_sum_h'] for r in results]) / np.array([r['count_h'] for r in results]) - mean_array_h ** 2)
        data_mean_h = summ_h / count_h
        data_std_h  = np.sqrt(sq_sum_h / count_h - data_mean_h ** 2)
        max_array_h = np.array([r['max_h'] for r in res_train])

  
        train_positions = [i for i in range(2000) if i in train_set]
        max_pos_h = train_positions[int(np.argmax(max_array_h))]
        hr_max_cube , _, _ = temp_dataset[max_pos_h]
        logging = Logging()
        hr_max_log = logging(hr_max_cube)
        hr_max_dwt = DWT(hr_max_log.unsqueeze(0)).squeeze(0)
        _, bins_h = np.histogram(hr_max_cube, bins=int(np.ceil(np.sqrt(hr_max_cube.numel()))))
        _, bins_h_log = np.histogram(hr_max_log, bins=int(np.ceil(np.sqrt(hr_max_log.numel()))))
        bins_h_dwt = dwt_bins(hr_max_dwt)


        # LR computations
        summ_l = sum(r['sum_l'] for r in res_train)
        sq_sum_l = sum(r['sq_sum_l'] for r in res_train)
        count_l = sum(r['count_l'] for r in res_train)
        mean_array_l = np.array([r['sum_l'] for r in results]) / np.array([r['count_l'] for r in results])
        std_array_l = np.sqrt(np.array([r['sq_sum_l'] for r in results]) / np.array([r['count_l'] for r in results]) - mean_array_l ** 2)

        data_mean_l = summ_l / count_l
        data_std_l  = np.sqrt(sq_sum_l / count_l - data_mean_l ** 2)
        max_array_l = np.array([r['max_l'] for r in res_train])
        max_pos_l = train_positions[int(np.argmax(max_array_l))]

        _ , lr_max_cube, _ = temp_dataset[max_pos_l]
        #logging_l = Logging(c_h, s_h, hr_max)
        lr_max_log = logging(lr_max_cube)
        lr_max_dwt = DWT(lr_max_log.unsqueeze(0)).squeeze(0)
        _, bins_l = np.histogram(lr_max_cube, bins=int(np.ceil(np.sqrt(lr_max_cube.numel()))))
        _, bins_l_log = np.histogram(lr_max_log, bins=int(np.ceil(np.sqrt(lr_max_log.numel()))))
        bins_l_dwt = dwt_bins(lr_max_dwt)


        hi_mask = high_band_mask(128, 64)
        print('Histograms for plotting')
        results_h = Parallel(n_jobs=num_workers)(delayed(parallel_hists)(i, temp_dataset, bins_l, bins_h, bins_h_log, bins_l_log, bins_h_dwt, bins_l_dwt, hi_mask)
                                               for i in tqdm(range(2000), desc="Creating Histograms"))

        res_train_h = [r for i, r in enumerate(results_h) if i in train_set]

        sigma_hi = np.sqrt(np.mean([r['sigma_hi_sq'] for r in res_train_h]))

        count_dwt = 64**3 * n_train

        # Normalization

        dwt_sum_h = sum(r['dwt_sum_h'] for r in res_train_h)
        dwt_sq_sum_h = sum(r['dwt_sq_sum_h'] for r in res_train_h)
        dwt_sum_l = sum(r['dwt_sum_l'] for r in res_train_h)
        dwt_sq_sum_l = sum(r['dwt_sq_sum_l'] for r in res_train_h)
        mean_dwt_h = np.array(dwt_sum_h) / count_dwt
        std_dwt_h  = np.sqrt(dwt_sq_sum_h / count_dwt - mean_dwt_h ** 2)
        mean_dwt_l = np.array(dwt_sum_l) / count_dwt
        std_dwt_l  = np.sqrt(dwt_sq_sum_l / count_dwt - mean_dwt_l ** 2)
        normalize_h = Normalize(mean_dwt_h, std_dwt_h)
        normalize_l = Normalize(mean_dwt_l, std_dwt_l)
        hr_max_dwt_norm = normalize_h(hr_max_dwt)
        lr_max_dwt_norm = normalize_l(lr_max_dwt)
        bins_h_dwt_norm = dwt_bins(hr_max_dwt_norm)
        bins_l_dwt_norm = dwt_bins(lr_max_dwt_norm)

        print('Computing DWT Hists')
        results_dwt = Parallel(n_jobs=num_workers)(delayed(parallel_hists_dwt)(i, temp_dataset, bins_h_dwt_norm, bins_l_dwt_norm, mean_dwt_h, std_dwt_h, mean_dwt_l, std_dwt_l)
                                               for i in tqdm(range(2000), desc="Computing Hists"))


        # Params Computations
        all_params_tr = torch.stack([torch.from_numpy(r['params']) for r in res_train])
        all_params_tot = torch.stack([torch.from_numpy(r['params']) for r in results])
        param_mean, param_std = torch.mean(all_params_tr, dim=0), torch.std(all_params_tr, dim=0)

        print('Saving Parameters')
        stats_params = {
            'train_indices': train_indices,
            'val_indices':   val_indices,
            'test_indices':  test_indices,
            'data_mean_h':   float(data_mean_h),
            'data_std_h':    float(data_std_h),
            'data_mean_l':   float(data_mean_l),
            'data_std_l':    float(data_std_l),
            'param_mean':    param_mean,
            'param_std':     param_std,
            'mean_dwt_hr':   mean_dwt_h,
            'std_dwt_hr':    std_dwt_h,
            'mean_dwt_lr':   mean_dwt_l,
            'std_dwt_lr':    std_dwt_l,
            'sigma_hi':     sigma_hi
        }

        file = open('param_stats', 'wb')
        pickle.dump(stats_params, file)
        file.close()

        print('Plotting')
        bins = (bins_h, bins_l)
        bins_log = (bins_h_log, bins_l_log)

        means = {'hr_tr': np.mean(np.array([r['hr_hist'] for r in results_h])[train_indices], axis = 0), 
                 'lr_tr': np.mean(np.array([r['lr_hist'] for r in results_h])[train_indices], axis = 0),
                 'hr_v': np.mean(np.array([r['hr_hist'] for r in results_h])[val_indices], axis = 0),
                 'lr_v': np.mean(np.array([r['lr_hist'] for r in results_h])[val_indices], axis = 0),
                 'hr_ts': np.mean(np.array([r['hr_hist'] for r in results_h])[test_indices], axis = 0),
                 'lr_ts': np.mean(np.array([r['lr_hist'] for r in results_h])[test_indices], axis = 0)
                 }

        means_log = {'hr_tr': np.mean(np.array([r['hr_log_hist'] for r in results_h])[train_indices], axis = 0), 
                 'lr_tr': np.mean(np.array([r['lr_log_hist'] for r in results_h])[train_indices], axis = 0),
                 'hr_v': np.mean(np.array([r['hr_log_hist'] for r in results_h])[val_indices], axis = 0),
                 'lr_v': np.mean(np.array([r['lr_log_hist'] for r in results_h])[val_indices], axis = 0),
                 'hr_ts': np.mean(np.array([r['hr_log_hist'] for r in results_h])[test_indices], axis = 0),
                 'lr_ts': np.mean(np.array([r['lr_log_hist'] for r in results_h])[test_indices], axis = 0)
                 }

        means_dwt = {'hr_tr': np.mean(np.array([r['hr_dwt_hist'] for r in results_h])[train_indices], axis = 0), 
                 'lr_tr': np.mean(np.array([r['lr_dwt_hist'] for r in results_h])[train_indices], axis = 0),
                 'hr_v': np.mean(np.array([r['hr_dwt_hist'] for r in results_h])[val_indices], axis = 0),
                 'lr_v': np.mean(np.array([r['lr_dwt_hist'] for r in results_h])[val_indices], axis = 0),
                 'hr_ts': np.mean(np.array([r['hr_dwt_hist'] for r in results_h])[test_indices], axis = 0),
                 'lr_ts': np.mean(np.array([r['lr_dwt_hist'] for r in results_h])[test_indices], axis = 0)
                 }

        means_dwt_norm = {'hr_tr': np.mean(np.array([r['hr_dwt_hist'] for r in results_dwt])[train_indices], axis = 0), 
                 'lr_tr': np.mean(np.array([r['lr_dwt_hist'] for r in results_dwt])[train_indices], axis = 0),
                 'hr_v': np.mean(np.array([r['hr_dwt_hist'] for r in results_dwt])[val_indices], axis = 0),
                 'lr_v': np.mean(np.array([r['lr_dwt_hist'] for r in results_dwt])[val_indices], axis = 0),
                 'hr_ts': np.mean(np.array([r['hr_dwt_hist'] for r in results_dwt])[test_indices], axis = 0),
                 'lr_ts': np.mean(np.array([r['lr_dwt_hist'] for r in results_dwt])[test_indices], axis = 0)
                 }


        bands = {'hr_tr': (means['hr_tr'] - np.std(np.array([r['hr_hist'] for r in results_h])[train_indices], axis = 0),
                            means['hr_tr'] + np.std(np.array([r['hr_hist'] for r in results_h])[train_indices], axis = 0)),
                 'lr_tr': (means['lr_tr'] - np.std(np.array([r['lr_hist'] for r in results_h])[train_indices], axis = 0),
                            means['lr_tr'] + np.std(np.array([r['lr_hist'] for r in results_h])[train_indices], axis = 0)),
                 'hr_v': (means['hr_v'] - np.std(np.array([r['hr_hist'] for r in results_h])[val_indices], axis = 0),
                            means['hr_v'] + np.std(np.array([r['hr_hist'] for r in results_h])[val_indices], axis = 0)),
                 'lr_v': (means['lr_v'] - np.std(np.array([r['lr_hist'] for r in results_h])[val_indices], axis = 0),
                            means['lr_v'] + np.std(np.array([r['lr_hist'] for r in results_h])[val_indices], axis = 0)),
                 'hr_ts': (means['hr_ts'] - np.std(np.array([r['hr_hist'] for r in results_h])[test_indices], axis = 0),
                            means['hr_ts'] + np.std(np.array([r['hr_hist'] for r in results_h])[test_indices], axis = 0)),
                 'lr_ts': (means['lr_ts'] - np.std(np.array([r['lr_hist'] for r in results_h])[test_indices], axis = 0),
                            means['lr_ts'] + np.std(np.array([r['lr_hist'] for r in results_h])[test_indices], axis = 0))
                }

        bands_log = {'hr_tr': (means_log['hr_tr'] - np.std(np.array([r['hr_log_hist'] for r in results_h])[train_indices], axis = 0),
                               means_log['hr_tr'] + np.std(np.array([r['hr_log_hist'] for r in results_h])[train_indices], axis = 0)),
                     'lr_tr': (means_log['lr_tr'] - np.std(np.array([r['lr_log_hist'] for r in results_h])[train_indices], axis = 0),
                                   means_log['lr_tr'] + np.std(np.array([r['lr_log_hist'] for r in results_h])[train_indices], axis = 0)),
                     'hr_v': (means_log['hr_v'] - np.std(np.array([r['hr_log_hist'] for r in results_h])[val_indices], axis = 0),
                                  means_log['hr_v'] + np.std(np.array([r['hr_log_hist'] for r in results_h])[val_indices], axis = 0)),
                 'lr_v': (means_log['lr_v'] - np.std(np.array([r['lr_log_hist'] for r in results_h])[val_indices], axis = 0),
                              means_log['lr_v'] + np.std(np.array([r['lr_log_hist'] for r in results_h])[val_indices], axis = 0)),
                 'hr_ts': (means_log['hr_ts'] - np.std(np.array([r['hr_log_hist'] for r in results_h])[test_indices], axis = 0),
                               means_log['hr_ts'] + np.std(np.array([r['hr_log_hist'] for r in results_h])[test_indices], axis = 0)),
                 'lr_ts': (means_log['lr_ts'] - np.std(np.array([r['lr_log_hist'] for r in results_h])[test_indices], axis = 0),
                               means_log['lr_ts'] + np.std(np.array([r['lr_log_hist'] for r in results_h])[test_indices], axis = 0))
                }

        bands_dwt = {'hr_tr': (means_dwt['hr_tr'] - np.std(np.array([r['hr_dwt_hist'] for r in results_h])[train_indices], axis = 0),
                            means_dwt['hr_tr'] + np.std(np.array([r['hr_dwt_hist'] for r in results_h])[train_indices], axis = 0)),
                 'lr_tr': (means_dwt['lr_tr'] - np.std(np.array([r['lr_dwt_hist'] for r in results_h])[train_indices], axis = 0),
                            means_dwt['lr_tr'] + np.std(np.array([r['lr_dwt_hist'] for r in results_h])[train_indices], axis = 0)),
                 'hr_v': (means_dwt['hr_v'] - np.std(np.array([r['hr_dwt_hist'] for r in results_h])[val_indices], axis = 0),
                            means_dwt['hr_v'] + np.std(np.array([r['hr_dwt_hist'] for r in results_h])[val_indices], axis = 0)),
                 'lr_v': (means_dwt['lr_v'] - np.std(np.array([r['lr_dwt_hist'] for r in results_h])[val_indices], axis = 0),
                            means_dwt['lr_v'] + np.std(np.array([r['lr_dwt_hist'] for r in results_h])[val_indices], axis = 0)),
                 'hr_ts': (means_dwt['hr_ts'] - np.std(np.array([r['hr_dwt_hist'] for r in results_h])[test_indices], axis = 0),
                            means_dwt['hr_ts'] + np.std(np.array([r['hr_dwt_hist'] for r in results_h])[test_indices], axis = 0)),
                 'lr_ts': (means_dwt['lr_ts'] - np.std(np.array([r['lr_dwt_hist'] for r in results_h])[test_indices], axis = 0),
                            means_dwt['lr_ts'] + np.std(np.array([r['lr_dwt_hist'] for r in results_h])[test_indices], axis = 0))
                }

        bands_dwt_norm = {'hr_tr': (means_dwt_norm['hr_tr'] - np.std(np.array([r['hr_dwt_hist'] for r in results_dwt])[train_indices], axis = 0),
                            means_dwt_norm['hr_tr'] + np.std(np.array([r['hr_dwt_hist'] for r in results_dwt])[train_indices], axis = 0)),
                 'lr_tr': (means_dwt_norm['lr_tr'] - np.std(np.array([r['lr_dwt_hist'] for r in results_dwt])[train_indices], axis = 0),
                            means_dwt_norm['lr_tr'] + np.std(np.array([r['lr_dwt_hist'] for r in results_dwt])[train_indices], axis = 0)),
                 'hr_v': (means_dwt_norm['hr_v'] - np.std(np.array([r['hr_dwt_hist'] for r in results_dwt])[val_indices], axis = 0),
                            means_dwt_norm['hr_v'] + np.std(np.array([r['hr_dwt_hist'] for r in results_dwt])[val_indices], axis = 0)),
                 'lr_v': (means_dwt_norm['lr_v'] - np.std(np.array([r['lr_dwt_hist'] for r in results_dwt])[val_indices], axis = 0),
                            means_dwt_norm['lr_v'] + np.std(np.array([r['lr_dwt_hist'] for r in results_dwt])[val_indices], axis = 0)),
                 'hr_ts': (means_dwt_norm['hr_ts'] - np.std(np.array([r['hr_dwt_hist'] for r in results_dwt])[test_indices], axis = 0),
                            means_dwt_norm['hr_ts'] + np.std(np.array([r['hr_dwt_hist'] for r in results_dwt])[test_indices], axis = 0)),
                 'lr_ts': (means_dwt_norm['lr_ts'] - np.std(np.array([r['lr_dwt_hist'] for r in results_dwt])[test_indices], axis = 0),
                            means_dwt_norm['lr_ts'] + np.std(np.array([r['lr_dwt_hist'] for r in results_dwt])[test_indices], axis = 0))
                }


        np.save('bins_h_norm.npy',bins_h_dwt_norm)
        np.save('bins_l_norm.npy',bins_l_dwt_norm)
        np.save('mean_norm.npy',means_dwt_norm, allow_pickle=True)
        np.save('bands_norm.npy',bands_dwt_norm, allow_pickle=True)
        np.save('bins_h.npy',bins_h_dwt)
        np.save('bins_l.npy',bins_l_dwt)
        np.save('bins.npy',bins)
        np.save('bins_log.npy',bins_log)
        np.save('mean_dwt.npy',means_dwt, allow_pickle=True)
        np.save('bands_dwt.npy',bands_dwt, allow_pickle=True)
        np.save('means_log.npy',means_log, allow_pickle=True)
        np.save('bands_log.npy',bands_log, allow_pickle=True)
        np.save('means.npy',means, allow_pickle=True)
        np.save('bands.npy',bands, allow_pickle=True)
        print(f"Params: Mean={param_mean}, Std={param_std}")
 
    hist_bands_plotter(means, means_log, bands, bands_log, bins, bins_log, "Data Distribution", "histogram_bands.png")
    #hist_bands_plotter(means_log, bands_log, bins_log, "Data Distribution Logged", "histogram_log_bands.png")
    hist_bands_plotter_dwt(means_dwt, bands_dwt, bins_h_dwt, 'DWT Distribution HR', 'Dwt_hr_hist.png', 'hr', log_scale_x = False, log_scale_y = False)
    hist_bands_plotter_dwt(means_dwt, bands_dwt, bins_l_dwt, 'DWT Distribution LR', 'Dwt_lr_hist.png', 'lr', log_scale_x = False, log_scale_y = False)
    hist_bands_plotter_dwt(means_dwt_norm, bands_dwt_norm, bins_h_dwt_norm, 'DWT Distribution HR Normed', 'Dwt_hr_norm_hist.png', 'hr', log_scale_x = False, log_scale_y = False)
    hist_bands_plotter_dwt(means_dwt_norm, bands_dwt_norm, bins_l_dwt_norm, 'DWT Distribution LR Normed', 'Dwt_lr_norm_hist.png', 'lr', log_scale_x = False, log_scale_y = False)
    #histogram_plotter(mean_array_h, train_indices, val_indices, test_indices, "Mean Distribution".format(percentile), "mean_hr_histogram.png")
    #histogram_plotter(mean_array_l, train_indices, val_indices, test_indices, "Mean Distribution".format(percentile), "mean_lr_histogram.png")
    #histogram_plotter(std_array_h, train_indices, val_indices, test_indices, "Standard Deviation Distribution".format(percentile), "std_hr_histogram.png")
    #histogram_plotter(std_array_l, train_indices, val_indices, test_indices, "Standard Deviation Distribution".format(percentile), "std_lr_histogram.png")
    #histogram_plotter(np.array([r['percentiles_h'] for r in results]), train_indices, val_indices, test_indices, "{0} Percentiles Distribution".format(percentile), "perc_hr_histogram.png")
    #histogram_plotter(np.array([r['percentiles_l'] for r in results]), train_indices, val_indices, test_indices, "{0} Percentiles Distribution".format(percentile), "perc_lr_histogram.png")
    #histogram_plotter(np.array([r['max_h'] for r in results]), train_indices, val_indices, test_indices, "Max HR Values Distribution", "max_hr_histogram.png")
    #histogram_plotter(np.array([r['max_l'] for r in results]), train_indices, val_indices, test_indices, "Max LR Values Distribution", "max_lr_histogram.png")
    #histogram_plotter(all_params_tot[:, 0], train_indices, val_indices, test_indices, "$\Omega_m$ Distribution", "om_histogram.png")
    #histogram_plotter(all_params_tot[:, 1], train_indices, val_indices, test_indices, "$\Omega_b$ Distribution", "ob_histogram.png")
    #histogram_plotter(all_params_tot[:, 2], train_indices, val_indices, test_indices, "$h$ Distribution", "h_histogram.png")
    #histogram_plotter(all_params_tot[:, 3], train_indices, val_indices, test_indices, "$n_s$ Distribution", "ns_histogram.png")
    #histogram_plotter(all_params_tot[:, 4], train_indices, val_indices, test_indices, "$\sigma_8$ Distribution", "sigma8_histogram.png")
            
    up_col = 1
    down_col = 1
    down_col_log = 1
    c_shift = +0.0
    idx = 663
    slice_idx = 80
 
    params = np.loadtxt('../../../../PCS/fourier3/data/latin_hypercube_params.txt')
    params_txt = '$\Omega_m$ = {0}, $\Omega_b$ = {1}, $h$ = {2}, $n_s$ = {3}, $\sigma_8$ = {4}'.format(*np.round(params[idx], 3))

    path_64 = f'../../../../PCS/df_m_{idx}_64_r_z=0.npy'
    path_128 = f'../../../../PCS/fourier3/data/df_m_{idx}_128_z=0.npy'
    path_64_aug = f'../../../../PCS/fourier3/data/df_m_{idx}_64_z=0.npy'
    df_64 = np.load(path_64)
    df_128 = np.load(path_128)
    df_64_aug = np.load(path_64_aug)
    df_64_repeated = np.repeat(np.repeat(np.repeat(df_64, 2, axis=0), 2, axis=1), 2, axis=2)
    df_64_trilinear = torch.nn.functional.interpolate(torch.tensor(df_64).unsqueeze(0).unsqueeze(0), size=(128,128,128), mode='trilinear', align_corners=False).squeeze().numpy()

    x = np.load('../../../../PCS/fourier3/data/df_m_663_64_z=0.npy')
    input_fields(df_64, df_64_aug, df_128, df_64_repeated, df_64_trilinear, logging, up_col, down_col_log, c_shift, slice_idx, params_txt, 'Input_fields.png')
    df_64_dwt = DWT(torch.tensor(df_64_aug).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
    df_128_dwt = DWT(torch.tensor(df_128).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
    df_64_back = IDWT(torch.tensor(df_64_dwt).unsqueeze(0)).squeeze().numpy()
    df_128_back = IDWT(torch.tensor(df_128_dwt).unsqueeze(0)).squeeze().numpy()
    df_64_normed = normalize_l(torch.tensor(df_64_dwt).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
    df_128_normed = normalize_h(torch.tensor(df_128_dwt).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
    
    plot_dwt(df_64_dwt, slice_idx, params_txt, 'DWT_fields_64.png')
    plot_dwt(df_128_dwt, slice_idx, params_txt, 'DWT_fields_128.png')
    plot_dwt(df_64_normed, slice_idx, params_txt, 'DWT_fields_64_normed.png')
    plot_dwt(df_128_normed, slice_idx, params_txt, 'DWT_fields_128_normed.png')

    clump = 20
    # Input fields PS plots
    fields_up = {"$64^3$ Repeated":  df_64_repeated,
                 "$64^3$ Trilinear": df_64_trilinear,
                 "$64^3$ Fourier":   df_64_aug,
                 "$64^3$ Back": df_64_back,
                 "$128^3$ Back": df_128_back}
    k_128, ps_128, spec = spectra_vs_reference(fields_up, df_128, config.BOX_SIZE, clump)
    #_,  ps_128_back, _, _ = power_spectrum_3d_torch(torch.tensor(df_128_back), torch.tensor(df_128_back), config.BOX_SIZE, clump)
    #_,  ps_64_back, _, _ = power_spectrum_3d_torch(torch.tensor(df_64_back), torch.tensor(df_64_back), config.BOX_SIZE, clump)

    ps_64_rep = spec["$64^3$ Repeated"]['P']
    ps_64_tri = spec["$64^3$ Trilinear"]['P']
    ps_64_aug = spec["$64^3$ Fourier"]['P']
    ps_64_back = spec["$64^3$ Back"]['P']
    ps_128_back = spec["$128^3$ Back"]['P']
    k_64,  ps_64,  _, _ = power_spectrum_3d_torch(torch.tensor(df_64), torch.tensor(df_64),
                                                  config.BOX_SIZE, MAS_index=4, clump=clump)
    # DWT round-trip checks unchanged
    ps_plots_input(k_128, k_64, ps_128, ps_64, ps_64_rep, ps_64_tri, ps_64_aug, ps_64_back, ps_128_back, 'PS_input.png')

    r_curves = {lab: s['r'] for lab, s in spec.items()}
    T_curves = {lab: s['T'] for lab, s in spec.items()}
    #r_curves["Fourier, half-voxel control"] = _r_and_T_shift(df_64_aug, df_128)  # optional
    cross_corr_plot(k_128, r_curves, 'cross_corr_input.png')
    transfer_plot(k_128, T_curves, 'transfer_input.png')

    dwt_ps_128 = []
    dwt_acc_128 = []
    dwt_ps_64 = []
    dwt_acc_64 = []
    indexes_swapped = [0, 1, 2, 4, 3, 5, 6, 7]
    indexes = [0, 1, 2, 3, 4, 5, 6, 7]

    for i in indexes_swapped:
        full_zero_128 = torch.zeros_like(torch.tensor(df_128_dwt))
        full_zero_64 = torch.zeros_like(torch.tensor(df_64_dwt))

        full_zero_128[indexes[i]] = torch.tensor(df_128_dwt)[i]
        full_zero_64[indexes[i]] = torch.tensor(df_64_dwt)[i]

        field_128 = IDWT(full_zero_128.unsqueeze(0)).squeeze(0).squeeze(0)
        field_64 = IDWT(full_zero_64.unsqueeze(0)).squeeze(0).squeeze(0)

        _, ps_i_128, _, _ = power_spectrum_3d_torch(field_128, field_128, config.BOX_SIZE, MAS_index = 4, clump = clump)
        _, ps_i_64, _, _ = power_spectrum_3d_torch(field_64, field_64, config.BOX_SIZE, MAS_index = 4, clump = clump)
        dwt_ps_128.append(ps_i_128)
        dwt_ps_64.append(ps_i_64)


    # DWT Subbands Contributions to PS
    acc_128 = torch.zeros_like(torch.tensor(df_128_dwt))
    acc_64 = torch.zeros_like(torch.tensor(df_64_dwt))
    for i in indexes_swapped:
        acc_128[indexes[i]] = torch.tensor(df_128_dwt)[i]    
        acc_64[indexes[i]] = torch.tensor(df_64_dwt)[i]                 
             
        field_128 = IDWT(acc_128.unsqueeze(0)).squeeze(0).squeeze(0)
        field__64 = IDWT(acc_64.unsqueeze(0)).squeeze(0).squeeze(0)
        _, ps_a_128, _, _ = power_spectrum_3d_torch(field_128, field_128, config.BOX_SIZE, MAS_index = 4, clump = clump)
        _, ps_a_64, _, _ = power_spectrum_3d_torch(field__64, field__64, config.BOX_SIZE, MAS_index = 4, clump = clump)
        dwt_acc_128.append(ps_a_128)
        dwt_acc_64.append(ps_a_64)
 
    dwt_contr(k_128, dwt_ps_128, dwt_acc_128, ps_128,'DWT_ind_128.png', 'Cumulative_dwt_128.png', "HR")
    dwt_contr(k_128, dwt_ps_64, dwt_acc_64, ps_64_aug,'DWT_ind_64.png', 'Cumulative_dwt_64.png', "LR")


    DK   = 3.0     
    BDEV = 'cpu'       

    fields_b = {
        "$128^3$":            df_128,
        "$64^3$ Fourier":     df_64_aug,
        "$64^3$ Trilinear":   df_64_trilinear,
        "$64^3$ Repeated":    df_64_repeated,
    }

    tri_eq = equilateral(n_min=4, n_max=60, dk=DK)
    res_eq = {lab: bispectrum(f, config.BOX_SIZE, tri_eq, dk=DK,
                                 MAS_index=4, clump=clump, device=BDEV)
              for lab, f in fields_b.items()}
    bispectrum_eq_plot(res_eq, 'bispectrum_equilateral.png', 'Equilateral bispectrum of the input fields')

    tri_ang, theta = angle_scan(n1=16, n2=28, n_theta=17, dk=DK)
    res_ang = {lab: bispectrum(f, config.BOX_SIZE, tri_ang, dk=DK,
                                  MAS_index=4, clump=clump, device=BDEV)
               for lab, f in fields_b.items()}
    kF = 2.0 * math.pi / config.BOX_SIZE
    bispectrum_angle_plot(res_ang, theta, 'bispectrum_angle.png', k1=16 * kF, k2=28 * kF, q_bool=False)

    tri_sq = squeezed(n_soft=6, n_min=8, n_max=56, dk=DK)
    res_sq = {lab: bispectrum(f, config.BOX_SIZE, tri_sq, dk=DK,
                                 MAS_index=4, clump=clump, device=BDEV)
              for lab, f in fields_b.items()}
    bispectrum_eq_plot(res_sq, 'bispectrum_squeezed.png', 'Squeezed bispectrum of the input fields')


    return stats_params
