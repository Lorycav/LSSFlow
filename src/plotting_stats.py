import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize, LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from config import K_NYQ_LR, K_NYQ_HR

BAND_NAMES = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
K_RANGES = [("k < kNyq_LR", -np.inf, K_NYQ_LR), ("generated band", K_NYQ_LR, K_NYQ_HR)]

# Color Palette
cosmic_indigo = LinearSegmentedColormap.from_list(
    "cosmic_indigo",
    [
        (0.00, "#000000"),
        (0.15, "#05050a"),
        (0.32, "#14142a"),
        (0.50, "#34214f"),
        (0.68, "#6a3d5a"),
        (0.82, "#a36a3a"),
        (0.93, "#e2c36b"),
        (1.00, "#ffffff"),
    ],
    N=256,
)

# Function to plot per simulation scalars distributions (like mean, max, std, percentiles, cosmological parameters)
def histogram_plotter(data, train_idx, val_idx, test_idx, title, filename, log_scale_x=False, log_scale_y=False):
    bins_t = int(np.ceil(np.sqrt(len(data))))
    bins_v = int(np.ceil(np.sqrt(len(data))))
    bins_ts = int(np.ceil(np.sqrt(len(data))))
    values_t,bin_edges_t = np.histogram(data[train_idx], bins=bins_t, density=True)
    values_v, bin_edges_v = np.histogram(data[val_idx], bins=bins_v, density=True)
    values_ts, bin_edges_ts = np.histogram(data[test_idx], bins=bins_ts, density=True)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(bin_edges_t[1:], values_t, alpha=0.7, label = 'Train')   
    ax.plot(bin_edges_v[1:], values_v, alpha=0.7, label = 'Validation')  
    ax.plot(bin_edges_ts[1:], values_ts, alpha=0.7, label = 'Test')  
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    if log_scale_x:
        ax.set_xscale('log')
    if log_scale_y:
        ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", linewidth=0.5) 
    plt.savefig(filename)

# Function to plot cubes distribution with std bands over the dataset
def hist_bands_plotter(data, data_log, bands, bands_log, bins, bins_log, title, filename, log_scale_x = True, log_scale_y = False):
    fig, ax = plt.subplots(figsize=(10, 8), nrows=2, ncols=1, sharex = True)
    fig.suptitle(title,fontsize=20)
    # HR
    ax[0].plot(bins[0][1:], data['hr_tr'], alpha=0.7, label = 'Train', color='lightblue')   
    ax[0].plot(bins[0][1:], data['hr_v'], alpha=0.7, label = 'Validation', color='lightgreen')  
    ax[0].plot(bins[0][1:], data['hr_ts'], alpha=0.7, label = 'Test', color='lightcoral')  
    ax[0].fill_between(bins[0][1:], bands['hr_tr'][0], bands['hr_tr'][1], color='lightblue', alpha=0.2)
    ax[0].fill_between(bins[0][1:], bands['hr_v'][0], bands['hr_v'][1], color='lightgreen', alpha=0.2)
    ax[0].fill_between(bins[0][1:], bands['hr_ts'][0], bands['hr_ts'][1], color='lightcoral', alpha=0.2)
    ax[0].axvline(-1, color='purple', linestyle='--', linewidth=1)
    ax[0].axvline(0,  color='purple', linestyle='--', linewidth=1)
    ax[0].axvline(1,  color='purple', linestyle='--', linewidth=1)
    # Log HR
    ax[0].plot(bins_log[0][1:], data_log['hr_tr'], alpha=0.7, label = 'Train Log', color='darkblue')   
    ax[0].plot(bins_log[0][1:], data_log['hr_v'], alpha=0.7, label = 'Validation Log', color='darkgreen')  
    ax[0].plot(bins_log[0][1:], data_log['hr_ts'], alpha=0.7, label = 'Test Log', color='darkred')  
    ax[0].fill_between(bins_log[0][1:], bands_log['hr_tr'][0], bands_log['hr_tr'][1], color='darkblue', alpha=0.2)
    ax[0].fill_between(bins_log[0][1:], bands_log['hr_v'][0], bands_log['hr_v'][1], color='darkgreen', alpha=0.2)
    ax[0].fill_between(bins_log[0][1:], bands_log['hr_ts'][0], bands_log['hr_ts'][1], color='darkred', alpha=0.2)
    ax[0].set_title('HR',fontsize=18)
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Density',fontsize=15)
    ax[0].legend(loc='best',fontsize=12)
    ax[0].grid(True, which="both", ls="--", linewidth=0.5)

    # LR
    ax[1].plot(bins[1][1:], data['lr_tr'], alpha=0.7, label = 'Train', color='lightblue')   
    ax[1].plot(bins[1][1:], data['lr_v'], alpha=0.7, label = 'Validation', color='lightgreen')  
    ax[1].plot(bins[1][1:], data['lr_ts'], alpha=0.7, label = 'Test', color='lightcoral')  
    ax[1].fill_between(bins[1][1:], bands['lr_tr'][0], bands['lr_tr'][1], color='lightblue', alpha=0.2)
    ax[1].fill_between(bins[1][1:], bands['lr_v'][0], bands['lr_v'][1], color='lightgreen', alpha=0.2)
    ax[1].fill_between(bins[1][1:], bands['lr_ts'][0], bands['lr_ts'][1], color='lightcoral', alpha=0.2)
    ax[1].axvline(-1, color='purple', linestyle='--', linewidth=1)
    ax[1].axvline(0,  color='purple', linestyle='--', linewidth=1)
    ax[1].axvline(1,  color='purple', linestyle='--', linewidth=1)
    # LR log
    ax[1].plot(bins_log[1][1:], data_log['lr_tr'], alpha=0.7, label = 'Train Log', color='darkblue')   
    ax[1].plot(bins_log[1][1:], data_log['lr_v'], alpha=0.7, label = 'Validation Log', color='darkgreen')  
    ax[1].plot(bins_log[1][1:], data_log['lr_ts'], alpha=0.7, label = 'Test Log', color='darkred')  
    ax[1].fill_between(bins_log[1][1:], bands_log['lr_tr'][0], bands_log['lr_tr'][1], color='darkblue', alpha=0.2)
    ax[1].fill_between(bins_log[1][1:], bands_log['lr_v'][0], bands_log['lr_v'][1], color='darkgreen', alpha=0.2)
    ax[1].fill_between(bins_log[1][1:], bands_log['lr_ts'][0], bands_log['lr_ts'][1], color='darkred', alpha=0.2)
    ax[1].set_title('LR',fontsize=18)
    ax[1].set_xlabel(r'$\delta$',fontsize=15)
    ax[1].set_ylabel('Density',fontsize=15)
    ax[1].legend(loc='best',fontsize=12)
    ax[1].grid(True, which="both", ls="--", linewidth=0.5)
   
    if log_scale_x:
        ticks = [-1, 0, 1, 10, 100]

        def format_tick(x, pos):
            if x == 0:
                return "0"
            elif abs(x) >= 1:
                return f"{x:g}"
            else:
                return f"{x:.2g}"

        for a in ax:
            a.set_xscale("asinh")
            a.xaxis.set_major_locator(FixedLocator(ticks))
            a.xaxis.set_major_formatter(FuncFormatter(format_tick))
    if log_scale_y:
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
    ax[0].tick_params(which='both',labelsize=12)
    ax[1].tick_params(which='both',labelsize=12)
    plt.savefig(filename)
    plt.close(fig)

# Function to plot DWT distributions per band
def hist_bands_plotter_dwt(data, bands, bins, title, filename, scale, log_scale_x = False, log_scale_y = False):
    fig, ax = plt.subplots(figsize=(20, 20), nrows=4, ncols=2)
    fig.suptitle(title, fontsize=35)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    tr_idx = scale + '_tr'
    v_idx = scale + '_v'
    ts_idx = scale + '_ts'
    for row in range(4):
        for col in range(2):
            idx = row * 2 + col
            #std
            bin_centers = (bins[idx][:-1] + bins[idx][1:]) / 2
            weighted_mean = np.average(bin_centers, weights=data[tr_idx][idx])
            std = np.sqrt(np.average((bin_centers - weighted_mean)**2, weights=data[tr_idx][idx]))
            
            ax[row, col].plot(bins[idx][1:], data[tr_idx][idx], alpha=0.7, label = 'Train', color='blue')   
            ax[row, col].plot(bins[idx][1:], data[v_idx][idx], alpha=0.7, label = 'Validation', color='green')  
            ax[row, col].plot(bins[idx][1:], data[ts_idx][idx], alpha=0.7, label = 'Test', color='red')  
            ax[row, col].fill_between(bins[idx][1:], bands[tr_idx][0][idx], bands[tr_idx][1][idx], color='blue', alpha=0.2)
            ax[row, col].fill_between(bins[idx][1:], bands[v_idx][0][idx], bands[v_idx][1][idx], color='green', alpha=0.2)
            ax[row, col].fill_between(bins[idx][1:], bands[ts_idx][0][idx], bands[ts_idx][1][idx], color='red', alpha=0.2)
            ax[row, col].set_title(BAND_NAMES[idx],fontsize=25)
            ax[row, col].set_xlabel('Value',fontsize=20)
            ax[row, col].set_ylabel('Frequency',fontsize=20)
            ax[row, col].set_xlim(weighted_mean-8*std, weighted_mean + 8*std)
            ax[row, col].legend(loc='best',fontsize=20)
            ax[row, col].grid(True, which="both", ls="--", linewidth=0.5)
            ax[row, col].vlines(x=-1, ymin=0, ymax=data[tr_idx][idx].max(), colors='purple', linestyles='dashed', label='Max ' + scale + ' Threshold')
            ax[row, col].vlines(x=0, ymin=0, ymax=data[tr_idx][idx].max(), colors='purple', linestyles='dashed', label='Max ' + scale + ' Threshold')
            ax[row, col].vlines(x=1, ymin=0, ymax=data[tr_idx][idx].max(), colors='purple', linestyles='dashed', label='Max ' + scale + ' Threshold')
            if log_scale_x:
                ax[row, col].set_xscale('asinh')
            if log_scale_y:
                ax[row, col].set_yscale('log')
            ax[row,col].tick_params(which='both',labelsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

# Function to plot the logged input fields (different upsamples)
def input_fields(df_64, df_64_aug, df_128, df_64_repeated, df_64_trilinear, data_transform, up_col, down_col_log, c_shift, slice_idx, params_txt, filename):

    df_64_log = data_transform(torch.tensor(df_64)).numpy()
    df_128_log = data_transform(torch.tensor(df_128)).numpy()
    df_64_aug_log = data_transform(torch.tensor(df_64_aug)).numpy()
    df_64_repeated_log = data_transform(torch.tensor(df_64_repeated)).numpy()
    df_64_trilinear_log = data_transform(torch.tensor(df_64_trilinear)).numpy()

    slice_64_log = df_64_log[slice_idx//2]
    slice_128_log = df_128_log[slice_idx]
    slice_64_aug_log = df_64_aug_log[slice_idx]
    slice_64_repeated_log = df_64_repeated_log[slice_idx]
    slice_64_trilinear_log = df_64_trilinear_log[slice_idx]

    norm_64_log = TwoSlopeNorm(vmin=np.min(slice_64_log)*down_col_log, vcenter=np.mean(slice_64_log)+c_shift, vmax=np.max(slice_64_log)*up_col)
    norm_128_log = TwoSlopeNorm(vmin=np.min(slice_128_log)*down_col_log, vcenter=np.mean(slice_128_log)+c_shift, vmax=np.max(slice_128_log)*up_col)
    norm_64_aug_log = TwoSlopeNorm(vmin=np.min(slice_64_aug_log)*down_col_log, vcenter=np.mean(slice_64_aug_log)+c_shift, vmax=np.max(slice_64_aug_log)*up_col)
    norm_64_repeated_log = TwoSlopeNorm(vmin=np.min(slice_64_repeated_log)*down_col_log, vcenter=np.mean(slice_64_repeated_log)+c_shift, vmax=np.max(slice_64_repeated_log)*up_col)
    norm_64_trilinear_log = TwoSlopeNorm(vmin=np.min(slice_64_trilinear_log)*down_col_log, vcenter=np.mean(slice_64_trilinear_log)+c_shift, vmax=np.max(slice_64_trilinear_log)*up_col)

    suptitle=f"2D slices of logged input density fields ({params_txt})"
    def show_with_cbar(ax, data, title, norm):
        im = ax.imshow(data, origin="lower", cmap=cosmic_indigo, norm=norm)
        ax.set_title(title,fontsize=20)
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
        cbar = ax.figure.colorbar(im, cax=cax)

        nticks = 5
        tmin = np.ceil(norm.vmin * 10) / 10
        tmax = np.floor(norm.vmax * 10) / 10
        if tmin >= tmax:
            ticks = np.array([norm.vmin, norm.vmax])
        else:
            grid = np.arange(tmin, tmax + 1e-9, 0.1)
            if nticks >= len(grid):
                ticks = grid
            else:
                idx = np.linspace(0, len(grid) - 1, nticks).round().astype(int)
                ticks = grid[idx]
                ticks = np.unique(ticks)

        cbar.set_ticks(ticks)

    # Three panels on the first row, two centred on the second
    layout = [["ax0", "ax0", "ax1", "ax1", "ax2", "ax2"],
              [".", "ax3", "ax3", "ax4", "ax4", "."]]
    fig, axs = plt.subplot_mosaic(layout, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(suptitle, fontsize=22)

    show_with_cbar(axs["ax0"], slice_64_log, "LR", norm_64_log)
    show_with_cbar(axs["ax1"], slice_64_repeated_log, "LR Repeated", norm_64_repeated_log)
    show_with_cbar(axs["ax2"], slice_64_trilinear_log, "LR Trilinear Interpolation", norm_64_trilinear_log)
    show_with_cbar(axs["ax3"], slice_64_aug_log, "LR Fourier Upsampled", norm_64_aug_log)
    show_with_cbar(axs["ax4"], slice_128_log, "HR", norm_128_log)
    fig.set_constrained_layout_pads(hspace=0.04, wspace=0.02, h_pad=0.15)

    plt.savefig(filename, dpi=130, bbox_inches="tight")
    plt.close(fig)

# Function to plot DWT bands examples
def plot_dwt(df_dwt, slice_idx, params_txt, filename):
    suptitle=f"2D slices of the DWT Transformed HR field ({params_txt})"
    
    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 9), constrained_layout=True)
    fig.suptitle(suptitle, fontsize=25,y=1.04)
    def show_with_cbar(ax, data, title, norm):
        im = ax.imshow(data, origin="lower", cmap=cosmic_indigo, norm=norm)
        ax.set_title(title,fontsize=24)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.tick_params(which='both',labelsize=14)
        cbar.ax.tick_params(which='both',labelsize=14)
    slice_dwt = df_dwt[:, slice_idx//2, :, :]
    norms = [TwoSlopeNorm(vmin=np.min(slice_dwt[i]), vcenter=np.mean(slice_dwt[i]), vmax=np.max(slice_dwt[i])) for i in range(len(slice_dwt))]
    for i, ax in enumerate(axs.flat):
        show_with_cbar(ax, slice_dwt[i], f"{BAND_NAMES[i]}", norms[i])

    plt.savefig(filename, dpi=130, bbox_inches="tight")
    plt.close(fig)

# Function to plot input fields PS (different upsamples)
def ps_plots_input(k_128, k_64, ps_128, ps_64, ps_64_rep, ps_64_tri, ps_64_aug, ps_64_back, ps_128_back, filename):

    fig, ax = plt.subplots(figsize=(12,9))
    ax.axvline(K_NYQ_LR, color="black", ls="--", lw=1.2,
                   label=r"$k_{\rm Nyq}^{\rm LR}=0.201$")
    ax.axvline(K_NYQ_HR, color="purple", ls="--", lw=1.2,
                   label=r"$k_{\rm Nyq}^{\rm HR}=0.402$")
    ax.plot(k_128,ps_128, label = '$128^3$', color = 'navy')
    ax.plot(k_128, ps_128_back, label = '$128^3$ DWT Check', linestyle = '--', color = 'dodgerblue')
    ax.plot(k_128,ps_64_rep, label = '$64^3$ Repeated', color = 'red')
    ax.plot(k_128,ps_64_tri, label = '$64^3$ Trilinear', color = 'orange')
    ax.plot(k_128,ps_64_aug, label = '$64^3$ Fourier', color = 'darkgreen')
    ax.plot(k_64,ps_64, label = '$64^3$', linestyle = '--', color = 'limegreen')
    ax.plot(k_128, ps_64_back, label = '$64^3$ DWT Check', linestyle = '--', color = 'limegreen')
    ax.set_title('Power Spectrum Comparison for Input Data', fontsize= 25)
    ax.set_ylabel('$P(k) \,\,\mathrm{[Mpc/h]^3}$', fontsize = 20)
    ax.set_xlabel('$k \,\,\mathrm{[h/Mpc]}$', fontsize=20)
    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(which='both',labelsize=14)
    ax.legend(loc='best',fontsize=17)
    plt.tight_layout()

    plt.savefig(filename)
    plt.close(fig)

# Function to plot dwt bands contributions to the PS
def dwt_contr(k, dwt_ps, dwt_acc, field, filename_1, filename_2, label):
    bands_swapped = ['LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HLH', 'HHL', 'HHH']

    fig, ax = plt.subplots(figsize=(16, 10))
    colors = plt.cm.plasma(np.linspace(0, 0.8, 8))
    ax.axvline(K_NYQ_LR, color="black", ls="--", lw=1.2,
                       label=r"$k_{\rm Nyq}^{\rm LR}=0.201$")
    ax.axvline(K_NYQ_HR, color="purple", ls="--", lw=1.2,
                       label=r"$k_{\rm Nyq}^{\rm HR}=0.402$")
    ax.plot(k, field, color='black', lw=2.5, label='Full field', zorder=10, ls = '--')
    for j, name in enumerate(bands_swapped):
        ax.plot(k, dwt_ps[j], color = colors[len(BAND_NAMES)-1 - j], label=name)
    ax.set_xlabel('k [h/Mpc]', fontsize = 24)
    ax.set_ylabel('P(k)', fontsize = 24) 
    ax.set_title(f'DWT Subbands Individual Contributions to {label} power spectrum', fontsize = 30, y=1.03)
    ax.legend(fontsize=18, ncol=2, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(which='both',labelsize=16)
    '''
    axins = ax.inset_axes([0.35, 0.05, 0.45, 0.45])
    axins.plot(k, field, color='black', lw=2.0, zorder=10, ls='--')
    for j, name in enumerate(bands_swapped):
        lw = 1.8 if name in ('LLL', 'HHH') else 0.9
        axins.plot(k, dwt_ps[j], color = colors[len(bands)-1 - j], lw=lw)
    axins.set_xlim(0.3, 0.8)
    axins.set_ylim(2, 7e3)
    axins.tick_params(labelsize=7)
    axins.grid(True, which='both', alpha=0.3)
    axins.set_title('High-k', fontsize=12)
    axins.set_yscale('log')
    axins.set_xscale('log')
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.8)
    '''
    plt.tight_layout()
    plt.savefig(filename_1, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 10))
    colors = plt.cm.plasma(np.linspace(0, 1, 8))
    ax.axvline(K_NYQ_LR, color="black", ls="--", lw=1.2,
                       label=r"$k_{\rm Nyq}^{\rm LR}=0.201$")
    ax.axvline(K_NYQ_HR, color="purple", ls="--", lw=1.2,
                       label=r"$k_{\rm Nyq}^{\rm HR}=0.402$")
    ax.plot(k, field, color='black', lw=2.5, ls='--', label='Full (target)', zorder=10)
    for j, name in enumerate(bands_swapped):
        label = '+' + name if j > 0 else name
        ax.plot(k, dwt_acc[j], color=colors[j], lw=1.3, label=label)
    ax.set_xlabel('k [h/Mpc]',fontsize=25)
    ax.set_ylabel('P(k)', fontsize=25)
    ax.set_title('Cumulative Reconstruction of Full Power Spectrum from DWT subbands', fontsize=30)
    ax.legend(fontsize=20, ncol=2, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(which='both',labelsize=18)
    '''
    axins = inset_axes(ax, width="60%", height="60%", loc='lower center',borderpad=1.2)
    axins.plot(k, field, color='black', lw=3.8, ls='--', zorder=10)
    axins.loglog(k)
    for j, name in enumerate(bands_swapped):
        axins.plot(k, dwt_acc[j], color=colors[j], lw = 2)
    axins.set_xlim(0.3, 0.8)
    axins.set_ylim(4, 7e3)
    axins.tick_params(labelsize=7)
    axins.grid(True, which='both', alpha=0.3)
    axins.set_title('High-k', fontsize=10)
    axins.set_yscale('log')
    axins.set_xscale('log')
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.8)
    '''
    plt.tight_layout()
    plt.savefig(filename_2, dpi=130)
    plt.close(fig)


# Function to plot the cross-correlation coefficient r(k) with the HR field
def cross_corr_plot(k, curves, filename, title="Cross-correlation of upsampled LR with HR"):
    fig, ax = plt.subplots(figsize=(10, 9))
    for label, r in curves.items():
        ax.plot(k, r, lw=1.8, label=label)
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.axvline(K_NYQ_LR, color="crimson", ls="--", lw=1.2, label=r"$k_{\rm Nyq}^{\rm LR}=0.201$")
    ax.axvline(K_NYQ_HR, color="purple", ls="--", lw=1.2, label=r"$k_{\rm Nyq}^{\rm HR}=0.402$")
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]", fontsize=14)
    ax.set_ylabel("r(k)", fontsize=14)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=18)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)

# Function to plot the transfer function T(k) with respect to the HR field
def transfer_plot(k, curves, filename, title=r"Transfer function $T(k)=\sqrt{P_{\rm up}/P_{\rm HR}}$"):
    fig, ax = plt.subplots(figsize=(10, 9))
    for label, T in curves.items():
        ax.plot(k, T, lw=1.8, label=label)
    ax.axhline(1.0, color="grey", lw=1.0)
    ax.axvline(K_NYQ_LR, color="crimson", ls="--", lw=1.2, label=r"$k_{\rm Nyq}^{\rm LR}$")
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]", fontsize=14)
    ax.set_ylabel("T(k)", fontsize=14)
    ax.set_ylim(0.0, 1.5)
    ax.set_title(title, fontsize=18)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)

# Function to plot the bispectrum (equilateral or squeezed configuration)
def bispectrum_eq_plot(results, filename, title, ref_label="$128^3$", q_bool=False):
    nrows = 3 if q_bool else 2
    ratios = [2, 1.4, 1.2] if q_bool else [2, 1.2]
    fig, axs = plt.subplots(nrows=nrows, figsize=(14, 16), sharex=True, gridspec_kw={"height_ratios": ratios})
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(results)))
    ref = results[ref_label]

    for i, (label, r) in enumerate(results.items()):
        is_ref = (label == ref_label)
        c = "black" if is_ref else colors[i]
        ls = "--" if is_ref else "-"
        lw = 2.5 if is_ref else 1.6
        axs[0].plot(r["k1"], np.abs(r["B"]), ls, color=c, lw=lw, label=label)
        if q_bool:
            axs[1].plot(r["k1"], r["Q"], ls, color=c, lw=lw, label=label)
        if not is_ref:
            axs[-1].plot(r["k1"], r["B"] / ref["B"], "-", color=c, lw=lw, label=label)

    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"$|B|$  $[(\mathrm{Mpc}/h)^6]$", fontsize=22)
    axs[0].set_title(title, fontsize=30)
    axs[0].legend(loc="best", fontsize=17)
    if q_bool:
        axs[1].set_ylabel(r"$Q$", fontsize=22)
    axs[-1].axhline(1.0, color="grey", lw=1.0)
    axs[-1].set_ylabel(r"$B / B_{\rm HR}$", fontsize=22)
    axs[-1].set_ylim(0.0, 1.6)
    axs[-1].set_xlabel("k [h/Mpc]", fontsize=22)
    for ax in axs:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.axvline(K_NYQ_LR, color="crimson", ls="--", lw=1.0)
        ax.tick_params(which='both', labelsize=15)

    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)

# Function to plot the angle spanning bispectrum
def bispectrum_angle_plot(results, theta, filename, k1, k2, ref_label="$128^3$", q_bool=False):
    fig, axs = plt.subplots(ncols=2 if q_bool else 1, figsize=(18, 7) if q_bool else (16, 9), squeeze=False)
    axs = axs[0]
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(results)))

    for i, (label, r) in enumerate(results.items()):
        th = theta[r["mask"]] / np.pi
        is_ref = (label == ref_label)
        c = "black" if is_ref else colors[i]
        ls = "--" if is_ref else "-"
        lw = 2.5 if is_ref else 1.6
        axs[0].plot(th, r["B"], ls, color=c, lw=lw, marker="o", ms=3, label=label)
        if q_bool:
            axs[1].plot(th, r["Q"], ls, color=c, lw=lw, marker="o", ms=3, label=label)

    axs[0].set_ylabel(r"$B(k_1,k_2,\theta)$  $[(\mathrm{Mpc}/h)^6]$", fontsize=14)
    if q_bool:
        axs[1].set_ylabel(r"$Q(k_1,k_2,\theta)$", fontsize=14)
    for ax in axs:
        ax.set_xlabel(r"$\theta/\pi$", fontsize=14)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=11)
        ax.text(0.05, 0.1, f"$k_3={k1 + k2:.3f}$", transform=ax.transAxes, fontsize=9, color="grey")
        ax.text(0.9, 0.05, f"$k_3={abs(k1 - k2):.3f}$", transform=ax.transAxes, fontsize=9, color="grey")

    fig.suptitle(rf"Bispectrum angle dependence, $k_1={k1:.3f}$, $k_2={k2:.3f}$ h/Mpc", fontsize=18)
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)

# Function to print min, max and average of a ratio inside some x ranges
def ratio_report(tag, label, x, ratio, ranges, drop_last=True):
    for name, lo, hi in ranges:
        sel = np.isfinite(ratio) & (x >= lo) & (x < hi)
        if not sel.any():
            continue
        r, xs = ratio[sel], x[sel]
        line = (f"{tag} | {label} | {name}: {r.min():.3f} to {r.max():.3f} "
                f"(min at {xs[r.argmin()]:.3f}, max at {xs[r.argmax()]:.3f}), avg {r.mean():.3f}")
        if drop_last and r.size > 2:
            line += (f" | without last bin: {r[:-1].min():.3f} to {r[:-1].max():.3f}, "
                     f"avg {r[:-1].mean():.3f}; last bin {r[-1]:.3f}")
        print(line)

# Angle ranges of a scan
def angle_ranges(k1, k2):
    cos_c = (K_NYQ_LR**2 - k1**2 - k2**2) / (2 * k1 * k2)
    if abs(cos_c) > 1:
        return [("all angles", -np.inf, np.inf)]
    theta_c = np.arccos(cos_c) / np.pi
    return [("all angles", -np.inf, np.inf), ("k3 generated", -np.inf, theta_c), ("k3 supplied", theta_c, np.inf)]

# Function to plot curves with std bands
def band_plot(x, curves, filename, xlabel, ylabel, title, ylog=False, hline=None, ylim=None, ratio_ref=None, ratio_ylim=(0.0, 1.6)):
    nrows = 2 if ratio_ref is not None else 1
    fig, axs = plt.subplots(nrows=nrows, figsize=(12, 6 + 5 * (nrows - 1)), sharex=True, squeeze=False,
                            gridspec_kw={"height_ratios": [2, 1]} if nrows == 2 else None)
    axs = axs[:, 0]
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(curves)))
    ref = curves[ratio_ref][0] if ratio_ref is not None else None

    for i, (label, (mean, std)) in enumerate(curves.items()):
        is_ref = (label == ratio_ref)
        c = "black" if is_ref else colors[i]
        axs[0].plot(x, mean, "--" if is_ref else "-", color=c, lw=2.4 if is_ref else 1.7, label=label)
        if std is not None:
            axs[0].fill_between(x, mean - std, mean + std, color=c, alpha=0.25, lw=0)
        if ref is None:
            ratio_report(filename, label, x, mean, K_RANGES)
        elif not is_ref:
            axs[1].plot(x, mean / ref, "-", color=c, lw=1.7, label=label)
            if std is not None:
                axs[1].fill_between(x, (mean - std) / ref, (mean + std) / ref, color=c, alpha=0.25, lw=0)
            ratio_report(filename, label, x, mean / ref, K_RANGES)

    if hline is not None:
        axs[0].axhline(hline, color="grey", lw=0.9)
    if ylim:
        axs[0].set_ylim(*ylim)
    axs[0].set_ylabel(ylabel, fontsize=20)
    axs[0].set_title(title, fontsize=18)
    axs[0].legend(loc="best", fontsize=16)
    if ylog:
        axs[0].set_yscale("log")
    axs[0].tick_params(which='both',labelsize=13)
 
    if ratio_ref is not None:
        axs[1].axhline(1.0, color="grey", lw=0.9)
        axs[1].set_ylim(*ratio_ylim)
        axs[1].set_ylabel(r"$P(k)/P_{\rm HR}(k)$", fontsize=20)
 
    for ax in axs:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.axvline(K_NYQ_LR, color="crimson", ls="--", lw=1.1)
        ax.axvline(K_NYQ_HR , color="purple", ls="--", lw=1.1)
    axs[-1].tick_params(which='both',labelsize=13)
    axs[-1].set_xlabel(xlabel, fontsize=20)
 
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)
 
# Function to plot one-point PDFs of delta with std bands
def pdf_plot(edges, curves, filename, title="One-point PDF", ratio_ref=None, ticks=(-1, 0, 1, 10, 100)):
    log_centres = 0.5 * (edges[:-1] + edges[1:])
    x = 10.0 ** log_centres - 1.0
    jac = 1.0 / ((1.0 + x) * np.log(10.0))

    nrows = 2 if ratio_ref is not None else 1
    fig, axs = plt.subplots(nrows=nrows, figsize=(14, 6 + 5 * (nrows - 1)), sharex=True, squeeze=False,
                            gridspec_kw={"height_ratios": [2, 1]} if nrows == 2 else None)
    axs = axs[:, 0]
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(curves)))
    ref = curves[ratio_ref][0] if ratio_ref is not None else None
    delta_ranges = [("delta < -0.8", -np.inf, -0.8), ("-0.8 < delta < 5", -0.8, 5.0), ("-0.8 < delta < 10", -0.8, 10.0)]

    for i, (label, (mean, std)) in enumerate(curves.items()):
        is_ref = (label == ratio_ref)
        c = "black" if is_ref else colors[i]
        axs[0].plot(x, mean * jac, "--" if is_ref else "-", color=c, lw=2.4 if is_ref else 1.7, label=label)
        if std is not None:
            axs[0].fill_between(x, (mean - std) * jac, (mean + std) * jac, color=c, alpha=0.25, lw=0)
        if ref is not None and not is_ref:
            den = np.where(ref > 1e-12, ref, np.nan)
            axs[1].plot(x, mean / den, "-", color=c, lw=1.7)
            if std is not None:
                axs[1].fill_between(x, (mean - std) / den, (mean + std) / den, color=c, alpha=0.25, lw=0)
            ratio_report(filename, label, x, mean / den, delta_ranges, drop_last=False)

    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"$p(\delta)$", fontsize=20)
    axs[0].set_title(title, fontsize=26)
    axs[0].legend(loc="best", fontsize=18)
    if ratio_ref is not None:
        axs[1].axhline(1.0, color="grey", lw=0.9)
        axs[1].set_ylim(0.0, 2.5)
        axs[1].set_ylabel(r"$p(\delta) / p(\delta_\mathrm{HR})$", fontsize=18)
    for ax in axs:
        ax.axvline(0.0, color="grey", lw=0.9)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(which='both', labelsize=14)
        ax.set_xscale("asinh")
        ax.set_xlim(-1.2, 48.5)
        ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    axs[-1].set_xlabel(r"$\delta$", fontsize=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)
 
# Function to plot an ensemble bispectrum
def bispectrum_band_plot(x, curves_B, curves_Q, filename, title, xlabel="k [h/Mpc]", ratio_ref=None, legs=None):
    nrows = 3 if ratio_ref is not None else 2
    fig, axs = plt.subplots(nrows=nrows, figsize=(16, 5 * nrows), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(curves_B)))
    ref = curves_B[ratio_ref][0] if ratio_ref is not None else None

    for i, label in enumerate(curves_B):
        is_ref = (label == ratio_ref)
        c = "black" if is_ref else colors[i]
        ls = "--" if is_ref else "-"
        lw = 2.4 if is_ref else 1.7
        B, B_std = curves_B[label]
        axs[0].plot(x, np.abs(B), ls, color=c, lw=lw, label=label)
        if B_std is not None:
            axs[0].fill_between(x, np.abs(B) - B_std, np.abs(B) + B_std, color=c, alpha=0.25, lw=0)
        if label in curves_Q: # The Q panel can show fewer curves than the B one
            Q, Q_std = curves_Q[label]
            axs[1].plot(x, Q, ls, color=c, lw=lw, label=label)
            if Q_std is not None:
                axs[1].fill_between(x, Q - Q_std, Q + Q_std, color=c, alpha=0.25, lw=0)

        if ref is not None and not is_ref:
            den = np.where(np.abs(ref) > 1e-30, ref, np.nan)
            axs[2].plot(x, B / den, "-", color=c, lw=1.7)
            if B_std is not None:
                axs[2].fill_between(x, (B - B_std) / den, (B + B_std) / den, color=c, alpha=0.25, lw=0)
            if legs is None:
                ratio_report(filename, label, x, B / den, K_RANGES)
            else:
                ratio_report(filename, label, x, B / den, angle_ranges(legs[0], legs[1]), drop_last=False)

    if legs is not None:
        k1, k2 = legs
        axs[-1].text(0.05, 0.1, f"$k_3={k1 + k2:.3f}$", transform=axs[-1].transAxes, fontsize=20, color="grey")
        axs[-1].text(0.9, 0.05, f"$k_3={abs(k1 - k2):.3f}$", transform=axs[-1].transAxes, fontsize=20, color="grey")

    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"$|B|\ [(\mathrm{Mpc}/h)^6]$", fontsize=25)
    axs[0].set_title(title, fontsize=26, y=1.1)
    axs[0].legend(loc="best", fontsize=24)
    axs[1].set_ylabel(r"$Q$", fontsize=25)
    if ratio_ref is not None:
        axs[2].axhline(1.0, color="grey", lw=0.9)
        axs[2].set_ylim(0.0, 2.0)
        axs[2].set_ylabel(r"$B/B_{\rm HR}$", fontsize=25)
    for ax in axs:
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(which='both', labelsize=16)
        if legs is None: # k axis
            ax.set_xscale("log")
            ax.axvline(K_NYQ_LR, color="crimson", ls="--", lw=1.1)
    axs[-1].set_xlabel(xlabel, fontsize=25)

    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)

# Log transform of the maps: ln(2 + delta) - 1
def log_field(x):
    return np.log(x + 2.0) - 1.0

# Two-slope colour norm centred on the mean of the panel
def get_per_panel_norm(data, down_col=1.0, up_col=1.15, c_shift=0.0):
    vmin = np.min(data) * down_col
    vmax = np.max(data) * up_col
    vcenter = np.mean(data) + c_shift
    
    eps = 1e-6 * (vmax - vmin) if vmax > vmin else 1e-6
    vcenter = np.clip(vcenter, vmin + eps, vmax - eps)
    
    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

# Function to show a map with its colorbar
def show_with_cbar(ax, data, label, norm, cmap=cosmic_indigo, fontsize=20):
    im = ax.imshow(data, origin="lower", cmap=cmap, norm=norm)
    ax.set_title(label, fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tmin = np.ceil(norm.vmin * 10) / 10
    tmax = np.floor(norm.vmax * 10) / 10
    if tmin >= tmax:
        ticks = np.array([norm.vmin, norm.vmax])
    else:
        ticks = np.round(np.linspace(tmin, tmax, 5), 2)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.minorticks_off()
    return im

# Function to plot generated, source and true maps, with their differences
def field_maps_plot(d, filename, title=None, diff_pct=99.999, cmap=cosmic_indigo):
    gen = log_field(d["map_gen0"])
    hr = log_field(d["map_hr"])
    src = log_field(d["map_src"])
    src_clean = log_field(d["map_src_clean"])
    fields = [("Clean source (upsampled LR)", src_clean), ("Noisy source", src), ("Generated (one member)", gen), ("True HR", hr)]
    diffs = [("Generated $-$ truth", gen - hr), ("Generated $-$ noisy source", gen - src),
             ("Generated $-$ clean source", gen - src_clean), ("Noisy $-$ clean source", src - src_clean)]

    fig, axs = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    for ax, (label, v) in zip(axs[0], fields):
        show_with_cbar(ax, v, label, get_per_panel_norm(v, up_col=1.0), cmap)

    dmax = max(np.percentile(np.abs(v), diff_pct) for _, v in diffs[1:])
    for ax, (label, v) in zip(axs[1], diffs):
        show_with_cbar(ax, v, label, TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax), "RdBu_r")

    fig.suptitle(title or f"2D slices of logged generated density fields ({param_title(d['params'])})", fontsize=30, y=1.05)
    fig.set_constrained_layout_pads(hspace=0.0, wspace=0.08, h_pad=0.15)
    plt.savefig(filename, dpi=130, bbox_inches='tight')
    plt.close(fig)

# Function to plot three ensemble members next to the sources, the truth and the ensemble mean
def members_plot(d, filename, title=None):
    panels = [("Clean source", d["map_src_clean"]), ("Member 1", d["map_gen0"]), ("Member 2", d["map_gen1"]),
              ("Member 3", d["map_gen2"]), ("Noisy source", d["map_src"]), ("True HR", d["map_hr"]),
              ("Ensemble mean", d["map_genmean"])]

    fig, axs = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    for ax, (label, v) in zip(axs.flat, panels):
        v = log_field(v)
        show_with_cbar(ax, v, label, get_per_panel_norm(v, up_col=1.0), fontsize=27)

    ax = axs[1, 3]
    im = ax.imshow(d["map_std_log"], origin="lower", cmap="magma")
    ax.set_title("Ensemble scatter", fontsize=25)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title or f"Ensemble members ({param_title(d['params'])})", fontsize=35, y=1.05)
    fig.set_constrained_layout_pads(hspace=0.0, wspace=0.08, h_pad=0.15)
    plt.savefig(filename, dpi=130, bbox_inches='tight')
    plt.close(fig)
 
# Function to plot P_cond
def conv_plot(d, filename):
    ns, P = d["conv_n"], d["conv_P"]
    k, P_true = d["k"], d["P_true"]
    band = (k >= K_NYQ_LR) & (k < K_NYQ_HR)

    fig, axs = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [2, 1]})
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ns)))
    for i, m in enumerate(ns):
        axs[0].plot(k, P[i] / P_true, color=colors[i], lw=1.7, label=f"$m={m}$")
    axs[0].axhline(1.0, color="grey", lw=0.9)
    axs[0].axvline(K_NYQ_LR, color="crimson", ls="--", lw=1.1)
    axs[0].axvline(K_NYQ_HR, color="purple", ls="--", lw=1.1)
    axs[0].set_xscale("log")
    axs[0].set_ylim(-0.05, 1.35)
    axs[0].set_xlabel("k [h/Mpc]", fontsize=14)
    axs[0].set_ylabel(r"$P_{\rm cond}/P_{\rm true}$", fontsize=14)
    axs[0].set_title("Conditional variance vs ensemble size", fontsize=16)
    axs[0].legend(fontsize=11)
    axs[0].grid(True, which="both", alpha=0.3)

    axs[1].plot(ns, (P[:, band] / P_true[band]).mean(axis=1), "o-", color="darkorange", lw=1.8)
    axs[1].set_xscale("log", base=2)
    axs[1].set_xlabel("members $m$", fontsize=14)
    axs[1].set_ylabel(r"$\langle P_{\rm cond}/P_{\rm true}\rangle$ in band", fontsize=13)
    axs[1].set_title("Band average", fontsize=16)
    axs[1].grid(True, which="both", alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)
 
# Function to plot one curve per checkpoint
def evolution_plot(epochs, x, ys, filename, xlabel, ylabel, title, xlog=True, hline=None, ylim=None):
    epochs = np.asarray(epochs)
    norm = LogNorm(epochs.min(), epochs.max()) if len(epochs) > 1 else Normalize(epochs.min(), epochs.min() + 1)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(13, 8))
    for e, y in zip(epochs, ys):
        ax.plot(x, y, "-", color=cmap(norm(e)), lw=1.5)
    if hline is not None:
        ax.axhline(hline, color="grey", lw=0.9)
    if ylim:
        ax.set_ylim(*ylim)
    if xlog:
        ax.set_xscale("log")
        ax.axvline(K_NYQ_LR, color="crimson", ls="--", lw=1.1)
        ax.axvline(K_NYQ_HR, color="purple", ls="--", lw=1.1)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=25)
    ax.grid(True, which="both", alpha=0.3)

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label("epoch", fontsize=18)
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)
 
# Function to plot band-averaged statistics against the training epoch
def evolution_scalar_plot(epochs, series, filename, title, loss=None, ylabel="Generated band average", hline=None, ncol=2):
    fig, ax = plt.subplots(figsize=(14, 9))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(series)))
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "p"]
    for i, (label, ys) in enumerate(series.items()):
        ax.plot(epochs, ys, "-", marker=markers[i % len(markers)], color=colors[i], lw=1.8, ms=5, label=label)
    if hline is not None:
        ax.axhline(hline, color="grey", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=27)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(which="both", labelsize=14)

    if loss is not None:
        ax2 = ax.twinx()
        ax2.plot(epochs, loss, "s--", color="crimson", lw=1.4, ms=4, label="validation loss", alpha=0.8)
        ax2.set_yscale("log")
        ax2.set_ylabel("validation loss", fontsize=20, color="crimson")
        ax2.tick_params(which='both', labelcolor="crimson", labelsize=14)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=15, ncol=ncol)
    else:
        ax.legend(loc="best", fontsize=12, ncol=ncol)

    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)
 
# Function to plot the response of the generated power spectrum to each cosmological parameter
def cosmology_response_plot(k, res, filename, title):
    n = len(res)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(7 * ncol, 6 * nrow), constrained_layout=True)

    gs = fig.add_gridspec(nrow, ncol * 2)
    axs = []
    for i in range(n):
        row = i // ncol
        panels_in_row = min(ncol, n - row * ncol)
        start_col = (ncol - panels_in_row) + (i % ncol) * 2
        axs.append(fig.add_subplot(gs[row, start_col:start_col + 2]))

    for ax, (label, r) in zip(axs, res.items()):
        colors = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(r["values"])))
        for i, v in enumerate(r["values"]):
            fid = (v == r["fid"])
            ratio, err = r["ratios"][i], r["ratios_err"][i]
            ax.plot(k, ratio, "--" if fid else "-", color=colors[i], lw=2.4 if fid else 1.6, label=f"{v:.3g}")
            ax.fill_between(k, ratio - err, ratio + err, color=colors[i], alpha=0.25, lw=0)

        ax.axhline(1.0, color="grey", lw=0.9)
        ax.axvline(K_NYQ_LR, color="crimson", ls="--", lw=1.1)
        ax.axvline(K_NYQ_HR, color="purple", ls="--", lw=1.1)
        ax.set_xscale("log")
        ax.set_ylim(0.5, 1.5)
        ax.set_xlabel("k [h/Mpc]", fontsize=24)
        ax.set_ylabel(r"$P_{\rm gen}/P_{\rm gen}^{\rm fid}$", fontsize=24)
        ax.set_title(label, fontsize=28)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(which='both', labelsize=17)
        ax.legend(fontsize=14, ncol=2)

    fig.suptitle(title, fontsize=35, y=1.05)
    fig.set_constrained_layout_pads(hspace=0.08, wspace=0.04, h_pad=0.15)
    plt.savefig(filename, dpi=130, bbox_inches='tight')
    plt.close(fig)  
 
def param_title(p):
    return rf"$\Omega_m={p[0]:.3f}$, $\Omega_b={p[1]:.3f}$, $h={p[2]:.3f}$, $n_s={p[3]:.3f}$, $\sigma_8={p[4]:.3f}$"
 
# Function to plot the maps along the ODE trajectory
def trajectory_maps_plot(d, filename, title=None, members=2):
    ts, maps = d["t"], d["maps"]
    nrow = min(members, maps.shape[1])
    ncol = len(ts) + 1
    truth = log_field(d["map_hr"])
    norm = get_per_panel_norm(truth)

    fig, axs = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.2 * nrow), constrained_layout=True, squeeze=False)
    for i in range(nrow):
        for j, t in enumerate(ts):
            im = axs[i, j].imshow(log_field(maps[j, i]), origin="lower", cmap=cosmic_indigo, norm=norm)
            if t == 0:
                axs[i, j].set_title(f"$t={t:g}$ (source)", fontsize=28)
            elif t == 1:
                axs[i, j].set_title(f"$t={t:g}$ (output)", fontsize=28)
            else:
                axs[i, j].set_title(f"$t={t:g}$", fontsize=28)
        axs[i, 0].set_ylabel(f"member {i + 1}", fontsize=24)
        axs[i, -1].imshow(truth, origin="lower", cmap=cosmic_indigo, norm=norm)
        axs[i, -1].set_title("True HR", fontsize=28)
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01, label=r"$\ln(2+\delta)-1$")
    cb.ax.yaxis.label.set_size(24)
    cb.ax.tick_params(which='both', labelsize=16)
    fig.suptitle(title or f"ODE trajectory ({param_title(d['params'])})", fontsize=33, y=1.08)
    plt.savefig(filename, dpi=130, bbox_inches='tight')
    plt.close(fig)
 
# Function to plot the one-point PDF of each DWT subband
def subband_pdf_plot(d, filename, title=None, floor=1e-7):
    edges = d["band_edges"]
    x = 0.5 * (edges[:-1] + edges[1:])
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, 4))
    curves = [("True HR", d["band_pdf_true"], None, "black", "--", 2.4),
              ("Generated", d["band_pdf_gen_mean"], d["band_pdf_gen_std"], colors[0], "-", 1.8),
              ("Ensemble mean", d["band_pdf_genmean"], None, colors[1], "-", 1.6),
              ("Noisy source", d["band_pdf_src_mean"], d["band_pdf_src_std"], colors[2], "-", 1.6),
              ("Clean source", d["band_pdf_srcclean"], None, colors[3], "-", 1.6)]

    fig, axs = plt.subplots(2, 4, figsize=(24, 11), constrained_layout=True)
    for b, ax in enumerate(axs.flat):
        filled = np.zeros(len(x), dtype=bool)
        for label, pdf, std, c, ls, lw in curves:
            ax.plot(x, np.where(pdf[b] > 0, pdf[b], np.nan), ls, color=c, lw=lw, label=label)
            if std is not None:
                ax.fill_between(x, np.maximum(pdf[b] - std[b], floor), pdf[b] + std[b], color=c, alpha=0.22, lw=0)
            filled |= pdf[b] > floor

        idx = np.flatnonzero(filled)
        lo, hi = x[idx[0]], x[idx[-1]]
        pad = 0.1 * max(abs(lo), abs(hi))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xscale("asinh")
        ax.xaxis.set_major_locator(FixedLocator([-100, -30, -10, -3, -1, 0, 1, 3, 10, 30, 100]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
        ax.set_title(BAND_NAMES[b], fontsize=20)
        ax.grid(True, which="both", alpha=0.3)
        if b >= 4:
            ax.set_xlabel(r"coefficient $/\ \sigma_{\rm band}^{\rm train}$", fontsize=20)
        if b % 4 == 0:
            ax.set_ylabel("density", fontsize=20)

    axs[0, 0].legend(fontsize=11)
    fig.suptitle(title or f"One-point PDF of the wavelet bands ({param_title(d['params'])})", fontsize=30, y=1.06)
    plt.savefig(filename, dpi=130, bbox_inches='tight')
    plt.close(fig)
 
