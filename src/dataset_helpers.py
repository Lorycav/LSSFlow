import config
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pathlib as Path
import numpy as np
from helpers import DWT, high_band_mask, high_band_noise
      
class QuijoteSuperresDataset(Dataset):
    def __init__(self, simulation_indices, params, hr_template, lr_template, 
                 transform_h=None, transform_l=None, transform_dwt_h = None, 
                 transform_dwt_l = None, param_transform=None, dwt_flag = None, 
                 hi_mask=None, sigma_hi=0.0, noise_scale=1.):
        
        self.indices = simulation_indices
        self.params = params[simulation_indices]
        self.hr_template = str(hr_template)
        self.lr_template = str(lr_template)
        self.transform_h = transform_h
        self.transform_l = transform_l
        self.transform_dwt_h = transform_dwt_h
        self.transform_dwt_l = transform_dwt_l
        self.param_transform = param_transform    
        self.dwt_flag = dwt_flag                   # Wether DWT mode is active or not
        self.hi_mask = hi_mask
        self.sigma_hi = float(sigma_hi)
        self.noise_scale = float(noise_scale)

    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        sim_idx = self.indices[idx]
        try:
            hr_data = torch.from_numpy(np.load(self.hr_template.format(sim_idx)).astype(np.float32)).unsqueeze(0)
            lr_data = torch.from_numpy(np.load(self.lr_template.format(sim_idx)).astype(np.float32)).unsqueeze(0)
            sim_params = torch.from_numpy(self.params[idx])
            if self.transform_h: 
                hr_data = self.transform_h(hr_data)
            if self.transform_l:
                lr_data = self.transform_l(lr_data)
            if self.param_transform: sim_params = self.param_transform(sim_params)

            if self.hi_mask is not None and self.noise_scale > 0:
                lr_data = lr_data + high_band_noise(self.hi_mask, self.noise_scale * self.sigma_hi,
                    shape=lr_data.shape[-3:], dtype=lr_data.dtype).unsqueeze(0)

            # DWT
            if self.dwt_flag == True:
                hr_data = DWT(hr_data.unsqueeze(0)).squeeze(0)
                lr_data = DWT(lr_data.unsqueeze(0)).squeeze(0)

                # Normalization for better training
                hr_data = self.transform_dwt_h(hr_data)
                lr_data = self.transform_dwt_l(lr_data)

            return hr_data, lr_data, sim_params
        except Exception as e:
            print(f"Warning: Error loading data for index {sim_idx}: {e}. Returning None.")
            return None, None, None

class Normalize:
    def __init__(self, mean, std, dwt_flag = None):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        self.dwt = dwt_flag
        if self.mean.ndim == 0: self.mean, self.std = self.mean.unsqueeze(0), self.std.unsqueeze(0)
        while self.mean.ndim < 4: self.mean, self.std = self.mean.unsqueeze(-1), self.std.unsqueeze(-1)
    def __call__(self, x):
        std_safe = torch.where(self.std == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), self.std)
        return (x - self.mean.to(x.device)) / std_safe.to(x.device)
    def inverse(self, x): return x * self.std.to(x.device) + self.mean.to(x.device)

class Logging:
    def log_transf_numpy (self, x: np.ndarray) -> np.ndarray:
        return np.log(2.0+x)-1
    
    def log_transf_torch (self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(2.0+x)-1
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_transf_torch(x)
    
    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        return self.log_transf_numpy(x)
    
    def inverse_numpy(self, y: np.ndarray) -> np.ndarray:
        return np.exp(y+1)-2.0
    
    def inverse_torch(self, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(y+1)-2.0
    
class ParamNormalize:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
    def __call__(self, p):
        std_safe = torch.where(self.std == 0, torch.tensor(1.0, device=p.device, dtype=p.dtype), self.std)
        return (p - self.mean.to(p.device)) / std_safe.to(p.device)
    def inverse(self, p): return p * self.std.to(p.device) + self.mean.to(p.device)

def load_params(param_file: Path, num_simulations: int, n_params: int) -> np.ndarray:
    try:
        params = np.loadtxt(param_file)
        if params.shape[0] > num_simulations: # Simple header check
             params = np.loadtxt(param_file, skiprows=1)
        if params.shape != (num_simulations, n_params):
            raise ValueError(f"Param shape mismatch: expected ({num_simulations}, {n_params}), got {params.shape}")
        return params.astype(np.float32)
    except Exception as e:
        print(f"Error loading parameters from {param_file}: {e}")
        raise

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None
    
def get_dataloaders(params_all, hr_template, lr_template, batch_size, parallel_flag, precomputed_stats, num_workers=2):
    
    train_indices = precomputed_stats['train_indices'] 
    val_indices = precomputed_stats['val_indices']
    test_indices = precomputed_stats['test_indices']
    print(f"Dataset splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # Load Parameters
    # HR
    data_mean_h = torch.as_tensor(precomputed_stats['data_mean_h'], dtype=torch.float32)
    data_std_h = torch.as_tensor(precomputed_stats['data_std_h'], dtype=torch.float32)
    mean_dwt_h = torch.as_tensor(precomputed_stats['mean_dwt_hr'], dtype=torch.float32)
    std_dwt_h = torch.as_tensor(precomputed_stats['std_dwt_hr'], dtype=torch.float32)
    sigma_hi = torch.as_tensor(precomputed_stats['sigma_hi'], dtype=torch.float32)
    # Parameters
    param_mean = torch.as_tensor(precomputed_stats['param_mean'], dtype=torch.float32)
    param_std = torch.as_tensor(precomputed_stats['param_std'], dtype=torch.float32)

    # Pixel Transform
    if (config.LOG_FLAG == True):
        data_transform_h = Logging()
        data_transform_l = data_transform_h
    else:
        data_transform_h = Normalize(data_mean_h, data_std_h)
        data_transform_l = data_transform_h

    # DWT Normalization
    transform_dwt_h = Normalize(mean_dwt_h, std_dwt_h, True)
    transform_dwt_l = transform_dwt_h

    # Parameters Normalization
    param_transform = ParamNormalize(param_mean, param_std)

    #hi_mask = None
    hi_mask  = high_band_mask(128, 64)
    
    train_ds = QuijoteSuperresDataset(train_indices, params_all, hr_template, lr_template, data_transform_h, data_transform_l, transform_dwt_h, transform_dwt_l, param_transform, True, hi_mask, sigma_hi)
    val_ds = QuijoteSuperresDataset(val_indices, params_all, hr_template, lr_template, data_transform_h, data_transform_l, transform_dwt_h, transform_dwt_l, param_transform, True, hi_mask, sigma_hi)
    test_ds = QuijoteSuperresDataset(test_indices, params_all, hr_template, lr_template, data_transform_h, data_transform_l, transform_dwt_h, transform_dwt_l, param_transform, True, hi_mask, sigma_hi)

    if parallel_flag == True and dist.is_initialized():

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        training_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        validation_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        testing_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        
        batches_per_gpu = max(1, config.BATCH_SIZE // world_size)
        train_loader = DataLoader(train_ds, batch_size=batches_per_gpu, sampler=training_sampler, num_workers=num_workers, persistent_workers=True,  prefetch_factor=2, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batches_per_gpu, sampler=validation_sampler, num_workers=num_workers, persistent_workers=True, prefetch_factor=2, pin_memory=True, collate_fn=collate_fn, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=1, sampler=testing_sampler, num_workers=2, persistent_workers=True, prefetch_factor=1, pin_memory=False, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True,  prefetch_factor=2, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True, prefetch_factor=2, pin_memory=True, collate_fn=collate_fn, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True, prefetch_factor=2, pin_memory=True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, data_transform_h, data_transform_l, transform_dwt_h, transform_dwt_l, param_transform
