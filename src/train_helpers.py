import config
import torch
import math
import torch.nn as nn
from contextlib  import contextmanager
 
# MSE normed on each subband std for equal subbands contributions to the loss
def normed_mse (pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    std = target.std(dim = (0, 2, 3, 4), keepdim = True)
    p_normed = pred / (std + eps)
    t_normed = target / (std + eps)
    loss = nn.functional.mse_loss(p_normed, t_normed)

    return loss

def get_module(model):
    if hasattr(model, 'module'):     model = model.module
    if hasattr(model, '_orig_mod'):  model = model._orig_mod
    return model

# Exponential Moving Average    
class EMA:
    def __init__(self, model, max_decay = 0.9999):
        self.max_decay = max_decay
        self.step_count = 0
        
        # Initilizes parameters dictionary
        mod = get_module(model)
        self.shadow = {
            name: param.data.clone().detach()
            for name, param in mod.named_parameters()
            if param.requires_grad
        }
    
    def get_decay(self):
        return min(self.max_decay, (1 + self.step_count) / (10 + self.step_count))

    # Updates parameters inside local dict with new model params
    @torch.no_grad()
    def update(self, model):
        self.step_count += 1
        decay = self.get_decay()
        mod =  get_module(model)
        for name, param in mod.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.data, alpha = 1.0 - decay)

    # Applies local dict params to a model params
    def apply_to(self, model):
        mod =  get_module(model)
        self._backup = {
            name: param.data.clone()
            for name, param in mod.named_parameters()
            if param.requires_grad
        }
        for name, param in mod.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    # Restores params from local backup dict to model params and erase backup dict
    def restore(self, model):
        mod =  get_module(model)
        for name, param in mod.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'step_count': self.step_count}

    # Loads dictionary
    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and 'shadow' in state_dict:
            self.shadow = {k: v.clone() for k, v in state_dict['shadow'].items()}
            self.step_count = state_dict.get('step_count', 0)
        else:
            self.shadow = {k: v.clone() for k, v in state_dict.items()}
            self.step_count = 0

def step_control(step, w_steps = config.WARMUP_STEPS):
    if step < w_steps:
        return float(step) / float(w_steps)
    else:
        return 1.0

def cosine_warmrestart_lambda(step, T_0, T_mult=2, eta_min_ratio=0.01):
    T_cur = step
    T_i = T_0

    while T_cur >= T_i:
        T_cur -= T_i
        T_i = int(T_i * T_mult) # 
    return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1 + math.cos(math.pi * T_cur / T_i))

@contextmanager
def weights_context(model, ema=None, use_ema=True):
    if ema is None or not use_ema:
        yield
        return
    ema.apply_to(model)
    try:
        yield
    finally:
        ema.restore(model)