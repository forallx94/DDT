import math
from src.diffusion.base.sampling import *
from src.diffusion.base.scheduling import *
from src.diffusion.pre_integral import *

from typing import Callable, List, Tuple

def ode_step_fn(x, v, dt, s, w):
    return x + v * dt

def t2snr(t):
    if isinstance(t, torch.Tensor):
        return (t.clip(min=1e-8)/(1-t + 1e-8))
    if  isinstance(t, List) or isinstance(t, Tuple):
        return [t2snr(t) for t in t]
    t = max(t, 1e-8)
    return (t/(1-t + 1e-8))

def t2logsnr(t):
    if isinstance(t, torch.Tensor):
        return torch.log(t.clip(min=1e-3)/(1-t + 1e-3))
    if  isinstance(t, List) or isinstance(t, Tuple):
        return [t2logsnr(t) for t in t]
    t = max(t, 1e-3)
    return math.log(t/(1-t + 1e-3))

def t2isnr(t):
   return 1/t2snr(t)

def nop(t):
    return t

def shift_respace_fn(t, shift=3.0):
    return t / (t + (1 - t) * shift)

import logging
logger = logging.getLogger(__name__)

class AdamLMSampler(BaseSampler):
    def __init__(
            self,
            order: int = 2,
            timeshift: float = 1.0,
            lms_transform_fn: Callable = nop,
            w_scheduler: BaseScheduler = None,
            step_fn: Callable = ode_step_fn,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.step_fn = step_fn
        self.w_scheduler = w_scheduler

        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn, ]
        self.order = order
        self.lms_transform_fn = lms_transform_fn

        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, timeshift)
        self.timedeltas = timesteps[1:] - self.timesteps[:-1]
        self._reparameterize_coeffs()

    def _reparameterize_coeffs(self):
        solver_coeffs = [[] for _ in range(self.num_steps)]
        for i in range(0, self.num_steps):
            pre_vs = [1.0, ]*(i+1)
            pre_ts = self.lms_transform_fn(self.timesteps[:i+1])
            int_t_start = self.lms_transform_fn(self.timesteps[i])
            int_t_end = self.lms_transform_fn(self.timesteps[i+1])

            order_annealing = self.order #self.num_steps - i
            order = min(self.order, i + 1, order_annealing)

            _, coeffs = lagrange_preint(order, pre_vs, pre_ts, int_t_start, int_t_end)
            solver_coeffs[i] = coeffs
        self.solver_coeffs = solver_coeffs

    def _impl_sampling(self, net, noise, condition, uncondition):
        """
        sampling process of Euler sampler
        -
        """
        batch_size = noise.shape[0]
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = x0 = noise
        pred_trajectory = []
        t_cur = torch.zeros([batch_size,]).to(noise.device, noise.dtype)
        timedeltas = self.timedeltas
        solver_coeffs = self.solver_coeffs
        for i  in range(self.num_steps):
            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)
            out = net(cfg_x, cfg_t, cfg_condition)
            out = self.guidance_fn(out, self.guidances[i])
            pred_trajectory.append(out)
            out = torch.zeros_like(out)
            order = len(self.solver_coeffs[i])
            for j in range(order):
                out += solver_coeffs[i][j] * pred_trajectory[-order:][j]
            v = out
            dt = timedeltas[i]
            x0 = self.step_fn(x, v, 1-t_cur[0], s=0, w=0)
            x = self.step_fn(x, v, dt, s=0, w=0)
            t_cur += dt
        return x