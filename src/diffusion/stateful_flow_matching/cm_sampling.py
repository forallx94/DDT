import torch

from src.diffusion.base.guidance import *
from src.diffusion.base.scheduling import *
from src.diffusion.base.sampling import *

from typing import Callable


def shift_respace_fn(t, shift=3.0):
    return t / (t + (1 - t) * shift)

def ode_step_fn(x, v, dt, s, w):
    return x + v * dt


import logging
logger = logging.getLogger(__name__)

class CMSampler(BaseSampler):
    def __init__(
            self,
            w_scheduler: BaseScheduler = None,
            timeshift=1.0,
            guidance_interval_min: float = 0.0,
            guidance_interval_max: float = 1.0,
            state_refresh_rate=1,
            last_step=None,
            step_fn=None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.last_step = last_step
        self.timeshift = timeshift
        self.state_refresh_rate = state_refresh_rate
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps

        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)

        assert self.last_step > 0.0
        assert self.scheduler is not None


    def _impl_sampling(self, net, noise, condition, uncondition):
        """
        sampling process of Euler sampler
        -
        """
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        state = None
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            cfg_t = t_cur.repeat(batch_size*2)
            cfg_x = torch.cat([x, x], dim=0)
            if i % self.state_refresh_rate == 0:
                state = None
            out, state = net(cfg_x, cfg_t, cfg_condition, state)
            if t_cur > self.guidance_interval_min and t_cur < self.guidance_interval_max:
                out = self.guidance_fn(out, self.guidance)
            else:
                out = self.guidance_fn(out, 1.0)
            v = out

            x0 = x + v * (1-t_cur)
            alpha_next = self.scheduler.alpha(t_next)
            sigma_next = self.scheduler.sigma(t_next)
            x = alpha_next * x0 + sigma_next * torch.randn_like(x)
            # print(alpha_next, sigma_next)
        return x