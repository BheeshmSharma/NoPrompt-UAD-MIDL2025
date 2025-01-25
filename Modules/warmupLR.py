import torch
import torch.optim as optim

class WarmUpLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, base_lr, max_lr, step_size=20, decay_factor=0.5, last_epoch=-1):
        """
        Parameters:
        - optimizer: the optimizer for which to adjust the learning rate.
        - warmup_steps: the number of steps over which to warm up the learning rate.
        - base_lr: the base learning rate to use after warm-up.
        - max_lr: the maximum learning rate (can be set to base_lr for no ramp-up).
        - step_size: the number of epochs after which to decay the learning rate (default 10).
        - decay_factor: the factor by which to decay the learning rate (default 0.5).
        - last_epoch: The index of the last epoch.
        """
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.decay_factor = decay_factor
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        This function computes the learning rate to be used at the current epoch.
        """
        if self.last_epoch < self.warmup_steps:
            # Linear warm-up
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.last_epoch / self.warmup_steps)
        else:
            # After warm-up, decay the learning rate every `step_size` epochs by `decay_factor`
            num_decay_steps = (self.last_epoch - self.warmup_steps) // self.step_size
            lr = self.base_lr * (self.decay_factor ** num_decay_steps)
        
        return [lr for _ in self.optimizer.param_groups]


