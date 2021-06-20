import math

class LinerWarmup:
    def __init__(self, optimizer, start_lr, target_lr, steps, **kwargs):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.steps = steps

        self.total_gain = self.target_lr - self.start_lr
        self.gain = self.total_gain / (self.steps + 1e-16)

        self.kwargs = kwargs

        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

    def step(self, **kwargs):
        history = kwargs['history']
        iteration = kwargs['iteration']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = iteration * self.gain + self.start_lr

class StepLRAfter:
    def __init__(self, optimizer, steps, factors, min_lr, **kwargs):
        self.optimizer = optimizer
        self.steps = steps
        self.factors = factors
        self.min_lr = min_lr
        self.kwargs = kwargs

        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

    def step(self, **kwargs):
        history = kwargs['history']
        iteration = kwargs['iteration']

        for param_group in self.optimizer.param_groups:
            for step, factor in zip(self.steps, self.factors):
                if iteration >= step:
                    lr = param_group['initial_lr'] * factor
                    param_group['lr'] = lr if lr > self.min_lr else self.min_lr


class StepLR:
    def __init__(self, optimizer, factor, patience, min_lr, **kwargs):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.kwargs = kwargs

        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

    def step(self, **kwargs):
        history = kwargs['history']
        iteration = kwargs['iteration']

        for param_group in self.optimizer.param_groups:
            lr = param_group['initial_lr'] * (self.factor ** (iteration // self.patience))
            param_group['lr'] = lr if lr > self.min_lr else self.min_lr


class CosineAnnealingWarmRestarts():

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, **kwargs):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.kwargs = kwargs
        self.T_cur = last_epoch


        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, **kwargs):
        history = kwargs['history']
        epoch = kwargs['iteration']

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr