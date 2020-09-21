import torch
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS coefficient
        max_iters (int, required): maximum iterations

    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large batch training of convolutional networks with layer-wise adaptive rate scaling. ICLR'18:
        https://openreview.net/pdf?id=rJ4uaX2aW

    The LARS algorithm can be written as
    .. math::
            \begin{aligned}
                {global_lr}_{t+1} &= {base_lr}_{t} * (1 - t / T)**2 \\
                {local_lr}_{t+1} &= \eta * ||w_{t}|| / (||g_{t}|| + \beta * ||w_{t}||) \\
                {actual_lr}_{t+1} &= {global_lr}_{t+1} * {local_lr}_{t+1} \\
                v_{t+1} & = \mu * v_{t} + {actual_lr}_{t+1} * (g_{t} + \beta * w_{t}), \\
                w_{t+1} & = w_{t} - v_{t+1},
            \end{aligned}
    where :math:`w`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3, max_iters=40000)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr=required, momentum=.9,
                 weight_decay=.0005, eta=0.001, max_iters=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))
        if max_iters is not required and max_iters < 0:
            raise ValueError("Invalid maximum iterations:%s"%max_iters)

        self._iter = 0
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        eta=eta, max_iters=max_iters)
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, iteration=None, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            iteration: current iteration to calculate polynomial LR decay schedule.
                   if None, uses self._iter and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if iteration is None:
            iteration = self._iter
            self._iter += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']
            max_iters = group['max_iters']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data
                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)

                # Global LR computed on polynomial decay schedule
                decay = (1 - float(iteration) / max_iters) ** 2
                global_lr = lr * decay

                # Compute local learning rate for this layer
                local_lr = eta * weight_norm / \
                    (1e-7 + grad_norm + weight_decay * weight_norm)

                actual_lr = local_lr * global_lr

                if weight_decay != 0:
                    d_p.add_(p, alpha=weight_decay)

                # Update the momentum
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']

                buf.mul_(momentum).add_(d_p, alpha=actual_lr)

                # Update the weight
                p.add_(-buf)

        return loss



