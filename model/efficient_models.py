import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, set_grad_enabled, grad
from operator import add

from .model import WN


class AffineCouplingBlock(nn.Module):
    def __init__(self,
                 *args,
                 memory_efficient=True,
                 last_layer=False,
                 **kwargs):
        super().__init__()

        self.F = WN(*args, **kwargs)
        if memory_efficient:
            self.efficient_forward = AffineCouplingFunc.apply
            self.efficient_inverse = InvAffineCouplingFunc.apply
            self.param_list = list(self.F.parameters())

    def forward(self, x, y):
        if hasattr(self, 'efficient_forward'):
            return self.efficient_forward(x, y, self.F, self.param_list)
        else:
            xa, xb = x.chunk(2, 1)
            za = xa
            log_s, t = self.F(xa, y)
            zb = xb * log_s.exp() + t
            z = torch.cat((za, zb), 1)
            return z, log_s

    def inverse(self, z, y):
        if hasattr(self, 'efficient_inverse'):
            return self.efficient_inverse(z, y, self.F, self.param_list)
        else:
            za, zb = z.chunk(2, 1)
            xa = za
            log_s, t = self.F(za, y)
            xb = (zb - t) / log_s.exp()
            x = torch.cat((xa, xb), 1)
            return x, -log_s


class AffineCouplingFunc(Function):
    @staticmethod
    def forward(ctx, x, y, F, *F_weights):
        ctx.F = F
        ctx.y = y

        with torch.no_grad():
            xa, xb = x.chunk(2, 1)
            log_s, t = F(xa, y)
            xb *= log_s.exp()
            xb += t
            z = x.clone()

        ctx.save_for_backward(z)
        return x, log_s

    @staticmethod
    def backward(ctx, z_grad, log_s_grad):
        F = ctx.F
        y = ctx.y
        z = ctx.saved_tensors
        za, zb = z.chunk(2, 1)
        dza, dzb = z_grad.chunk(2, 1)

        xa, dxa = za, dza
        xa.requires_grad = True
        with set_grad_enabled(True):
            log_s, t = F(xa, y)

        with torch.no_grad():
            s = log_s.exp()
            xb = (zb - t) / s

        param_list = [xa] + list(F.parameters())
        with set_grad_enabled(True):
            tgrads = grad(t, param_list, grad_outputs=dzb)
            sgrads = grad(log_s, param_list, grad_outputs=dzb * xb * s + log_s_grad)

            dw = tuple(map(add, tgrads[1:], sgrads[1:]))
            dxa += tgrads[0] + sgrads[0]
            dxb = dzb * s
            dx = torch.cat((dxa, dxb), 1)
        return (dx, None, None) + dw


class InvAffineCouplingFunc(Function):
    @staticmethod
    def forward(ctx, z, y, F, *F_weights):
        ctx.F = F
        ctx.y = y

        with torch.no_grad():
            za, zb = z.chunk(2, 1)
            log_s, t = F(za, y)
            zb -= t
            zb /= log_s.exp()
            x = z.clone()

        ctx.save_for_backward(x)
        return z, -log_s

    @staticmethod
    def backward(ctx, x_grad, log_s_grad):
        F = ctx.F
        y = ctx.y
        x = ctx.saved_tensors
        xa, xb = x.chunk(2, 1)
        dxa, dxb = x_grad.chunk(2, 1)

        za, dza = xa, dxa
        za.requires_grad = True
        with set_grad_enabled(True):
            log_s, t = F(za, y)

        with torch.no_grad():
            s = log_s.exp()
            zb = xb * s + t

        param_list = [za] + list(F.parameters())
        with set_grad_enabled(True):
            tgrads = grad(-t, param_list, grad_outputs=dxb / s)
            sgrads = grad(-log_s, param_list, grad_outputs=dxb * zb / s + log_s_grad)

            dw = tuple(map(add, tgrads[1:], sgrads[1:]))
            dza += tgrads[0] + sgrads[0]
            dzb = dxb / s
            dz = torch.cat((dza, dzb), 1)
        return (dz, None, None) + dw
