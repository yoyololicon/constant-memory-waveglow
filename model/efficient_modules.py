import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, set_grad_enabled, grad, gradcheck

from functools import reduce
from operator import mul


class InvertibleConv1x1(nn.Conv1d):
    def __init__(self, c, memory_efficient=False):
        super().__init__(c, c, 1, bias=False)
        W = torch.randn(c, c).qr()[0]
        self.weight.data = W[..., None]
        if memory_efficient:
            self.efficient_forward = Conv1x1Func.apply
            self.efficient_inverse = InvConv1x1Func.apply

    def forward(self, x):
        if hasattr(self, 'efficient_forward'):
            z, log_det_W = self.efficient_forward(x, self.weight)
            x.storage().resize_(0)
            return z, log_det_W
        else:
            *_, n_of_groups = x.shape
            log_det_W = n_of_groups * self.weight.squeeze().slogdet()[1]  # should fix nan logdet
            z = super().forward(x)
            return z, log_det_W

    def inverse(self, z):
        if hasattr(self, 'efficient_inverse'):
            x, log_det_W = self.efficient_inverse(z, self.weight)
            z.storage().resize_(0)
            return x, log_det_W
        else:
            weight = self.weight.squeeze()
            *_, n_of_groups = z.shape
            log_det_W = -n_of_groups * weight.slogdet()[1]  # should fix nan logdet
            x = F.conv1d(z, weight.inverse().unsqueeze(-1))
            return x, log_det_W


class AffineCouplingBlock(nn.Module):
    def __init__(self,
                 transform_type,
                 memory_efficient=True,
                 **kwargs):
        super().__init__()

        self.F = transform_type(**kwargs)
        if memory_efficient:
            self.efficient_forward = AffineCouplingFunc.apply
            self.efficient_inverse = InvAffineCouplingFunc.apply
            self.param_list = list(self.F.parameters())

    def forward(self, x, y):
        if hasattr(self, 'efficient_forward'):
            z, log_s = self.efficient_forward(x, y, self.F, *self.param_list)
            x.storage().resize_(0)
            return z, log_s
        else:
            xa, xb = x.chunk(2, 1)
            za = xa
            log_s, t = self.F(xa, y)
            zb = xb * log_s.exp() + t
            z = torch.cat((za, zb), 1)
            return z, log_s

    def inverse(self, z, y):
        if hasattr(self, 'efficient_inverse'):
            x, log_s = self.efficient_inverse(z, y, self.F, *self.param_list)
            z.storage().resize_(0)
            return x, log_s
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
        with torch.no_grad():
            xa, xb = x.chunk(2, 1)
            xa, xb = xa.contiguous(), xb.contiguous()

            log_s, t = F(xa, y)
            zb = xb * log_s.exp() + t
            za = xa
            z = torch.cat((za, zb), 1)

        ctx.save_for_backward(x.data, y, z)
        return z, log_s

    @staticmethod
    def backward(ctx, z_grad, log_s_grad):
        F = ctx.F
        x, y, z = ctx.saved_tensors

        za, zb = z.chunk(2, 1)
        za, zb = za.contiguous(), zb.contiguous()
        dza, dzb = z_grad.chunk(2, 1)
        dza, dzb = dza.contiguous(), dzb.contiguous()

        with set_grad_enabled(True):
            xa = za
            xa.requires_grad = True
            log_s, t = F(xa, y)

        with torch.no_grad():
            s = log_s.exp()
            xb = (zb - t) / s
            xout = torch.cat((xa, xb), 1)  # .contiguous()
            x.storage().resize_(reduce(mul, xout.shape))
            x.copy_(xout)  # .detach()

        with set_grad_enabled(True):
            param_list = [xa] + list(F.parameters())
            if ctx.needs_input_grad[1]:
                param_list += [y]
            dtsdxa, *dw = grad(torch.cat((log_s, t), 1), param_list,
                               grad_outputs=torch.cat((dzb * xb * s + log_s_grad, dzb), 1))

            dxa = dza + dtsdxa
            dxb = dzb * s
            dx = torch.cat((dxa, dxb), 1)
            if ctx.needs_input_grad[1]:
                *dw, dy = dw
            else:
                dy = None

        return (dx, dy, None) + tuple(dw)


class InvAffineCouplingFunc(Function):
    @staticmethod
    def forward(ctx, z, y, F, *F_weights):
        ctx.F = F
        with torch.no_grad():
            za, zb = z.chunk(2, 1)
            za, zb = za.contiguous(), zb.contiguous()

            log_s, t = F(za, y)
            xb = (zb - t) / log_s.exp()
            xa = za
            x = torch.cat((xa, xb), 1)

        ctx.save_for_backward(z.data, y, x)
        return x, -log_s

    @staticmethod
    def backward(ctx, x_grad, log_s_grad):
        F = ctx.F
        z, y, x = ctx.saved_tensors

        xa, xb = x.chunk(2, 1)
        xa, xb = xa.contiguous(), xb.contiguous()
        dxa, dxb = x_grad.chunk(2, 1)
        dxa, dxb = dxa.contiguous(), dxb.contiguous()

        with set_grad_enabled(True):
            za = xa
            za.requires_grad = True
            log_s, t = F(za, y)
            s = log_s.exp()

        with torch.no_grad():
            zb = xb * s + t
            zout = torch.cat((za, zb), 1)
            z.storage().resize_(reduce(mul, zout.shape))
            z.copy_(zout)

        with set_grad_enabled(True):
            param_list = [za] + list(F.parameters())
            if ctx.needs_input_grad[1]:
                param_list += [y]
            dtsdza, *dw = grad(torch.cat((-log_s, -t / s), 1), param_list,
                               grad_outputs=torch.cat((dxb * zb / s.detach() + log_s_grad, dxb), 1))

            dza = dxa + dtsdza
            dzb = dxb / s.detach()
            dz = torch.cat((dza, dzb), 1)
            if ctx.needs_input_grad[1]:
                *dw, dy = dw
            else:
                dy = None
        return (dz, dy, None) + tuple(dw)


class Conv1x1Func(Function):
    @staticmethod
    def forward(ctx, x, weight):
        with torch.no_grad():
            *_, n_of_groups = x.shape
            log_det_W = weight.squeeze().slogdet()[1]
            log_det_W *= n_of_groups
            z = F.conv1d(x, weight)

        ctx.save_for_backward(x.data, weight, z)
        return z, log_det_W

    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        x, weight, z = ctx.saved_tensors
        *_, n_of_groups = z.shape

        with torch.no_grad():
            inv_weight = weight.squeeze().inverse()
            xout = F.conv1d(z, inv_weight.unsqueeze(-1))

            x.storage().resize_(reduce(mul, xout.shape))
            x.copy_(xout)

            dx = F.conv1d(z_grad, weight[..., 0].t().unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight.shape[0], -1) @ xout.transpose(1, 2).contiguous().view(
                -1, weight.shape[1])
            dw += inv_weight.t() * log_det_W_grad * n_of_groups

        return dx, dw.unsqueeze(-1)


class InvConv1x1Func(Function):
    @staticmethod
    def forward(ctx, x, inv_weight):
        with torch.no_grad():
            sqr_inv_weight = inv_weight.squeeze()
            *_, n_of_groups = x.shape
            log_det_W = -sqr_inv_weight.slogdet()[1]
            log_det_W *= n_of_groups
            z = F.conv1d(x, sqr_inv_weight.inverse().unsqueeze(-1))

        ctx.save_for_backward(x.data, inv_weight, z)
        return z, log_det_W

    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        x, inv_weight, z = ctx.saved_tensors
        *_, n_of_groups = z.shape

        with torch.no_grad():
            xout = F.conv1d(z, inv_weight)

            x.storage().resize_(reduce(mul, xout.shape))
            x.copy_(xout)

            inv_weight = inv_weight.squeeze()
            weight_T = inv_weight.inverse().t()
            dx = F.conv1d(z_grad, weight_T.unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight_T.shape[0], -1) @ \
                 xout.transpose(1, 2).contiguous().view(-1, weight_T.shape[1])
            dinvw = - weight_T @ dw @ weight_T
            dinvw -= weight_T * log_det_W_grad * n_of_groups

        return dx, dinvw.unsqueeze(-1)
