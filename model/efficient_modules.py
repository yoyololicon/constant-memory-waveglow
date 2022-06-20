import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
from torch.autograd import Function, set_grad_enabled, grad, gradcheck
from torch.cuda.amp import custom_fwd, custom_bwd

from .base import Reversible


__all__ = [
    'InvertibleConv1x1', 'AffineCouplingBlock'
]


class InvertibleConv1x1(Reversible, nn.Conv1d):
    def __init__(self, c, memory_efficient=False, reverse_mode=False):
        super().__init__(in_channels=c, out_channels=c,
                         kernel_size=1, bias=False, reverse_mode=reverse_mode)

        W = torch.linalg.qr(torch.randn(c, c))[0]
        if torch.det(W) < 0:
            W[:, 0] = -W[:, 0]
        # W = torch.eye(c).flip(0)
        self.weight.data[:] = W.contiguous().unsqueeze(-1)
        if memory_efficient:
            self._efficient_forward = Conv1x1Func.apply
            self._efficient_reverse = InvConv1x1Func.apply

    def forward_computation(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if hasattr(self, '_efficient_forward'):
            z, log_det_W = self._efficient_forward(x, self.weight)
            x.storage().resize_(0)
            return z, log_det_W
        else:
            *_, n_of_groups = x.shape
            # should fix nan logdet
            log_det_W = n_of_groups * self.weight.squeeze().logdet()
            z = F.conv1d(x, self.weight)
            return z, log_det_W

    def reverse_computation(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        if hasattr(self, '_efficient_reverse'):
            x, log_det_W = self._efficient_reverse(z, self.weight)
            z.storage().resize_(0)
            return x, log_det_W
        else:
            weight = self.weight.squeeze()
            *_, n_of_groups = z.shape
            log_det_W = -n_of_groups * \
                weight.logdet()  # should fix nan logdet
            x = F.conv1d(z, weight.inverse().unsqueeze(-1))
            return x, log_det_W


class AffineCouplingBlock(Reversible):
    def __init__(self,
                 transform_type,
                 memory_efficient=True,
                 reverse_mode=False,
                 **kwargs):
        super().__init__(reverse_mode)

        self.F = transform_type(**kwargs)
        if memory_efficient:
            self._efficient_forward = AffineCouplingFunc.apply
            self._efficient_reverse = InvAffineCouplingFunc.apply

    def forward_computation(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        if hasattr(self, '_efficient_forward'):
            z, log_s = self._efficient_forward(
                x, y, self.F, *self.F.parameters())
            x.storage().resize_(0)
            return z, log_s
        else:
            xa, xb = x.chunk(2, 1)
            za = xa
            log_s, t = self.F(xa, y)
            zb = xb * log_s.exp() + t
            z = torch.cat((za, zb), 1)
            return z, log_s

    def reverse_computation(self, z: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        if hasattr(self, '_efficient_reverse'):
            x, log_s = self._efficient_reverse(
                z, y, self.F, *self.F.parameters())
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
    @custom_fwd
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
    @custom_bwd
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
            x.storage().resize_(xb.numel() * 2)
            torch.cat((xa, xb), 1, out=x)  # .contiguous()
            # x.copy_(xout)  # .detach()

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
    @custom_fwd
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
    @custom_bwd
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

            z.storage().resize_(zb.numel() * 2)
            torch.cat((za, zb), 1, out=z)
            # z.copy_(zout)

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
    @custom_fwd
    def forward(ctx, x, weight):
        with torch.no_grad():
            *_, n_of_groups = x.shape
            log_det_W = weight.squeeze().logdet()
            log_det_W *= n_of_groups
            z = F.conv1d(x, weight)

        ctx.save_for_backward(x.data, weight, z)
        return z, log_det_W

    @staticmethod
    @custom_bwd
    def backward(ctx, z_grad, log_det_W_grad):
        x, weight, z = ctx.saved_tensors
        *_, n_of_groups = z.shape

        with torch.no_grad():
            inv_weight = weight.squeeze().inverse()
            x.storage().resize_(z.numel())
            x[:] = F.conv1d(z, inv_weight.unsqueeze(-1))

            dx = F.conv1d(z_grad, weight.transpose(0, 1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight.shape[0], -1) @ x.transpose(1, 2).contiguous().view(
                -1, weight.shape[1])
            dw += inv_weight.t() * log_det_W_grad * n_of_groups

        return dx, dw.unsqueeze(-1)


class InvConv1x1Func(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, inv_weight):
        with torch.no_grad():
            sqr_inv_weight = inv_weight.squeeze()
            *_, n_of_groups = x.shape
            log_det_W = -sqr_inv_weight.logdet()
            log_det_W *= n_of_groups
            z = F.conv1d(x, sqr_inv_weight.inverse().unsqueeze(-1))

        ctx.save_for_backward(x.data, inv_weight, z)
        return z, log_det_W

    @staticmethod
    @custom_bwd
    def backward(ctx, z_grad, log_det_W_grad):
        x, inv_weight, z = ctx.saved_tensors
        *_, n_of_groups = z.shape

        with torch.no_grad():
            x.storage().resize_(z.numel())
            x[:] = F.conv1d(z, inv_weight)

            inv_weight = inv_weight.squeeze()
            weight_T = inv_weight.inverse().t()
            dx = F.conv1d(z_grad, weight_T.unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight_T.shape[0], -1) @ \
                x.transpose(1, 2).contiguous().view(-1, weight_T.shape[1])
            dinvw = - weight_T @ dw @ weight_T
            dinvw -= weight_T * log_det_W_grad * n_of_groups

        return dx, dinvw.unsqueeze(-1)
