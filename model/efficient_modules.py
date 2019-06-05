import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, set_grad_enabled, grad, gradcheck

from functools import reduce
from operator import mul


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
            x.storage().resize_(reduce(mul, xb.shape) * 2)
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

            z.storage().resize_(reduce(mul, zb.shape) * 2)
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


class SqueezeStrideConv(nn.Conv1d):
    def __init__(self, in_channels, stride=1, memory_efficient=False, bias=False, just_squeeze=False):
        super().__init__(in_channels, in_channels * stride, stride, stride=stride, bias=bias and not just_squeeze)
        if just_squeeze:
            del self._parameters['weight']
            W = torch.eye(in_channels * stride)
            self.register_buffer('weight', W.view(-1, stride, in_channels).transpose(1, 2))
        else:
            W = torch.randn(in_channels * stride, in_channels * stride).qr()[0]
            self.weight.data.copy_(W.view(-1, stride, in_channels).transpose(1, 2))
        if memory_efficient and not just_squeeze:
            self.efficient_forward = SqueezeStrideConvFunc.apply
            self.efficient_inverse = InvSqueezeStrideConvFunc.apply

    def forward(self, x):
        if hasattr(self, 'efficient_forward'):
            z, log_det_W = self.efficient_forward(x, self.weight, self.bias)
            x.storage().resize_(0)
            return z, log_det_W
        else:
            log_det_W = self.weight.transpose(1, 2).contiguous().view(self.out_channels, -1).slogdet()[1]
            z = super().forward(x)
            *_, n_of_groups = z.shape
            return z, log_det_W * n_of_groups

    def inverse(self, z):
        if hasattr(self, 'efficient_inverse'):
            x, log_det_W = self.efficient_inverse(z, self.weight, self.bias)
            z.storage().resize_(0)
            return x, log_det_W
        else:
            weight = self.weight.transpose(1, 2).contiguous().view(self.out_channels, -1)
            *_, n_of_groups = z.shape
            log_det_W = -n_of_groups * weight.slogdet()[1]
            if self.bias is not None:
                z = z - self.bias.unsqueeze(-1)
            weight = weight.inverse().view(self.stride[0], self.in_channels, -1).transpose(0, 2)
            x = F.conv_transpose1d(z, weight, stride=self.stride)
            return x, log_det_W


class SqueezeStrideConvFunc(Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        out_channels, in_channels, stride = weight.shape
        with torch.no_grad():
            log_det_W = weight.transpose(1, 2).contiguous().view(weight.shape[0], -1).slogdet()[1]
            z = F.conv1d(x, weight, stride=stride, bias=bias)
            *_, n_of_groups = z.shape
            log_det_W *= n_of_groups

        ctx.save_for_backward(x.data, weight, bias, z)
        return z, log_det_W

    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        x, weight, bias, z = ctx.saved_tensors
        batch, *_, n_of_groups = z.shape
        out_channels, in_channels, stride = weight.shape

        with torch.no_grad():
            weight = weight.transpose(1, 2).contiguous().view(out_channels, -1)
            inv_weight = weight.inverse()
            weight_T = weight.t()
            if bias is None:
                db = None
            else:
                z = z - bias.unsqueeze(-1)
                db = z_grad.sum((0, 2))
            x.storage().resize_(reduce(mul, z.shape))
            x[:] = F.conv_transpose1d(z, inv_weight.view(stride, in_channels, out_channels).transpose(0, 2),
                                      stride=stride)

            dx = F.conv_transpose1d(z_grad, weight_T.view(stride, in_channels, out_channels).transpose(0, 2),
                                    stride=stride)
            dw = z_grad.transpose(0, 1).contiguous().view(out_channels, -1) @ \
                 x.view(batch, in_channels, -1, stride).permute(0, 2, 3, 1).contiguous().view(-1, out_channels)
            dw += inv_weight.t() * log_det_W_grad * n_of_groups

        return dx, dw.view(out_channels, stride, in_channels).transpose(1, 2), db


class InvSqueezeStrideConvFunc(Function):
    @staticmethod
    def forward(ctx, z, inv_weight, bias):
        out_channels, in_channels, stride = inv_weight.shape
        with torch.no_grad():
            sqr_inv_weight = inv_weight.transpose(1, 2).contiguous().view(out_channels, -1)
            *_, n_of_groups = z.shape
            log_det_W = -sqr_inv_weight.slogdet()[1]
            log_det_W *= n_of_groups
            if bias is not None:
                z = z - bias.unsqueeze(-1)
            x = F.conv_transpose1d(z, sqr_inv_weight.inverse().view(stride, in_channels, -1).transpose(0, 2),
                                   stride=stride)

        ctx.save_for_backward(z.data, inv_weight, bias, x)
        return x, log_det_W

    @staticmethod
    def backward(ctx, x_grad, log_det_W_grad):
        z, inv_weight, bias, x = ctx.saved_tensors
        out_channels, in_channels, stride = inv_weight.shape
        batch, *_ = x.shape

        with torch.no_grad():
            z.storage().resize_(reduce(mul, x.shape))
            *_, n_of_groups = z.shape
            z[:] = F.conv1d(x, inv_weight, stride=stride, bias=bias)

            inv_weight = inv_weight.transpose(1, 2).contiguous().view(out_channels, -1)
            weight_T = inv_weight.inverse().t()
            dz = F.conv1d(x_grad, weight_T.view(out_channels, stride, in_channels).transpose(1, 2), stride=stride)
            if bias is None:
                db = None
            else:
                db = -dz.sum((0, 2))
            dw = x_grad.view(batch, in_channels, -1, stride).permute(3, 1, 0, 2).contiguous().view(out_channels, -1) \
                 @ (z - bias.unsqueeze(-1)).transpose(1, 2).contiguous().view(-1, out_channels)
            dinvw = - weight_T @ dw @ weight_T
            dinvw -= weight_T * log_det_W_grad * n_of_groups

        return dz, dinvw.view(out_channels, stride, in_channels).transpose(1, 2), db


if __name__ == '__main__':
    m1 = SqueezeStrideConv(2, 2, just_squeeze=False, bias=True, memory_efficient=True)
    m2 = SqueezeStrideConv(2, 2, just_squeeze=False, bias=True, memory_efficient=False)
    m2.load_state_dict(m1.state_dict())

    from copy import deepcopy
    x = torch.arange(20.).view(1, 5, 4).transpose(1, 2)
    x.requires_grad = True
    xclone = deepcopy(x)
    print(x)
    x1, logdet = m1.inverse(x)
    loss = x1.pow(2).sum() #- logdet.sum()
    loss.backward()
    print(x.grad, m1.weight.grad, m1.bias.grad)

    x2, logdet = m2.inverse(xclone)
    print('hey')
    loss = x2.pow(2).sum() #- logdet.sum()

    loss.backward()
    print(xclone.grad, m2.weight.grad, m2.bias.grad)
    # x = m.inverse(x)[0]
    # print(x)
