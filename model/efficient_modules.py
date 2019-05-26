import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, set_grad_enabled, grad, gradcheck
from operator import add

from utils.util import add_weight_norms


class _NonCausalLayer(nn.Module):
    def __init__(self,
                 dilation,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 aux_channels,
                 radix,
                 bias,
                 last_layer=False):
        super().__init__()
        pad_size = dilation * (radix - 1) // 2
        self.WV = nn.Conv1d(residual_channels + aux_channels, dilation_channels * 2, kernel_size=radix,
                            padding=pad_size, dilation=dilation, bias=bias)

        self.chs_split = [skip_channels]
        if last_layer:
            self.W_o = nn.Conv1d(dilation_channels, skip_channels, 1, bias=bias)
        else:
            self.W_o = nn.Conv1d(dilation_channels, residual_channels + skip_channels, 1, bias=bias)
            self.chs_split.insert(0, residual_channels)

    def forward(self, x, y):
        xy = torch.cat((x, y), 1)
        zw, zf = self.WV(xy).chunk(2, 1)
        z = zw.tanh_() * zf.sigmoid_()
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        return z[0] + x if len(z) else None, skip


class WN(nn.Module):
    def __init__(self,
                 in_channels,
                 aux_channels,
                 dilation_channels=512,
                 residual_channels=512,
                 skip_channels=256,
                 depth=8,
                 radix=3,
                 bias=False):
        super().__init__()
        dilations = radix ** torch.arange(depth)
        self.dilations = dilations.tolist()
        self.in_chs = in_channels
        self.res_chs = residual_channels
        self.dil_chs = dilation_channels
        self.skp_chs = skip_channels
        self.aux_chs = aux_channels
        self.rdx = radix
        self.r_field = sum(self.dilations) + 1

        self.start = nn.Conv1d(in_channels, residual_channels, 1, bias=bias)
        self.start.apply(add_weight_norms)

        self.layers = nn.ModuleList(_NonCausalLayer(d,
                                                    dilation_channels,
                                                    residual_channels,
                                                    skip_channels,
                                                    aux_channels,
                                                    radix,
                                                    bias) for d in self.dilations[:-1])
        self.layers.append(_NonCausalLayer(self.dilations[-1],
                                           dilation_channels,
                                           residual_channels,
                                           skip_channels,
                                           aux_channels,
                                           radix,
                                           bias,
                                           last_layer=True))
        self.layers.apply(add_weight_norms)

        self.end = nn.Conv1d(skip_channels, in_channels * 2, 1, bias=bias)
        self.end.weight.data.zero_()
        if bias:
            self.end.bias.data.zero_()

    def forward(self, x, y):
        x = self.start(x)
        cum_skip = 0
        for layer in self.layers:
            x, skip = layer(x, y)
            cum_skip = cum_skip + skip
        return self.end(cum_skip).chunk(2, 1)


class InvertibleConv1x1(nn.Conv1d):
    def __init__(self, c, memory_efficient=True):
        super().__init__(c, c, 1, bias=False)
        W = torch.randn(c, c).qr()[0]
        self.weight.data = W[..., None]
        if memory_efficient:
            self.efficient_forward = Invertible1x1Func.apply
            self.efficient_inverse = Invertible1x1Func.apply

    def forward(self, z):
        if hasattr(self, 'efficient_forward'):
            return self.efficient_forward(z, self.weight)
        else:
            *_, n_of_groups = z.shape
            log_det_W = n_of_groups * self.weight.squeeze().slogdet()[1]  # should fix nan logdet
            z = super().forward(z)
            return z, log_det_W

    def inverse(self, z):
        if hasattr(self, 'efficient_inverse'):
            return self.efficient_inverse(z, self.weight.squeeze().inverse().unsqueeze(-1))
        else:
            weight = self.weight.squeeze().inverse()
            *_, n_of_groups = z.shape
            log_det_W = n_of_groups * weight.slogdet()[1]  # should fix nan logdet
            z = F.conv1d(z, weight.unsqueeze(-1))
            return z, log_det_W


class AffineCouplingBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 *args,
                 memory_efficient=True,
                 last_layer=False,
                 **kwargs):
        super().__init__()

        self.F = WN(in_channels // 2, *args, **kwargs)
        if memory_efficient:
            self.efficient_forward = AffineCouplingFunc.apply
            self.efficient_inverse = InvAffineCouplingFunc.apply
            self.param_list = list(self.F.parameters())

    def forward(self, x, y):
        if hasattr(self, 'efficient_forward'):
            return self.efficient_forward(x, y, self.F, *self.param_list)
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
        z, = ctx.saved_tensors
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
            tgrads = grad(t, param_list, grad_outputs=dzb, retain_graph=True)
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
        x, = ctx.saved_tensors
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
            tgrads = grad(-t, param_list, grad_outputs=dxb / s, retain_graph=True)
            sgrads = grad(-log_s, param_list, grad_outputs=dxb * zb / s + log_s_grad)

            dw = tuple(map(add, tgrads[1:], sgrads[1:]))
            dza += tgrads[0] + sgrads[0]
            dzb = dxb / s
            dz = torch.cat((dza, dzb), 1)
        return (dz, None, None) + dw


class Invertible1x1Func(Function):
    @staticmethod
    def forward(ctx, z, weight):
        with torch.no_grad():
            *_, n_of_groups = z.shape
            log_det_W = weight.squeeze().slogdet()[1]
            log_det_W *= n_of_groups
            z = F.conv1d(z, weight)

        ctx.save_for_backward(z, weight)
        return z, log_det_W

    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        z, weight = ctx.saved_tensors
        *_, n_of_groups = z.shape

        with torch.no_grad():
            inv_weight = weight.squeeze().inverse()
            z2 = F.conv1d(z, inv_weight.unsqueeze(-1))
            dz = F.conv1d(z_grad, weight[..., 0].t().unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight.shape[0], -1) @ z2.transpose(1, 2).contiguous().view(
                -1, weight.shape[1])
            dw += inv_weight.t() * log_det_W_grad * n_of_groups

        return dz, dw.unsqueeze(-1)


if __name__ == '__main__':
    """
    x = torch.randn(10, 100, 100)
    conv1 = InvertibleConv1x1(100, False)
    # conv1.weight.data.normal_()
    conv2 = InvertibleConv1x1(100, True)
    conv2.weight.data.copy_(conv1.weight.data)
    y1, log1 = conv1.inverse(x)
    y2, log2 = conv2.inverse(x)
    # log2 = log1 = 0
    #print(y1, y2, log1.item(), log2.item())
    y1.sum().add(log1).backward()
    y2.sum().add(log2).backward()
    #print(conv1.weight.grad, conv2.weight.grad)
    print((conv1.weight.grad - conv2.weight.grad).pow(2).mean().sqrt())
    """
    x, y = torch.randn(1, 10, 100).double(), torch.randn(1, 2, 100).double()
    conv = AffineCouplingBlock(10, 2).double()
    x.requires_grad = True
    print(gradcheck(conv, (x, y)))
