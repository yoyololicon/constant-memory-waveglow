import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, set_grad_enabled, grad, gradcheck


class InvertibleConv1x1(nn.Conv1d):
    def __init__(self, c, memory_efficient=False):
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

        ctx.save_for_backward(x.clone())
        return x, log_s

    @staticmethod
    def backward(ctx, z_grad, log_s_grad):
        F = ctx.F
        y = ctx.y
        z, = ctx.saved_tensors

        za, zb = z.chunk(2, 1)
        dza, dzb = z_grad.chunk(2, 1)

        xa = za
        xa.requires_grad = True
        with set_grad_enabled(True):
            log_s, t = F(xa, y)

        with torch.no_grad():
            s = log_s.exp()
            xb = (zb - t) / s

        param_list = [xa, y] + list(F.parameters())
        with set_grad_enabled(True):
            # tgrads = grad(t, param_list, grad_outputs=dzb, retain_graph=True)
            dtsdxa, dy, *dw = grad(torch.cat((log_s, t), 1), param_list,
                                   grad_outputs=torch.cat((dzb * xb * s + log_s_grad, dzb), 1))

            dxa = dza + dtsdxa
            dxb = dzb * s
            dx = torch.cat((dxa, dxb), 1)
        return (dx, dy, None) + tuple(dw)


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

        ctx.save_for_backward(z.clone())
        return z, -log_s

    @staticmethod
    def backward(ctx, x_grad, log_s_grad):
        F = ctx.F
        y = ctx.y
        x, = ctx.saved_tensors
        xa, xb = x.chunk(2, 1)
        dxa, dxb = x_grad.chunk(2, 1)

        za = xa
        za.requires_grad = True
        with set_grad_enabled(True):
            log_s, t = F(za, y)

        with torch.no_grad():
            s = log_s.exp()
            zb = xb * s + t

        param_list = [za, y] + list(F.parameters())
        with set_grad_enabled(True):
            # tgrads = grad(-t, param_list, grad_outputs=dxb / s, retain_graph=True)
            dtsdza, dy, *dw = grad(torch.cat((-log_s, -t), 1), param_list,
                                   grad_outputs=torch.cat((dxb * zb / s + log_s_grad, dxb / s), 1))

            dza = dxa + dtsdza
            dzb = dxb / s
            dz = torch.cat((dza, dzb), 1)
        return (dz, dy, None) + tuple(dw)


class Invertible1x1Func(Function):
    @staticmethod
    def forward(ctx, z, weight):
        with torch.no_grad():
            *_, n_of_groups = z.shape
            log_det_W = weight.squeeze().slogdet()[1]
            log_det_W *= n_of_groups
            z[:] = F.conv1d(z, weight)

        ctx.save_for_backward(z.clone(), weight)
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
    x = torch.randn(10, 100, 100)
    conv1 = InvertibleConv1x1(100, False)
    # conv1.weight.data.normal_()
    conv2 = InvertibleConv1x1(100, True)
    conv2.weight.data.copy_(conv1.weight.data)
    y1, log1 = conv1.inverse(x)
    y2, log2 = conv2.inverse(x.clone())
    # log2 = log1 = 0
    # print(y1, y2, log1.item(), log2.item())
    y1.sum().add(log1).backward()
    y2.sum().add(log2).backward()
    # print(conv1.weight.grad, conv2.weight.grad)
    print((conv1.weight.grad - conv2.weight.grad).pow(2).mean().sqrt())

    # x, y = torch.randn(1, 10, 100).double(), torch.randn(1, 2, 100).double()
    # conv = AffineCouplingBlock(10, 2).double()
    # x.requires_grad = True
    # print(gradcheck(conv, (x, y)))
