import pytest
import torch
import numpy as np
from torch import nn

from model.efficient_modules import AffineCouplingBlock, InvertibleConv1x1
from model.model import WN
from model.loss import WaveGlowLoss


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.mark.parametrize('batch', list(2 ** i for i in range(6)))
@pytest.mark.parametrize('channels', list(2 ** i for i in range(1, 4)))
@pytest.mark.parametrize('length', [2000])
def test_conv1x1_fwd_bwd(batch, channels, length):
    data = torch.rand(batch, channels, length) * 2 - 1
    weights = InvertibleConv1x1(channels).state_dict()

    loss_func = WaveGlowLoss().cuda()

    for seed in range(10):
        set_seed(seed)
        for bwd in [False, True]:
            impl_out, impl_grad = [], []
            for keep_input in [True, False]:
                model = InvertibleConv1x1(channels, not keep_input)
                model.load_state_dict(weights)
                model = model.cuda()
                model.train()
                model.zero_grad()

                x = data.cuda()

                if bwd:
                    xin = x.clone()
                    y, log1 = model.inverse(xin)
                    yrev = y.clone()
                    xinv, log2 = model(yrev)
                else:
                    xin = x.clone()
                    y, log1 = model(xin)
                    yrev = y.clone()
                    xinv, log2 = model.inverse(yrev)

                # assert log1.item() == -log2.item()
                loss = loss_func(y.view(batch, -1), log1)

                if keep_input:
                    assert xin.data.shape == x.shape
                    assert y.data.shape == yrev.shape
                else:
                    assert len(xin.data.shape) == 0 \
                           or (len(xin.data.shape) == 0 and xin.data.shape[0] == 0) \
                           or xin.storage().size() == 0

                    assert len(yrev.data.shape) == 0 \
                           or (len(yrev.data.shape) == 0 and yrev.data.shape[0] == 0) \
                           or yrev.storage().size() == 0

                loss.backward()

                assert y.shape == x.shape
                assert x.shape == data.shape
                assert torch.allclose(x.cpu(), data)
                print(torch.abs(x - xinv).max().item())
                assert torch.allclose(x, xinv, atol=1e-6, rtol=0)

                impl_out.append(y.detach().cpu())
                impl_grad.append([p.grad.cpu() for p in model.parameters()])

            for i in range(len(impl_grad) // 2):
                print(impl_grad[i * 2][0].view(-1)[:5], impl_grad[i * 2 + 1][0].view(-1)[:5])
                assert torch.allclose(impl_grad[i * 2][0], impl_grad[i * 2 + 1][0], atol=1e-6, rtol=0)
                assert torch.allclose(impl_out[2 * i], impl_out[2 * i + 1])


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('channels', list(2 ** i for i in range(4, 6)))
@pytest.mark.parametrize('WN_channels', [128])
@pytest.mark.parametrize('depth', list(range(1, 5)))
@pytest.mark.parametrize('aux_channels', [20, 40])
@pytest.mark.parametrize('length', [4000])
def test_affine_fwd_bwd(batch, channels, WN_channels, depth, aux_channels, length):
    data = torch.rand(batch, channels, length) * 2 - 1
    condition = torch.randn(batch, aux_channels, length)

    weights = AffineCouplingBlock(WN, False, in_channels=channels // 2, aux_channels=aux_channels,
                                  zero_init=False,
                                  dilation_channels=WN_channels,
                                  residual_channels=WN_channels,
                                  skip_channels=WN_channels,
                                  depth=depth).state_dict()

    loss_func = WaveGlowLoss().cuda()

    for seed in range(10):
        set_seed(seed)
        for bwd in [False, True]:
            impl_out, impl_grad = [], []
            for keep_input in [True, False]:
                model = AffineCouplingBlock(WN, not keep_input, in_channels=channels // 2, aux_channels=aux_channels,
                                            zero_init=False,
                                            dilation_channels=WN_channels,
                                            residual_channels=WN_channels,
                                            skip_channels=WN_channels,
                                            depth=depth)
                model.load_state_dict(weights)
                model = model.cuda()
                model.train()
                model.zero_grad()

                x = data.cuda()
                h = condition.cuda()

                if bwd:
                    xin = x.clone()
                    y, log1 = model.inverse(xin, h)
                    yrev = y.clone()
                    xinv, log2 = model(yrev, h)
                else:
                    xin = x.clone()
                    y, log1 = model(xin, h)
                    yrev = y.clone()
                    xinv, log2 = model.inverse(yrev, h)

                assert log1.sum().item() == -log2.sum().item()
                loss = loss_func(y.view(2, -1), log1.sum((1, 2)))

                if keep_input:
                    assert xin.data.shape == x.shape
                    assert y.data.shape == yrev.shape
                else:
                    assert len(xin.data.shape) == 0 \
                           or (len(xin.data.shape) == 0 and xin.data.shape[0] == 0) \
                           or xin.storage().size() == 0

                    assert len(yrev.data.shape) == 0 \
                           or (len(yrev.data.shape) == 0 and yrev.data.shape[0] == 0) \
                           or yrev.storage().size() == 0
                    assert h.shape == condition.shape
                    assert torch.allclose(h.cpu(), condition)

                loss.backward()

                assert y.shape == x.shape
                assert x.data.shape == data.shape
                assert torch.allclose(x.cpu(), data)
                print(torch.abs(x - xinv).max().item())
                assert torch.allclose(x, xinv, atol=1e-7)

                impl_out.append(y.cpu().detach())
                impl_grad.append([p.grad.cpu() for p in model.parameters()])

            for i in range(len(impl_grad) // 2):
                assert torch.allclose(impl_grad[i * 2][0], impl_grad[i * 2 + 1][0])
                assert torch.allclose(impl_out[2 * i], impl_out[2 * i + 1])


@pytest.mark.parametrize('batch', list(2 ** i for i in range(6)))
@pytest.mark.parametrize('channels', list(2 ** i for i in range(1, 4)))
@pytest.mark.parametrize('length', [2000])
def test_complx_chained(batch, channels, length):
    data = torch.rand(batch, channels, length) * 2 - 1

    model1 = nn.ModuleList([InvertibleConv1x1(channels, True),
                            InvertibleConv1x1(channels, False),
                            InvertibleConv1x1(channels, True)])
    model2 = nn.ModuleList([InvertibleConv1x1(channels, False),
                            InvertibleConv1x1(channels, True),
                            InvertibleConv1x1(channels, False)])
    model2.load_state_dict(model1.state_dict())
    loss_func = WaveGlowLoss().cuda()

    for seed in range(10):
        set_seed(seed)
        impl_grad = []
        for model in [model1, model2]:
            model = model.cuda()
            model.train()
            model.zero_grad()

            x = data.cuda()

            xin = x.clone()
            logdet = 0
            for layer in model:
                xin, _ = layer.inverse(xin)
                logdet = logdet + _

            loss = loss_func(xin.view(batch, -1), logdet)

            loss.backward()
            impl_grad.append([p.grad.cpu() for p in model.parameters()])

        for p_grad1, p_grad2 in zip(impl_grad[0], impl_grad[1]):
            assert torch.allclose(p_grad1, p_grad2, atol=1e-7, rtol=0)
