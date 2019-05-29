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


def test_conv1x1_fwd_bwd():
    in_size = 10
    length = 1000

    data = np.random.randn(2, in_size, length)
    weights = InvertibleConv1x1(in_size).state_dict()

    loss_func = WaveGlowLoss().cuda()

    for seed in range(10):
        set_seed(seed)
        for bwd in [False, True]:
            impl_out, impl_grad = [], []
            for keep_input in [True, False]:
                model = InvertibleConv1x1(in_size, not keep_input)
                model.load_state_dict(weights)
                s_grad = [p.data.numpy().copy() for p in model.parameters()]
                x = torch.Tensor(data.copy()).cuda()

                model = model.cuda()
                model.train()

                optim = torch.optim.Adam(model.parameters())
                optim.zero_grad()

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

                assert log1.item() == -log2.item()
                loss = loss_func(y.view(2, -1), log1)

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
                optim.step()

                x = x.cpu()
                y = y.cpu()
                xinv = xinv.cpu()

                assert y.shape == x.shape
                assert x.data.numpy().shape == data.shape
                assert np.allclose(x.data.numpy(), data)
                print(x[0, 0, :10], xinv[0, 0, :10])
                assert np.allclose(x.data.numpy(), xinv.data.numpy(), atol=1e-6)

                impl_out.append(y.data.numpy().copy())
                impl_grad.append([p.data.cpu().numpy().copy() for p in model.parameters()])
                assert not np.allclose(s_grad[0], impl_grad[-1][0])

            for i in range(len(impl_grad) // 2):
                print(impl_grad[i * 2][0].reshape(-1)[:5], impl_grad[i * 2 + 1][0].reshape(-1)[:5])
                assert np.allclose(impl_grad[i * 2][0], impl_grad[i * 2 + 1][0])
                assert np.allclose(impl_out[2 * i], impl_out[2 * i + 1])


def test_affine_fwd_bwd():
    in_size = 10
    aux_size = 40
    length = 1000

    data = np.random.randn(2, in_size, length)
    condition = np.random.randn(2, aux_size, length)

    weights = AffineCouplingBlock(WN, False, in_channels=in_size // 2, aux_channels=aux_size,
                                  zero_init=False).state_dict()

    loss_func = WaveGlowLoss().cuda()

    for seed in range(10):
        set_seed(seed)
        for bwd in [False, True]:
            impl_out, impl_grad = [], []
            for keep_input in [True, False]:
                model = AffineCouplingBlock(WN, not keep_input, in_channels=in_size // 2, aux_channels=aux_size,
                                            zero_init=False)
                model.load_state_dict(weights)
                s_grad = [p.data.numpy().copy() for p in model.parameters()]
                x = torch.Tensor(data.copy()).cuda()
                h = torch.Tensor(condition.copy()).cuda()

                model = model.cuda()
                model.train()

                optim = torch.optim.Adam(model.parameters())
                optim.zero_grad()

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

                loss.backward()
                optim.step()

                x = x.cpu()
                y = y.cpu()
                xinv = xinv.cpu()

                assert y.shape == x.shape
                assert x.data.numpy().shape == data.shape
                assert np.allclose(x.data.numpy(), data)
                assert np.allclose(x.data.numpy(), xinv.data.numpy())

                impl_out.append(y.data.numpy().copy())
                impl_grad.append([p.data.cpu().numpy().copy() for p in model.parameters()])
                assert not np.allclose(s_grad[0], impl_grad[-1][0])

            for i in range(len(impl_grad) // 2):
                assert np.allclose(impl_grad[i * 2][0], impl_grad[i * 2 + 1][0])
                assert np.allclose(impl_out[2 * i], impl_out[2 * i + 1])
