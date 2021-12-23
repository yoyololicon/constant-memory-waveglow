import os
import argparse
import torch
from torch.cuda import amp
import torchaudio
from utils import remove_weight_norms
from time import time
import math

from model import LightModel, condition


def main(ckpt, infile, outfile, sigma, half, n_group=None):
    lit_model = LightModel.load_from_checkpoint(ckpt, map_location='cpu')
    model = lit_model.model
    conditioner = lit_model.conditioner
    model.apply(remove_weight_norms)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    conditioner = conditioner.to(device)
    model.eval()

    y, sr = torchaudio.load(infile)
    y = y.mean(0, keepdim=True).to(device)

    if n_group:
        offset = y.shape[1] % n_group
        if offset:
            y = y[:, :-offset]
    cond = conditioner(y)

    if half:
        model = model.half()
        cond = cond.half()
        y = y.half()

    with torch.no_grad():
        start = time()
        z, logdet = model(y, cond)
        cost = time() - start
        z = z.squeeze()

    print(z.mean().item(), z.std().item())
    print("Forward LL:", logdet.mean().item() / z.size(0) - 0.5 *
          (z.pow(2).mean().item() / sigma ** 2 + math.log(2 * math.pi) + 2 * math.log(sigma)))
    print("Time cost: {:.4f}, Speed: {:.4f} kHz".format(
        cost, z.numel() / cost / 1000))

    with torch.no_grad():
        start = time()
        x = model.infer(cond, sigma)
        cost = time() - start

    print("Time cost: {:.4f}, Speed: {:.4f} kHz".format(
        cost, x.numel() / cost / 1000))
    print(x.max().item(), x.min().item())

    torchaudio.save(outfile, x.unsqueeze(0).cpu(), sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inferencer')
    parser.add_argument('ckpt', type=str)
    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('-s', '--sigma', type=float, default=0.6)
    parser.add_argument('-n', '--n-group', type=int, default=None)
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    main(args.ckpt, args.infile, args.outfile,
         args.sigma, args.half, args.n_group)
