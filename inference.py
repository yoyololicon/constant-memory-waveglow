import os
import argparse
import torch
import model.model as module_arch
from utils.util import remove_weight_norms
from train import get_instance
from librosa import load
import soundfile as sf
from time import time
import math


def main(config, resume, infile, outfile, sigma, dur, half):
    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    if config['n_gpu'] > 1:
        model = model.module
    model.apply(remove_weight_norms)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    sr = config['arch']['args']['sr']
    n_group = config['arch']['args']['n_group']
    y, _ = load(infile, sr=sr, duration=dur)
    
    offset = len(y) % n_group
    if offset:
        y = y[:-offset]

    y = torch.Tensor(y).to(device)

    # get mel before turn to half, because sparse.half is not implement yet
    mel = model.get_mel(y[None, :])

    if half:
        model = model.half()
        mel = mel.half()
        y = y.half()

    with torch.no_grad():
        start = time()
        z, logdet, _ = model(y[None, :], mel)
        cost = time() - start
        z = z.squeeze()
    
    print(z.mean().item(), z.std().item())
    print("Forward LL:", logdet.mean().item() / z.size(0) - 0.5 * (z.pow(2).mean().item() / sigma ** 2 + math.log(2 * math.pi) + 2 * math.log(sigma)))
    print("Time cost: {:.4f}, Speed: {:.4f} kHz".format(cost, z.numel() / cost / 1000))

    start = time()
    x, logdet = model.infer(mel, sigma)
    cost = time() - start

    print("Backward LL:", -logdet.mean().item() / x.size(0) - 0.5 * (1 + math.log(2 * math.pi) + 2 * math.log(sigma)))

    print("Time cost: {:.4f}, Speed: {:.4f} kHz".format(cost, x.numel() / cost / 1000))
    print(x.max().item(), x.min().item())
    sf.write(outfile, x.cpu().float().numpy(), sr, subtype='PCM_16')
    #write_wav(outfile, x.cpu().float().numpy(), sr, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WaveGlow inference')
    parser.add_argument('infile', type=str, help='wave file to generate mel-spectrogram')
    parser.add_argument('outfile', type=str, help='output file name')
    parser.add_argument('--duration', type=float, help='duration of audio, in seconds')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('-s', '--sigma', type=float, default=1.0)
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume, args.infile, args.outfile, args.sigma, args.duration, args.half)
