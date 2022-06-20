import argparse
import torch
from pathlib import Path
import argparse
import torchaudio
from kazane import Decimate
from functools import partial
from tqdm import tqdm
from model.condition import STFTDecimate
from model import LightModel


class LSD(torch.nn.Module):
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, y_hat, y):
        Y_hat = torch.stft(y_hat, self.n_fft, hop_length=self.hop_length,
                           window=self.window, return_complex=True)
        Y = torch.stft(y, self.n_fft, hop_length=self.hop_length,
                       window=self.window, return_complex=True)
        sp = Y_hat.abs().square_().clamp_(min=1e-8).log10_()
        st = Y.abs().square_().clamp_(min=1e-8).log10_()
        return (sp - st).square_().mean(0).sqrt_().mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vctk', type=str)
    parser.add_argument('--ckpt', type=str,
                        default='../WSRGlow/ckpt/x2_best.pt')
    parser.add_argument('--downsample-type', type=str,
                        choices=['sinc', 'stft'], default='stft')

    args = parser.parse_args()

    checkpoint = args.ckpt
    model = LightModel.load_from_checkpoint(checkpoint)
    model.eval()
    model = model.cuda()

    sinc_kwargs = {
        'q': 2,
        'roll_off': 0.962,
        'num_zeros': 128,
        'window_func': partial(torch.kaiser_window, periodic=False,
                               beta=14.769656459379492),
    }

    if args.downsample_type == 'sinc':
        downsampler = Decimate(**sinc_kwargs)
    else:
        downsampler = STFTDecimate(sinc_kwargs['q'])
    downsampler = downsampler.cuda()
    evaluater = LSD().cuda()
    vctk_path = Path(args.vctk)
    test_files = list(vctk_path.glob('*/*.wav'))

    pbar = tqdm(total=len(test_files))

    lsd_list = []
    for filename in test_files:
        raw_y, sr = torchaudio.load(filename)
        raw_y = raw_y.cuda()
        offset = raw_y.shape[1] % 16
        if offset > 0:
            y = raw_y[:, :-offset]
        else:
            y = raw_y

        y_lowpass = downsampler(y)
        with torch.no_grad():
            y_hat, _ = model.model.reverse(torch.randn_like(y), y_lowpass)
        y_hat = y_hat.squeeze()

        if offset > 0:
            y_hat = torch.cat([y_hat, y_hat.new_zeros(offset)], dim=0)
        raw_y = raw_y.squeeze()
        lsd = evaluater(y_hat, raw_y).item()
        lsd_list.append(lsd)
        pbar.set_postfix(lsd=lsd)
        pbar.update(1)

    print(sum(lsd_list) / len(lsd_list))
