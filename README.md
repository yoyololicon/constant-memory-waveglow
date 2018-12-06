# WaveGlow

Another PyTorch implementation of [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002).

Most of the codes are based on NVIDIA [official implementation](https://github.com/NVIDIA/waveglow), and the project structure 
is brought from [pytorch-template](https://github.com/victoresque/pytorch-template).

## Quick Start

Modify the `data_dir` in the json file to a directory which has a bunch of wave files with the same sampling rate, 
then your are good to go. The mel-spectrogram will be computed on the fly.

```json
{
  "data_loader": {
    "type": "RandomWaveFileLoader",
    "args": {
      "data_dir": "/your/data/wave/files",
      "batch_size": 8,
      "num_workers": 2,
      "segment": 16000
    }
  }
}
```

```
python train.py -c config.json
```


## Result

I trained the model on some cello music pieces from MusicNet using the `musicnet_config.json`.
The clips in the `samples` folder is what I got. Although the audio quality is not very good, it's possible to use 
WaveGlow on music generation as well. 
The generation speed is around 470kHz on a 1080ti.