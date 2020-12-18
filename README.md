# Constant Memory WaveGlow
[![DOI](https://zenodo.org/badge/159754913.svg)](https://zenodo.org/badge/latestdoi/159754913)

A PyTorch implementation of
[WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002)
using constant memory method described in [Training Glow with Constant
Memory Cost](http://bayesiandeeplearning.org/2018/papers/37.pdf).

The model implementation details are slightly differed from the
[official implementation](https://github.com/NVIDIA/waveglow) based on
personal favor, and the project structure is brought from
[pytorch-template](https://github.com/victoresque/pytorch-template).

Besides, we also add implementations of Baidu's [WaveFlow](https://arxiv.org/abs/1912.01219), and [MelGlow](https://arxiv.org/abs/2012.01684), 
which are easier to train and more memory fiendly.

## Requirements

After install the requirements from [pytorch-template](https://github.com/victoresque/pytorch-template#requirements):

```commandline
pip install nnAudio torch_optimizer
```

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

## Memory consumption of model training in PyTorch


| Model                                            |  Memory (MB)  |
---------------------------------------------------|:-------------:|
| WaveGlow, channels=256, batch size=24 (naive)    |    N.A.       |
| WaveGlow, channels=256, batch size=24 (efficient)|    4951       |



## Result

### WaveGlow

I trained the model on some cello music pieces from MusicNet using the `musicnet_config.json`.
The clips in the `samples` folder is what I got. Although the audio quality is not very good, it's possible to use 
WaveGlow on music generation as well. 
The generation speed is around 470kHz on a 1080ti.


### WaveFlow

I trained on full LJ speech dataset using the `waveflow_LJ_speech.json`. The settings are corresponding to the **64 residual channels, h=64** model in the paper. After training about 1.25M steps, the audio quality is very similiar to their official examples.
Samples generated from training data can be listened [here](samples/waveflow_64chs).

### MelGlow

Coming soon.


## Citation
If you use our code on any project and research, please cite:

```bibtex
@misc{memwaveglow,
  doi          = {10.5281/zenodo.3874330},
  author       = {Chin Yun Yu},
  title        = {Constant Memory WaveGlow: A PyTorch implementation of WaveGlow with constant memory cost},
  howpublished = {\url{https://github.com/yoyololicon/constant-memory-waveglow}},
  year         = {2019}
}
```
