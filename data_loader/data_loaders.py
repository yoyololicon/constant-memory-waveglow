from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import os
import pandas as pd
import random
from librosa import load
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor


class _MusicNetDataset(Dataset):
    """
    MusicNet Dataset.
    """
    train_data = 'train_data'
    test_data = 'test_data'
    metafile = 'musicnet_metadata.csv'

    def __init__(self,
                 data_dir,
                 n_workers,
                 size,
                 sr=None,
                 segment=16384,
                 training=True,
                 category='all'):
        self.segment = segment
        self.sr = sr
        self.data_dir = os.path.expanduser(data_dir)
        self.size = size

        metadata = pd.read_csv(os.path.join(data_dir, self.metafile))
        if category == 'all':
            ids = metadata['id'].values.tolist()
        else:
            idx = metadata.index[metadata['ensemble'] == category]
            ids = metadata.loc[idx]['id'].values.tolist()

        if not training:
            self.data_path = os.path.join(self.data_dir, self.test_data)
        else:
            self.data_path = os.path.join(self.data_dir, self.train_data)

        self.waves = []
        load_fn = partial(load, sr=self.sr)
        with ProcessPoolExecutor(n_workers) as executor:
            futures = [executor.submit(load_fn, os.path.join(self.data_path, f)) for f in os.listdir(self.data_path) if
                       f.endswith('.wav') and int(f[:-4]) in ids]
            for future in tqdm(futures):
                y, _ = future.result()
                self.waves.append(y)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        wav = random.choice(self.waves)
        pos = random.randint(0, len(wav) - self.segment - 1)
        x = wav[pos:pos + self.segment]
        return x


class _WAVDataset(Dataset):
    """
    Wave-file-only Dataset.
    """

    def __init__(self,
                 data_dir,
                 size,
                 segment):
        self.segment = segment
        self.data_path = os.path.expanduser(data_dir)
        self.size = size

        self.waves = []
        self.sr = None
        for f in os.listdir(self.data_path):
            if f.endswith('.wav'):
                y, sr = load(os.path.join(self.data_path, f), sr=None)
                if not self.sr:
                    self.sr = sr
                else:
                    assert sr == self.sr
                self.waves.append(y)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        wav = random.choice(self.waves)
        pos = random.randint(0, len(wav) - self.segment - 1)
        x = wav[pos:pos + self.segment]
        return x


class MusicNetDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, **kwargs):
        self.data_dir = data_dir
        self.dataset = _MusicNetDataset(data_dir, num_workers, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class RandomWaveFileLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, steps, data_dir, batch_size, num_workers, **kwargs):
        self.data_dir = data_dir
        self.dataset = _WAVDataset(data_dir, batch_size * steps, ** kwargs)
        super().__init__(self.dataset, batch_size, False, 0., num_workers)
