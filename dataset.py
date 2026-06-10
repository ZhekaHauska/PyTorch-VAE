import os
import torch
import random
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
import numpy as np


# Add your custom dataset class here
class AnimalAI(Dataset):
    def __init__(self,
                 data,
                 transform: Callable,
                 **kwargs):
        self.transforms = transform
        self.im = data['s']
        self.rew = data['r']
        self.rst = data.get('reset', np.zeros(len(self.im), dtype=np.float32))

    def __len__(self):
        return len(self.im)

    def __getitem__(self, idx):
        img = self.im[idx]
        rew = self.rew[idx]
        rst = self.rst[idx]

        if self.transforms is not None:
            img = self.transforms(img.astype(np.float32))

        return img, rew.astype(np.float32), rst.astype(np.float32)


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 


class AnimalAIEpisodeParallel(IterableDataset):
    def __init__(self,
                 data,
                 transform: Callable,
                 batch_size: int,
                 **kwargs):
        super().__init__()
        self.data = data
        self.transforms = transform
        self.batch_size = batch_size

    def _split_into_episodes(self):
        resets = self.data.get('reset', np.zeros(len(self.data['s']), dtype=np.float32))
        indices = np.where(resets > 0)[0]
        if len(indices) == 0 or indices[0] != 0:
            indices = np.concatenate([[0], indices])
        episodes = []
        for i, start in enumerate(indices):
            end = indices[i + 1] if i + 1 < len(indices) else len(self.data['s'])
            if end - start > 0:
                episodes.append({
                    's': self.data['s'][start:end],
                    'r': self.data['r'][start:end],
                })
        return episodes

    def __iter__(self):
        episodes = self._split_into_episodes()
        random.shuffle(episodes)

        effective_batch_size = min(self.batch_size, len(episodes))
        if effective_batch_size == 0:
            return

        track_episodes = [[] for _ in range(effective_batch_size)]
        for i, ep in enumerate(episodes):
            track_episodes[i % effective_batch_size].append(ep)

        track_ep_idx = [0] * effective_batch_size
        track_frame_idx = [0] * effective_batch_size

        while True:
            batch_imgs = []
            batch_rews = []
            batch_resets = []
            valid = True

            for i in range(effective_batch_size):
                while True:
                    if track_ep_idx[i] >= len(track_episodes[i]):
                        valid = False
                        break
                    ep = track_episodes[i][track_ep_idx[i]]
                    pos = track_frame_idx[i]
                    if pos < len(ep['s']):
                        break
                    track_ep_idx[i] += 1
                    track_frame_idx[i] = 0

                if not valid:
                    break

                ep = track_episodes[i][track_ep_idx[i]]
                pos = track_frame_idx[i]
                img = ep['s'][pos]
                rew = ep['r'][pos]
                rst = 1.0 if pos == 0 else 0.0

                if self.transforms is not None:
                    img = self.transforms(img.astype(np.float32))
                else:
                    img = torch.from_numpy(img.astype(np.float32))

                batch_imgs.append(img)
                batch_rews.append(rew)
                batch_resets.append(rst)
                track_frame_idx[i] += 1

            if not valid or len(batch_imgs) < effective_batch_size:
                break

            yield (torch.stack(batch_imgs),
                   torch.tensor(batch_rews, dtype=torch.float32),
                   torch.tensor(batch_resets, dtype=torch.float32))


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        trace_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.trace_decay = trace_decay

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([transforms.ToTensor()])
        
        val_transforms = transforms.Compose([transforms.ToTensor()])

        data_dir = Path(self.data_dir)
        data = np.load(data_dir)

        data_keys = ['s', 'r']
        if 'reset' in data:
            data_keys.append('reset')

        if self.trace_decay > 0:
            all_resets = data.get('reset', np.zeros(len(data['s']), dtype=np.float32))
            ep_starts = np.where(all_resets > 0)[0]
            if len(ep_starts) == 0 or ep_starts[0] != 0:
                ep_starts = np.concatenate([[0], ep_starts])

            n = len(data['s'])
            n_train = int(n * 0.75)
            train_ep_idx = np.searchsorted(ep_starts, n_train, side='right')
            train_start_idx = 0
            train_end_idx = ep_starts[train_ep_idx] if train_ep_idx < len(ep_starts) else n
            val_start_idx = train_end_idx
            val_end_idx = n

            train_data = {name: data[name][train_start_idx:train_end_idx] for name in data_keys}
            test_data = {name: data[name][val_start_idx:val_end_idx] for name in data_keys}

            self.train_dataset = AnimalAIEpisodeParallel(
                train_data,
                transform=train_transforms,
                batch_size=self.train_batch_size,
            )
        else:
            split_indexes = np.arange(len(data['s']))
            np.random.shuffle(split_indexes)
            train_data = {name: data[name][split_indexes[:int(len(split_indexes) * 0.75)]] for name in data_keys}
            test_data = {name: data[name][split_indexes[int(len(split_indexes) * 0.75):]] for name in data_keys}

            self.train_dataset = AnimalAI(
                train_data,
                transform=train_transforms,
            )

        self.val_dataset = AnimalAI(
            test_data,
            transform=val_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        if self.trace_decay > 0:
            return DataLoader(
                self.train_dataset,
                batch_size=1,
                num_workers=0,
                pin_memory=self.pin_memory,
                collate_fn=lambda x: x[0],
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     