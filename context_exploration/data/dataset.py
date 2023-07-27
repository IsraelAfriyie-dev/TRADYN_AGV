"""
`torch.data.Dataset` from numpy containers

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import os
import re
from collections import defaultdict

import numpy as np
from torch.utils import data
from tqdm import tqdm

__all__ = ["NumpyDataset"]


class NumpyDataset(data.Dataset):
    """ Dataset from numpy containers in directory """

    def __init__(
        self,
        data_dir,
        subsequence_transform=None,
        subsequence_length=None,
        seed_min_incl=None,
        seed_max_incl=None,
        verbose=False,
    ):
        self.data_dir = data_dir
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        self.subsequence_transform = subsequence_transform

        self.seed_min_incl = seed_min_incl
        self.seed_max_incl = seed_max_incl
        self.verbose = verbose
        self.data = self.load_data()
        if subsequence_length is None:
            subsequence_length = self.rollout_length
        self.subsequence_length = subsequence_length

        print("Loaded numpy dataset: ")
        for k in self.data.keys():
            print(f"  {k} - {self.data[k].shape}, {self.data[k].dtype}")
        print(f"Sequence length: {self.subsequence_length}")
        print(f"Total items: {len(self)}")

    def load_data(self):
        seeds = []
        containers = []
        for filename in tqdm(os.listdir(self.data_dir)):
            container_match = re.match("np_container_(\d+)-(\d+)\.npz", filename)
            if container_match is not None:
                start_seed = int(container_match.group(1))
                end_seed = int(container_match.group(2))
            else:
                continue

            seeds.extend(s for s in range(start_seed, end_seed + 1))
            containers.extend(filename for _ in range(start_seed, end_seed + 1))

        if len(containers) == 0:
            raise ValueError("No containers found")

        seed_container = zip(seeds, containers)

        # sort by seed
        seed_container = sorted(seed_container, key=lambda t: t[0])

        # check that seeds are contiguous and unique
        seeds = set([t[0] for t in seed_container])
        if not len(seeds) == len(seed_container):
            raise ValueError("Seeds are not unique")
        if not np.max(list(seeds)) - np.min(list(seeds)) == len(seed_container) - 1:
            raise ValueError("Seeds are not contiguous")

        if self.seed_min_incl is not None or self.seed_max_incl is not None:
            if self.seed_min_incl is not None and self.seed_max_incl is not None:
                assert self.seed_max_incl >= self.seed_min_incl
            if self.seed_min_incl is not None:
                if min(seeds) > self.seed_min_incl:
                    raise ValueError("seed_min not available")
            if self.seed_max_incl is not None:
                if max(seeds) < self.seed_max_incl:
                    raise ValueError("seed_max not available")

            seed_container_filtered = []
            for seed, container in seed_container:
                if self.seed_min_incl is not None and seed < self.seed_min_incl:
                    continue
                if self.seed_max_incl is not None and seed > self.seed_max_incl:
                    continue
                seed_container_filtered.append((seed, container))
            seed_container = seed_container_filtered

        # build dict with list of seeds to load per container
        seeds_per_container = defaultdict(list)
        for seed, container in seed_container:
            seeds_per_container[container].append(seed)

        np_data = {"action": [], "reward": [], "done": [], "observation": []}
        for container_filename, seed_list in seeds_per_container.items():
            container_path = os.path.join(self.data_dir, container_filename)
            filesize_mb = os.stat(container_path).st_size / 1024 / 1024
            print(f"Loading {container_filename} ({filesize_mb} MiB)")
            container_data = np.load(container_path)
            # assert seed list is contiguous
            seed_min = min(seed_list)
            seed_max = max(seed_list)
            assert len(seed_list) == seed_max - seed_min + 1
            for k in np_data.keys():
                v = container_data[k]
                np_data[k].append(v[seed_min : seed_max + 1])

        for k, v in np_data.items():
            np_data[k] = np.concatenate(v)

        return np_data

    @property
    def n_rollouts(self):
        return self.data["observation"].shape[0]

    @property
    def rollout_length(self):
        return self.data["observation"].shape[1]

    @property
    def chunks_per_rollout(self):
        return self.rollout_length - self.subsequence_length + 1

    def __getitem__(self, item):
        n_rollouts, rollout_length, _ = self.data["observation"].shape
        chunks_per_rollout = self.chunks_per_rollout
        rollout_idx = item // chunks_per_rollout
        offset = item - rollout_idx * chunks_per_rollout
        subsequence_data = {}
        for k, v in self.data.items():
            subsequence_data[k] = v[rollout_idx][
                offset : offset + self.subsequence_length
            ]
        subsequence_data = dict(
            **subsequence_data,
            **{
                "sequence_start": offset,
                "rollout_idx": rollout_idx,
                "item_idx": item,
            },
        )
        if self.subsequence_transform:
            subsequence_data = self.subsequence_transform(subsequence_data)
        return subsequence_data

    def __len__(self):
        n_rollout = self.n_rollouts
        chunks_per_rollout = self.chunks_per_rollout
        return n_rollout * chunks_per_rollout
