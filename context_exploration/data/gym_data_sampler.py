"""
Sample data from OpenAI gyms

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from context_exploration.data.dataset import NumpyDataset

__all__ = ["GymDataSampler"]


class GymDataSampler(object):
    def __init__(
        self,
        dataset_name,
        state_dim,
        action_dim,
        obs_n_transitions,
        ctx_card,
        batchsize,
        seed_min_incl,
        seed_max_incl,
        ctx_from_train_prob,
    ):
        super(GymDataSampler, self).__init__()

        if not isinstance(ctx_card, (tuple, list)):
            raise ValueError("ctx_card must be a list of observation cardinalities")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_n_transitions = obs_n_transitions
        self.ctx_card = np.array(ctx_card)
        self.batchsize = batchsize
        self.ctx_from_train_prob = ctx_from_train_prob

        experience_dir = Path(__file__).parent.parent.parent.joinpath(
            "data", "experience", dataset_name
        )
        print(f"Experience directory is at {experience_dir}")

        self.dataset = NumpyDataset(
            experience_dir,
            subsequence_transform=self._subsequence_transform,
            seed_min_incl=seed_min_incl,
            seed_max_incl=seed_max_incl,
        )
        self.data_loader = None
        self.data_iter = None
        # default numpy rng
        self.rng = np.random.random.__self__
        self._get_data_loader()

    def _stack_context(self, list_of_arrays, default_dim):
        """ Stack context, even if eventually empty """
        if len(list_of_arrays) == 0:
            return np.empty((0, default_dim))
        else:
            return np.stack(list_of_arrays)

    def _subsequence_transform(self, subsequence):
        ctx_size = []
        rollout_length = len(subsequence["observation"])

        observation = np.stack(subsequence["observation"])
        observation_ctx = observation[..., : self.state_dim]
        observation_train = observation[..., self.state_dim :]
        action_ctx = subsequence["action"][..., : self.action_dim]
        action_train = subsequence["action"][..., self.action_dim :]
        assert observation_ctx.shape[-1] == self.state_dim
        assert observation_train.shape[-1] == self.state_dim
        assert action_ctx.shape[-1] == self.action_dim
        assert action_train.shape[-1] == self.action_dim

        # extract training chunk randomly (obs_x, obs_u)
        first_transition_idx = self.rng.randint(
            0, rollout_length - self.obs_n_transitions + 1
        )
        s_ = np.index_exp[
            first_transition_idx : first_transition_idx + self.obs_n_transitions
        ]
        obs_x = observation_train[s_]
        obs_u = action_train[s_]

        # extract context samples
        # sample size of context set
        ctx_cardinality = self.rng.choice(self.ctx_card, size=None)
        ctx_size.append(ctx_cardinality)

        # choose for each context sample if from 'train' or 'ctx' rollout
        ctx_from_train = self.rng.rand(ctx_cardinality) < self.ctx_from_train_prob
        # if ctx_from_train is 1 for a particular context sample,
        # add an offset of 'rollout_length' to the index,
        # for sampling from the concatenated (observation_ctx, observation_train)
        ctx_idxs = (
            self.rng.choice(np.arange(rollout_length - 1), size=ctx_cardinality)
            + ctx_from_train * rollout_length
        )
        observation_all = np.concatenate((observation_ctx, observation_train))
        action_all = np.concatenate((action_ctx, action_train))
        ctx_x = observation_all[ctx_idxs]
        ctx_u = action_all[ctx_idxs]
        ctx_x_next = observation_all[ctx_idxs + 1]

        subsequence = {
            "obs_x": obs_x.astype(np.float32),
            "obs_u": obs_u.astype(np.float32),
            "ctx_x": ctx_x.astype(np.float32),
            "ctx_u": ctx_u.astype(np.float32),
            "ctx_x_next": ctx_x_next.astype(np.float32),
            "ctx_size": ctx_cardinality,
        }

        return subsequence

    def _concat_collate(self, batch):
        elem = batch[0]
        keys = list(elem.keys())
        collate_batch = {}
        for key in keys:
            if key in ["obs_x", "obs_u"]:
                batch_elem = torch.stack(
                    [torch.as_tensor(item[key]) for item in batch], dim=1
                )
            elif isinstance(elem[key], np.ndarray):
                batch_elem = torch.cat([torch.as_tensor(item[key]) for item in batch])
            elif isinstance(elem[key], np.int64):
                batch_elem = torch.as_tensor([item[key] for item in batch])
            else:
                raise TypeError(
                    f"Type {type(elem[key])} not handled in custom collation"
                )
            collate_batch[key] = batch_elem
        return collate_batch

    def _get_data_loader(self):
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=self._concat_collate,
        )
        self.data_iter = iter(self.data_loader)

    def process_batch(self, batch, device):
        for key, elem in batch.items():
            batch[key] = elem.to(device)

        batchsize = len(batch["ctx_size"])
        batch["ctx_assignments"] = torch.repeat_interleave(
            torch.arange(batchsize, out=torch.LongTensor()).to(device),
            batch["ctx_size"],
        )
        return batch

    def sample_batch(self, device):
        try:
            rollout_batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            rollout_batch = next(self.data_iter)

        return self.process_batch(rollout_batch, device)

    def sample_validation_data(self, n_batches, device):
        backup_rng = self.rng
        self.rng = np.random.RandomState(42)
        batches = []
        for batch_idx in range(n_batches):
            data = []
            for idx in range(self.batchsize):
                data.append(self.dataset[batch_idx * self.batchsize + idx])
            raw_batch = self._concat_collate(data)
            batch = self.process_batch(raw_batch, device)
            batches.append(batch)
        self.rng = backup_rng
        return batches
