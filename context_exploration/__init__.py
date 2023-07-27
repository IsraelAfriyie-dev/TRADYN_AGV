"""
Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""

from pathlib import Path

from torch.distributions import Normal

# Patch normal distribution with 'shape' property and expand(*shape) methods


def normal_shape(self):
    return self.mean.shape


def normal_expand(self, *target_shape):
    return Normal(
        loc=self.loc.expand(*target_shape), scale=self.scale.expand(*target_shape)
    )


Normal.shape = property(normal_shape)
Normal.expand = normal_expand


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data")
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent.joinpath("experiments")
