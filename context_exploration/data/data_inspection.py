"""
Inspect training data script

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

from context_exploration.data.dataset import NumpyDataset

# This file is intended to be run as a script
__all__ = []


def main():
    experience_dir = (
        Path(__file__)
        .resolve()
        .parent.parent.parent.joinpath(
            "data", "experience", "unicycle_robotvary_terrainpatches"
        )
    )
    n_rollouts = 20
    dataset = NumpyDataset(experience_dir)
    actions = []
    observations = []

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    for idx in tqdm(range(n_rollouts)):
        data_item = dataset[idx]
        actions.append(data_item["action"][:-1])
        observations.append(data_item["observation"][:-1])
        ax.plot(data_item["observation"][:, 0], data_item["observation"][:, 1])
        ax.scatter(data_item["observation"][:, 0], data_item["observation"][:, 1])

    plt.show()


if __name__ == "__main__":
    main()
