"""
Unicycle, Landscape, and UnicycleLandscape envs

Copyright 2023 Max-Planck-Gesellschaft
Code authors: Jan Achterhold, jan.achterhold@tuebingen.mpg.de, Suresh Guttikonda
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import os
import warnings
from multiprocessing import Pool
from pathlib import Path
from shutil import copytree
from typing import Dict, Tuple

import gym
import numpy as np
import torch
from gym import spaces
from numba import float32, float64, jit
from PIL import Image
from scipy.integrate import solve_ivp

from context_exploration.data.envs.action_sampler import (
    GaussianCorrelatedExcitationController,
)
from context_exploration.data.envs.common import (
    ParametrizedEnvWrapper,
    SampleActionMixin,
    SizeProperties,
)
from context_exploration.data.wrappers import MaxDurationWrapper

__all__ = ["ContextUnicycleLandscapeEnv"]


def normalize(
    input_val: np.ndarray,
    input_min: np.ndarray,
    input_max: np.ndarray,
    new_min: np.ndarray,
    new_max: np.ndarray,
) -> np.ndarray:
    """
    Rescale(Normalize) the input from [input_min, input_max] to [new_min, new_max].

    Parameters
    ----------
    input_val: `np.ndarray`
    input_min: `np.ndarray`
    input_max: `np.ndarray`
    new_min: `np.ndarray`
    new_max: `np.ndarray`

    Returns
    -------
    rescaled_input: `np.ndarray`
    """
    input_val = np.clip(input_val, input_min, input_max)
    rescaled_input = new_min + (new_max - new_min) * (
        (input_val - input_min) / (input_max - input_min)
    )
    return rescaled_input


class Landscape:
    """
    Landscape class. The extent of the landscape is always [0, 1]^2.
    The feature value of a point on the landscape is [0, 1]^3 (normalized RGB).
    Landscapes can be cached to a memdisk for faster data generation.
    """

    observation_space = spaces.Box(
        low=np.zeros(3).astype(np.float32), high=np.ones(3).astype(np.float32)
    )

    _curr_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    _shm_basedir = Path("/dev/shm")
    _shm_landscapes = _shm_basedir.joinpath("landscapes")
    _local_landscapes = _curr_dir.joinpath("landscapes")

    def __init__(
        self,
        is_vary_terrain: bool,
        terrain_pattern_seed: int,
        mu_min: float,
        mu_max: float,
        has_noisy_observations: bool,
    ):
        """
        Initialize `Landscape`

        Parameters
        ----------
        is_vary_terrain: bool
            If True, use landscape image.
            If False, use a fixed RGB observation.
        terrain_pattern_seed: int
            If `is_vary_terrain`, the landscape image to load (0, ..., 99).
            Else, a seed for the RNG generating the terrain RGB observation.
        mu_min: Minimal friction coefficient
        mu_max: Maximal friction coefficient
        has_noisy_observations
        """
        """The Landscape class initializer.

        @param is_uni_terrain Whether to use uniform terrain landscape or not.
        @param terrain_pattern_seed: Seed of terrain pattern

        @return An instance of Landscape class.
        """
        self._is_vary_terrain = is_vary_terrain
        self._mu_min = mu_min
        self._mu_max = mu_max
        self._has_noisy_observations = has_noisy_observations

        if is_vary_terrain:
            image = self._load_landscape_img(terrain_pattern_seed)
            self._feature_image = 1 - np.array(image).astype(np.uint8) / 255
            self._feature_image = self._feature_image[20:-20, 20:-20]
            # overall image size: 460px x 460px
            self._feature_image_torch = None
            self._feature_image_shape = None
        else:
            color_rng = np.random.RandomState(terrain_pattern_seed)
            self._uni_feature = color_rng.rand(3)

    @classmethod
    def cache_landscapes(cls):
        """
        Cache landscapes into memdisk.
        This is not automatically done by load_landscape_img,
        as this would need to be thread-safe then (e.g., using
        this env within joblib or a vector env) - different processes
        may try to write the landscapes into the memdisk concurrently.
        """
        if not cls._shm_basedir.is_dir():
            # /dev/shm does not exist, do not try copying
            warnings.warn(f"Memdisk at {cls._shm_basedir} does not exist")
        elif not cls._shm_landscapes.is_dir():
            # /dev/shm does exist, but no subfolder 'landscapes' exist
            try:
                copytree(cls._local_landscapes, cls._shm_landscapes)
            except Exception as e:
                warnings.warn(f"Caching landscapes failed: {e}")

    def _load_landscape_img(self, terrain_pattern_seed):
        # query /dev/shm first for image file, if copy did not fail
        img_loc = ["imgs", f"landscape{terrain_pattern_seed}.png"]
        shm_img_path = self._shm_landscapes.joinpath(*img_loc)
        local_img_path = self._local_landscapes.joinpath(*img_loc)
        if shm_img_path.is_file():
            img_path = shm_img_path
        else:
            img_path = local_img_path
        image = Image.open(img_path).convert("RGB")
        image = image.resize((500, 500))
        return image

    def plot_friction(self, ax, **kwargs):
        """
        Plot friction coefficients on `ax`

        Parameters
        ----------
        ax: `matplotlib.axes.Axes`
            Matplotlib axes
        kwargs: dict
            kwargs for ax.imshow()

        Returns
        -------
        imshow_obj: object
            Return value of ax.imshow()
        """
        if self._is_vary_terrain:
            feature_img = self._feature_image
            mu = self.get_mu(feature_img)
            return ax.imshow(
                np.flipud(mu.T),
                extent=[0, 1, 0, 1],
                vmin=self._mu_min,
                vmax=self._mu_max,
                **kwargs,
            )

    def plot_landscape(self, ax, **kwargs):
        """
        Plot landscape on `ax`

        Parameters
        ----------
        ax: `matplotlib.axes.Axes`
            Matplotlib axes
        kwargs: dict
            kwargs for ax.imshow()

        Returns
        -------
        imshow_obj: object
            Return value of ax.imshow()
        """
        if self._is_vary_terrain:
            feature_img = self._feature_image
            return ax.imshow(
                np.flipud(np.transpose(feature_img, (1, 0, 2))),
                extent=[0, 1, 0, 1],
                vmin=0,
                vmax=1,
                **kwargs,
            )

    def get_feature(self, position: np.ndarray) -> np.array:
        """
        Get the terrain feature at the input's location.

        Parameters
        ----------
        position: `np.ndarray` or `torch.Tensor` with shape [<bs>, 2]
            Query position

        Returns
        -------
        feature: `np.ndarray` or `torch.Tensor` with shape [<bs>, 3]
            Terrain feature
        """
        if self._is_vary_terrain:
            feature_img_shape = np.array(self._feature_image.shape[:2])

        if isinstance(position, torch.Tensor) and self._is_vary_terrain:
            position = torch.clip(position, 0, 1)
            device = position.device
            if (
                self._feature_image_torch is None
                or self._feature_image_torch.device != device
            ):
                self._feature_image_torch = torch.Tensor(self._feature_image).to(device)

            xpos = (position[..., 0] * (self._feature_image_torch.shape[0] - 1)).long()
            ypos = (position[..., 1] * (self._feature_image_torch.shape[1] - 1)).long()
            feature = self._feature_image_torch[xpos, ypos, :]

        elif isinstance(position, torch.Tensor) and not self._is_vary_terrain:
            device = position.device
            feature = self._uni_feature.copy()
            feature = torch.Tensor(feature).to(device)
            feature = feature.expand(*position.shape[:-1], feature.shape[-1])

        elif isinstance(position, np.ndarray) and self._is_vary_terrain:
            position = np.clip(position, 0, 1)
            pixel_idx = position * (feature_img_shape - 1)
            pixel_idx = pixel_idx.astype(int)
            feature = self._feature_image[pixel_idx[..., 0], pixel_idx[..., 1], :]

        elif isinstance(position, np.ndarray) and not self._is_vary_terrain:
            feature = self._uni_feature.copy()
            feature = np.broadcast_to(
                feature, position.shape[:-1] + (feature.shape[-1],)
            )
        else:
            raise ValueError

        if self._has_noisy_observations:
            if isinstance(feature, torch.Tensor):
                feature = feature + 0.01 * torch.randn_like(feature)
                feature = torch.clip(feature, 0, 1)
            elif isinstance(feature, np.ndarray):
                feature = feature + 0.01 * np.random.randn(*feature.shape)
                feature = np.clip(feature, 0, 1)
            else:
                raise ValueError

        return feature

    def get_mu(self, feature):
        """
        Get the friction coefficient for the given terrain feature.

        Parameters
        ----------
        feature: `np.ndarray`
            Array of terrain features (from `get_feature`)
        Returns
        -------
        mu: `np.ndarray`
            Array of friction coefficients
        """
        feature = (255 * feature).astype(np.uint32)

        def rgb_to_hex(r, g, b):
            return (r << 16) + (g << 8) + b

        mu_0_1 = normalize(
            rgb_to_hex(feature[..., 0], feature[..., 1], feature[..., 2]),
            rgb_to_hex(0, 0, 0),
            rgb_to_hex(255, 255, 255),
            0,
            1,
        )

        mu = self._mu_min + (mu_0_1 ** 2) * (self._mu_max - self._mu_min)
        return mu

    def close(self):
        if self._is_vary_terrain:
            del self._feature_image
            del self._feature_image_torch


class Unicycle:
    """
    The Unicycle class.
    The position is bounded to [0, 1]
    The velocity is bounded to [-5, 5] m/s (it needs 20 steps to traverse the env)
    """

    # Simulation timestep: 10ms
    DT = 0.01
    # Positions are bound to [0, 1]
    _p_min = 0
    _p_max = 1
    # Velocities are bound to [-5, 5]
    _v_min = -5
    _v_max = 5

    # cos(th), sin(th) are in [-1, 1]
    observation_space = spaces.Box(
        low=np.array([_p_min, _p_min, _v_min, -1, -1]).astype(np.float32),
        high=np.array([_p_max, _p_max, _v_max, 1, 1]).astype(np.float32),
    )

    def __init__(
        self,
        unconstrained_dynamics,
        parameters: np.ndarray,
        has_noisy_observations: bool,
        has_noisy_actions: bool,
    ):
        """
        Initialize unicycle

        Parameters
        ----------
        parameters: tuple
            Parameters 'mass', 'throttle_mag', 'steer_mag' (actuator gains)
        has_noisy_observations: bool
            Add noise to observations
        has_noisy_actions: bool
            Add noise to actions before executing
        """
        self._unconstrained_dynamics = unconstrained_dynamics
        self._mass, self._throttle_mag, self._steer_mag = parameters
        self._has_noisy_observations = has_noisy_observations
        self._has_noisy_actions = has_noisy_actions
        self._state = None

    def _get_obs(self):
        return self.obs_from_state(self._state)

    def state_from_obs(self, obs):
        """
        Get the 'state' usable for simulation from an 'observation'

        Parameters
        ----------
        obs: `np.ndarray`, shape [<bs> x 5]
            Observation (x, y, v, cos(th), sin(th))

        Returns
        ----------
        state: `np.ndarray`, shape [<bs> x 4]
            State (x, y, v, th)
        """
        assert obs.shape[-1] == 5  # x, y, v, cos(th), sin(th)
        theta = np.arctan2(obs[..., 4], obs[..., 3])
        return np.concatenate((obs[..., :3], theta[..., None]), axis=-1)

    def obs_from_state(self, state):
        """
        Get the 'observation' for a 'state'. Eventually, add noise.

        Parameters
        ----------
        state: `np.ndarray`, shape [<bs> x 4]
            State (x, y, v, th)

        Returns
        -------
        obs: `np.ndarray`, shape [<bs> x 5]
            Observation (x, y, v, cos(th), sin(th))
        """
        obs = np.concatenate(
            [
                state[..., :3],
                np.cos(state[..., 3])[..., None],
                np.sin(state[..., 3])[..., None],
            ],
            axis=-1,
        )
        if self._has_noisy_observations:
            obs = obs + 0.01 * np.random.randn(*obs.shape)
            obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def reset(self, init_state: np.ndarray = None):
        """ Reset unicycle to given `init_state` (or all-zero state) """
        if init_state is None:
            # x, y, v, theta
            init_state = np.array([0, 0, 0, 0])
        self._state = init_state
        return self._get_obs()

    def step_fun(self, state, throttle, steer, terrain_fcn):
        """
        Advance `state` one step forward in time

        Parameters
        ----------
        state: `np.ndarray`, shape [<bs> x 4]
            State array, (x, y, v, theta)
        throttle: `np.ndarray`, shape [<bs>]
        steer: `np.ndarray`, shape [<bs>]
        terrain_fcn: function terrain_fcn(x: np.ndarray)
            See documentation of 'unconstrained_dynamics'

        Returns
        -------
        new_state: `np.ndarray`, shape [<bs> x 4]
            State array, (x, y, v, theta)
        """
        assert np.all(throttle >= -1)
        assert np.all(throttle <= 1)
        assert np.all(steer >= -1)
        assert np.all(steer <= 1)
        assert state.shape[-1] == 4  # x, y, v, theta

        if self._has_noisy_actions:
            throttle = throttle + 0.01 * np.random.randn(*throttle.shape)
            steer = steer + 0.01 * np.random.randn(*steer.shape)

        x, y, v, theta = (state[..., k] for k in range(4))

        mass = (np.ones_like(x) * self._mass).astype(np.float32)
        throttle = throttle * self._throttle_mag
        steer = steer * self._steer_mag
        x_new, y_new, v_new, theta_new = self._unconstrained_dynamics.step(
            x, y, v, theta, throttle, steer, terrain_fcn, mass, self.DT
        )

        x_new = np.clip(x_new, self._p_min, self._p_max)
        y_new = np.clip(y_new, self._p_min, self._p_max)
        v_new = np.clip(v_new, self._v_min, self._v_max)

        new_state = np.stack((x_new, y_new, v_new, theta_new), axis=-1)
        return new_state

    def step(self, throttle, steer, terrain_fcn):
        """
        Advance internal state of environment one step forward in time

        Parameters
        ----------
        throttle: float in [-1, 1]
        steer: float in [-1, 1]
        terrain_fcn: function terrain_fcn(x: np.ndarray)
            See documentation of 'unconstrained_dynamics'

        Returns
        -------
        observation: `np.ndarray`, shape [<bs> x 5]
            Observation (x, y, v, cos(th), sin(th))
        """
        self._state = self.step_fun(self._state, throttle, steer, terrain_fcn)
        return self._get_obs()


@jit(float64[:](float64[:], float32, float32, float32, float32, float32))
def deriv(y, mass, theta, mu_terrain, F_n, F_ext):
    v = y[2:3]  # atleast_1d
    dpxdt = v * np.cos(theta)
    dpydt = v * np.sin(theta)
    F_fric_terrain = -np.tanh((1e3) * v) * mu_terrain * F_n
    a = (1 / mass) * (F_ext + F_fric_terrain)
    a[np.logical_and(np.abs(v) < 1e-3, np.abs(a) < 1e-3)] = 0
    dvdt = a
    dydt = np.concatenate((dpxdt, dpydt, dvdt), axis=-1)
    return dydt


class IntegrableFun:
    def __init__(self, mass, theta, mu_terrain, F_n, F_ext):
        self._mass = mass
        self._theta = theta
        self._mu_terrain = mu_terrain
        self._F_n = F_n
        self._F_ext = F_ext

    def __call__(self, t, y):
        # y: (N, 3) or (3,): current 2D position + velocity
        return deriv(
            y, self._mass, self._theta, self._mu_terrain, self._F_n, self._F_ext
        )


class SolveIvpMapWorker:
    def __init__(self, t_span):
        self._t_span = t_span

    def __call__(self, y0, mass, theta, mu_terrain, F_n, F_ext):
        solve_obj = solve_ivp(
            IntegrableFun(mass, theta, mu_terrain, F_n, F_ext), self._t_span, y0
        )
        yT = solve_obj.y[:, -1]
        return yT


class SolveIvpMapIterable:
    def __init__(self, *args):
        self._arg_array = args
        self._counter = 0

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter < len(self._arg_array[0]):
            retval = [a[self._counter] for a in self._arg_array]
            self._counter += 1
            return retval
        else:
            raise StopIteration


class UnicycleColoumbDynamics:
    """
    Unconstrained unicycle dynamics with coloumb friction
    """

    def __init__(self):
        pass

    def step(self, x, y, v, theta, throttle, steer, terrain_fcn, mass, dt):
        # dv/dt = (1/m) * (F_ext + F_fric_terrain)
        # F_ext = throttle_mag * throttle
        # F_fric_terrain = -sgn(v) * mu_terrain * F_n
        # F_n = m * g

        x, y, v, theta, throttle, steer = (
            val.astype(np.float32) for val in (x, y, v, theta, throttle, steer)
        )

        g = 9.81

        thetadot = steer.astype(np.float32)
        theta = theta + thetadot

        mu_terrain = terrain_fcn(np.stack((x, y), axis=-1)).astype(np.float32)

        F_n = mass * g
        F_ext = throttle.astype(np.float32)

        y0 = np.stack((x, y, v), axis=-1)

        solve_obj = solve_ivp(
            IntegrableFun(mass, theta, mu_terrain, F_n, F_ext), (0, dt), y0
        )
        yT = solve_obj.y[:, -1]

        x_new, y_new, v_new = (yT[..., k] for k in range(3))
        return x_new, y_new, v_new, theta


class UnicycleLandscapeEnv(gym.Env):
    """
    Merge given unicycle and landscape into a single env.

    Observation:
        Observation is a (5, ) ndarray.
        All observations are *normalized*, see the last two columns.

                                                     Raw         Norm
        | Number | Observation                  | Min | Max | Min | Max |
        |--------|------------------------------|-----|-----|-----|-----|
        | 0      | x-position [m]               | 0   | 1   | 0   | 1  |
        | 1      | y-position [m]               | 0   | 1   | 0   | 1  |
        | 2      | velocity [m/s]               | -5  | 5   | -1  | 1  |
        | 3      | x-rotation_vector (cos)      | -1  | +1  | -1  | 1  |
        | 4      | y-rotation_vector (sin)      | -1  | +1  | -1  | 1  |
        | 5,6,7  | terrain feat                 | 0   | 1   | 0   | 1  |

    Actions:
        The agent take a (2,) ndarray for actions, where elements correspond
        to the following:
        | Number | Action             | Min | Max |
        |--------|--------------------|-----|-----|
        | 0      | throttle [m/s^2]   | -1  | +1  |
        | 1      | steer [rad/s]      | -1  | +1  |
    """

    def __init__(self, unicycle: Unicycle, landscape: Landscape):
        """
        Initialize `ContextUnicycleLandscapeEnvBase`

        Parameters
        ----------
        unicycle: `Unicycle`
            Unicycle
        landscape: `Landscape`
            Landscape
        """
        super().__init__()

        obs_low = np.concatenate(
            [Unicycle.observation_space.low, Landscape.observation_space.low],
            axis=0,
        )
        obs_high = np.concatenate(
            [Unicycle.observation_space.high, Landscape.observation_space.high],
            axis=0,
        )

        self.raw_observation_space = spaces.Box(
            low=obs_low.astype(np.float32), high=obs_high.astype(np.float32)
        )
        # normalized observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1, -1, 0, 0, 0]).astype(np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1]).astype(np.float32),
        )

        self.action_space = spaces.Box(
            low=-np.ones(2).astype(np.float32), high=np.ones(2).astype(np.float32)
        )

        self._rng = None
        self._robot = unicycle
        self._landscape = landscape
        # x, y, v, theta
        self._robot_state_low = np.array([0, 0, unicycle._v_min, 0])
        self._robot_state_high = np.array([1, 1, unicycle._v_max, 2 * np.pi])

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)

    def reset(self, init_mode=None, init_robot_state=None):
        assert self._landscape is not None
        assert self._robot is not None

        if init_robot_state is None:
            robot_state_space = spaces.Box(
                low=self._robot_state_low.astype(np.float32),
                high=self._robot_state_high.astype(np.float32),
            )
            init_robot_state = robot_state_space.low + self._rng.rand(
                robot_state_space.shape[0]
            ) * (robot_state_space.high - robot_state_space.low)

        robot_obs = self._robot.reset(init_state=init_robot_state)
        # Get terrain observation at (x, y) coordinate of robot
        terrain_obs = self._landscape.get_feature(robot_obs[:2])
        obs_raw = np.concatenate([robot_obs, terrain_obs], axis=0)
        obs_norm = self._normalize_obs(obs_raw)
        return obs_norm

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        throttle, steer = action
        throttle = np.clip(throttle, -1, 1)
        steer = np.clip(steer, -1, 1)

        def terrain_fcn(robot_pos):
            terrain_obs = self._landscape.get_feature(robot_pos)
            terrain_mu = self._landscape.get_mu(terrain_obs)
            return terrain_mu

        next_robot_obs = self._robot.step(throttle, steer, terrain_fcn)
        terrain_obs = self._landscape.get_feature(next_robot_obs[:2])
        obs_raw = np.concatenate([next_robot_obs, terrain_obs], axis=0)
        obs_norm = self._normalize_obs(obs_raw)
        info = {
            "is_valid_obs": True,
        }
        return obs_norm, 0, False, info

    def _normalize_obs(self, obs_raw: np.ndarray) -> np.ndarray:
        obs_norm = normalize(
            obs_raw,
            self.raw_observation_space.low,
            self.raw_observation_space.high,
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32)
        return obs_norm

    def query_terrain_state(self, robot_obs):
        return self._landscape.get_feature(robot_obs[..., :2])

    def plot_landscape(self, ax, **kwargs):
        return self._landscape.plot_landscape(ax, **kwargs)

    def plot_friction(self, ax, **kwargs):
        return self._landscape.plot_friction(ax, **kwargs)


class UnicycleLandscapeContextSpace:
    """
    Context space for the merged unicycle/landscape env

    The robot context is a (3, ) ndarray

    | Number | Context           | Min    | Max     |
    |--------|-------------------|--------|---------|
    | 0      | Mass [kg]         | 1      | 4       |
    | 1      | throttle gain     | 500    | 1000    |
    | 2      | steer gain        | pi/8   | pi/4    |

    If not `is_vary_robot`, the average of min and max is set as context.

    The landscape context is a (1, ) ndarray

    If `is_vary_terrain` and `reset_split`==train:
    | 3      | landscape seed    | 0      | 49    |
    If `is_vary_terrain` and `reset_split`==test:
    | 3      | landscape seed    | 50     | 99    |
    If not `is_vary_terrain` and `reset_split`==train:
    | 3      | landscape seed    | 0      | 4999  |
    If not `is_vary_terrain` and `reset_split`==test:
    | 3      | landscape seed    | 5000   | 9999  |
    """

    def __init__(self, is_vary_robot, is_vary_terrain, reset_split, param_range):
        """ Initialize context space. """
        self._is_vary_robot = is_vary_robot
        self._is_vary_terrain = is_vary_terrain
        self._reset_split = reset_split
        self._param_range = param_range
        self._rng = None

    @property
    def shape(self):
        return np.array([4])

    def seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def sample(self):
        if self._is_vary_terrain:
            terrain_train_idxs = list(range(0, 50))
            terrain_test_idxs = list(range(50, 100))
            if self._reset_split == "train":
                terrain_idxs = terrain_train_idxs
            elif self._reset_split == "test":
                terrain_idxs = terrain_test_idxs
            else:
                raise ValueError
        else:
            if self._reset_split == "train":
                terrain_idxs = list(range(0, 5000))
            elif self._reset_split == "test":
                terrain_idxs = list(range(5000, 10_000))
            else:
                raise ValueError

        terrain_pattern_idx = self._rng.choice(terrain_idxs)

        if self._param_range == "default":
            ctx_low = np.array(
                [
                    1,
                    500,
                    np.pi / 8,
                ]  # mass, throttle coeff, steer coeff
            )
            ctx_high = np.array(
                [
                    4,
                    1000,
                    np.pi / 4,
                ]  # mass, throttle coeff, steer coeff
            )
        else:
            raise ValueError

        default_robot_parameters = (ctx_low + ctx_high) / 2

        if self._is_vary_robot:
            robot_parameters = ctx_low + self._rng.rand(ctx_low.shape[0]) * (
                ctx_high - ctx_low
            )
        else:
            robot_parameters = default_robot_parameters

        return np.concatenate((robot_parameters, np.array([terrain_pattern_idx])))


class ContextUnicycleLandscapeEnv(
    ParametrizedEnvWrapper, SizeProperties, SampleActionMixin
):
    """
    Contextual environment, merging unicycle and landscape
    """

    def __init__(
        self,
        is_vary_robot,
        is_vary_terrain,
        reset_split,
        param_range="default",
        max_duration=100,
        is_noisy=False,
    ):
        """
        Initialize `ContextUnicycleLandscapeEnv`

        Parameters
        ----------
        is_vary_robot: bool
            If False, robot properties are the midpoint of their value ranges (see `UnicycleLandscapeContextSpace`)
        is_vary_terrain: bool
            If True, terrain consists of patches.
            If False, terrain feature is constant.
        reset_split: str
            Reset split in ["train", "test"]. Only affects the landscape.
        param_range: str
            For future use to implement different value ranges
        max_duration: int
            Max lifetime of environment, default: 100
        is_noisy: bool
            If True, add noise to actions and observations
        """
        self._is_vary_robot = is_vary_robot
        self._is_vary_terrain = is_vary_terrain
        self._reset_split = reset_split
        self._is_noisy = is_noisy

        self.max_duration = max_duration

        state_dim = 5 + 3  # robot (5) + landscape (3)
        action_dim = 2  # throttle, steer

        self._mu_min = 0.1
        self._mu_max = 10
        context_space = UnicycleLandscapeContextSpace(
            is_vary_robot, is_vary_terrain, reset_split, param_range
        )
        self.excitation_controller = GaussianCorrelatedExcitationController(
            self, mean=np.zeros(2), std=0.5 * np.ones(2), beta=0.5
        )
        ParametrizedEnvWrapper.__init__(self, context_space)
        SizeProperties.__init__(self, state_dim, action_dim)

    @property
    def local_context_dim(self):
        # Dimensionality of the terrain feature
        return 3

    def reset(self, init_mode=None, init_robot_state=None):
        return super().reset(init_mode=init_mode, init_robot_state=init_robot_state)

    def _construct_env(self, context):
        robot_parameters, terrain_pattern_idx = context[:-1], int(context[-1])
        unconstrained_dynamics = UnicycleColoumbDynamics()
        self._robot = Unicycle(
            unconstrained_dynamics=unconstrained_dynamics,
            parameters=robot_parameters,
            has_noisy_observations=self._is_noisy,
            has_noisy_actions=self._is_noisy,
        )
        self._landscape = Landscape(
            self._is_vary_terrain,
            terrain_pattern_idx,
            self._mu_min,
            self._mu_max,
            has_noisy_observations=self._is_noisy,
        )
        env = UnicycleLandscapeEnv(self._robot, self._landscape)
        env = MaxDurationWrapper(env, self.max_duration)
        return env

    def seed(self, seed=None):
        super().seed(seed)

    def is_transition_informative(self, x, u, x_next):
        return np.zeros(x.shape[:-1])

    def query_terrain_state(self, robot_obs):
        return self.env.query_terrain_state(robot_obs)

    def plot_landscape(self, ax, **kwargs):
        return self.env.plot_landscape(ax, **kwargs)

    def plot_friction(self, ax, **kwargs):
        return self.env.plot_friction(ax, **kwargs)

    def close(self):
        if hasattr(self, "_landscape"):
            self._landscape.close()


def example():
    env = ContextUnicycleLandscapeEnv(
        is_vary_robot=True, is_vary_terrain=True, reset_split="train", is_noisy=True
    )

    n_rollouts = 50
    n_steps = 20
    obs_xy = np.zeros((n_rollouts, n_steps, 2))
    actions = np.zeros((n_rollouts, n_steps, 2))

    for rollout_idx in range(n_rollouts):
        env.initialize_context(rollout_idx)

        env.seed(rollout_idx)
        action_iterator = env.excitation_controller.get_iterator(
            excitation_seed=rollout_idx
        )
        env.reset()
        for step_idx in range(n_steps):
            action = next(action_iterator)
            obs, _, _, _ = env.step(action)
            obs_xy[rollout_idx, step_idx, :] = obs[:2]
            actions[rollout_idx, step_idx, :] = action

        env.release_context()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for rollout_idx in range(n_rollouts):
        ax.plot(obs_xy[rollout_idx, :, 0], obs_xy[rollout_idx, :, 1])
        ax.scatter(obs_xy[rollout_idx, :, 0], obs_xy[rollout_idx, :, 1])
    plt.show()

    env.close()


if __name__ == "__main__":
    example()
