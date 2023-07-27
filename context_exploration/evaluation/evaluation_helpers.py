"""
Helper functions for evaluation

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import os
from contextlib import ContextDecorator

import matplotlib.pyplot as plt
import numpy as np
import torch

from context_exploration.data.envs import make_env
from context_exploration.model.context_encoder import ContextSet


class NullContext(ContextDecorator):
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def plot_context_prediction(
    context_distribution,
    context_size,
    context_sse,
    step,
    run_directory,
    writer,
):
    context_entropy = context_distribution.entropy().sum(dim=-1).detach().cpu().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(context_size, context_entropy)
    ax.set_title(f"Step {step}, SSE {context_sse}")
    plot_directory = os.path.join(run_directory, "plots")
    os.makedirs(plot_directory, exist_ok=True)
    plt.savefig(os.path.join(plot_directory, f"step_{step}.png"))
    if writer:
        writer.add_figure("fig/context_set", fig, step, close=False)
    plt.close()


def generate_rollout(env, env_seed, actions, max_transitions=None, reset_kwargs=None):
    env.seed(env_seed)
    obs = env.reset(**({} if reset_kwargs is None else reset_kwargs))
    obs_list = [obs]
    done = False
    if hasattr(actions, "__next__"):
        action_iter = actions
    else:
        action_iter = iter(actions)

    applied_actions = []
    step = 0
    while not done:
        try:
            action = next(action_iter)
        except StopIteration:
            break
        obs_new, _, done, _ = env.step(action)
        step += 1
        if max_transitions is not None and step >= max_transitions:
            done = True
        obs_list.append(obs_new)
        applied_actions.append(action)
    return {"observation": np.stack(obs_list), "action": np.stack(applied_actions)}


def generate_context_set(
    env,
    env_seed,
    actions,
    max_transitions=None,
    context_size=None,
    return_rollout=False,
):
    reset_kwargs = {"init_mode": "calibration"}
    rollout = generate_rollout(env, env_seed, actions, max_transitions, reset_kwargs)
    context_set = ContextSet.from_trajectory(
        rollout["observation"], rollout["action"], context_size=context_size
    )
    if return_rollout:
        return context_set, rollout
    else:
        return context_set


def generate_prediction_rollout(validation_rollout, transition_model, context_latent):
    device = transition_model.device
    state = torch.tensor(validation_rollout["observation"]).float().to(device)
    actions = torch.tensor(validation_rollout["action"]).float().to(device)
    # Introduce batch dims for state and actions
    state = state.unsqueeze(1)
    actions = actions.unsqueeze(1)
    state, local_ctx = transition_model.extract_local_context(state)
    # Make predictions starting from first state
    state = state[0]
    # Local context of last observation is not needed
    local_ctx = local_ctx[:-1] if local_ctx is not None else None
    context_latent = context_latent.mean.to(device)
    observation_distribution = transition_model.forward_multi_step(
        state, actions, context_latent, local_ctx, return_mean_only=True
    )
    predicted_observation = observation_distribution.detach().cpu().squeeze(1).numpy()
    return {"observation": predicted_observation}


def plot_rollouts(
    env_name,
    transition_model,
    context_encoder,
    step,
    run_directory,
    writer,
    n_rollouts=5,
):
    envs = [make_env(env_name, reset_split="train") for _ in range(n_rollouts)]
    state_dim = transition_model.state_dim

    for env_idx, env in enumerate(envs):
        env.initialize_context(int(1e6) + env_idx)

    context_size_list = [1, 5, 10]

    # get context latents from random rollouts
    # (initial state / control) seed for context rollouts is env_idx + 1e6
    # (initial state / control) seed for validation rollouts is env_idx + 1e7
    context_latents = {}
    for env_idx, env in enumerate(envs):
        context_latents[f"env_{env_idx}"] = {}
        for context_size in context_size_list:
            context_set = generate_context_set(
                env,
                env_seed=env_idx + int(1e6),
                actions=env.excitation_controller.get_iterator(env_idx + int(1e6)),
                context_size=context_size,
            )
            with torch.no_grad():
                context_latent = context_encoder.forward_set(context_set)
            context_latents[f"env_{env_idx}"][context_size] = context_latent

    # generate validation rollouts
    validation_rollouts = {}
    for env_idx, env in enumerate(envs):
        if hasattr(env, "inhibit_disturbance"):
            inhibit_disturbance = env.inhibit_disturbance
        else:
            inhibit_disturbance = NullContext()
        with inhibit_disturbance:
            validation_rollouts[f"env_{env_idx}"] = generate_rollout(
                env,
                env_seed=env_idx + int(1e7),
                actions=env.excitation_controller.get_iterator(env_idx + int(1e7)),
            )

    # compute predictions for validation rollouts
    prediction_rollouts = {}
    for env_idx, env in enumerate(envs):
        prediction_rollouts[f"env_{env_idx}"] = {}
        for context_size in context_size_list:
            prediction_rollouts[f"env_{env_idx}"][
                context_size
            ] = generate_prediction_rollout(
                validation_rollouts[f"env_{env_idx}"],
                transition_model,
                context_latents[f"env_{env_idx}"][context_size],
            )

    fig, ax = plt.subplots(
        nrows=state_dim, ncols=len(context_size_list), sharex=True, sharey="row"
    )
    for env_idx in range(n_rollouts):
        validation_traj = validation_rollouts[f"env_{env_idx}"]
        for context_size_idx in range(len(context_size_list)):
            context_size = context_size_list[context_size_idx]
            prediction_traj = prediction_rollouts[f"env_{env_idx}"][context_size]
            for state_dim_idx in range(state_dim):
                validation_states = validation_traj["observation"][:20, state_dim_idx]
                predicted_states = prediction_traj["observation"][:20, state_dim_idx]
                line = ax[state_dim_idx, context_size_idx].plot(
                    np.arange(len(validation_states)), validation_states
                )[0]
                ax[state_dim_idx, context_size_idx].plot(
                    np.arange(len(predicted_states)),
                    predicted_states,
                    c=line.get_color(),
                    linestyle="--",
                )

    plot_directory = os.path.join(run_directory, "plots")
    os.makedirs(plot_directory, exist_ok=True)
    plt.savefig(os.path.join(plot_directory, f"prediction_{step}.png"))
    if writer:
        writer.add_figure("fig/predictions", fig, step, close=False)
    plt.close()
