import os
import pickle
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import d4rl
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import optax
import tyro
import wandb
from flax.training.train_state import TrainState
from termination_fns import get_termination_fn

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    dataset: str = "halfcheetah-medium-v2"
    algorithm: str = "enn-dynamics"
    eval_interval: int = 10_000
    # --- Logging ---
    log: bool = False
    wandb_project: str = "unifloral"
    wandb_team: str = "flair"
    wandb_group: str = "debug"
    model_path: str = "dynamics_models"
    # --- Generic optimization ---
    lr: float = 0.001
    batch_size: int = 256
    # --- Dynamics training ---
    n_layers: int = 4
    layer_size: int = 200
    num_ensemble: int = 7
    z_dim: int = 100
    num_epochs: int = 400
    logvar_diff_coef: float = 0.01
    weight_decay: float = 2.5e-5
    validation_split: float = 0.2
    precompute_term_stats: bool = False


r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Preliminaries
"""

Transition = namedtuple("Transition", "obs action reward next_obs next_action done")


class SingleENNDynamicsModel(nn.Module):
    obs_dim: int
    n_layers: int
    layer_size: int
    max_logvar_init: float = 0.5
    min_logvar_init: float = -10.0

    @nn.compact
    def __call__(self, delta_obs_action, z):
        x = jnp.concatenate([delta_obs_action, z], axis=-1)
        for _ in range(self.n_layers):
            x = nn.relu(nn.Dense(self.layer_size)(x))

        # --- Do all postprocessing here ---
        obs_reward_stats = nn.Dense(2 * (self.obs_dim + 1))(x)
        pred_mean, logvar = jnp.split(obs_reward_stats, 2, axis=-1)

        # --- Soft clamp log-variance ---
        max_logvar = self.param(
            "max_logvar",
            init_fn=lambda key: jnp.full((self.obs_dim + 1,), self.max_logvar_init),
        )
        min_logvar = self.param(
            "min_logvar",
            init_fn=lambda key: jnp.full((self.obs_dim + 1,), self.min_logvar_init),
        )
        logvar = max_logvar - nn.softplus(max_logvar - logvar)
        logvar = min_logvar + nn.softplus(logvar - min_logvar)
        return pred_mean, logvar


class ENNDynamics:
    """Wrapper class for ENN-based dynamics model that handles sampling and rollouts."""

    def __init__(
        self,
        dynamics_model,
        params,
        termination_fn,
        discrepancy: Optional[float] = None,
        min_r: Optional[float] = None,
    ):
        self.dynamics_model = dynamics_model
        self.termination_fn = termination_fn
        self.discrepancy = discrepancy
        self.min_r = min_r
        self.dataset = None

        # do not need to clip to ensemble indexes so just set it.
        self.params = params

    def make_rollout_fn(
        self,
        batch_size,
        rollout_length,
        step_penalty_coef=0.0,
        term_penalty_offset=None,
        threshold_coef=1.0,
    ):
        """Make buffer update function."""

    def _sample_transition(
        self,
        rng,
        policy,
        obs,
        z,
        step_penalty_coef=0.0,
        term_penalty_offset=None,
        threshold_coef=1.0,
    ):
        """Sample transition from policy and dynamics model."""
        rng_action, rng_dynamics, rng_noise, rng_next_action = jax.random.split(rng, 4)

        # --- Sample action and model predictions ---
        action = policy(obs, rng_action)
        obs_action = jnp.concatenate([obs, action], axis=-1)
        next_obs_reward_mean, next_obs_reward_logvar = self.dynamics_model.apply(
            self.params, obs_action, z
        )
        net_obs_reward_logvar = jnp.exp(0.5 * net_obs_reward_logvar)
