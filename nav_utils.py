import logging
import torch
import torch.nn as nn
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from omni_drones.utils.torchrl import RenderCallback
from torchrl.envs.utils import ExplorationType, set_exploration_type
import collections


class ValueNorm(nn.Module):

    def __init__(
        self, input_shape: Union[int, Iterable], beta=0.995, epsilon=1e-05
    ) -> None:
        super().__init__()
        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta
        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=0.01)
        return (debiased_mean, debiased_var)

    def _check_input_shape(self, input_vector: torch.Tensor):
        actual_shape = input_vector.shape[-len(self.input_shape) :]
        if actual_shape != self.input_shape:
            raise ValueError(
                f"Expected trailing shape {self.input_shape}, got {actual_shape}"
            )

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        self._check_input_shape(input_vector)
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)
        weight = self.beta
        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        self._check_input_shape(input_vector)
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        self._check_input_shape(input_vector)
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out


class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {
        "alpha": torch.distributions.constraints.positive,
        "beta": torch.distributions.constraints.positive,
    }

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)


class BetaActor(nn.Module):

    def __init__(self, action_dim: int, input_feature_dim: int):
        super().__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_feature_dim, 72),
            nn.LayerNorm(72),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.05),
            nn.Linear(72, 36),
            nn.LayerNorm(36),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.05),
        )
        self.alpha_transition = nn.Sequential(nn.Linear(36, 18), nn.Tanh())
        self.beta_transition = nn.Sequential(nn.Linear(36, 18), nn.Tanh())
        self.alpha_layer = nn.Linear(18, action_dim)
        self.beta_layer = nn.Linear(18, action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()

    def forward(self, features: torch.Tensor):
        x = self.hidden_layer(features)
        alpha_feat = self.alpha_transition(x)
        beta_feat = self.beta_transition(x)
        alpha = 1.0 + self.alpha_softplus(self.alpha_layer(alpha_feat)) + 1e-06
        beta = 1.0 + self.beta_softplus(self.beta_layer(beta_feat)) + 1e-06
        return (alpha, beta)


class GAE(nn.Module):

    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor

    def forward(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor,
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step]
                + self.gamma * next_value[:, step] * not_done[:, step]
                - value[:, step]
            )
            advantages[:, step] = gae = (
                delta + self.gamma * self.lmbda * not_done[:, step] * gae
            )
        returns = advantages + value
        return (advantages, returns)


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        tensordict.shape[0] // num_minibatches * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]


@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int = 0,
    exploration_type: ExplorationType = ExplorationType.MEAN,
):
    logging.info("\n" + "=" * 30 + " Evaluation " + "=" * 30)
    env.eval()
    env.set_seed(seed)
    should_record = bool(cfg.get("eval", {}).get("record", False))
    render_callback = RenderCallback(interval=1) if should_record else None
    if should_record:
        env.enable_render(True)
    if hasattr(env.base_env, "finished_stats"):
        env.base_env.finished_stats.clear()
        logging.info("[EVAL] cleared finished_stats buffer.")
    max_steps = cfg.get("eval", {}).get("steps") or env.max_episode_length
    logging.info(f"[EVAL] rollout max_steps={max_steps}...")
    rollout_kwargs = {
        "max_steps": max_steps,
        "policy": policy,
        "auto_reset": True,
        "break_when_any_done": False,
        "return_contiguous": False,
    }
    if render_callback is not None:
        rollout_kwargs["callback"] = render_callback
    with set_exploration_type(exploration_type):
        env.rollout(**rollout_kwargs)
    logging.info("[EVAL] rollout complete.")
    if should_record:
        env.enable_render(not cfg.headless)
    env.reset()

    wandb_info = {}
    default_stats = {
        "episode_len": 0.0,
        "return": 0.0,
        "success": 0.0,
        "timeout": 0.0,
        "escaped": 0.0,
        "pursuer_collision": 0.0,
        "evader_collision": 0.0,
    }
    final_chosen_stats_list = []
    if not hasattr(env.base_env, "finished_stats") or not env.base_env.finished_stats:
        logging.info("[EVAL] finished_stats is empty; using default zero metrics.")
        raw_stats_avg = default_stats.copy()
    else:
        all_finished_stats = list(env.base_env.finished_stats)
        env.base_env.finished_stats.clear()
        logging.info(f"[EVAL] collected {len(all_finished_stats)} finished episodes.")
        env_episodes = collections.defaultdict(list)
        for stats in all_finished_stats:
            env_id = stats.get("env_id", -1)
            if env_id != -1:
                env_episodes[env_id].append(stats)
        for env_id, episodes_list in env_episodes.items():
            first_episode = episodes_list[0]
            ep_len = first_episode.get("episode_len", 0)
            if ep_len < 5 and len(episodes_list) > 1:
                chosen_episode = episodes_list[1]
                logging.info(
                    f"[EVAL] env {env_id}: first episode too short "
                    f"(len={ep_len:.0f}); using second episode."
                )
            else:
                chosen_episode = first_episode
            filled_stats = default_stats.copy()
            filled_stats.update(chosen_episode)
            final_chosen_stats_list.append(filled_stats)

        num_valid_episodes = len(final_chosen_stats_list)
        logging.info(f"[EVAL] selected {num_valid_episodes} valid episodes.")
        if num_valid_episodes == 0:
            raw_stats_avg = default_stats.copy()
        else:
            stats_sum = collections.defaultdict(float)
            for stats in final_chosen_stats_list:
                for key, value in stats.items():
                    if key != "env_id":
                        stats_sum[key] += value
            raw_stats_avg = default_stats.copy()
            for key, value_sum in stats_sum.items():
                raw_stats_avg[key] = value_sum / num_valid_episodes

    final_stats_agg = collections.defaultdict(list)
    for stats in final_chosen_stats_list:
        for key, value in stats.items():
            final_stats_agg[key].append(value)
    for key, avg_value in raw_stats_avg.items():
        wandb_info[f"eval/stats.{key}"] = float(avg_value)
    env.train()
    logging.info(f"[EVAL] wandb summary: {wandb_info}")
    logging.info("=" * 30 + " End Evaluation " + "=" * 30 + "\n")
    return wandb_info, raw_stats_avg, final_stats_agg


def my_vec_to_world_yaw_only(vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    if quat.ndim == vec.ndim + 1:
        quat = quat.squeeze(-2)
    euler_angles = quaternion_to_euler(quat)
    yaw = euler_angles[2]
    vx_body = vec[..., 0]
    vy_body = vec[..., 1]
    vz_body = vec[..., 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    vx_world = vx_body * cos_yaw - vy_body * sin_yaw
    vy_world = vx_body * sin_yaw + vy_body * cos_yaw
    vz_world = vz_body
    return torch.stack([vx_world, vy_world, vz_world], dim=-1)


def my_world_to_vec(vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    if quat.ndim == vec.ndim + 1:
        quat = quat.squeeze(-2)
    quat_normalized = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-09)
    q_w = quat_normalized[..., 0]
    q_vec = -quat_normalized[..., 1:]
    t = 2 * torch.cross(q_vec, vec, dim=-1)
    rotated_vec = vec + q_w.unsqueeze(-1) * t + torch.cross(q_vec, t, dim=-1)
    return rotated_vec


def quaternion_to_euler(quat):
    w, x, y, z = (quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3])
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1, torch.sign(sinp) * (torch.pi / 2), torch.asin(sinp)
    )
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)
