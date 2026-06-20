import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from tensordict.tensordict import TensorDict
from torchrl.modules import ProbabilisticActor

from lstm import TrajectoryPredictor
from models.model import FullModel, STATE_DIM
from nav_utils import (
    BetaActor,
    GAE,
    IndependentBeta,
    ValueNorm,
    make_batch,
    my_vec_to_world_yaw_only,
)


class MAPPO(TensorDictModuleBase):
    def __init__(self, cfg, device, num_agents: int, action_dim: int):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.training = True

        self.gamma = getattr(cfg, "gamma", 0.995)
        self.gae_lambda = getattr(cfg, "gae_lambda", 0.95)
        self.gae = GAE(self.gamma, self.gae_lambda)
        self.value_norm = ValueNorm(1).to(self.device)

        self._build_networks()
        self.critic_loss_fn = nn.HuberLoss(delta=10)

        sl_future_len = getattr(cfg, "sl_future_len", 10)
        self.lstm = TrajectoryPredictor(
            input_dim=4, hidden_dim=64, output_dim=sl_future_len * 3
        ).to(self.device)
        self.sl_loss_fn = nn.MSELoss()

        actor_lr = getattr(cfg, "actor_lr", getattr(cfg, "learning_rate", 1e-4))
        critic_lr = getattr(cfg, "critic_lr", getattr(cfg, "learning_rate", 1e-4))
        encoder_lr = getattr(cfg, "encoder_lr", actor_lr)
        sl_lr = getattr(cfg, "sl_lr", 4e-4)

        self.encoder_opt = torch.optim.AdamW(
            self.encoder.parameters(), lr=encoder_lr, weight_decay=2e-4
        )
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=actor_lr, weight_decay=2e-4
        )
        self.critic_encoder_opt = torch.optim.AdamW(
            self.critic_encoder.parameters(), lr=critic_lr, weight_decay=1e-4
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=critic_lr, weight_decay=1e-4
        )
        self.lstm_opt = torch.optim.Adam(self.lstm.parameters(), lr=sl_lr)

        total_steps = getattr(cfg, "scheduler_steps", 600)
        warmup_steps = min(getattr(cfg, "warmup_steps", 21), max(total_steps - 1, 1))
        warmup_start_factor = getattr(cfg, "warmup_start_factor", 0.1)
        lr_min = getattr(cfg, "lr_min", 3e-6)

        self.encoder_scheduler = lr_scheduler.SequentialLR(
            self.encoder_opt,
            schedulers=[
                lr_scheduler.LinearLR(
                    self.encoder_opt,
                    start_factor=warmup_start_factor,
                    total_iters=warmup_steps,
                ),
                lr_scheduler.CosineAnnealingLR(
                    self.encoder_opt,
                    T_max=max(total_steps - warmup_steps, 1),
                    eta_min=lr_min,
                ),
            ],
            milestones=[warmup_steps],
        )
        self.actor_scheduler = lr_scheduler.SequentialLR(
            self.actor_opt,
            schedulers=[
                lr_scheduler.LinearLR(
                    self.actor_opt,
                    start_factor=warmup_start_factor,
                    total_iters=warmup_steps,
                ),
                lr_scheduler.CosineAnnealingLR(
                    self.actor_opt,
                    T_max=max(total_steps - warmup_steps, 1),
                    eta_min=lr_min,
                ),
            ],
            milestones=[warmup_steps],
        )
        self.critic_encoder_scheduler = lr_scheduler.ConstantLR(
            self.critic_encoder_opt, factor=1.0
        )
        self.critic_scheduler = lr_scheduler.ConstantLR(self.critic_opt, factor=1.0)
        self.lstm_scheduler = lr_scheduler.StepLR(
            self.lstm_opt,
            step_size=getattr(cfg, "lstm_lr_decay_steps", 1000),
            gamma=getattr(cfg, "lstm_lr_decay_rate", 0.95),
        )

        self.clip_ratio = getattr(cfg, "clip_ratio", 0.1)
        self.entropy_coef = getattr(cfg, "entropy_loss_coefficient", 0.01)
        self.action_limit = torch.tensor(
            getattr(cfg, "action_limit", [0.8, 2.0]), device=device
        )
        self.value_clip_ratio = getattr(
            cfg, "value_clip_ratio", getattr(cfg, "value_clip_range", self.clip_ratio)
        )

    def _build_networks(self):
        self.encoder = FullModel().to(self.device)
        with torch.no_grad():
            lidar_vbeams = getattr(self.cfg, "lidar_vbeams", 24)
            lidar_hres = getattr(self.cfg, "lidar_hres", 0.75)
            downrate = getattr(self.cfg, "downrate", 4)
            vbeams_down = lidar_vbeams // downrate
            hbeams_down = int(360 / lidar_hres) // downrate
            fake_fused_image = torch.zeros(
                1, 2, vbeams_down, hbeams_down, device=self.device
            )
            fake_state = torch.zeros(1, STATE_DIM, device=self.device)
            feature = self.encoder(fake_fused_image, fake_state)

        feature_dim = feature.shape[-1]
        self.actor_module = BetaActor(self.action_dim, feature_dim)
        self.actor = ProbabilisticActor(
            TensorDictModule(
                self.actor_module, ["combined_feature"], ["alpha", "beta"]
            ),
            in_keys=["alpha", "beta"],
            out_keys=["action_normalized"],
            distribution_class=IndependentBeta,
            return_log_prob=True,
        ).to(self.device)

        num_pursuers = getattr(self.cfg, "num_pursuers", self.num_agents)
        num_evaders = getattr(self.cfg, "num_evaders", 1)
        num_drones = num_pursuers + num_evaders
        critic_state_dim = (
            num_drones * 3 + num_drones * 10 + num_drones * 3 + num_pursuers * 18
        )

        self.critic_encoder = nn.Sequential(
            nn.Linear(critic_state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        ).to(self.device)
        self.critic = TensorDictModule(
            nn.Sequential(
                nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 1)
            ),
            ["_critic_feature"],
            ["state_value"],
        ).to(self.device)

        with torch.no_grad():
            dummy_feature = torch.zeros(1, feature_dim, device=self.device)
            _ = self.actor(
                TensorDict({"combined_feature": dummy_feature}, batch_size=[1])
            )
            dummy_global_state = torch.zeros(1, critic_state_dim, device=self.device)
            dummy_critic_feature = self.critic_encoder(dummy_global_state)
            _ = self.critic(
                TensorDict({"_critic_feature": dummy_critic_feature}, batch_size=[1])
            )

        def init_(module):
            if isinstance(module, nn.Linear) and module.weight.data.numel() > 0:
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.actor.apply(init_)
        self.critic.apply(init_)
        self.critic_encoder.apply(init_)

    def __call__(self, tensordict: TensorDict):
        agents_td = tensordict["agents"]
        observation_td = agents_td["observation"]
        features = self.encoder(observation_td["fused_image"], observation_td["state"])
        actor_input = TensorDict(
            {"combined_feature": features}, batch_size=features.shape[0]
        )
        agents_td.update(self.actor(actor_input))

        global_state = tensordict["state"]
        critic_feature = self.critic_encoder(global_state)
        critic_input = TensorDict(
            {"_critic_feature": critic_feature}, batch_size=global_state.shape[:-1]
        )
        tensordict.set("state_value", self.critic(critic_input)["state_value"])
        return self._process_actions(tensordict)

    def _process_actions(self, tensordict: TensorDict):
        action_normalized = tensordict["agents", "action_normalized"]
        action_vector = action_normalized[..., 0]
        angle = math.pi * (1 - 2 * action_vector)
        unit_vector = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        action_length = 0.1 * action_normalized[..., 1] + 0.9
        vel_body = action_length.unsqueeze(-1) * unit_vector * self.action_limit[0]
        vel_body = torch.cat(
            [vel_body, torch.zeros(*vel_body.shape[:-1], 1, device=vel_body.device)],
            dim=-1,
        )
        quat = tensordict["agents", "observation"]["state"][..., 0:4]
        vel_world = torch.vmap(my_vec_to_world_yaw_only, in_dims=(0, 0))(
            vel_body.reshape(-1, 3), quat.reshape(-1, 4)
        )
        actions = vel_world.reshape(*vel_body.shape)
        tensordict.set(("agents", "action"), actions)
        tensordict.set(("agents", "actionforstate"), vel_body)
        return tensordict

    def train(self, tensordict: TensorDict, current_step: int, total_steps: int):
        tensordict = self._compute_returns(tensordict)
        logs = []
        lstm_train_counter = 0
        for _ in range(self.cfg.training_epoch_num):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                log = self._update_rl(minibatch)
                if lstm_train_counter % getattr(self.cfg, "lstm_train_every", 3) == 0:
                    log.update(self._update_sl(minibatch))
                else:
                    log.update({"sl_loss": 0.0, "lstm_grad_norm": 0.0})
                lstm_train_counter += 1
                logs.append(log)
        return {
            k: torch.mean(torch.tensor([item[k] for item in logs])) for k in logs[0]
        }

    def step_schedulers(self):
        self.encoder_scheduler.step()
        self.actor_scheduler.step()
        self.critic_encoder_scheduler.step()
        self.critic_scheduler.step()
        self.lstm_scheduler.step()

    def _compute_returns(self, tensordict: TensorDict):
        with torch.no_grad():
            next_global_state = tensordict["next"]["state"]
            next_critic_feature = self.critic_encoder(next_global_state)
            next_values = self.critic(
                TensorDict(
                    {"_critic_feature": next_critic_feature},
                    batch_size=next_global_state.shape[:-1],
                )
            )["state_value"]

        rewards = tensordict["next", ("agents", "reward")][..., 0, :]
        dones = tensordict["next", "terminated"]
        values = self.value_norm.denormalize(tensordict["state_value"])
        next_values = self.value_norm.denormalize(next_values)
        adv, ret = self.gae(rewards, dones, values, next_values)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)
        tensordict.set("adv", adv.unsqueeze(2).expand(-1, -1, self.num_agents, -1))
        tensordict.set("ret", ret.unsqueeze(2).expand(-1, -1, self.num_agents, -1))
        return tensordict

    def _update_rl(self, minibatch: TensorDict):
        observation_td_flat = minibatch[("agents", "observation")].reshape(-1)
        features = self.encoder(
            observation_td_flat["fused_image"], observation_td_flat["state"]
        )
        actor_input = TensorDict(
            {"combined_feature": features}, batch_size=features.shape[0]
        )
        dist = self.actor.get_dist(actor_input)

        action_normalized_flat = minibatch[("agents", "action_normalized")].reshape(
            -1, self.action_dim
        )
        log_probs = dist.log_prob(action_normalized_flat)
        entropy = dist.entropy()
        adv_flat = minibatch["adv"].reshape(-1, 1)
        old_log_probs_flat = minibatch[("agents", "sample_log_prob")].reshape(-1)
        ratio = torch.exp(log_probs - old_log_probs_flat).unsqueeze(-1)
        policy_loss = -torch.mean(
            torch.min(
                adv_flat * ratio,
                adv_flat * ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio),
            )
        )
        entropy_loss = -self.entropy_coef * torch.mean(entropy)
        actor_loss = policy_loss + entropy_loss

        global_state = minibatch["state"]
        returns = minibatch["ret"][:, 0, :]
        critic_feature = self.critic_encoder(global_state)
        new_values = self.critic(
            TensorDict(
                {"_critic_feature": critic_feature}, batch_size=global_state.shape[:-1]
            )
        )["state_value"]
        old_values = minibatch["state_value"]
        values_denorm = self.value_norm.denormalize(new_values)
        old_values_denorm = self.value_norm.denormalize(old_values)
        values_clipped = old_values_denorm + (values_denorm - old_values_denorm).clamp(
            -self.value_clip_ratio, self.value_clip_ratio
        )
        values_clipped_norm = self.value_norm.normalize(values_clipped)
        value_loss = torch.max(
            self.critic_loss_fn(new_values, returns),
            self.critic_loss_fn(values_clipped_norm, returns),
        )
        total_loss = actor_loss + value_loss

        self.encoder_opt.zero_grad()
        self.actor_opt.zero_grad()
        self.critic_encoder_opt.zero_grad()
        self.critic_opt.zero_grad()
        total_loss.backward()
        encoder_grad_norm = nn.utils.clip_grad_norm_(
            self.encoder.parameters(), self.cfg.max_grad_norm
        )
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.max_grad_norm
        )
        critic_encoder_grad_norm = nn.utils.clip_grad_norm_(
            self.critic_encoder.parameters(), self.cfg.max_grad_norm
        )
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.max_grad_norm
        )
        self.encoder_opt.step()
        self.actor_opt.step()
        self.critic_encoder_opt.step()
        self.critic_opt.step()

        explained_var = 1 - F.mse_loss(
            new_values.detach(), returns
        ) / returns.var().clamp(min=1e-7)
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "entropy_loss": entropy_loss.item(),
            "entropy_coef": self.entropy_coef,
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "critic_encoder_grad_norm": critic_encoder_grad_norm.item(),
            "encoder_grad_norm": encoder_grad_norm.item(),
            "explained_var": explained_var.item(),
        }

    def _update_sl(self, minibatch: TensorDict):
        history_pos = minibatch.get(("info", "sl_history_input"))
        ground_truth = minibatch.get(("info", "sl_future_ground_truth"))
        sl_history_mask = minibatch.get(("info", "sl_history_mask"))
        agent_dt_value = minibatch.get(("info", "agent_dt"))
        has_valid_history = sl_history_mask.any(dim=1)
        if not has_valid_history.any():
            return {"sl_loss": 0.0, "lstm_grad_norm": 0.0}

        valid_history_pos = history_pos[has_valid_history]
        valid_ground_truth = ground_truth[has_valid_history]
        valid_history_mask = sl_history_mask[has_valid_history]
        valid_agent_dt = agent_dt_value[has_valid_history]
        dt_tensor = valid_agent_dt.view(-1, 1, 1).expand(
            -1, valid_history_pos.shape[1], 1
        )
        lstm_input = torch.cat([valid_history_pos, dt_tensor], dim=-1)
        predicted_trajectory = self.lstm(lstm_input, mask=valid_history_mask).view_as(
            valid_ground_truth
        )
        future_mask = torch.abs(valid_ground_truth).sum(dim=-1) > 1e-6
        loss = (
            F.mse_loss(
                predicted_trajectory[future_mask], valid_ground_truth[future_mask]
            )
            if future_mask.any()
            else torch.tensor(0.0, device=self.device)
        )

        if loss.item() > 1e-6:
            self.lstm_opt.zero_grad()
            loss.backward()
            lstm_grad_norm = nn.utils.clip_grad_norm_(
                self.lstm.parameters(),
                max_norm=getattr(self.cfg, "lstm_max_grad_norm", 1.0),
            )
            self.lstm_opt.step()
        else:
            lstm_grad_norm = torch.tensor(0.0, device=self.device)
        return {"sl_loss": loss.item(), "lstm_grad_norm": lstm_grad_norm.item()}

    # Checkpoints are stored per module so evaluation can load model weights without optimizer state.
    def save_state(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "critic_encoder": self.critic_encoder.state_dict(),
                "critic": self.critic.state_dict(),
                "lstm": self.lstm.state_dict(),
                "encoder_opt": self.encoder_opt.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_encoder_opt": self.critic_encoder_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "lstm_opt": self.lstm_opt.state_dict(),
                "value_norm": self.value_norm.state_dict(),
            },
            path,
        )

    def load_model_weights(
        self, path: str, strict: bool = True, load_optimizers: bool = False
    ):
        state = torch.load(path, map_location=self.device)
        for name, model in {
            "encoder": self.encoder,
            "actor": self.actor,
            "critic_encoder": self.critic_encoder,
            "critic": self.critic,
            "lstm": self.lstm,
        }.items():
            if name not in state:
                raise KeyError(f"Checkpoint is missing '{name}' weights")
            model.load_state_dict(state[name], strict=strict)
        if "value_norm" in state:
            self.value_norm.load_state_dict(state["value_norm"])
        if load_optimizers:
            for key, opt in {
                "encoder_opt": self.encoder_opt,
                "actor_opt": self.actor_opt,
                "critic_encoder_opt": self.critic_encoder_opt,
                "critic_opt": self.critic_opt,
                "lstm_opt": self.lstm_opt,
            }.items():
                if key in state:
                    opt.load_state_dict(state[key])
        logging.info("Loaded checkpoint from %s", path)

    def train_mode(self):
        self.training = True
        self.encoder.train()
        self.actor.train()
        self.critic_encoder.train()
        self.critic.train()
        self.lstm.train()

    def eval_mode(self):
        self.training = False
        self.encoder.eval()
        self.actor.eval()
        self.critic_encoder.eval()
        self.critic.eval()
        self.lstm.eval()

    def load_model_weights_eval(self, path: str):
        self.load_model_weights(path, strict=True, load_optimizers=False)
        self.eval_mode()
