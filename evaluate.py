import logging
import os
from collections import defaultdict
from typing import List, Tuple

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from setproctitle import setproctitle
from torchrl.envs.transforms import Compose, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from mappo import MAPPO
from nav_utils import evaluate
from omni_drones import init_simulation_app
from omni_drones.utils.wandb import init_wandb


def _setup_env(cfg: DictConfig) -> TransformedEnv:
    from env import NavigationEnv

    return TransformedEnv(NavigationEnv(cfg), Compose()).eval()


def _setup_policy(cfg: DictConfig, env: TransformedEnv) -> MAPPO:
    return MAPPO(
        cfg=cfg.algo,
        device=env.device,
        num_agents=env.base_env.num_pursuers,
        action_dim=env.action_spec.shape[-1],
    )


def _safe_mean(data_list: List[float]) -> Tuple[float, int]:
    if not data_list:
        return 0.0, 0
    return float(np.mean(data_list)), len(data_list)


def _run_evaluation(cfg: DictConfig, env: TransformedEnv, policy: MAPPO, run):
    with set_exploration_type(ExplorationType.MODE):
        wandb_info, raw_stats_avg, final_stats_agg = evaluate(env, policy, cfg=cfg)
    if final_stats_agg is None:
        final_stats_agg = defaultdict(list)
    if run:
        run.log(wandb_info)

    all_lengths = final_stats_agg.get("episode_len", final_stats_agg.get("l", []))
    avg_len_success, count_success = _safe_mean(
        [l for l, s in zip(all_lengths, final_stats_agg.get("success", [])) if s > 0.5]
    )
    avg_len_p_collision, count_p_collision = _safe_mean(
        [
            l
            for l, c in zip(all_lengths, final_stats_agg.get("pursuer_collision", []))
            if c > 0.5
        ]
    )
    avg_len_e_collision, count_e_collision = _safe_mean(
        [
            l
            for l, c in zip(all_lengths, final_stats_agg.get("evader_collision", []))
            if c > 0.5
        ]
    )
    avg_len_timeout, count_timeout = _safe_mean(
        [l for l, t in zip(all_lengths, final_stats_agg.get("timeout", [])) if t > 0.5]
    )
    avg_len_escaped, count_escaped = _safe_mean(
        [l for l, e in zip(all_lengths, final_stats_agg.get("escaped", [])) if e > 0.5]
    )

    logging.info("\nEvaluation summary")
    logging.info(f"checkpoint: {os.path.basename(cfg.eval.checkpoint_path)}")
    logging.info(f"episodes: {env.num_envs}")
    logging.info(f"return: {raw_stats_avg.get('return', 0.0):.2f}")
    logging.info(f"length: {raw_stats_avg.get('episode_len', 0.0):.1f}")
    logging.info(f"success_rate: {raw_stats_avg.get('success', 0.0) * 100:.1f}%")
    logging.info(
        f"pursuer_collision_rate: {raw_stats_avg.get('pursuer_collision', 0.0) * 100:.1f}%"
    )
    logging.info(
        f"evader_collision_rate: {raw_stats_avg.get('evader_collision', 0.0) * 100:.1f}%"
    )
    logging.info(f"timeout_rate: {raw_stats_avg.get('timeout', 0.0) * 100:.1f}%")
    logging.info(f"escaped_rate: {raw_stats_avg.get('escaped', 0.0) * 100:.1f}%")
    if (
        count_success
        + count_p_collision
        + count_e_collision
        + count_timeout
        + count_escaped
        > 0
    ):
        logging.info(f"length_success: {avg_len_success:.1f} ({count_success})")
        logging.info(
            f"length_pursuer_collision: {avg_len_p_collision:.1f} ({count_p_collision})"
        )
        logging.info(
            f"length_evader_collision: {avg_len_e_collision:.1f} ({count_e_collision})"
        )
        logging.info(f"length_timeout: {avg_len_timeout:.1f} ({count_timeout})")
        logging.info(f"length_escaped: {avg_len_escaped:.1f} ({count_escaped})")


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    setproctitle(f"eval_{cfg.task.name}")
    logging.info("\n%s", OmegaConf.to_yaml(cfg))
    run = init_wandb(cfg)

    env = _setup_env(cfg)
    policy = _setup_policy(cfg, env)
    env.base_env.set_policy_lstm(policy.lstm)

    checkpoint_path = cfg.eval.checkpoint_path
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"eval.checkpoint_path does not exist: {checkpoint_path}"
        )
    policy.load_model_weights_eval(checkpoint_path)
    policy.eval_mode()

    with torch.no_grad():
        _run_evaluation(cfg, env, policy, run)
    wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()
