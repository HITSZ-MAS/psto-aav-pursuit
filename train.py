import logging
import os

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from setproctitle import setproctitle
from torchrl.envs.transforms import Compose, TransformedEnv
from torchrl.envs.utils import ExplorationType
from tqdm import tqdm

from mappo import MAPPO
from omni_drones import init_simulation_app
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.wandb import init_wandb


def _setup_env(cfg: DictConfig) -> TransformedEnv:
    from env import NavigationEnv

    return TransformedEnv(NavigationEnv(cfg), Compose()).train()


def _setup_policy(cfg: DictConfig, env: TransformedEnv) -> MAPPO:
    return MAPPO(
        cfg=cfg.algo,
        device=env.device,
        num_agents=env.base_env.num_pursuers,
        action_dim=env.action_spec.shape[-1],
    )


def _mean_stats(stats, key):
    values = [item[key] for item in stats if key in item]
    return float(np.mean(values)) if values else 0.0


def _run_training_loop(
    cfg: DictConfig, env: TransformedEnv, policy: MAPPO, run, start_frames: int
):
    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = int(cfg.total_frames)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
        exploration_type=ExplorationType.RANDOM,
    )
    if start_frames > 0:
        collector.update_frames(start_frames)

    finished_episode_stats = []
    save_interval = int(cfg.get("save_interval", 0))
    log_episode_count = max(env.num_envs, int(cfg.get("log_episode_count", 400)))
    pbar = tqdm(collector, total=total_frames)
    pbar.update(start_frames)

    for iteration, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        train_loss_stats = policy.train(
            data,
            current_step=iteration,
            total_steps=max(total_frames // max(frames_per_batch, 1), 1),
        )
        policy.step_schedulers()
        info.update(train_loss_stats)

        if save_interval > 0 and iteration > 0 and iteration % save_interval == 0:
            policy.save_state(os.path.join(run.dir, f"checkpoint_{iteration}.pt"))

        if hasattr(env.base_env, "finished_stats") and env.base_env.finished_stats:
            finished_episode_stats.extend(env.base_env.finished_stats)
            env.base_env.finished_stats.clear()

        if len(finished_episode_stats) >= log_episode_count:
            info.update(
                {
                    "episode/return": _mean_stats(finished_episode_stats, "return"),
                    "episode/length": _mean_stats(
                        finished_episode_stats, "episode_len"
                    ),
                    "episode/success_rate": _mean_stats(
                        finished_episode_stats, "success"
                    ),
                    "episode/pursuer_collision_rate": _mean_stats(
                        finished_episode_stats, "pursuer_collision"
                    ),
                    "episode/evader_collision_rate": _mean_stats(
                        finished_episode_stats, "evader_collision"
                    ),
                    "episode/timeout_rate": _mean_stats(
                        finished_episode_stats, "timeout"
                    ),
                    "episode/escaped_rate": _mean_stats(
                        finished_episode_stats, "escaped"
                    ),
                }
            )
            finished_episode_stats.clear()

        run.log(info)
        pbar.set_postfix(
            {
                "fps": f"{collector._fps:.1f}",
                "policy": f"{float(train_loss_stats['policy_loss']):.4f}",
                "value": f"{float(train_loss_stats['value_loss']):.4f}",
                "entropy": f"{float(train_loss_stats['entropy']):.3f}",
                "sl": f"{float(train_loss_stats['sl_loss']):.4f}",
            }
        )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    logging.info("\n%s", OmegaConf.to_yaml(cfg))

    env = _setup_env(cfg)
    policy = _setup_policy(cfg, env)
    env.base_env.set_policy_lstm(policy.lstm)

    checkpoint_path = cfg.get("checkpoint_path")
    if checkpoint_path:
        policy.load_model_weights(
            checkpoint_path,
            strict=True,
            load_optimizers=bool(cfg.get("resume_optimizer", False)),
        )

    _run_training_loop(cfg, env, policy, run, start_frames=0)
    policy.save_state(os.path.join(run.dir, "checkpoint_final.pt"))
    wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()
