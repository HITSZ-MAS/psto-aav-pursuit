import torch
import math
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    UnboundedContinuousTensorSpec,
    CompositeSpec,
)
from omni_drones.envs.isaac_env import IsaacEnv
import omni.isaac.lab.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.terrains import (
    TerrainImporterCfg,
    TerrainImporter,
    TerrainGeneratorCfg,
    HfDiscreteObstaclesTerrainCfg,
)
from omni_drones.utils.torch import euler_to_quaternion
from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.prims import XFormPrimView
import omni.physx as _physx
import torch.nn.functional as F
from omni_drones.controllers import LeePositionController
from nav_utils import my_world_to_vec
from omni.isaac.lab.utils.math import quat_conjugate, quat_rotate

STATE_DIM = 13
TIME_STEPS = 1
ACTION_DIM = 4
DOWNRATE = 4


def downsample_lidar_scan(
    lidar_scan, kernel_size=(DOWNRATE, DOWNRATE), stride=(DOWNRATE, DOWNRATE)
):
    if lidar_scan.dim() == 3:
        lidar_scan = lidar_scan.unsqueeze(1)
    return F.max_pool2d(lidar_scan, kernel_size=kernel_size, stride=stride)


class NavigationEnv(IsaacEnv):

    def __init__(self, cfg):
        self.num_pursuers = cfg.get("num_pursuers", 2)
        self.num_evaders = cfg.get("num_evaders", 1)
        self.num_drones = self.num_pursuers + self.num_evaders
        reward_cfg = cfg.get("reward", {})
        self.escape_min_distance = float(reward_cfg.get("escape_min_distance", 7.5))
        self.escape_max_distance = float(reward_cfg.get("escape_max_distance", 10.0))
        self.capture_threshold = float(reward_cfg.get("capture_threshold", 1.0))
        self.capture_bonus = float(reward_cfg.get("capture_bonus", 1800.0))
        self.pursuer_collision_distance = float(
            reward_cfg.get("pursuer_collision_distance", 0.35)
        )
        self.evader_collision_distance = float(
            reward_cfg.get("evader_collision_distance", 0.32)
        )
        self.pursuer_collision_penalty = float(
            reward_cfg.get("pursuer_collision_penalty", -900.0)
        )
        self.evader_collision_bonus = float(
            reward_cfg.get("evader_collision_bonus", 100.0)
        )
        self.escape_penalty = float(reward_cfg.get("escape_penalty", -700.0))
        self.below_bound_altitude = float(reward_cfg.get("below_bound_altitude", 0.5))
        self.approach_reward_min = float(reward_cfg.get("approach_reward_min", -3.0))
        self.coordination_optimal_distance = float(
            reward_cfg.get("coordination_optimal_distance", 4.0)
        )
        self.coordination_sigma = float(reward_cfg.get("coordination_sigma", 2.0))
        self.obstacle_penalty_sensitive_range = float(
            reward_cfg.get("obstacle_penalty_sensitive_range", 2.5)
        )
        self.obstacle_penalty_log_power = float(
            reward_cfg.get("obstacle_penalty_log_power", 5.0)
        )
        self.obstacle_penalty_log_offset = float(
            reward_cfg.get("obstacle_penalty_log_offset", 1.6)
        )
        self.obstacle_penalty_quantile = float(
            reward_cfg.get("obstacle_penalty_quantile", 0.03)
        )
        self.time_penalty_start_step = int(
            reward_cfg.get("time_penalty_start_step", 200)
        )
        self.time_penalty_value = float(reward_cfg.get("time_penalty_value", -0.05))
        self.pursuit_weight = float(reward_cfg.get("pursuit_weight", 1.1))
        self.pursuit_offset = float(reward_cfg.get("pursuit_offset", 0.1))
        self.coordination_weight = float(reward_cfg.get("coordination_weight", 0.2))
        self.obstacle_weight = float(reward_cfg.get("obstacle_weight", 0.5))
        self.formation_weight = float(reward_cfg.get("formation_weight", 1.3))
        evader_cfg = cfg.get("evader", {})
        self.evader_pursuer_force_multiplier = float(
            evader_cfg.get("pursuer_force_multiplier", 18.0)
        )
        self.max_evader_speed = float(evader_cfg.get("max_speed", 1.7))
        self.evader_near_boundary_threshold = float(
            evader_cfg.get("near_boundary_threshold", 2.2)
        )
        self.evader_obstacle_awareness_distance = float(
            evader_cfg.get("obstacle_awareness_distance", 2.0)
        )
        self.evader_obstacle_repulsion_weight = float(
            evader_cfg.get("obstacle_repulsion_weight", 0.1)
        )
        self.evader_inward_bias_strength = float(
            evader_cfg.get("inward_bias_strength", 0.1)
        )
        self.evader_target_altitude = float(evader_cfg.get("target_altitude", 2.0))
        self.evader_altitude_gain = float(evader_cfg.get("altitude_gain", 2.0))
        controller_cfg = cfg.get("controller", {})
        self.yaw_speed_threshold = float(controller_cfg.get("yaw_speed_threshold", 0.1))
        visibility_cfg = cfg.get("visibility", {})
        self.max_visibility_distance = float(visibility_cfg.get("max_distance", 15.0))
        self.visibility_ray_alignment_threshold = float(
            visibility_cfg.get("ray_alignment_threshold", 0.7)
        )
        self.visibility_obstacle_clearance = float(
            visibility_cfg.get("obstacle_clearance", 0.5)
        )
        self.sl_history_len = 10
        self.sl_future_len = 10
        self.pursuer_ids = torch.arange(self.num_pursuers, device=self.device)
        self.evader_id = self.num_pursuers
        self.lidar_range = cfg.sensor.lidar_range
        vfov_tuple = cfg.sensor.lidar_vfov
        self.lidar_vfov = (max(-89.0, vfov_tuple[0]), min(89.0, vfov_tuple[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360 / self.lidar_hres)
        self.downrate = DOWNRATE
        self.lidar_hbeams_down = self.lidar_hbeams // self.downrate
        self.lidar_vbeams_down = self.lidar_vbeams // self.downrate
        self.env_spacing = 20
        super().__init__(cfg, cfg.headless)
        self.frame_skip = cfg.get("frame_skip", 1)
        self.agent_dt = self.dt * self.frame_skip
        self._lidar_layout = "hv"
        self.flip_v_axis = False
        self.all_drone_prims = XFormPrimView(f"/World/envs/env_.*/{self.drone.name}_*")
        self.all_drone_prims.initialize()
        self.drone.n = self.num_drones
        self.drone.initialize()
        self.evader_full_history = torch.zeros(
            self.num_envs,
            self.sl_history_len + self.sl_future_len,
            3,
            device=self.device,
        )
        all_terrain_prim_paths = ["/World/envs"]
        self.mesh_prim_paths = all_terrain_prim_paths
        self.finished_stats = []
        self.lidar = []
        for i in range(self.num_pursuers):
            prim_path = f"/World/envs/env_.*/{self.drone.name}_{i}/base_link"
            ray_cfg = RayCasterCfg(
                prim_path=prim_path,
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
                attach_yaw_only=False,
                pattern_cfg=patterns.BpearlPatternCfg(
                    horizontal_res=self.lidar_hres,
                    vertical_ray_angles=torch.linspace(
                        *self.lidar_vfov, self.lidar_vbeams
                    ),
                ),
                mesh_prim_paths=self.mesh_prim_paths,
                debug_vis=False,
            )
            lidar = RayCaster(ray_cfg)
            lidar._initialize_impl()
            self.lidar.append(lidar)
        evader_ray_caster_cfg = RayCasterCfg(
            prim_path=f"/World/envs/env_.*/{self.drone.name}_{self.evader_id}/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=False,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres,
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams),
            ),
            mesh_prim_paths=self.mesh_prim_paths,
            debug_vis=False,
        )
        self.evader_lidar = RayCaster(evader_ray_caster_cfg)
        self.evader_lidar._initialize_impl()
        self.physx_iface = _physx.acquire_physx_interface()
        self.physics_context = self.sim._physics_context
        self.state_buffer = torch.zeros(
            self.num_envs, self.num_pursuers, TIME_STEPS, STATE_DIM, device=self.device
        )
        self.action = torch.zeros(
            self.num_envs, self.num_drones, ACTION_DIM, device=self.device
        )
        self.prev_3d_actions = torch.zeros(
            self.num_envs, self.num_drones, 3, device=self.device
        )
        self.drone_state = torch.zeros(
            self.num_envs, self.num_drones, STATE_DIM, device=self.device
        )
        self.lidar_buffer = torch.zeros(
            self.num_envs,
            self.num_pursuers,
            self.lidar_vbeams_down,
            self.lidar_hbeams_down,
            device=self.device,
        )
        self.last_side_step_dir = torch.zeros(self.num_envs, 3, device=self.device)
        self.was_head_on = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.policy_lstm = None
        self.evader_history_buffer = torch.zeros(
            self.num_envs, self.sl_history_len, 3, device=self.device
        )
        self.controller = LeePositionController(9.81, self.cfg.drone.uav_params).to(
            self.device
        )
        self.stats = TensorDict(
            {
                "return": torch.zeros(self.num_envs, 1, device=self.device),
                "episode_len": torch.zeros(self.num_envs, 1, device=self.device),
                "success": torch.zeros(self.num_envs, 1, device=self.device),
                "pursuer_collision": torch.zeros(self.num_envs, 1, device=self.device),
                "evader_collision": torch.zeros(self.num_envs, 1, device=self.device),
                "timeout": torch.zeros(self.num_envs, 1, device=self.device),
                "escaped": torch.zeros(self.num_envs, 1, device=self.device),
            },
            batch_size=[self.num_envs],
        )
        self.collision_history = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_collision_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.step1_collision_ever = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.collision_reset_positions = torch.tensor(
            [
                [[0.0, 2.3, 2.0], [0.0, -2.3, 2.0], [3.5, 0.0, 2.0]],
                [[0.0, 0.3, 2.0], [0.0, -4.3, 2.0], [3.5, -2.0, 2.0]],
                [[-3.5, 1.5, 2.0], [-3.5, -1.5, 2.0], [1.2, 0.0, 2.0]],
                [[-2.0, 3.0, 2.0], [-2.0, -1.0, 2.0], [2.0, -0.0, 2.0]],
                [[-1.5, 1.5, 2.0], [-1.5, -2.5, 2.0], [2.5, 0.0, 2.0]],
                [[-4.0, 1.0, 2.0], [-4.0, -3.0, 2.0], [-0.5, -1.0, 2.0]],
                [[-3.0, 3.5, 2.0], [-3.0, -1.5, 2.0], [1.5, 2.0, 2.0]],
                [[2.0, 4.0, 2.0], [2.0, -1.0, 2.0], [5.5, 2.5, 2.0]],
            ],
            device=self.device,
        )
        self.alt_reset_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.num_alt_positions = int(self.collision_reset_positions.shape[0] - 1)
        self.last_reset_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._initialize_evader_policy_states()

    def set_policy_lstm(self, policy_lstm: torch.nn.Module):
        self.policy_lstm = policy_lstm

    def _apply_lidar_layout(self, scan_flat: torch.Tensor) -> torch.Tensor:
        B, P = (self.num_envs, self.num_pursuers)
        V, H = (self.lidar_vbeams, self.lidar_hbeams)
        if scan_flat.dim() != 3 or scan_flat.shape[-1] != V * H:
            raise RuntimeError(
                f"scan_flat shape mismatch: got {tuple(scan_flat.shape)}, expect [B,P,{V * H}]"
            )
        img_vh = scan_flat.reshape(B, P, H, V).permute(0, 1, 3, 2)
        if self.flip_v_axis:
            img_vh = torch.flip(img_vh, dims=[2])
        return img_vh

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name]
        drone_cfg = drone_model.cfg_cls()
        self.drone = drone_model(name="Hummingbird", cfg=drone_cfg)
        self.drone.n = self.num_drones
        all_drone_paths = []
        all_initial_positions_local = []
        all_initial_positions_world = []
        base_drone_positions = torch.tensor(
            [[-3.0, 2.0, 2.0], [-3.0, -2.0, 2.0], [0.0, 0.0, 2.0]], device=self.device
        )
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        cfg_ground = sim_utils.GroundPlaneCfg(
            color=(0.1, 0.1, 0.1), size=(10000.0, 12000.0)
        )
        cfg_ground.func(
            "/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01)
        )
        self.mapx = 9
        self.mapy = 9
        self.mapz = 4.5
        self.map_range = [self.mapx, self.mapy, self.mapz]
        terrain_generator_cfg = TerrainGeneratorCfg(
            seed=23233,
            curriculum=False,
            size=(self.map_range[0] * 2, self.map_range[1] * 2),
            border_width=0.0,
            num_rows=1,
            num_cols=self.num_envs,
            horizontal_scale=0.1,
            vertical_scale=0.1,
            slope_threshold=0.75,
            use_cache=False,
            color_scheme="height",
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=0.0,
                    num_obstacles=18,
                    obstacle_height_mode="range",
                    obstacle_width_range=(0.4, 0.8),
                    obstacle_height_range=[3.0, 3.5, 3.6, 3.8, 6.0],
                    obstacle_height_probability=[0.1, 0.15, 0.2, 0.55],
                    platform_width=0.0,
                )
            },
        )
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/envs/terrain",
            terrain_type="generator",
            terrain_generator=terrain_generator_cfg,
            num_envs=1,
            collision_group=-1,
            env_spacing=self.env_spacing,
        )
        TerrainImporter(terrain_cfg)
        center_offset = (self.num_envs - 1) / 2.0
        for env_idx in range(self.num_envs):
            env_path = f"/World/envs/env_{env_idx}"
            env_offset = [0.0, (env_idx - center_offset) * self.mapy * 2.0, 0.0]
            for drone_idx in range(self.num_drones):
                prim_path = f"{env_path}/{self.drone.name}_{drone_idx}"
                all_drone_paths.append(prim_path)
            all_initial_positions_local.append(base_drone_positions)
            env_offset_tensor = torch.tensor(env_offset, device=self.device)
            all_initial_positions_world.append(base_drone_positions + env_offset_tensor)
        local_positions_tensor = torch.cat(all_initial_positions_local, dim=0)
        self.drone.spawn(
            prim_paths=all_drone_paths, translations=local_positions_tensor
        )
        self.world_positions_tensor = torch.cat(all_initial_positions_world, dim=0)
        self.world_orientations_tensor = torch.zeros(
            self.num_envs * self.num_drones, 4, device=self.device
        )
        self.world_orientations_tensor[:, 0] = 1.0

    # IsaacEnv calls this hook once after scene construction to register multi-agent specs.
    def _set_specs(self):
        agent_obs_spec = CompositeSpec(
            {
                "state": UnboundedContinuousTensorSpec(shape=(STATE_DIM,)),
                "state_buffer": UnboundedContinuousTensorSpec(
                    shape=(TIME_STEPS, STATE_DIM)
                ),
                "fused_image": UnboundedContinuousTensorSpec(
                    shape=(2, self.lidar_vbeams_down, self.lidar_hbeams_down)
                ),
                "direction": UnboundedContinuousTensorSpec(shape=(3,)),
            }
        )
        single_action_spec = UnboundedContinuousTensorSpec(shape=(2,))
        central_state_dim = 84
        stats_spec = CompositeSpec(
            {
                "return": UnboundedContinuousTensorSpec(shape=(1,)),
                "episode_len": UnboundedContinuousTensorSpec(shape=(1,)),
                "success": UnboundedContinuousTensorSpec(shape=(1,)),
                "collision": UnboundedContinuousTensorSpec(shape=(1,)),
                "timeout": UnboundedContinuousTensorSpec(shape=(1,)),
            }
        )
        info_spec = CompositeSpec(
            {
                "sl_history_input": UnboundedContinuousTensorSpec(
                    shape=(self.sl_history_len, 3)
                ),
                "sl_future_ground_truth": UnboundedContinuousTensorSpec(
                    shape=(self.sl_future_len, 3)
                ),
                "sl_history_mask": UnboundedContinuousTensorSpec(
                    shape=(self.sl_history_len,)
                ),
                "agent_dt": UnboundedContinuousTensorSpec(shape=(1,)),
                "drone_state": UnboundedContinuousTensorSpec(
                    shape=(self.num_drones, STATE_DIM)
                ),
                "stats": stats_spec,
            }
        )
        self.observation_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec({"observation": agent_obs_spec}).expand(
                        self.num_pursuers
                    ),
                    "state": UnboundedContinuousTensorSpec(shape=(central_state_dim,)),
                    "info": info_spec,
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.action_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec({"action": single_action_spec}).expand(
                        self.num_pursuers
                    )
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.reward_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {"reward": UnboundedContinuousTensorSpec(shape=(1,))}
                    ).expand(self.num_pursuers)
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.state_spec = (
            CompositeSpec(
                {"state": UnboundedContinuousTensorSpec(shape=(central_state_dim,))}
            )
            .expand(self.num_envs)
            .to(self.device)
        )

    def _normalize_safe(self, v, dim=-1, keepdim=False):
        norm = torch.norm(v, dim=dim, keepdim=True)
        return torch.where(norm > 1e-06, v / (norm + 1e-06), torch.zeros_like(v))

    # Environment-side evader controller used to generate the pursuit target dynamics.
    def _evader_policy_vec(self, pursuer_pos, evader_pos):
        B = self.num_envs
        device = self.device
        evader_world_pos = evader_pos.squeeze(1)
        center_offset = (B - 1) / 2.0
        y_offsets = (
            (torch.arange(B, device=device, dtype=torch.float32) - center_offset)
            * self.mapy
            * 2.0
        )
        arena_centers = torch.zeros_like(evader_world_pos)
        arena_centers[:, 1] = y_offsets
        arena_radius = self.mapx
        vec_to_center = arena_centers - evader_world_pos
        vec_to_center[:, 2] = 0
        dist_to_wall = arena_radius - torch.norm(vec_to_center[:, :2], dim=-1)
        flee_vectors = evader_world_pos.unsqueeze(1) - pursuer_pos
        flee_vectors[..., 2] = 0
        distances_p = torch.norm(flee_vectors, dim=-1)
        repulsion_strength_p = torch.clamp(
            self.evader_pursuer_force_multiplier
            / (distances_p.clamp(min=0.1) ** 2 + 1e-06),
            max=50.0,
        )
        pursuer_force_vec = torch.sum(
            self._normalize_safe(flee_vectors, dim=-1)
            * repulsion_strength_p.unsqueeze(-1),
            dim=1,
        )
        obstacle_repulsion_vec = torch.zeros_like(evader_world_pos)
        lidar_hits = self.evader_lidar.data.ray_hits_w
        if lidar_hits is not None and self.evader_lidar.data.pos_w is not None:
            relative_hits = lidar_hits - self.evader_lidar.data.pos_w.unsqueeze(1)
            distances_lidar = torch.norm(relative_hits, dim=-1)
            hit_dist_from_center = torch.norm(
                (lidar_hits - arena_centers.unsqueeze(1))[:, :, :2], dim=-1
            )
            is_internal_obstacle = hit_dist_from_center < arena_radius - 1.5
            close_points_mask = (
                (distances_lidar < self.evader_obstacle_awareness_distance)
                & torch.isfinite(distances_lidar)
                & is_internal_obstacle
            )
            if close_points_mask.any():
                inf_mask = ~torch.isfinite(relative_hits)
                relative_hits[inf_mask] = 0.0
                ray_directions = self._normalize_safe(relative_hits, dim=-1)
                repulsion_forces = torch.where(
                    close_points_mask, 1.0 / distances_lidar.clamp(min=0.01) ** 2, 0.0
                ).unsqueeze(-1)
                obstacle_repulsion_vec = torch.sum(
                    -ray_directions * repulsion_forces, dim=1
                )
        is_near_wall = dist_to_wall < self.evader_near_boundary_threshold
        effective_force = torch.zeros_like(evader_world_pos)
        normal_mode_mask = ~is_near_wall
        if torch.any(normal_mode_mask):
            force_in_normal_mode = (
                pursuer_force_vec[normal_mode_mask]
                + obstacle_repulsion_vec[normal_mode_mask]
                * self.evader_obstacle_repulsion_weight
            )
            effective_force[normal_mode_mask] = force_in_normal_mode
        if torch.any(is_near_wall):
            wall_mode_mask = is_near_wall
            desired_force = (
                pursuer_force_vec[wall_mode_mask]
                + obstacle_repulsion_vec[wall_mode_mask]
                * self.evader_obstacle_repulsion_weight
            )
            radial_outward_vec = -self._normalize_safe(vec_to_center[wall_mode_mask])
            dot_product = torch.sum(
                desired_force * radial_outward_vec, dim=-1, keepdim=True
            )
            projection_scalar = torch.clamp(dot_product, min=0)
            force_into_wall_component = projection_scalar * radial_outward_vec
            safe_projected_force = desired_force - force_into_wall_component
            safe_direction = self._normalize_safe(safe_projected_force)
            inward_radial_vec = -radial_outward_vec
            inward_bias_force_component = (
                inward_radial_vec * self.evader_inward_bias_strength
            )
            biased_direction = safe_direction + inward_bias_force_component
            effective_force[wall_mode_mask] = self._normalize_safe(biased_direction)
        desired_speed = torch.norm(effective_force, dim=-1, keepdim=True)
        clamped_speed = torch.clamp(desired_speed, max=self.max_evader_speed)
        final_speed = torch.where(
            is_near_wall.unsqueeze(-1), self.max_evader_speed, clamped_speed
        )
        target_vel = self._normalize_safe(effective_force) * final_speed
        target_vel[:, 2] += (
            self.evader_target_altitude - evader_world_pos[:, 2]
        ) * self.evader_altitude_gain
        return target_vel

    def _get_1d_scan_from_2d_image(self, num_bins=18):
        flat_scan = torch.max(self.lidar_buffer, dim=2).values
        bn, h = (flat_scan.shape[0] * flat_scan.shape[1], flat_scan.shape[2])
        pooled_scan = F.adaptive_max_pool1d(
            flat_scan.view(bn, 1, h), output_size=num_bins
        )
        return pooled_scan.view(self.num_envs, self.num_pursuers, num_bins)

    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        num_resets = len(env_ids)
        env_list = env_ids.detach().cpu().tolist()
        all_pos_local = self._get_smart_reset_positions(env_ids)
        env_ids_float = env_ids.float().to(self.device)
        center_offset = (self.num_envs - 1) / 2.0
        y_offsets = (env_ids_float - center_offset) * self.mapy * 2.0
        env_offsets = torch.stack(
            [torch.zeros_like(y_offsets), y_offsets, torch.zeros_like(y_offsets)],
            dim=-1,
        ).unsqueeze(1)
        all_pos_world = all_pos_local + env_offsets
        all_rpy = torch.zeros(num_resets, self.num_drones, 3, device=self.device)
        all_rot = euler_to_quaternion(all_rpy.reshape(-1, 3)).reshape(
            num_resets, self.num_drones, 4
        )
        self.drone.set_world_poses(
            positions=all_pos_world.reshape(-1, 3),
            orientations=all_rot.reshape(-1, 4),
            env_indices=env_ids,
        )
        self.drone.set_velocities(
            torch.zeros(num_resets * self.num_drones, 6, device=self.device),
            env_indices=env_ids,
        )
        for env_idx in env_list:
            if bool(self.step1_collision_ever[env_idx].item()):
                self.collision_history[env_idx] = False
            else:
                keep_history = (
                    bool(self.collision_history[env_idx].item())
                    and int(self.last_collision_step[env_idx].item()) == 1
                )
                if not keep_history:
                    self.collision_history[env_idx] = False
                    self.last_collision_step[env_idx] = -1
        self.evader_history_buffer[env_ids] = 0.0
        self.evader_full_history[env_ids] = 0.0
        self.state_buffer[env_ids] = 0.0
        self.lidar_buffer[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.corner_stuck_counter[env_ids] = 0
        self.last_corner_position[env_ids] = 0.0
        self._prev_corner_state[env_ids] = False
        self._prev_escape_direction[env_ids] = 0
        self.escape_momentum_counter[env_ids] = 0
        self.escape_momentum_direction[env_ids] = 0.0
        self.preferred_escape_direction[env_ids] = 0
        self.direction_consistency_counter[env_ids] = 0
        if hasattr(self, "stats") and self.stats is not None:
            self.stats["return"][env_ids] = 0.0
            self.stats["episode_len"][env_ids] = 0.0
            self.stats["success"][env_ids] = 0.0
            self.stats["pursuer_collision"][env_ids] = 0.0
            self.stats["evader_collision"][env_ids] = 0.0
            self.stats["timeout"][env_ids] = 0.0
            self.stats["escaped"][env_ids] = 0.0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        self.root_state = self.drone.get_state(env_frame=False)
        all_states = self.root_state
        if torch.isnan(all_states).any() or torch.isinf(all_states).any():
            raise FloatingPointError("Invalid drone states before controller")
        all_pos = all_states[..., :3]
        pursuer_target_vels_world = tensordict["agents", "action"]
        pursuer_pos = all_pos[:, self.pursuer_ids]
        evader_pos = all_pos[:, self.evader_id].unsqueeze(1)
        evader_target_vel = self._evader_policy_vec(pursuer_pos, evader_pos)
        if (
            torch.isnan(evader_target_vel).any()
            or torch.isinf(evader_target_vel).any()
            or torch.abs(evader_target_vel).max() > 1000.0
        ):
            raise FloatingPointError("Invalid evader target velocity")
        if evader_target_vel.ndim == 2:
            evader_target_vel = evader_target_vel.unsqueeze(1)
        full_target_vel = torch.cat(
            [pursuer_target_vels_world, evader_target_vel], dim=1
        )
        from omni_drones.utils.torch import quaternion_to_euler

        horizontal_speed = torch.norm(full_target_vel[..., :2], dim=-1)
        current_yaw = quaternion_to_euler(all_states[..., 3:7])[..., -1]
        yaw_from_vel = torch.atan2(full_target_vel[..., 1], full_target_vel[..., 0])
        angle_diff = yaw_from_vel - current_yaw
        angle_diff_wrapped = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        clamped_diff = angle_diff_wrapped.clamp(-0.6, 0.6)
        yaw_from_vel = current_yaw + clamped_diff
        target_yaw = torch.where(
            horizontal_speed > self.yaw_speed_threshold, yaw_from_vel, current_yaw
        )
        final_motor_commands = self.controller(
            all_states[..., :13], target_vel=full_target_vel, target_yaw=target_yaw
        )
        self.drone.apply_action(final_motor_commands)
        self.prev_3d_actions.zero_()
        if ("agents", "actionforstate") in tensordict.keys(include_nested=True):
            self.prev_3d_actions[:, : self.num_pursuers] = tensordict[
                "agents", "actionforstate"
            ]
        else:
            self.prev_3d_actions[:, : self.num_pursuers] = pursuer_target_vels_world
        evader_quat = all_states[:, self.evader_id, 3:7]
        evader_target_vel_body = my_world_to_vec(
            evader_target_vel.squeeze(1), evader_quat
        )
        self.prev_3d_actions[:, self.evader_id] = evader_target_vel_body

    def _post_sim_step(self, tensordict: TensorDictBase):
        for lidar_sensor in self.lidar:
            lidar_sensor.update(self.dt)
        self.evader_lidar.update(self.dt)
        self.root_state = self.drone.get_state(env_frame=False)
        self.drone_state = torch.cat(
            [
                self.root_state[..., 3:7],
                self.root_state[..., 7:10],
                self.root_state[..., 10:13],
                self.prev_3d_actions,
            ],
            dim=-1,
        )

    # Visibility gates the evader-related PSTO channels using LiDAR occlusion checks.
    def _check_evader_visibility(self):
        self.root_state = self.drone.get_state(env_frame=False)
        all_pos = self.root_state[..., :3]
        pursuer_pos = all_pos[:, self.pursuer_ids]
        evader_pos = all_pos[:, self.evader_id]
        rel_vectors = evader_pos.unsqueeze(1) - pursuer_pos
        distances = torch.norm(rel_vectors, dim=-1)
        directions_to_evader = rel_vectors / (distances.unsqueeze(-1) + 1e-06)
        visibility_mask = torch.ones(
            self.num_envs, self.num_pursuers, dtype=torch.bool, device=self.device
        )
        visibility_mask = visibility_mask & (distances < self.max_visibility_distance)
        for env_idx in range(self.num_envs):
            for p_idx in range(self.num_pursuers):
                if not visibility_mask[env_idx, p_idx]:
                    continue
                pursuer_lidar_hits = self.lidar[p_idx].data.ray_hits_w[env_idx]
                pursuer_lidar_origin = self.lidar[p_idx].data.pos_w[env_idx]
                if pursuer_lidar_hits is None:
                    continue
                finite_mask = torch.isfinite(pursuer_lidar_hits).all(dim=-1)
                if not finite_mask.any():
                    continue
                hits_valid = pursuer_lidar_hits[finite_mask]
                ray_vectors = hits_valid - pursuer_lidar_origin.unsqueeze(0)
                ray_distances = torch.norm(ray_vectors, dim=-1)
                dist_mask = torch.isfinite(ray_distances) & (ray_distances > 1e-06)
                if not dist_mask.any():
                    continue
                ray_vectors = ray_vectors[dist_mask]
                ray_distances = ray_distances[dist_mask]
                ray_directions = ray_vectors / ray_distances.unsqueeze(-1)
                target_direction = directions_to_evader[env_idx, p_idx]
                dot_products = (ray_directions * target_direction.unsqueeze(0)).sum(
                    dim=-1
                )
                relevant_mask = dot_products > self.visibility_ray_alignment_threshold
                if relevant_mask.any():
                    relevant_dist = ray_distances[relevant_mask]
                    min_relevant_distance = relevant_dist.min().item()
                    distance_to_evader = distances[env_idx, p_idx].item()
                    if (
                        min_relevant_distance
                        < distance_to_evader - self.visibility_obstacle_clearance
                    ):
                        visibility_mask[env_idx, p_idx] = False
        return visibility_mask

    # Vectorized PSTO heatmap projection: future entities are accumulated in LiDAR image coordinates.
    def _generate_trajectory_heatmap_vectorized(
        self, observer_pos, observer_quat, trajectories, value_signs, gammas
    ):
        device = self.device
        B, P, T_total = observer_pos.shape[:3]
        Vd, Hd = (int(self.lidar_vbeams_down), int(self.lidar_hbeams_down))
        q_conj = quat_conjugate(observer_quat.reshape(-1, 4))
        rel_w = (trajectories - observer_pos).reshape(-1, 3)
        rel_b = quat_rotate(q_conj, rel_w).reshape(B, P, T_total, 3)
        x, y, z = (rel_b[..., 0], rel_b[..., 1], rel_b[..., 2])
        az = torch.atan2(y, x)
        az_deg = az * 180.0 / torch.pi % 360.0
        hyp = torch.sqrt(torch.clamp(x * x + y * y, min=1e-12))
        el = torch.atan2(z, hyp)
        el_deg = el * 180.0 / torch.pi
        hres = float(self.lidar_hres)
        h_beams = int(self.lidar_hbeams)
        v_beams = int(self.lidar_vbeams)
        vmin, vmax = (float(self.lidar_vfov[0]), float(self.lidar_vfov[1]))
        v_step = (vmax - vmin) / (v_beams - 1) if v_beams > 1 else 1.0
        idx_h_native = torch.floor(az_deg / hres).long() % h_beams
        idx_v_native = (
            torch.round((el_deg - vmin) / max(v_step, 1e-06)).long()
            if v_beams > 1
            else torch.zeros_like(idx_h_native)
        )
        idx_v_native = torch.clamp(idx_v_native, 0, v_beams - 1)
        down = int(self.downrate)
        idx_h = torch.clamp(idx_h_native // down, 0, Hd - 1)
        idx_v = torch.clamp(idx_v_native // down, 0, Vd - 1)
        dist = torch.norm(rel_b, dim=-1).clamp_min(1e-06)
        values = value_signs * (10.0 / dist) * gammas
        flat_indices = idx_v * Hd + idx_h
        heatmap_pos = torch.full(
            (B, P, Vd * Hd), -torch.inf, device=device, dtype=torch.float32
        )
        heatmap_neg = torch.full(
            (B, P, Vd * Hd), torch.inf, device=device, dtype=torch.float32
        )
        pos_values_src = torch.where(values > 0, values, -torch.inf)
        neg_values_src = torch.where(values < 0, values, torch.inf)
        heatmap_pos.scatter_reduce_(
            2, flat_indices, pos_values_src, reduce="amax", include_self=False
        )
        heatmap_neg.scatter_reduce_(
            2, flat_indices, neg_values_src, reduce="amin", include_self=False
        )
        heatmap_pos[torch.isinf(heatmap_pos)] = 0
        heatmap_neg[torch.isinf(heatmap_neg)] = 0
        final_heatmap = (heatmap_pos + heatmap_neg).view(B, P, Vd, Hd)
        if torch.isnan(final_heatmap).any():
            raise FloatingPointError("NaN detected in trajectory heatmap")
        return final_heatmap

    # Builds decentralized actor observations and the centralized critic state for MAPPO.
    def _compute_state_and_obs(self):
        pursuer_hits_list = [lidar.data.ray_hits_w for lidar in self.lidar]
        if any((hits is None for hits in pursuer_hits_list)):
            raise RuntimeError("One or more pursuer LiDAR sensors are not ready.")
        pursuer_hits_tensor = torch.stack(pursuer_hits_list, dim=1)
        pursuer_origins_tensor = torch.stack(
            [lidar.data.pos_w for lidar in self.lidar], dim=1
        )
        lidar_vec = pursuer_hits_tensor - pursuer_origins_tensor.unsqueeze(2)
        lidar_dist = torch.norm(lidar_vec, dim=-1).clamp_max(self.lidar_range)
        lidar_flat = self.lidar_range - lidar_dist
        lidar_img = self._apply_lidar_layout(lidar_flat)
        down_lidar_scan = downsample_lidar_scan(
            lidar_img.reshape(-1, 1, self.lidar_vbeams, self.lidar_hbeams),
            kernel_size=(self.downrate, self.downrate),
            stride=(self.downrate, self.downrate),
        ).reshape(
            self.num_envs,
            self.num_pursuers,
            self.lidar_vbeams_down,
            self.lidar_hbeams_down,
        )
        self.lidar_buffer = down_lidar_scan
        self.root_state = self.drone.get_state(env_frame=False)
        pursuer_current_full_state = self.root_state[:, self.pursuer_ids, :]
        pursuer_current_pos = pursuer_current_full_state[..., :3]
        pursuer_current_quat = pursuer_current_full_state[..., 3:7]
        pursuer_current_vel = pursuer_current_full_state[..., 7:10]
        B, P, T_future = (self.num_envs, self.num_pursuers, self.sl_future_len)
        current_evader_pos_true = (
            self.root_state[:, self.evader_id, :3]
            .unsqueeze(1)
            .expand(-1, P, -1)
            .unsqueeze(2)
        )
        actual_evader_pos_for_history = self.root_state[
            :, self.evader_id, :3
        ].unsqueeze(1)
        self.evader_full_history = torch.cat(
            [self.evader_full_history[:, 1:, :], actual_evader_pos_for_history], dim=1
        )
        history_for_heatmap = self.evader_full_history[:, -self.sl_history_len :, :]
        mask_for_heatmap = torch.abs(history_for_heatmap).sum(dim=-1) > 1e-06
        pred_evader_traj = (
            self.policy_lstm(
                torch.cat(
                    [
                        history_for_heatmap,
                        torch.full(
                            (B, self.sl_history_len, 1),
                            self.agent_dt,
                            device=self.device,
                        ),
                    ],
                    dim=-1,
                ),
                mask=mask_for_heatmap,
            )
            .view(B, 1, T_future, 3)
            .expand(-1, P, -1, -1)
        )
        identity_mask = torch.eye(P, device=self.device, dtype=torch.bool)
        teammate_mask = ~identity_mask
        all_vs_all_pos = pursuer_current_pos.unsqueeze(1).expand(-1, P, -1, -1)
        teammate_pos_flat = all_vs_all_pos[teammate_mask.expand(B, -1, -1)]
        teammate_pos = teammate_pos_flat.view(B, P, P - 1, 3)
        all_vs_all_vel = pursuer_current_vel.unsqueeze(1).expand(-1, P, -1, -1)
        teammate_vel_flat = all_vs_all_vel[teammate_mask.expand(B, -1, -1)]
        teammate_vel = teammate_vel_flat.view(B, P, P - 1, 3)
        future_steps_sequence = torch.arange(1, T_future + 1, device=self.device)
        dt = self.agent_dt
        time_offsets = future_steps_sequence * dt
        time_offsets_reshaped = time_offsets.view(1, 1, 1, T_future, 1)
        teammate_traj = (
            teammate_pos.unsqueeze(3)
            + teammate_vel.unsqueeze(3) * time_offsets_reshaped
        )
        trajectories = torch.cat(
            [
                current_evader_pos_true,
                pred_evader_traj,
                teammate_pos.view(B, P, P - 1, 1, 3)
                .expand(-1, -1, -1, 1, -1)
                .flatten(start_dim=2, end_dim=3),
                teammate_traj.flatten(start_dim=2, end_dim=3),
            ],
            dim=2,
        )
        T_total = trajectories.shape[2]
        visibility_mask = self._check_evader_visibility()
        val_evader_pos = (
            torch.full([B, P, 1], 2.0, device=self.device)
            * visibility_mask.unsqueeze(-1).float()
        )
        val_evader_traj = (
            torch.full([B, P, T_future], 1.5, device=self.device)
            * visibility_mask.unsqueeze(-1).float()
        )
        val_teammate_pos = torch.full([B, P, P - 1], -2.0, device=self.device)
        val_teammate_traj = torch.full(
            [B, P, (P - 1) * T_future], -1.5, device=self.device
        )
        value_signs = torch.cat(
            [
                val_evader_pos,
                val_evader_traj,
                val_teammate_pos.flatten(start_dim=2),
                val_teammate_traj,
            ],
            dim=2,
        )
        gamma_evader_pos = torch.ones_like(val_evader_pos)
        gamma_evader_traj = 0.95 ** torch.arange(T_future, device=self.device)
        gamma_evader_traj = gamma_evader_traj.view(1, 1, -1).expand(B, P, -1)
        gamma_teammate_pos = torch.full([B, P, P - 1], 1.0, device=self.device)
        gamma_teammate_traj = 0.9 ** torch.arange(T_future, device=self.device)
        gamma_teammate_traj = (
            gamma_teammate_traj.view(1, 1, 1, -1)
            .expand(B, P, P - 1, -1)
            .flatten(start_dim=2)
        )
        gammas = torch.cat(
            [
                gamma_evader_pos,
                gamma_evader_traj,
                gamma_teammate_pos.flatten(start_dim=2),
                gamma_teammate_traj,
            ],
            dim=2,
        )
        observer_pos = pursuer_current_pos.unsqueeze(2).expand(-1, -1, T_total, -1)
        observer_quat = pursuer_current_quat.unsqueeze(2).expand(-1, -1, T_total, -1)
        individual_heatmaps = self._generate_trajectory_heatmap_vectorized(
            observer_pos, observer_quat, trajectories, value_signs, gammas
        )
        fused_image = torch.stack([down_lidar_scan, individual_heatmaps], dim=2)
        pursuer_world_vel = pursuer_current_full_state[..., 7:10]
        pursuer_quat = pursuer_current_full_state[..., 3:7]
        pursuer_body_vel = my_world_to_vec(pursuer_world_vel, pursuer_quat)
        pursuer_state_no_pos = torch.cat(
            [
                pursuer_current_full_state[..., 3:7],
                pursuer_body_vel,
                pursuer_current_full_state[..., 10:13],
                self.prev_3d_actions[:, : self.num_pursuers],
            ],
            dim=-1,
        )
        self.state_buffer = torch.cat(
            [self.state_buffer[:, :, 1:], pursuer_state_no_pos.unsqueeze(2)], dim=2
        )
        rpos_p_e = (
            self.root_state[:, self.evader_id, :3].unsqueeze(1) - pursuer_current_pos
        )
        rpos_p_e_normalized = self._normalize_safe(rpos_p_e)
        obs = TensorDict(
            {
                "state": pursuer_state_no_pos,
                "state_buffer": self.state_buffer,
                "fused_image": fused_image,
                "direction": rpos_p_e_normalized,
            },
            batch_size=[B, P],
        )
        all_drone_positions_world = self.root_state[..., :3]
        env_indices = torch.arange(B, device=self.device, dtype=torch.float32)
        center_offset = (B - 1) / 2.0
        y_offsets = (env_indices - center_offset) * self.mapy * 2.0
        env_offsets_tensor = torch.zeros_like(all_drone_positions_world)
        env_offsets_tensor[..., 1] = y_offsets.view(-1, 1)
        all_drone_positions_local_flat = (
            all_drone_positions_world - env_offsets_tensor
        ).reshape(B, -1)
        all_drone_other_states = self.root_state[..., 3:13].reshape(B, -1)
        all_prev_actions = self.prev_3d_actions.reshape(B, -1)
        downsampled_scans_flat = self._get_1d_scan_from_2d_image(num_bins=18).reshape(
            B, -1
        )
        global_state_full = torch.cat(
            [
                all_drone_positions_local_flat,
                all_drone_other_states,
                all_prev_actions,
                downsampled_scans_flat,
            ],
            dim=-1,
        )
        history_for_training = self.evader_full_history[:, : self.sl_history_len, :]
        future_for_training = self.evader_full_history[:, self.sl_history_len :, :]
        sl_history_mask = torch.abs(history_for_training).sum(dim=-1) > 1e-6
        info_td = TensorDict(
            {
                "sl_history_input": history_for_training,
                "sl_history_mask": sl_history_mask,
                "sl_future_ground_truth": future_for_training,
                "agent_dt": torch.full((B, 1), self.agent_dt, device=self.device),
                "drone_state": self.drone_state,
            },
            batch_size=[B],
        )
        output_td = TensorDict(
            {
                "agents": TensorDict({"observation": obs}, batch_size=[B, P]),
                "state": global_state_full,
                "info": info_td,
            },
            batch_size=[B],
        )
        return output_td

    # Reward terms correspond to pursuit, coordination, obstacle safety, formation, and terminal events.
    def _compute_reward_and_done(self):
        pursuer_current_pos = self.root_state[:, self.pursuer_ids, :3]
        rpos_p_e = (
            self.root_state[:, self.evader_id, :3].unsqueeze(1) - pursuer_current_pos
        )
        dist_p_e = torch.norm(rpos_p_e, dim=-1)
        dist_p1_p2 = torch.norm(
            pursuer_current_pos[:, 0, :3] - pursuer_current_pos[:, 1, :3], dim=-1
        )
        min_dist_to_evader, _ = torch.min(dist_p_e, dim=1)
        max_dist_to_evader, _ = torch.max(dist_p_e, dim=1)
        failure_by_min_dist = min_dist_to_evader > self.escape_min_distance
        failure_by_max_dist = max_dist_to_evader > self.escape_max_distance
        is_escaped = failure_by_min_dist | failure_by_max_dist
        is_captured = (dist_p_e < self.capture_threshold).any(dim=1)
        reward_capture = torch.where(is_captured, self.capture_bonus, 0.0)
        pursuer_current_vel = self.root_state[:, self.pursuer_ids, 7:10]
        evader_vel = (
            self.root_state[:, self.evader_id, 7:10]
            .unsqueeze(1)
            .expand_as(pursuer_current_vel)
        )
        rel_vel = evader_vel - pursuer_current_vel
        rpos_p_e = (
            self.root_state[:, self.evader_id, :3].unsqueeze(1)
            - self.root_state[:, self.pursuer_ids, :3]
        )
        rpos_p_e_norm = rpos_p_e / (torch.norm(rpos_p_e, dim=-1, keepdim=True) + 1e-06)
        approach_speed = torch.sum(rel_vel * rpos_p_e_norm, dim=-1)
        approach_reward = torch.clamp(-approach_speed, min=self.approach_reward_min)
        reward_pursuit = (
            approach_reward.mean(dim=1) + torch.min(approach_reward, dim=1)[0]
        )
        sigma = torch.tensor(self.coordination_sigma, device=self.device)
        reward_coordination = torch.exp(
            -torch.square(dist_p1_p2 - self.coordination_optimal_distance)
            / (2 * torch.square(sigma))
        )
        pursuer_lidar_flat = self.lidar_buffer.view(self.num_envs, -1)
        pursuer_collision = (
            pursuer_lidar_flat.max(dim=-1)[0]
            > self.lidar_range - self.pursuer_collision_distance
        )
        lidar_proximity = self.lidar_buffer
        actual_distances = self.lidar_range - lidar_proximity
        actual_distances = actual_distances.clamp(min=0.0)
        sensitive_range_tensor = torch.tensor(
            self.obstacle_penalty_sensitive_range,
            dtype=torch.float32,
            device=self.device,
        ).clamp(min=1e-09)
        d_eff_for_scaling = actual_distances.clamp(
            max=self.obstacle_penalty_sensitive_range
        )
        scaled_distance = d_eff_for_scaling / sensitive_range_tensor
        penalized_distance = (
            scaled_distance.clamp(min=1e-06) ** self.obstacle_penalty_log_power
        )
        log_penalties_per_beam = torch.log(penalized_distance)
        B, P, Vd, Hd = log_penalties_per_beam.shape
        flat_penalties = log_penalties_per_beam.view(B, P, -1)
        if torch.isnan(flat_penalties).any():
            raise FloatingPointError("NaN detected in obstacle penalty reward")
        flat_penalties_safe = torch.where(
            torch.isposinf(flat_penalties),
            torch.zeros_like(flat_penalties),
            flat_penalties,
        )
        flat_penalties_safe = torch.where(
            torch.isneginf(flat_penalties_safe),
            torch.full_like(flat_penalties_safe, -100.0),
            flat_penalties_safe,
        )
        reward_per_pursuer = torch.quantile(
            flat_penalties_safe,
            self.obstacle_penalty_quantile,
            dim=-1,
            interpolation="linear",
        )
        reward_per_pursuer_offset = (
            reward_per_pursuer + self.obstacle_penalty_log_offset
        )
        obstacle_penalty = reward_per_pursuer_offset.mean(dim=1)
        reward_formation_penalty = torch.zeros(self.num_envs, device=self.device)
        if self.num_pursuers > 1:
            unit_vecs_to_evader = self._normalize_safe(rpos_p_e)
            closest_pursuer_indices = torch.argmin(dist_p_e, dim=1)
            closest_pursuer_vec = torch.gather(
                unit_vecs_to_evader,
                1,
                closest_pursuer_indices.view(-1, 1, 1).expand(-1, 1, 3),
            ).squeeze(1)
            dot_products = torch.sum(
                closest_pursuer_vec.unsqueeze(1) * unit_vecs_to_evader, dim=-1
            )
            angles_rad = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
            mean_angle_rad = torch.mean(angles_rad, dim=1)
            reward_formation_penalty = mean_angle_rad / math.pi * 2.0 - 0.5
        time_penalty = torch.zeros(self.num_envs, device=self.device)
        time_penalty = torch.where(
            self.progress_buf > self.time_penalty_start_step,
            self.time_penalty_value,
            0.0,
        )
        reward = (
            self.pursuit_weight * (reward_pursuit + self.pursuit_offset)
            + self.coordination_weight * reward_coordination
            + reward_capture
            + self.obstacle_weight * obstacle_penalty
            + reward_formation_penalty * self.formation_weight
            + time_penalty
        )
        evader_collision = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if self.evader_lidar.data.ray_hits_w is not None:
            evader_origins = self.evader_lidar.data.pos_w
            evader_hits = self.evader_lidar.data.ray_hits_w
            evader_distances = torch.norm(
                evader_hits - evader_origins.unsqueeze(1), dim=-1
            )
            evader_collision = (
                evader_distances.min(dim=-1)[0] < self.evader_collision_distance
            )
        reward[pursuer_collision] += self.pursuer_collision_penalty
        reward[evader_collision] += self.evader_collision_bonus
        reward[is_escaped] += self.escape_penalty
        below_bound = (self.root_state[:, :, 2] < self.below_bound_altitude).any(dim=1)
        collision = pursuer_collision | evader_collision
        self._update_collision_history(collision)
        bad_start_mask = (self.progress_buf <= 1) & (collision | below_bound)
        if bad_start_mask.any():
            envs_with_bad_start = torch.where(bad_start_mask)[0]
            self.step1_collision_ever[envs_with_bad_start] = True
            indices_that_caused_conflict = self.last_reset_indices[envs_with_bad_start]
            num_total_positions = self.collision_reset_positions.shape[0]
            next_indices = (indices_that_caused_conflict + 1) % num_total_positions
            self.alt_reset_index[envs_with_bad_start] = next_indices
        terminated = is_captured | collision | below_bound | is_escaped
        is_initial_step = self.progress_buf == 0
        if is_initial_step.any():
            terminated[is_initial_step] = False
        truncated = self.progress_buf >= self.max_episode_length
        self.stats["return"].add_(reward.unsqueeze(1))
        done = terminated | truncated
        finished_env_indices = torch.where(done)[0]
        if len(finished_env_indices) > 0:
            self.stats["success"][finished_env_indices] = 0.0
            self.stats["pursuer_collision"][finished_env_indices] = 0.0
            self.stats["evader_collision"][finished_env_indices] = 0.0
            self.stats["escaped"][finished_env_indices] = 0.0
            self.stats["timeout"][finished_env_indices] = 0.0
            for env_idx in finished_env_indices:
                if is_captured[env_idx]:
                    self.stats["success"][env_idx] = 1.0
                elif pursuer_collision[env_idx] | below_bound[env_idx]:
                    self.stats["pursuer_collision"][env_idx] = 1.0
                elif evader_collision[env_idx]:
                    self.stats["evader_collision"][env_idx] = 1.0
                elif is_escaped[env_idx]:
                    self.stats["escaped"][env_idx] = 1.0
                elif truncated[env_idx]:
                    self.stats["timeout"][env_idx] = 1.0
            for env_idx in finished_env_indices:
                stats_dict = {
                    k: self.stats[k][env_idx].item() for k in self.stats.keys()
                }
                stats_dict["env_id"] = env_idx.item()
                self.finished_stats.append(stats_dict)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)
        reward_expanded = (
            reward.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_pursuers, 1)
        )
        done = (terminated | truncated).unsqueeze(-1)
        reward_expanded = (
            reward.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_pursuers, 1)
        )
        done_expanded = done.unsqueeze(-1)
        terminated_expanded = terminated.unsqueeze(-1)
        truncated_expanded = truncated.unsqueeze(-1)
        return TensorDict(
            {
                "agents": TensorDict(
                    {"reward": reward_expanded},
                    batch_size=[self.num_envs, self.num_pursuers],
                ),
                "done": done_expanded,
                "terminated": terminated_expanded,
                "truncated": truncated_expanded,
                "stats": self.stats,
            },
            batch_size=[self.num_envs],
        )

    def _get_smart_reset_positions(self, env_ids: torch.Tensor):
        num_resets = len(env_ids)
        selected_positions = torch.zeros(
            num_resets, self.num_drones, 3, device=self.device
        )
        num_total_positions = self.collision_reset_positions.shape[0]
        for i, env_idx_t in enumerate(env_ids.detach().cpu().tolist()):
            env_idx = int(env_idx_t)
            chosen_idx = 0
            if bool(self.step1_collision_ever[env_idx].item()):
                chosen_idx = int(self.alt_reset_index[env_idx].item())
            elif (
                bool(self.collision_history[env_idx].item())
                and int(self.last_collision_step[env_idx].item()) == 1
            ):
                chosen_idx = 1
            else:
                chosen_idx = torch.randint(
                    0, num_total_positions, (1,), device=self.device
                ).item()
            selected_positions[i] = self.collision_reset_positions[chosen_idx][
                : self.num_drones, :
            ]
            self.last_reset_indices[env_idx] = chosen_idx
        return selected_positions

    def _update_collision_history(self, collision_mask: torch.Tensor):
        collision_envs = torch.where(collision_mask)[0]
        if len(collision_envs) > 0:
            self.collision_history[collision_envs] = True
            self.last_collision_step[collision_envs] = self.progress_buf[
                collision_envs
            ].long()
            step1_mask = (self.progress_buf <= 3) & collision_mask
            if step1_mask.any():
                self.step1_collision_ever[step1_mask] = True
        else:
            current_step = self.progress_buf.max().item()
            if current_step > 0 and current_step % 500 == 0:
                previous_collision_mask = (
                    self.collision_history
                    & (self.last_collision_step < current_step - 100)
                    & (self.last_collision_step != 1)
                )
                if previous_collision_mask.any():
                    cleared_envs = torch.where(previous_collision_mask)[0]
                    self.collision_history[cleared_envs] = False
                    self.last_collision_step[cleared_envs] = -1

    def _initialize_evader_policy_states(self):
        B, device = (self.num_envs, self.device)
        self.corner_stuck_counter = torch.zeros(B, dtype=torch.long, device=device)
        self.last_corner_position = torch.zeros(B, 3, device=device)
        self.corner_escape_memory = torch.zeros(B, 8, device=device)
        self._prev_corner_state = torch.zeros(B, dtype=torch.bool, device=device)
        self._prev_escape_direction = torch.zeros(B, dtype=torch.long, device=device)
        self.escape_momentum_counter = torch.zeros(B, dtype=torch.long, device=device)
        self.escape_momentum_direction = torch.zeros(B, 3, device=device)
        self.preferred_escape_direction = torch.zeros(
            B, dtype=torch.long, device=device
        )
        self.direction_consistency_counter = torch.zeros(
            B, dtype=torch.long, device=device
        )
