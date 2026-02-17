# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# import mujoco
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical
# import imageio
# import gc


# # XML GENERATOR / CURRICULUM

# def generate_curriculum_xml():
#     print("HIDE AND SEEK RL MODEL")
#     xml_content = r"""
# <mujoco model="HideAndSeek_Curriculum">
#     <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
#     <option timestep="0.005" gravity="0 0 -9.81" density="1.2" viscosity="0.0"/>

#     <visual>
#         <global offwidth="1920" offheight="1080" fovy="45"/>
#         <quality shadowsize="4096"/>
#         <map fogstart="40" fogend="80"/>
#         <rgba fog="0.95 0.96 0.98 1"/>
#     </visual>

#     <default>
#         <joint armature="0" damping="2.0" limited="true"/>
#         <geom conaffinity="1" condim="3" density="5.0" friction="0 0 0" margin="0.01"/>
#     </default>

#     <asset>
#         <texture type="skybox" builtin="gradient" rgb1="0.9 0.92 0.95" rgb2="1.0 1.0 1.0" width="512" height="512"/>
#         <texture name="tex_floor" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="1024" height="1024" mark="edge" markrgb="0.1 0.1 0.1"/>
#         <material name="mat_floor" texture="tex_floor" reflectance="0.8" shininess="1.0" specular="1.0" texrepeat="8 8"/>
#         <material name="mat_wall" rgba="0.95 0.95 0.95 1" reflectance="0.1" shininess="0.2"/>
#         <material name="mat_seeker" rgba="0.2 0.6 1.0 1" reflectance="0.5" shininess="1.0" specular="1.0"/>
#         <material name="mat_hider" rgba="0.9 0.1 0.1 1" reflectance="0.5" shininess="1.0" specular="1.0"/>
#         <material name="mat_box" rgba="0.6 0.4 0.2 1" reflectance="0.3" shininess="0.1"/>
#     </asset>

#     <worldbody>
#         <light cutoff="90" diffuse="0.8 0.8 0.85" dir="-0.3 0.3 -1" directional="true" pos="0 0 20" specular="0.6 0.6 0.6" castshadow="true"/>
#         <light cutoff="90" diffuse="0.3 0.3 0.35" dir="0.3 -0.3 -1" directional="false" pos="0 0 20" specular="0.2 0.2 0.2"/>

#         <camera name="closeup" pos="0 -26 24" xyaxes="1 0 0 0 0.6 0.8"/>
#         <geom material="mat_floor" name="floor" pos="0 0 0" size="15 15 0.1" type="plane"/>

#         <body name="walls_static" pos="0 0 0">
#             <geom name="wall_top" material="mat_wall" pos="0 15.0 0.5" size="15.0 0.5 0.8" type="box"/>
#             <geom name="wall_btm" material="mat_wall" pos="0 -15.0 0.5" size="15.0 0.5 0.8" type="box"/>
#             <geom name="wall_lft" material="mat_wall" pos="-15.0 0 0.5" size="0.5 15.0 0.8" type="box"/>
#             <geom name="wall_rgt" material="mat_wall" pos="15.0 0 0.5" size="0.5 15.0 0.8" type="box"/>
#         </body>

#         <!-- Barriers (randomly enabled/positioned every 200 episodes) -->
#         <body name="barrier0" pos="0 0 0.5">
#             <geom name="geom_barrier0" material="mat_wall" size="3.0 0.2 0.8" type="box"/>
#         </body>
#         <body name="barrier1" pos="0 0 0.5">
#             <geom name="geom_barrier1" material="mat_wall" size="3.0 0.2 0.8" type="box"/>
#         </body>
#         <body name="barrier2" pos="0 0 0.5">
#             <geom name="geom_barrier2" material="mat_wall" size="3.0 0.2 0.8" type="box"/>
#         </body>
#         <body name="barrier3" pos="0 0 0.5">
#             <geom name="geom_barrier3" material="mat_wall" size="3.0 0.2 0.8" type="box"/>
#         </body>
#         <body name="barrier4" pos="0 0 0.5">
#             <geom name="geom_barrier4" material="mat_wall" size="3.0 0.2 0.8" type="box"/>
#         </body>

#         <!-- Multiple boxes for obstacle scaling -->
#         <body name="moveable_box0" pos="5 0 0.5">
#             <joint name="box0_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="box0_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_box0" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
#         </body>
#         <body name="moveable_box1" pos="6 2 0.5">
#             <joint name="box1_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="box1_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_box1" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
#         </body>
#         <body name="moveable_box2" pos="-6 -2 0.5">
#             <joint name="box2_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="box2_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_box2" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
#         </body>
#         <body name="moveable_box3" pos="-4 4 0.5">
#             <joint name="box3_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="box3_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_box3" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
#         </body>
#         <body name="moveable_box4" pos="4 -4 0.5">
#             <joint name="box4_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="box4_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_box4" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
#         </body>

#         <body name="seeker1" pos="0 0 0.5">
#             <joint name="s1_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="s1_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_seeker1" material="mat_seeker" type="sphere" size="0.5" mass="1.0"/>
#         </body>
#         <body name="seeker2" pos="0 0 0.5">
#             <joint name="s2_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="s2_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_seeker2" material="mat_seeker" type="sphere" size="0.5" mass="1.0"/>
#         </body>
#         <body name="hider" pos="0 0 0.5">
#             <joint name="h_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="h_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_hider" material="mat_hider" type="sphere" size="0.5" mass="1.0"/>
#         </body>
#     </worldbody>

#     <actuator>
#         <motor name="thrust_s1_x" joint="s1_x" gear="15000"/>
#         <motor name="thrust_s1_y" joint="s1_y" gear="15000"/>
#         <motor name="thrust_s2_x" joint="s2_x" gear="15000"/>
#         <motor name="thrust_s2_y" joint="s2_y" gear="15000"/>
#         <motor name="thrust_h_x" joint="h_x" gear="15000"/>
#         <motor name="thrust_h_y" joint="h_y" gear="15000"/>
#     </actuator>
# </mujoco>
# """
#     return xml_content


# XML_PATH = "mappo_curriculum.xml"
# with open(XML_PATH, "w", encoding="utf-8") as f:
#     f.write(generate_curriculum_xml())


# #  ENVIRONMENT

# class CurriculumEnv(gym.Env):
#     metadata = {"render_modes": ["rgb_array"]}

#     def __init__(self, xml_path):
#         super().__init__()
#         self.model = mujoco.MjModel.from_xml_path(xml_path)
#         self.data = mujoco.MjData(self.model)
#         self.renderer = mujoco.Renderer(self.model, height=1080, width=1920)

#         self.level = 1
#         self.episode_idx = 0
#         self.n_rays = 8
#         self.max_sight = 15.0

#         # Agents: seeker1 (5), seeker2 (5), hider (7)
#         self.action_space = spaces.MultiDiscrete([5, 5, 7])

#         # Observation: [vx, vy] + lidar (n_rays * 3)
#         self.obs_dim = 2 + (self.n_rays * 3)
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(3, self.obs_dim), dtype=np.float32
#         )

#         # IDs
#         self.id_hider_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_hider")
#         self.id_seeker1_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_seeker1")
#         self.id_seeker2_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_seeker2")

#         # NOTE: using only top wall geom id for "wall" classification in lidar (as in your original code)
#         self.id_wall = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "wall_top")

#         self.barrier_ids = [
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier0"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier1"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier2"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier3"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier4"),
#         ]

#         self.box_ids = [
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box0"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box1"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box2"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box3"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box4"),
#         ]

#         # direct hider control over box0 joints (hacky but kept close to your design)
#         self.box0_x_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box0_x")
#         self.box0_y_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box0_y")
#         self.box0_x_dof = int(self.model.jnt_dofadr[self.box0_x_joint])
#         self.box0_y_dof = int(self.model.jnt_dofadr[self.box0_y_joint])

#     def set_level(self, level: int):
#         self.level = int(level)
#         print(f"LEVEL UP! Advanced to Level {self.level}")

#     def _seeker_hider_contact(self):
#         for i in range(self.data.ncon):
#             c = self.data.contact[i]
#             g1, g2 = c.geom1, c.geom2
#             if (g1 == self.id_hider_geom and g2 in [self.id_seeker1_geom, self.id_seeker2_geom]) or \
#                (g2 == self.id_hider_geom and g1 in [self.id_seeker1_geom, self.id_seeker2_geom]):
#                 return True
#         return False

#     def _hider_visible(self):
#         h_pos = self.data.qpos[4:6]
#         for seeker_base in [0, 2]:
#             s_pos = self.data.qpos[seeker_base:seeker_base + 2]
#             vec = np.array([h_pos[0] - s_pos[0], h_pos[1] - s_pos[1], 0.0], dtype=np.float64)
#             dist = np.linalg.norm(vec[:2])
#             if dist < 1e-6:
#                 return True
#             vec = vec / (dist + 1e-9)
#             start_pos = np.array([s_pos[0], s_pos[1], 0.5], dtype=np.float64)
#             geomid = np.zeros(1, dtype=np.int32)
#             d = mujoco.mj_ray(self.model, self.data, start_pos, vec, None, 1, -1, geomid)
#             if d > 0 and d < self.max_sight and int(geomid[0]) == int(self.id_hider_geom):
#                 return True
#         return False

#     def set_barriers(self):
#         # curriculum based on episode index (as in your original code)
#         count = min(len(self.barrier_ids), self.episode_idx // 200 + 1)
#         for i, body_id in enumerate(self.barrier_ids):
#             if i < count:
#                 x = np.random.uniform(-6, 6)
#                 y = np.random.uniform(-6, 6)
#                 self.model.body_pos[body_id] = np.array([x, y, 0.5])
#             else:
#                 # hide far away
#                 self.model.body_pos[body_id] = np.array([0.0, 0.0, -100.0])

#     def set_obstacles(self):
#         obstacle_count = min(len(self.box_ids), self.episode_idx // 100 + 1)
#         for i, body_id in enumerate(self.box_ids):
#             if i < obstacle_count:
#                 self.model.body_pos[body_id][2] = 0.5
#             else:
#                 self.model.body_pos[body_id][2] = -100.0

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         mujoco.mj_resetData(self.model, self.data)

#         self.set_barriers()
#         self.set_obstacles()

#         # Random start positions
#         self.data.qpos[0] = np.random.uniform(-8, 8)
#         self.data.qpos[1] = np.random.uniform(-8, -2)
#         self.data.qpos[2] = np.random.uniform(-8, 8)
#         self.data.qpos[3] = np.random.uniform(-8, -2)
#         self.data.qpos[4] = np.random.uniform(-8, 8)
#         self.data.qpos[5] = np.random.uniform(2, 8)

#         # Important after mutating model positions
#         mujoco.mj_forward(self.model, self.data)

#         return self._get_obs(), {}

#     def step(self, actions):
#         # actions: [a_s1, a_s2, a_h] where a_s* in [0..4], a_h in [0..6]
#         self.data.ctrl[:] = 0.0

#         # seekers: 0=no-op, 1:+x, 2:-x, 3:+y, 4:-y
#         for i in range(2):
#             base_ctrl = i * 2
#             a = int(actions[i])
#             if a == 1:
#                 self.data.ctrl[base_ctrl] = 1.0
#             elif a == 2:
#                 self.data.ctrl[base_ctrl] = -1.0
#             elif a == 3:
#                 self.data.ctrl[base_ctrl + 1] = 1.0
#             elif a == 4:
#                 self.data.ctrl[base_ctrl + 1] = -1.0

#         # hider action: 0=no-op, 1:+x, 2:-x, 3:+y, 4:-y, 5:box0 +x vel, 6:box0 +y vel
#         h_act = int(actions[2])
#         if h_act == 1:
#             self.data.ctrl[4] = 1.0
#         elif h_act == 2:
#             self.data.ctrl[4] = -1.0
#         elif h_act == 3:
#             self.data.ctrl[5] = 1.0
#         elif h_act == 4:
#             self.data.ctrl[5] = -1.0

#         # direct box velocity control (kept from original; can be replaced by actuators later)
#         self.data.qvel[self.box0_x_dof] = 0.0
#         self.data.qvel[self.box0_y_dof] = 0.0
#         if h_act == 5:
#             self.data.qvel[self.box0_x_dof] = 2.5
#         elif h_act == 6:
#             self.data.qvel[self.box0_y_dof] = 2.5

#         # simulate a few substeps
#         for _ in range(5):
#             mujoco.mj_step(self.model, self.data)

#         obs = self._get_obs()

#         s1_pos = self.data.qpos[0:2]
#         s2_pos = self.data.qpos[2:4]
#         h_pos = self.data.qpos[4:6]

#         d1 = float(np.linalg.norm(s1_pos - h_pos))
#         d2 = float(np.linalg.norm(s2_pos - h_pos))
#         min_dist = min(d1, d2)

#         # Individual seeker rewards (you selected 2.B)
#         # Encourage each seeker to get close to hider.
#         r_seek1 = -0.1 * d1
#         r_seek2 = -0.1 * d2
#         if d1 < 1.0:
#             r_seek1 += 1.0
#         if d2 < 1.0:
#             r_seek2 += 1.0
#         if d1 < 0.6:
#             r_seek1 += 10.0
#         if d2 < 0.6:
#             r_seek2 += 10.0

#         # Hider reward: maximize min distance, penalize being too close
#         r_hide = 0.1 * min_dist
#         if min_dist < 0.6:
#             r_hide -= 10.0

#         contact_1 = (min_dist < 1.0)
#         contact_06 = (min_dist < 0.6)
#         collision = self._seeker_hider_contact()

#         if collision:
#             r_seek1 += 5.0
#             r_seek2 += 5.0
#             r_hide -= 2.0

#         # boundary penalty for hider
#         if abs(h_pos[0]) > 9.5 or abs(h_pos[1]) > 9.5:
#             r_hide -= 5.0

#         # hide bonus if not visible
#         if not self._hider_visible():
#             r_hide += 1.0

#         terminated = False
#         truncated = False

#         info = {
#             "collision": collision,
#             "contact_1": contact_1,
#             "contact_06": contact_06,
#             "d1": d1,
#             "d2": d2,
#         }

#         # IMPORTANT: return 3 rewards for 3 agents
#         return obs, [r_seek1, r_seek2, r_hide], terminated, truncated, info

#     def _get_obs(self):
#         observations = []
#         for i in range(3):
#             base_idx = i * 2
#             pos = self.data.qpos[base_idx:base_idx + 2]
#             vel = self.data.qvel[base_idx:base_idx + 2]

#             lidar_data = []
#             for r in range(self.n_rays):
#                 angle = (r / self.n_rays) * 2 * np.pi
#                 vec = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
#                 start_pos = np.array([pos[0], pos[1], 0.5], dtype=np.float64)

#                 geomid = np.zeros(1, dtype=np.int32)
#                 d = mujoco.mj_ray(self.model, self.data, start_pos, vec, None, 1, -1, geomid)

#                 g = int(geomid[0])
#                 is_hider = 0.0
#                 is_wall = 0.0

#                 if d == -1 or d > self.max_sight:
#                     d = self.max_sight
#                 else:
#                     if g == int(self.id_hider_geom):
#                         is_hider = 1.0
#                     elif g == int(self.id_wall):
#                         is_wall = 1.0

#                 d_norm = float(d) / self.max_sight
#                 lidar_data.extend([d_norm, is_hider, is_wall])

#             observations.append(np.concatenate([vel, np.array(lidar_data, dtype=np.float32)]))

#         return np.stack(observations).astype(np.float32)

#     def render(self):
#         self.renderer.update_scene(self.data, camera="closeup")
#         return self.renderer.render()

#     def close(self):
#         if hasattr(self, "renderer"):
#             self.renderer.close()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# #  ICM

# class ICM(nn.Module):
#     def __init__(self, obs_dim, act_dim):
#         super().__init__()
#         self.act_dim = int(act_dim)
#         self.encoder = nn.Sequential(
#             nn.Linear(obs_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#         )
#         self.forward_model = nn.Sequential(
#             nn.Linear(64 + self.act_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#         )
#         self.inverse_model = nn.Sequential(
#             nn.Linear(64 + 64, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.act_dim),
#         )

#     def forward(self, state, next_state, action_idx):
#         """
#         state:      (B, obs_dim)
#         next_state: (B, obs_dim)
#         action_idx: (B,) long
#         """
#         act_onehot = torch.zeros(state.size(0), self.act_dim, device=state.device)
#         act_onehot.scatter_(1, action_idx.unsqueeze(1), 1.0)

#         curr_feat = self.encoder(state)
#         next_feat = self.encoder(next_state).detach()

#         pred_next = self.forward_model(torch.cat([curr_feat, act_onehot], dim=1))
#         forward_loss = (pred_next - next_feat).pow(2).mean(dim=1)  # per-sample

#         pred_action = self.inverse_model(torch.cat([curr_feat, next_feat], dim=1))
#         inverse_loss = nn.functional.cross_entropy(pred_action, action_idx, reduction="none")

#         intrinsic_reward = forward_loss.detach()
#         icm_loss = forward_loss.mean() + inverse_loss.mean()
#         return intrinsic_reward, icm_loss


# # ---------------------------
# # 5) MAPPO MODEL
# # ---------------------------
# class MAPPO(nn.Module):
#     def __init__(self, obs_dim, n_agents=3, act_dims=(5, 5, 7)):
#         super().__init__()
#         self.n_agents = int(n_agents)
#         self.act_dims = tuple(act_dims)

#         self.actors = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(obs_dim, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, self.act_dims[i]),
#             )
#             for i in range(self.n_agents)
#         ])

#         self.critic = nn.Sequential(
#             nn.Linear(obs_dim * n_agents, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, n_agents),
#         )

#     def act(self, obs, agent_idx):
#         logits = self.actors[agent_idx](obs)
#         dist = Categorical(logits=logits)
#         action = dist.sample()
#         return action, dist.log_prob(action), dist.entropy()

#     def value(self, global_obs):
#         return self.critic(global_obs)


# # ---------------------------
# # 6) PPO UTILS (with bootstrap)
# # ---------------------------
# def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
#     """
#     rewards:    (T,) tensor
#     values:     (T,) tensor
#     next_value: scalar tensor
#     dones:      (T,) tensor (0/1)
#     """
#     advantages = []
#     gae = torch.zeros((), device=rewards.device, dtype=torch.float32)

#     for t in reversed(range(len(rewards))):
#         delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
#         gae = delta + gamma * lam * (1.0 - dones[t]) * gae
#         advantages.insert(0, gae)
#         next_value = values[t]

#     return torch.stack(advantages, dim=0)


# def ppo_update(
#     model,
#     optimizer,
#     obs_batch,
#     global_obs_batch,
#     act_batch,
#     logp_old,
#     adv,
#     returns,
#     agent_idx,
#     clip_eps=0.2,
# ):
#     logits = model.actors[agent_idx](obs_batch)
#     dist = Categorical(logits=logits)

#     logp = dist.log_prob(act_batch)
#     ratio = torch.exp(logp - logp_old)

#     surr1 = ratio * adv
#     surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
#     actor_loss = -torch.min(surr1, surr2).mean()

#     values = model.value(global_obs_batch)[:, agent_idx]
#     critic_loss = nn.functional.mse_loss(values, returns)

#     entropy = dist.entropy().mean()
#     loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#     optimizer.step()


# # ---------------------------
# # 7) TRAINING LOOP
# # ---------------------------
# def main():
#     gc.collect()

#     env = CurriculumEnv(XML_PATH)

#     model = MAPPO(env.obs_dim, n_agents=3, act_dims=(5, 5, 7)).to(device)

#     # You selected 1.B: ICM for all agents, but with correct act_dims
#     icm_seek = ICM(env.obs_dim, act_dim=5).to(device)
#     icm_hide = ICM(env.obs_dim, act_dim=7).to(device)

#     opt = optim.Adam(model.parameters(), lr=3e-4)
#     opt_icm_seek = optim.Adam(icm_seek.parameters(), lr=1e-3)
#     opt_icm_hide = optim.Adam(icm_hide.parameters(), lr=1e-3)

#     EPISODES = 3000
#     STEPS_PER_EP = 150

#     frames = []
#     level = 1

#     for episode in range(EPISODES):
#         env.episode_idx = episode
#         obs, _ = env.reset()

#         rollout = {
#             "obs": [],
#             "global_obs": [],
#             "acts": [],
#             "logp": [],
#             "rews": [],
#             "dones": [],
#             "values": [],
#             "collision": False,
#             "contact_1": False,
#             "contact_06": False,
#         }

#         record = (episode % 200 == 199)

#         for _step in range(STEPS_PER_EP):
#             if record:
#                 frames.append(env.render())

#             obs_t = torch.tensor(obs, device=device, dtype=torch.float32)  # (3, obs_dim)

#             actions = []
#             logps = []
#             for i in range(3):
#                 a, lp, _ent = model.act(obs_t[i], i)
#                 actions.append(int(a.item()))
#                 logps.append(lp)

#             next_obs, rewards, terminated, truncated, info = env.step(actions)
#             done = bool(terminated or truncated)

#             if info.get("contact_1"):
#                 rollout["contact_1"] = True
#             if info.get("contact_06"):
#                 rollout["contact_06"] = True
#             if info.get("collision"):
#                 rollout["collision"] = True

#             next_obs_t = torch.tensor(next_obs, device=device, dtype=torch.float32)

#             # --- ICM per agent type ---
#             # seekers (batch=2)
#             seek_state = obs_t[0:2]
#             seek_next = next_obs_t[0:2]
#             seek_actions = torch.tensor(actions[0:2], device=device, dtype=torch.long)
#             int_seek, icm_seek_loss = icm_seek(seek_state, seek_next, seek_actions)

#             opt_icm_seek.zero_grad(set_to_none=True)
#             icm_seek_loss.backward()
#             opt_icm_seek.step()

#             # hider (batch=1)
#             hide_state = obs_t[2:3]
#             hide_next = next_obs_t[2:3]
#             hide_action = torch.tensor([actions[2]], device=device, dtype=torch.long)
#             int_hide, icm_hide_loss = icm_hide(hide_state, hide_next, hide_action)

#             opt_icm_hide.zero_grad(set_to_none=True)
#             icm_hide_loss.backward()
#             opt_icm_hide.step()

#             # combine extrinsic + intrinsic (per-agent now, consistent)
#             # keep your 0.3 scaling idea
#             total_r1 = float(rewards[0]) + 0.3 * float(int_seek[0].item())
#             total_r2 = float(rewards[1]) + 0.3 * float(int_seek[1].item())
#             total_rh = float(rewards[2]) + 0.3 * float(int_hide[0].item())

#             global_obs = obs_t.flatten()  # (3*obs_dim,)
#             values = model.value(global_obs.unsqueeze(0)).squeeze(0).detach().cpu().numpy()

#             rollout["obs"].append(obs_t.detach().cpu())
#             rollout["global_obs"].append(global_obs.detach().cpu())
#             rollout["acts"].append(torch.tensor(actions, dtype=torch.long))
#             rollout["logp"].append(torch.stack(logps).detach().cpu())
#             rollout["rews"].append([total_r1, total_r2, total_rh])
#             rollout["dones"].append(1.0 if done else 0.0)
#             rollout["values"].append(values)

#             obs = next_obs
#             if done:
#                 break

#         # curriculum level-up trigger (kept from your logic)
#         if rollout["contact_1"] or rollout["contact_06"] or rollout["collision"]:
#             level += 1
#             env.set_level(level)

#         # --- Prepare tensors ---
#         obs_batch = torch.stack(rollout["obs"]).to(device, dtype=torch.float32)                # (T, 3, obs_dim)
#         global_obs_batch = torch.stack(rollout["global_obs"]).to(device, dtype=torch.float32)  # (T, 3*obs_dim)
#         act_batch = torch.stack(rollout["acts"]).to(device, dtype=torch.long)                  # (T, 3)
#         logp_old = torch.stack(rollout["logp"]).to(device, dtype=torch.float32)                # (T, 3)
#         rewards_t = torch.tensor(rollout["rews"], device=device, dtype=torch.float32)          # (T, 3)
#         values_t = torch.tensor(rollout["values"], device=device, dtype=torch.float32)         # (T, 3)
#         dones_t = torch.tensor(rollout["dones"], device=device, dtype=torch.float32)           # (T,)

#         # bootstrap next_value from last state if not done
#         with torch.no_grad():
#             last_obs_t = torch.tensor(obs, device=device, dtype=torch.float32)
#             last_global = last_obs_t.flatten().unsqueeze(0)
#             last_values = model.value(last_global).squeeze(0)  # (3,)
#             next_values = last_values * (1.0 - dones_t[-1]) if len(dones_t) > 0 else last_values

#         # PPO update per agent
#         for agent_idx in range(3):
#             adv = compute_gae(
#                 rewards=rewards_t[:, agent_idx],
#                 values=values_t[:, agent_idx],
#                 next_value=next_values[agent_idx],
#                 dones=dones_t,
#             )
#             returns = adv + values_t[:, agent_idx]

#             # (optional) advantage normalization helps stability
#             adv = (adv - adv.mean()) / (adv.std() + 1e-8)

#             ppo_update(
#                 model=model,
#                 optimizer=opt,
#                 obs_batch=obs_batch[:, agent_idx, :],
#                 global_obs_batch=global_obs_batch,
#                 act_batch=act_batch[:, agent_idx],
#                 logp_old=logp_old[:, agent_idx],
#                 adv=adv,
#                 returns=returns,
#                 agent_idx=agent_idx,
#             )

#         if episode % 50 == 0:
#             ep_ret = rewards_t.sum().item() if rewards_t.numel() else 0.0
#             print(f"Ep {episode} | Lvl {env.level} | Return(sum over agents/steps): {ep_ret:.1f}")

#     env.close()

#     if frames:
#         imageio.mimsave("hideandseek.mp4", frames, fps=40)
#         print("DONE! Video Saved: hideandseek.mp4")
#     else:
#         print("DONE! (No frames recorded; set record frequency to capture video.)")


# if __name__ == "__main__":
#     main()
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# import mujoco
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical
# import imageio
# import gc

# # ============================================================
# # 1) XML: CURRICULUM ARENA (UPDATED: velocity actuators + friction)
# # ============================================================
# def generate_curriculum_xml():
#     print("HIDE AND SEEK RL MODEL (regen: velocity actuators, friction fix, robust motion)")
#     xml_content = f"""
# <mujoco model="HideAndSeek_Curriculum">
#     <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
#     <option timestep="0.0025" gravity="0 0 -9.81" density="1.2" viscosity="0.18"/>

#     <visual>
#         <global offwidth="1920" offheight="1080" fovy="45"/>
#         <quality shadowsize="2048"/>
#         <map fogstart="40" fogend="80"/>
#         <rgba fog="0.95 0.96 0.98 1"/>
#     </visual>

#     <default>
#         <joint armature="0" damping="6.0" frictionloss="0.4" limited="true"/>
#         <!-- FIX: nonzero friction so contacts behave normally -->
#         <geom conaffinity="1" condim="3" density="5.0" friction="1.0 0.005 0.0001" margin="0.01"/>
#     </default>

#     <asset>
#         <texture type="skybox" builtin="gradient" rgb1="0.9 0.92 0.95" rgb2="1.0 1.0 1.0" width="256" height="256"/>
#         <texture name="tex_floor" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="512" height="512" mark="edge" markrgb="0.1 0.1 0.1"/>
#         <material name="mat_floor" texture="tex_floor" reflectance="0.8" shininess="1.0" specular="1.0" texrepeat="8 8"/>
#         <material name="mat_wall" rgba="0.95 0.95 0.95 1" reflectance="0.1" shininess="0.2"/>
#         <material name="mat_seeker" rgba="0.2 0.6 1.0 1" reflectance="0.5" shininess="1.0" specular="1.0"/>
#         <material name="mat_hider" rgba="0.9 0.1 0.1 1" reflectance="0.5" shininess="1.0" specular="1.0"/>
#         <material name="mat_box" rgba="0.6 0.4 0.2 1" reflectance="0.3" shininess="0.1"/>
#     </asset>

#     <worldbody>
#         <light cutoff="90" diffuse="0.8 0.8 0.85" dir="-0.3 0.3 -1" directional="true" pos="0 0 20" specular="0.6 0.6 0.6" castshadow="true"/>
#         <light cutoff="90" diffuse="0.3 0.3 0.35" dir="0.3 -0.3 -1" directional="false" pos="0 0 20" specular="0.2 0.2 0.2"/>

#         <camera name="closeup" pos="0 -26 24" xyaxes="1 0 0 0 0.6 0.8"/>
#         <geom material="mat_floor" name="floor" pos="0 0 0" size="15 15 0.1" type="plane"/>

#         <!-- Boundary walls aligned to floor edges -->
        # <body name="walls_static" pos="0 0 0">
        #     <geom name="wall_top" material="mat_wall" pos="0 15.0 0.5" size="15.0 0.5 0.8" type="box"/>
        #     <geom name="wall_btm" material="mat_wall" pos="0 -15.0 0.5" size="15.0 0.5 0.8" type="box"/>
        #     <geom name="wall_lft" material="mat_wall" pos="-15.0 0 0.5" size="0.5 15.0 0.8" type="box"/>
        #     <geom name="wall_rgt" material="mat_wall" pos="15.0 0 0.5" size="0.5 15.0 0.8" type="box"/>
        # </body>

#         <!-- Barriers (random square with a gap set at runtime) -->
#         <body name="barrier0" pos="0 6 0.5">
#             <geom name="geom_barrier0" material="mat_wall" size="6.0 0.2 0.8" type="box"/>
#         </body>
#         <body name="barrier1" pos="0 -6 0.5">
#             <geom name="geom_barrier1" material="mat_wall" size="6.0 0.2 0.8" type="box"/>
#         </body>
#         <body name="barrier2" pos="-6 0 0.5">
#             <geom name="geom_barrier2" material="mat_wall" size="0.2 6.0 0.8" type="box"/>
#         </body>
#         <body name="barrier3" pos="6 4 0.5">
#             <geom name="geom_barrier3" material="mat_wall" size="0.2 3.0 0.8" type="box"/>
#         </body>
#         <body name="barrier4" pos="6 -4 0.5">
#             <geom name="geom_barrier4" material="mat_wall" size="0.2 3.0 0.8" type="box"/>
#         </body>

#         <!-- Multiple boxes for obstacle scaling -->
#         <body name="moveable_box0" pos="5 0 0.5">
#             <joint name="box0_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="box0_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_box0" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
#         </body>
#         <body name="moveable_box1" pos="6 2 0.5">
#             <joint name="box1_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="box1_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_box1" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
#         </body>
#         <body name="moveable_box2" pos="-6 -2 0.5">
#             <joint name="box2_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="box2_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_box2" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
#         </body>
#         <body name="moveable_box3" pos="-4 4 0.5">
#             <joint name="box3_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="box3_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_box3" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
#         </body>
#         <body name="moveable_box4" pos="4 -4 0.5">
#             <joint name="box4_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="box4_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_box4" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
#         </body>

#         <body name="seeker1" pos="0 0 0.5">
#             <joint name="s1_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="s1_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_seeker1" material="mat_seeker" type="sphere" size="0.5" mass="1.0"/>
#         </body>
#         <body name="seeker2" pos="0 0 0.5">
#             <joint name="s2_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="s2_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_seeker2" material="mat_seeker" type="sphere" size="0.5" mass="1.0"/>
#         </body>

#         <body name="hider1" pos="0 0 0.5">
#             <joint name="h1_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="h1_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_hider1" material="mat_hider" type="sphere" size="0.5" mass="1.0"/>
#         </body>
#         <body name="hider2" pos="0 0 0.5">
#             <joint name="h2_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="h2_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_hider2" material="mat_hider" type="sphere" size="0.5" mass="1.0"/>
#         </body>
#         <body name="hider3" pos="0 0 0.5">
#             <joint name="h3_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="h3_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_hider3" material="mat_hider" type="sphere" size="0.5" mass="1.0"/>
#         </body>
#         <body name="hider4" pos="0 0 0.5">
#             <joint name="h4_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
#             <joint name="h4_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
#             <geom name="geom_hider4" material="mat_hider" type="sphere" size="0.5" mass="1.0"/>
#         </body>
#     </worldbody>

#     <!-- FIX: Use velocity actuators for reliable top-down motion on slide joints -->
#     <actuator>
#         <velocity name="thrust_s1_x" joint="s1_x" kv="7"/>
#         <velocity name="thrust_s1_y" joint="s1_y" kv="7"/>
#         <velocity name="thrust_s2_x" joint="s2_x" kv="7"/>
#         <velocity name="thrust_s2_y" joint="s2_y" kv="7"/>

#         <velocity name="thrust_h1_x" joint="h1_x" kv="7"/>
#         <velocity name="thrust_h1_y" joint="h1_y" kv="7"/>
#         <velocity name="thrust_h2_x" joint="h2_x" kv="7"/>
#         <velocity name="thrust_h2_y" joint="h2_y" kv="7"/>
#         <velocity name="thrust_h3_x" joint="h3_x" kv="7"/>
#         <velocity name="thrust_h3_y" joint="h3_y" kv="7"/>
#         <velocity name="thrust_h4_x" joint="h4_x" kv="7"/>
#         <velocity name="thrust_h4_y" joint="h4_y" kv="7"/>
#     </actuator>
# </mujoco>
# """
#     return xml_content


# with open("mappo_curriculum.xml", "w") as f:
#     f.write(generate_curriculum_xml())

# # ============================================================
# # 2) ENVIRONMENT (UPDATED: robust ctrl mapping like base=i*2)
# # ============================================================
# class CurriculumEnv(gym.Env):
#     def __init__(self, xml_path):
#         self.model = mujoco.MjModel.from_xml_path(xml_path)
#         self.data = mujoco.MjData(self.model)
#         self.renderer = mujoco.Renderer(self.model, height=240, width=320)

#         self.n_seekers = 2
#         self.n_hiders = 4
#         self.n_agents = self.n_seekers + self.n_hiders

#         self.level = 1
#         self.episode_idx = 0
#         self.n_rays = 8
#         self.max_sight = 15.0
#         self.debug_printed = False

#         # hiders have 7 actions (5 move + 2 box controls), seekers have 5
#         self.act_dims = [5] * self.n_seekers + [7] * self.n_hiders
#         self.action_space = spaces.MultiDiscrete(self.act_dims)

#         # obs: vel(2) + lidar(n_rays * 4) [dist, is_hider, is_seeker, is_wall]
#         self.obs_dim = 2 + (self.n_rays * 4)
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(self.n_agents, self.obs_dim), dtype=np.float32
#         )

#         # --- IDs for ray/collision classification ---
#         self.id_seeker_geoms = [
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_seeker1"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_seeker2"),
#         ]
#         self.id_hider_geoms = [
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_hider1"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_hider2"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_hider3"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_hider4"),
#         ]
#         self.id_wall_geoms = [
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "wall_top"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "wall_btm"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "wall_lft"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "wall_rgt"),
#         ]

#         self.barrier_body_ids = [
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier0"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier1"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier2"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier3"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier4"),
#         ]
#         self.barrier_geom_ids = [
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_barrier0"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_barrier1"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_barrier2"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_barrier3"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_barrier4"),
#         ]

#         self.box_ids = [
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box0"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box1"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box2"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box3"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box4"),
#         ]
#         self.box0_body_id = self.box_ids[0]
#         self.prev_box0_pos = np.zeros(2, dtype=np.float64)

#         # hider1 controls box0 joints
#         self.box0_x_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box0_x")
#         self.box0_y_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box0_y")
#         self.box0_x_dof = self.model.jnt_dofadr[self.box0_x_joint]
#         self.box0_y_dof = self.model.jnt_dofadr[self.box0_y_joint]

#         # increase damping for box joints (body-level damping)
#         self.model.dof_damping[self.box0_x_dof] += 1.5
#         self.model.dof_damping[self.box0_y_dof] += 1.5

#         # agent qpos/qvel indices (unchanged)
#         self.agent_joint_names = [
#             ("s1_x", "s1_y"),
#             ("s2_x", "s2_y"),
#             ("h1_x", "h1_y"),
#             ("h2_x", "h2_y"),
#             ("h3_x", "h3_y"),
#             ("h4_x", "h4_y"),
#         ]

#         self.agent_qpos_indices = []
#         self.agent_qvel_indices = []
#         for (jx, jy) in self.agent_joint_names:
#             jx_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jx)
#             jy_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jy)

#             self.agent_qpos_indices.append(
#                 (self.model.jnt_qposadr[jx_id], self.model.jnt_qposadr[jy_id])
#             )
#             self.agent_qvel_indices.append(
#                 (self.model.jnt_dofadr[jx_id], self.model.jnt_dofadr[jy_id])
#             )

#         # --- FIX: deterministic ctrl mapping (like the working 3-agent code) ---
#         expected_nu = self.n_agents * 2
#         assert self.model.nu == expected_nu, f"Expected {expected_nu} actuators, got {self.model.nu}"
#         self.agent_ctrl_indices = [(2 * i, 2 * i + 1) for i in range(self.n_agents)]

#         self.prev_min_dists = [None] * self.n_seekers

#     def set_level(self, level):
#         self.level = level

#     def _agent_pos(self, agent_idx):
#         ix, iy = self.agent_qpos_indices[agent_idx]
#         return np.array([self.data.qpos[ix], self.data.qpos[iy]], dtype=np.float64)

#     def _agent_vel(self, agent_idx):
#         ix, iy = self.agent_qvel_indices[agent_idx]
#         return np.array([self.data.qvel[ix], self.data.qvel[iy]], dtype=np.float64)

#     def _seeker_hider_contact(self):
#         for i in range(self.data.ncon):
#             c = self.data.contact[i]
#             g1, g2 = c.geom1, c.geom2
#             if (g1 in self.id_hider_geoms and g2 in self.id_seeker_geoms) or \
#                (g2 in self.id_hider_geoms and g1 in self.id_seeker_geoms):
#                 return True
#         return False

#     def _is_visible(self, src_pos, target_geom_id):
#         vec = np.array([0.0, 0.0, 0.0], dtype=np.float64)
#         vec[:2] = self.data.geom_xpos[target_geom_id][:2] - src_pos[:2]
#         dist = np.linalg.norm(vec[:2])
#         if dist < 1e-6:
#             return True
#         vec = vec / (dist + 1e-9)
#         start_pos = np.array([src_pos[0], src_pos[1], 0.5], dtype=np.float64)
#         geomid = np.zeros(1, dtype=np.int32)
#         d = mujoco.mj_ray(self.model, self.data, start_pos, vec, None, 1, -1, geomid)
#         return d > 0 and d < self.max_sight and geomid[0] == target_geom_id

#     def _hider_visible(self, hider_geom_id):
#         for s_idx in range(self.n_seekers):
#             s_pos = self._agent_pos(s_idx)
#             if self._is_visible(s_pos, hider_geom_id):
#                 return True
#         return False

#     def set_barriers(self):
#         cx = np.random.uniform(-3.0, 3.0)
#         cy = np.random.uniform(-3.0, 3.0)
#         half = np.random.uniform(4.5, 7.0)
#         gap_center = np.random.uniform(-half * 0.4, half * 0.4)
#         gap_half = half * 0.25

#         horiz_len = half
#         vert_len = half
#         right_seg_len = max(0.8, half - gap_half)

#         self.model.geom_size[self.barrier_geom_ids[0]] = np.array([horiz_len, 0.2, 0.8])
#         self.model.geom_size[self.barrier_geom_ids[1]] = np.array([horiz_len, 0.2, 0.8])
#         self.model.body_pos[self.barrier_body_ids[0]] = np.array([cx, cy + half, 0.5])
#         self.model.body_pos[self.barrier_body_ids[1]] = np.array([cx, cy - half, 0.5])

#         self.model.geom_size[self.barrier_geom_ids[2]] = np.array([0.2, vert_len, 0.8])
#         self.model.body_pos[self.barrier_body_ids[2]] = np.array([cx - half, cy, 0.5])

#         self.model.geom_size[self.barrier_geom_ids[3]] = np.array([0.2, right_seg_len, 0.8])
#         self.model.geom_size[self.barrier_geom_ids[4]] = np.array([0.2, right_seg_len, 0.8])
#         self.model.body_pos[self.barrier_body_ids[3]] = np.array(
#             [cx + half, cy + gap_center + (gap_half + right_seg_len), 0.5]
#         )
#         self.model.body_pos[self.barrier_body_ids[4]] = np.array(
#             [cx + half, cy + gap_center - (gap_half + right_seg_len), 0.5]
#         )

#     def set_obstacles(self):
#         obstacle_count = min(len(self.box_ids), self.episode_idx // 100 + 1)
#         for i, body_id in enumerate(self.box_ids):
#             if i < obstacle_count:
#                 self.model.body_pos[body_id][2] = 0.5
#             else:
#                 self.model.body_pos[body_id][2] = -100.0

#     def reset(self, seed=None):
#         mujoco.mj_resetData(self.model, self.data)

#         self.set_barriers()
#         self.set_obstacles()

#         if not self.debug_printed:
#             print("nu (actuators):", self.model.nu)
#             print("agent_ctrl_indices:", self.agent_ctrl_indices)
#             self.debug_printed = True

#         # seekers start in lower half, hiders in upper half
#         for s_idx in range(self.n_seekers):
#             ix, iy = self.agent_qpos_indices[s_idx]
#             self.data.qpos[ix] = np.random.uniform(-8, 8)
#             self.data.qpos[iy] = np.random.uniform(-8, -2)

#         for h_idx in range(self.n_hiders):
#             agent_idx = self.n_seekers + h_idx
#             ix, iy = self.agent_qpos_indices[agent_idx]
#             self.data.qpos[ix] = np.random.uniform(-8, 8)
#             self.data.qpos[iy] = np.random.uniform(2, 8)

#         mujoco.mj_forward(self.model, self.data)
#         self.prev_min_dists = [None] * self.n_seekers
#         self.prev_box0_pos = self.data.xpos[self.box0_body_id][:2].copy()
#         return self._get_obs(), {}

#     def step(self, actions):
#         self.data.ctrl[:] = 0.0

#         # With velocity actuators, ctrl is target velocity.
#         v_cmd = 3.0  # tune for speed

#         # apply movement actions to each agent
#         for agent_idx in range(self.n_agents):
#             cx, cy = self.agent_ctrl_indices[agent_idx]
#             a = int(actions[agent_idx])

#             if a == 1:
#                 self.data.ctrl[cx] = +v_cmd
#             elif a == 2:
#                 self.data.ctrl[cx] = -v_cmd
#             elif a == 3:
#                 self.data.ctrl[cy] = +v_cmd
#             elif a == 4:
#                 self.data.ctrl[cy] = -v_cmd
#             # action 0 = no-op; actions 5/6 reserved for hider1 box control

#         # hider1 controls box0 joints (unchanged behavior)
#         self.data.qvel[self.box0_x_dof] = 0.0
#         self.data.qvel[self.box0_y_dof] = 0.0
#         h1_act = int(actions[self.n_seekers])  # hider1 index
#         if h1_act == 5:
#             self.data.qvel[self.box0_x_dof] = 0.8
#         elif h1_act == 6:
#             self.data.qvel[self.box0_y_dof] = 0.8

#         for _ in range(5):
#             mujoco.mj_step(self.model, self.data)

#         # box movement reward
#         box0_pos = self.data.xpos[self.box0_body_id][:2]
#         box0_delta = np.linalg.norm(box0_pos - self.prev_box0_pos)
#         self.prev_box0_pos = box0_pos.copy()

#         obs = self._get_obs()

#         seeker_positions = [self._agent_pos(i) for i in range(self.n_seekers)]
#         hider_positions = [self._agent_pos(self.n_seekers + i) for i in range(self.n_hiders)]

#         rewards = [0.0] * self.n_agents
#         min_dists = []
#         for s_idx in range(self.n_seekers):
#             dists = [np.linalg.norm(seeker_positions[s_idx] - h_pos) for h_pos in hider_positions]
#             min_dist = min(dists)
#             min_dists.append(min_dist)

#             r_seek = -0.2 * min_dist
#             if min_dist < 1.0:
#                 r_seek += 1.0
#             if min_dist < 0.6:
#                 r_seek += 10.0

#             prev = self.prev_min_dists[s_idx]
#             if prev is not None:
#                 delta = min_dist - prev
#                 if delta > 0:
#                     r_seek -= 2.0 * delta
#                 else:
#                     r_seek += 2.5 * (-delta)

#             if min_dist < 3.0:
#                 r_seek += (3.0 - min_dist) * 1.2

#             barrier_positions = [self.model.body_pos[b][:2] for b in self.barrier_body_ids]
#             cover_dist = min(np.linalg.norm(seeker_positions[s_idx] - b) for b in barrier_positions)
#             cover_reward = max(0.0, (2.5 - cover_dist) / 2.5) * 0.4
#             r_seek += cover_reward

#             rewards[s_idx] = r_seek

#         for h_idx in range(self.n_hiders):
#             h_pos = hider_positions[h_idx]
#             dists = [np.linalg.norm(h_pos - s_pos) for s_pos in seeker_positions]
#             min_dist = min(dists)

#             r_hide = 0.2 * min_dist
#             if min_dist < 0.6:
#                 r_hide -= 10.0
#             if min_dist < 1.0:
#                 r_hide -= 1.0

#             if abs(h_pos[0]) > 9.5 or abs(h_pos[1]) > 9.5:
#                 r_hide -= 5.0

#             if not self._hider_visible(self.id_hider_geoms[h_idx]):
#                 r_hide += 1.0

#             rewards[self.n_seekers + h_idx] = r_hide

#         # box0 exploration reward for hider1
#         rewards[self.n_seekers] += 0.5 * box0_delta

#         contact_1 = any(d < 1.0 for d in min_dists)
#         contact_06 = any(d < 0.6 for d in min_dists)
#         collision = self._seeker_hider_contact()

#         if collision:
#             for s_idx in range(self.n_seekers):
#                 rewards[s_idx] += 5.0
#             for h_idx in range(self.n_hiders):
#                 rewards[self.n_seekers + h_idx] -= 2.0

#         self.prev_min_dists = min_dists

#         terminated = collision
#         truncated = False

#         return obs, rewards, terminated, truncated, {
#             "collision": collision,
#             "contact_1": contact_1,
#             "contact_06": contact_06
#         }

#     def _get_obs(self):
#         observations = []
#         for i in range(self.n_agents):
#             ix, iy = self.agent_qpos_indices[i]
#             vx, vy = self.agent_qvel_indices[i]
#             pos = np.array([self.data.qpos[ix], self.data.qpos[iy]])
#             vel = np.array([self.data.qvel[vx], self.data.qvel[vy]])

#             lidar_data = []
#             for r in range(self.n_rays):
#                 angle = (r / self.n_rays) * 2 * np.pi
#                 vec = np.array([np.cos(angle), np.sin(angle), 0], dtype=np.float64)
#                 start_pos = np.array([pos[0], pos[1], 0.5], dtype=np.float64)

#                 geomid = np.zeros(1, dtype=np.int32)
#                 d = mujoco.mj_ray(self.model, self.data, start_pos, vec, None, 1, -1, geomid)

#                 g = geomid[0]
#                 is_hider = 0.0
#                 is_seeker = 0.0
#                 is_wall = 0.0
#                 if d == -1 or d > self.max_sight:
#                     d = self.max_sight
#                 else:
#                     if g in self.id_hider_geoms:
#                         is_hider = 1.0
#                     elif g in self.id_seeker_geoms:
#                         is_seeker = 1.0
#                     elif g in self.id_wall_geoms:
#                         is_wall = 1.0

#                 d_norm = d / self.max_sight
#                 lidar_data.extend([d_norm, is_hider, is_seeker, is_wall])

#             observations.append(np.concatenate([vel, np.array(lidar_data)]))
#         return np.stack(observations).astype(np.float32)

#     def render(self):
#         self.renderer.update_scene(self.data, camera="closeup")
#         return self.renderer.render()

#     def close(self):
#         if hasattr(self, "renderer"):
#             self.renderer.close()


# # ============================================================
# # 3) ICM
# # ============================================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class ICM(nn.Module):
#     def __init__(self, obs_dim, act_dim=7):
#         super().__init__()
#         self.act_dim = act_dim
#         self.encoder = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64))
#         self.forward_model = nn.Sequential(
#             nn.Linear(64 + act_dim, 128), nn.ReLU(), nn.Linear(128, 64)
#         )
#         self.inverse_model = nn.Sequential(
#             nn.Linear(64 + 64, 128), nn.ReLU(), nn.Linear(128, act_dim)
#         )

#     def forward(self, state, next_state, action_idx):
#         act_onehot = torch.zeros(state.size(0), self.act_dim, device=state.device)
#         act_onehot.scatter_(1, action_idx.unsqueeze(1), 1.0)

#         curr_feat = self.encoder(state)
#         next_feat = self.encoder(next_state).detach()

#         pred_next = self.forward_model(torch.cat([curr_feat, act_onehot], dim=1))
#         forward_loss = (pred_next - next_feat).pow(2).mean(dim=1)

#         pred_action = self.inverse_model(torch.cat([curr_feat, next_feat], dim=1))
#         inverse_loss = nn.functional.cross_entropy(pred_action, action_idx, reduction="none")

#         intrinsic_reward = forward_loss.detach()
#         icm_loss = forward_loss.mean() + inverse_loss.mean()
#         return intrinsic_reward, icm_loss


# # ============================================================
# # 4) MAPPO MODEL
# # ============================================================
# class MAPPO(nn.Module):
#     def __init__(self, obs_dim, n_agents=6, act_dims=(5, 5, 7, 7, 7, 7)):
#         super().__init__()
#         self.actors = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Linear(obs_dim, 128),
#                     nn.ReLU(),
#                     nn.Linear(128, 128),
#                     nn.ReLU(),
#                     nn.Linear(128, act_dims[i]),
#                 )
#                 for i in range(n_agents)
#             ]
#         )
#         self.critic = nn.Sequential(
#             nn.Linear(obs_dim * n_agents, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, n_agents),
#         )

#     def act(self, obs, agent_idx):
#         logits = self.actors[agent_idx](obs)
#         dist = Categorical(logits=logits)
#         action = dist.sample()
#         return action, dist.log_prob(action), dist.entropy()

#     def value(self, global_obs):
#         return self.critic(global_obs)


# # ============================================================
# # 5) PPO UTILS
# # ============================================================
# def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
#     advantages = []
#     gae = 0
#     for t in reversed(range(len(rewards))):
#         delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
#         gae = delta + gamma * lam * (1 - dones[t]) * gae
#         advantages.insert(0, gae)
#         next_value = values[t]
#     return torch.tensor(advantages, device=device, dtype=torch.float32)


# def ppo_update(
#     model,
#     optimizer,
#     obs_batch,
#     global_obs_batch,
#     act_batch,
#     logp_old,
#     adv,
#     returns,
#     agent_idx,
#     clip_eps=0.2,
# ):
#     logits = model.actors[agent_idx](obs_batch)
#     dist = Categorical(logits=logits)
#     logp = dist.log_prob(act_batch)
#     ratio = torch.exp(logp - logp_old)

#     surr1 = ratio * adv
#     surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
#     actor_loss = -torch.min(surr1, surr2).mean()

#     values = model.value(global_obs_batch)[:, agent_idx]
#     critic_loss = nn.functional.mse_loss(values, returns)

#     entropy = dist.entropy().mean()
#     loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

#     optimizer.zero_grad()
#     loss.backward()
#     nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#     optimizer.step()


# # ============================================================
# # 6) TRAINING LOOP
# # ============================================================
# gc.collect()
# env = CurriculumEnv("mappo_curriculum.xml")

# model = MAPPO(env.obs_dim, n_agents=env.n_agents, act_dims=tuple(env.act_dims)).to(device)
# icm = ICM(env.obs_dim, act_dim=7).to(device)

# opt = optim.Adam(model.parameters(), lr=0.0003)
# opt_icm = optim.Adam(icm.parameters(), lr=0.001)

# EPISODES = 300
# frames = []
# icm_weight = 0.05

# int_rew_mean = 0.0
# int_rew_var = 1.0
# int_rew_alpha = 0.99

# for episode in range(EPISODES):
#     env.episode_idx = episode
#     obs, _ = env.reset()

#     rollout = {
#         "obs": [],
#         "global_obs": [],
#         "acts": [],
#         "logp": [],
#         "rews": [],
#         "dones": [],
#         "values": [],
#         "collision": False,
#         "contact_1": False,
#         "contact_06": False,
#     }

#     record = (episode % 200 == 199)

#     for step in range(150):
#         if record:
#             frames.append(env.render())

#         obs_t = torch.tensor(obs, device=device, dtype=torch.float32)

#         actions = []
#         logps = []
#         for i in range(env.n_agents):
#             a, lp, _ = model.act(obs_t[i], i)
#             actions.append(a.item())
#             logps.append(lp)

#         if episode == 0 and step < 3:
#             print("Debug actions:", actions)

#         next_obs, rewards, terminated, truncated, info = env.step(actions)
#         if info.get("contact_1"):
#             rollout["contact_1"] = True
#         if info.get("contact_06"):
#             rollout["contact_06"] = True
#         if info.get("collision"):
#             rollout["collision"] = True

#         next_obs_t = torch.tensor(next_obs, device=device, dtype=torch.float32)

#         a_tensors = torch.tensor(actions, device=device, dtype=torch.long)
#         int_rew, icm_loss = icm(obs_t, next_obs_t, a_tensors)

#         # running mean/std normalization for intrinsic reward
#         batch_mean = float(int_rew.mean().item())
#         batch_var = float(int_rew.var(unbiased=False).item())
#         int_rew_mean = int_rew_alpha * int_rew_mean + (1 - int_rew_alpha) * batch_mean
#         int_rew_var = int_rew_alpha * int_rew_var + (1 - int_rew_alpha) * batch_var
#         int_rew_std = float(np.sqrt(int_rew_var + 1e-8))
#         int_rew_norm = (int_rew - int_rew_mean) / int_rew_std
#         int_rew_norm = torch.clamp(int_rew_norm, -2.0, 2.0)

#         opt_icm.zero_grad()
#         icm_loss.backward()
#         opt_icm.step()

#         total_rewards = []
#         for i in range(env.n_agents):
#             total_rewards.append(rewards[i] + icm_weight * float(int_rew_norm[i].item()))

#         global_obs = obs_t.flatten()
#         values = model.value(global_obs.unsqueeze(0)).squeeze(0).detach().cpu().numpy()

#         rollout["obs"].append(obs_t.cpu().float())
#         rollout["global_obs"].append(global_obs.cpu().float())
#         rollout["acts"].append(torch.tensor(actions, dtype=torch.long))
#         rollout["logp"].append(torch.stack(logps).detach().cpu())
#         rollout["rews"].append(total_rewards)
#         rollout["dones"].append(1 if terminated else 0)
#         rollout["values"].append(values)

#         obs = next_obs

#         if terminated or truncated:
#             break

#     obs_batch = torch.stack(rollout["obs"]).to(device, dtype=torch.float32)
#     global_obs_batch = torch.stack(rollout["global_obs"]).to(device, dtype=torch.float32)
#     act_batch = torch.stack(rollout["acts"]).to(device, dtype=torch.long)
#     logp_old = torch.stack(rollout["logp"]).to(device)
#     rewards_t = torch.tensor(rollout["rews"], device=device, dtype=torch.float32)
#     values_t = torch.tensor(rollout["values"], device=device, dtype=torch.float32)

#     for agent_idx in range(env.n_agents):
#         adv = compute_gae(rewards_t[:, agent_idx], values_t[:, agent_idx], 0, rollout["dones"])
#         returns = adv + values_t[:, agent_idx]

#         ppo_update(
#             model,
#             opt,
#             obs_batch[:, agent_idx, :],
#             global_obs_batch,
#             act_batch[:, agent_idx],
#             logp_old[:, agent_idx],
#             adv,
#             returns,
#             agent_idx,
#         )

#     if episode % 50 == 0:
#         print(f"Ep {episode} | Lvl {env.level} | Reward: {rewards_t.sum().item():.1f}")

# env.close()
# imageio.mimsave("hideandseek.mp4", frames, fps=40)
# print("DONE! Video Saved.")
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import gymnasium.vector as vector
import mujoco
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import imageio
import gc
import warnings

# --- 1. SETUP ---
# os.environ['MUJOCO_GL'] = 'egl'
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. XML GENERATOR (UPDATED: MOTOR ACTUATORS)
# ==============================================================================
def generate_curriculum_xml():
    xml_content = f"""
<mujoco model="HideAndSeek_Curriculum_6Agents">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    <option timestep="0.005" gravity="0 0 -9.81" density="1.2" viscosity="0.0"/>

    <visual>
        <global offwidth="1920" offheight="1080" fovy="45"/>
        <quality shadowsize="2048"/>
        <map fogstart="40" fogend="80"/>
        <rgba fog="0.95 0.96 0.98 1"/>
    </visual>

    <default>
        <joint armature="0" damping="0.5" frictionloss="0.1" limited="true"/>
        <geom conaffinity="1" condim="3" density="5.0" friction="1.0 0.005 0.0001" margin="0.01"/>
    </default>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.9 0.92 0.95" rgb2="1.0 1.0 1.0" width="256" height="256"/>
        <texture name="tex_floor" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="512" height="512" mark="edge" markrgb="0.1 0.1 0.1"/>
        <material name="mat_floor" texture="tex_floor" reflectance="0.8" shininess="1.0" specular="1.0" texrepeat="8 8"/>
        <material name="mat_wall" rgba="0.95 0.95 0.95 1" reflectance="0.1" shininess="0.2"/>
        <material name="mat_seeker" rgba="0.2 0.6 1.0 1" reflectance="0.5" shininess="1.0" specular="1.0"/>
        <material name="mat_hider" rgba="0.9 0.1 0.1 1" reflectance="0.5" shininess="1.0" specular="1.0"/>
        <material name="mat_box" rgba="0.6 0.4 0.2 1" reflectance="0.3" shininess="0.1"/>
    </asset>

    <worldbody>
        <light cutoff="90" diffuse="0.8 0.8 0.85" dir="-0.3 0.3 -1" directional="true" pos="0 0 20" specular="0.6 0.6 0.6" castshadow="true"/>
        <camera name="closeup" pos="0 -26 24" xyaxes="1 0 0 0 0.6 0.8"/>
        <geom material="mat_floor" name="floor" pos="0 0 0" size="15 15 0.1" type="plane"/>

        <body name="walls_static" pos="0 0 0">
            <geom name="wall_top" material="mat_wall" pos="0 15.0 0.5" size="15.0 0.5 0.8" type="box"/>
            <geom name="wall_btm" material="mat_wall" pos="0 -15.0 0.5" size="15.0 0.5 0.8" type="box"/>
            <geom name="wall_lft" material="mat_wall" pos="-15.0 0 0.5" size="0.5 15.0 0.8" type="box"/>
            <geom name="wall_rgt" material="mat_wall" pos="15.0 0 0.5" size="0.5 15.0 0.8" type="box"/>
        </body>
        <body name="barrier0" pos="0 6 0.5">
            <geom name="geom_barrier0" material="mat_wall" size="6.0 0.2 0.8" type="box"/>
        </body>
        <body name="barrier1" pos="0 -6 0.5">
            <geom name="geom_barrier1" material="mat_wall" size="6.0 0.2 0.8" type="box"/>
        </body>
        <body name="moveable_box0" pos="5 0 0.5">
            <joint name="box0_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="box0_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_box0" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
        </body>
        <body name="moveable_box1" pos="-5 0 0.5">
            <joint name="box1_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="box1_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_box1" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
        </body>

        <body name="seeker1" pos="0 0 0.5">
            <joint name="s1_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="s1_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_seeker1" material="mat_seeker" type="sphere" size="0.5" mass="1.0"/>
        </body>
        <body name="seeker2" pos="0 0 0.5">
            <joint name="s2_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="s2_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_seeker2" material="mat_seeker" type="sphere" size="0.5" mass="1.0"/>
        </body>
        <body name="hider1" pos="0 0 0.5">
            <joint name="h1_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="h1_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_hider1" material="mat_hider" type="sphere" size="0.5" mass="1.0"/>
        </body>
        <body name="hider2" pos="0 0 0.5">
            <joint name="h2_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="h2_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_hider2" material="mat_hider" type="sphere" size="0.5" mass="1.0"/>
        </body>
        <body name="hider3" pos="0 0 0.5">
            <joint name="h3_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="h3_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_hider3" material="mat_hider" type="sphere" size="0.5" mass="1.0"/>
        </body>
        <body name="hider4" pos="0 0 0.5">
            <joint name="h4_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="h4_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_hider4" material="mat_hider" type="sphere" size="0.5" mass="1.0"/>
        </body>
    </worldbody>
    <actuator>
        <!-- SWITCHED TO MOTOR CONTROL (FORCE) WITH HIGH GEAR RATIO -->
        <motor name="thrust_s1_x" joint="s1_x" gear="20000"/>
        <motor name="thrust_s1_y" joint="s1_y" gear="20000"/>
        <motor name="thrust_s2_x" joint="s2_x" gear="20000"/>
        <motor name="thrust_s2_y" joint="s2_y" gear="20000"/>
        
        <motor name="thrust_h1_x" joint="h1_x" gear="20000"/>
        <motor name="thrust_h1_y" joint="h1_y" gear="20000"/>
        <motor name="thrust_h2_x" joint="h2_x" gear="20000"/>
        <motor name="thrust_h2_y" joint="h2_y" gear="20000"/>
        <motor name="thrust_h3_x" joint="h3_x" gear="20000"/>
        <motor name="thrust_h3_y" joint="h3_y" gear="20000"/>
        <motor name="thrust_h4_x" joint="h4_x" gear="20000"/>
        <motor name="thrust_h4_y" joint="h4_y" gear="20000"/>
    </actuator>
</mujoco>
"""
    return xml_content

with open("mappo_curriculum.xml", "w") as f:
    f.write(generate_curriculum_xml())

# ==============================================================================
# 3. ENVIRONMENT CLASS
# ==============================================================================
class CurriculumEnv(gym.Env):
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.level = 1
        self.n_seekers = 2
        self.n_hiders = 4
        self.n_agents = self.n_seekers + self.n_hiders
        self.n_rays = 16 
        self.max_sight = 15.0
        self.act_dims = [5] * self.n_seekers + [7] * self.n_hiders
        self.action_space = spaces.MultiDiscrete(self.act_dims)
        self.obs_dim = 2 + (self.n_rays * 4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_agents, self.obs_dim), dtype=np.float32
        )
        self.id_seeker_geoms = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_seeker{i+1}") for i in range(2)]
        self.id_hider_geoms = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_hider{i+1}") for i in range(4)]
        self.id_wall_geoms = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ["wall_top", "wall_btm", "wall_lft", "wall_rgt"]]
        self.box0_x_dof = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box0_x")]
        self.box0_y_dof = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box0_y")]
        self.agent_qpos_indices = []
        self.agent_qvel_indices = []
        names = ["s1", "s2", "h1", "h2", "h3", "h4"]
        for n in names:
            jx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_x")
            jy = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_y")
            self.agent_qpos_indices.append((self.model.jnt_qposadr[jx], self.model.jnt_qposadr[jy]))
            self.agent_qvel_indices.append((self.model.jnt_dofadr[jx], self.model.jnt_dofadr[jy]))
        self.agent_ctrl_indices = [(2 * i, 2 * i + 1) for i in range(self.n_agents)]

    def set_level(self, level):
        self.level = level

    def _get_global_state(self):
        state = []
        for i in range(self.n_agents):
            pix, piy = self.agent_qpos_indices[i]
            vix, viy = self.agent_qvel_indices[i]
            state.extend([self.data.qpos[pix], self.data.qpos[piy], self.data.qvel[vix], self.data.qvel[viy]])
        for b_idx in range(2):
            bx_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"box{b_idx}_x")
            qp_x = self.model.jnt_qposadr[bx_id]
            qv_x = self.model.jnt_dofadr[bx_id]
            state.extend([self.data.qpos[qp_x], self.data.qpos[qp_x+1], self.data.qvel[qv_x], self.data.qvel[qv_x+1]])
        return np.array(state, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        for i in range(self.n_agents):
            pix, piy = self.agent_qpos_indices[i]
            if i < self.n_seekers:
                self.data.qpos[pix] = np.random.uniform(-8, 8)
                self.data.qpos[piy] = np.random.uniform(-8, -2)
            else:
                self.data.qpos[pix] = np.random.uniform(-8, 8)
                self.data.qpos[piy] = np.random.uniform(2, 8)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {'global_state': self._get_global_state()}

    def step(self, actions):
        self.data.ctrl[:] = 0.0
        # For Motor Actuators with Gear=20000, 1.0 is a strong force (20kN).
        # We assume the user wants this strength.
        force_cmd = 1.0 
        
        for i in range(self.n_agents):
            cx, cy = self.agent_ctrl_indices[i]
            a = int(actions[i])
            if a == 1: self.data.ctrl[cx] = force_cmd
            elif a == 2: self.data.ctrl[cx] = -force_cmd
            elif a == 3: self.data.ctrl[cy] = force_cmd
            elif a == 4: self.data.ctrl[cy] = -force_cmd
        
        h1_idx = self.n_seekers
        h1_act = int(actions[h1_idx])
        
        self.data.qvel[self.box0_x_dof] *= 0.95
        self.data.qvel[self.box0_y_dof] *= 0.95
        if h1_act == 5: self.data.qvel[self.box0_x_dof] = 1.0
        elif h1_act == 6: self.data.qvel[self.box0_y_dof] = 1.0
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        s_pos = [np.array([self.data.qpos[self.agent_qpos_indices[i][0]], self.data.qpos[self.agent_qpos_indices[i][1]]]) for i in range(self.n_seekers)]
        h_pos = [np.array([self.data.qpos[self.agent_qpos_indices[i][0]], self.data.qpos[self.agent_qpos_indices[i][1]]]) for i in range(self.n_seekers, self.n_agents)]
        min_hider_dists = []
        for h in range(self.n_hiders):
            dists = [np.linalg.norm(h_pos[h] - s) for s in s_pos]
            min_dist = min(dists)
            min_hider_dists.append(min_dist)
            r_h = min_dist * 0.1
            is_visible = False
            for s_idx in range(self.n_seekers):
                if self._check_vision(s_pos[s_idx], self.id_hider_geoms[h]):
                    is_visible = True
                    break
            if is_visible: r_h -= 1.0
            else: r_h += 1.0
            if min_dist < 0.6: r_h -= 5.0 
            rewards[self.n_seekers + h] = r_h
        r_s = 0.0
        for md in min_hider_dists:
            r_s -= md * 0.1
            if md < 0.6: r_s += 5.0
        for i in range(self.n_seekers):
            rewards[i] = r_s
        info = {'global_state': self._get_global_state()}
        return obs, rewards, False, False, info

    def _check_vision(self, src_pos, target_geom_id):
        start = np.array([src_pos[0], src_pos[1], 0.5])
        target_pos = self.data.geom_xpos[target_geom_id]
        diff = target_pos - start
        dist = np.linalg.norm(diff)
        if dist < 0.1: return True
        vec = diff / (dist + 1e-6)
        geomid = np.zeros(1, dtype=np.int32)
        mujoco.mj_ray(self.model, self.data, start, vec, None, 1, -1, geomid)
        return geomid[0] == target_geom_id

    def _get_obs(self):
        observations = []
        for i in range(self.n_agents):
            pix, piy = self.agent_qpos_indices[i]
            vix, viy = self.agent_qvel_indices[i]
            pos = np.array([self.data.qpos[pix], self.data.qpos[piy]])
            vel = np.array([self.data.qvel[vix], self.data.qvel[viy]])
            lidar_data = []
            for r in range(self.n_rays):
                angle = (r / self.n_rays) * 2 * np.pi
                vec = np.array([np.cos(angle), np.sin(angle), 0], dtype=np.float64)
                start_pos = np.array([pos[0], pos[1], 0.5], dtype=np.float64)
                geomid = np.zeros(1, dtype=np.int32)
                d = mujoco.mj_ray(self.model, self.data, start_pos, vec, None, 1, -1, geomid)
                g = geomid[0]
                is_hider = 1.0 if g in self.id_hider_geoms else 0.0
                is_seeker = 1.0 if g in self.id_seeker_geoms else 0.0
                is_wall = 1.0 if g in self.id_wall_geoms else 0.0
                if d == -1 or d > self.max_sight: d = self.max_sight
                d_norm = d / self.max_sight
                lidar_data.extend([d_norm, is_hider, is_seeker, is_wall])
            observations.append(np.concatenate([vel, np.array(lidar_data)]))
        return np.stack(observations).astype(np.float32)

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.renderer.update_scene(self.data, camera="closeup")
        return self.renderer.render()

    def close(self):
        if self.renderer is not None: self.renderer.close()


# ==============================================================================
# 4. NETWORKS
# ==============================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorGRU(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.actor_fc = nn.Sequential(
            layer_init(nn.Linear(obs_dim, config.fc_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(config.fc_dim, config.fc_dim)),
            nn.ReLU(),
        )
        self.actor_gru = nn.GRU(config.fc_dim, config.hidden_dim, batch_first=True)
        self.actor_head = layer_init(nn.Linear(config.hidden_dim, action_dim), std=0.01)

    def forward(self, obs, hidden):
        if obs.dim() == 2: obs = obs.unsqueeze(1)
        x = self.actor_fc(obs)
        out, new_hidden = self.actor_gru(x, hidden)
        logits = self.actor_head(out)
        return logits, new_hidden

class CentralCriticGRU(nn.Module):
    def __init__(self, global_state_dim, num_agents, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.critic_fc = nn.Sequential(
            layer_init(nn.Linear(global_state_dim, config.fc_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(config.fc_dim, config.fc_dim)),
            nn.ReLU(),
        )
        self.critic_gru = nn.GRU(config.fc_dim, config.hidden_dim, batch_first=True)
        self.critic_head = layer_init(nn.Linear(config.hidden_dim, num_agents), std=1.0)

    def forward(self, global_state, hidden):
        if global_state.dim() == 2: global_state = global_state.unsqueeze(1)
        y = self.critic_fc(global_state)
        out, new_hidden = self.critic_gru(y, hidden)
        vals = self.critic_head(out)
        return vals, new_hidden

class SmallConfig:
    def __init__(self):
        self.hidden_dim = 64
        self.fc_dim = 128


# ==============================================================================
# 5. EXECUTION & TRAINING LOOP
# ==============================================================================
def make_env(xml_path):
    def _env(): return CurriculumEnv(xml_path)
    return _env

def compute_returns_from_rewards(reward_list, gamma=0.99):
    R = np.zeros_like(reward_list[0])
    returns = []
    for r in reversed(reward_list):
        R = r + gamma * R
        returns.insert(0, R)
    return np.array(returns)

if __name__ == "__main__":
    gc.collect()
    xml_path = "mappo_curriculum.xml"
    NUM_ENVS = 8
    
    print(f"🚀 INITIALIZING Parallel Training ({NUM_ENVS} Envs | 6 Agents | Motor Ctrl)...")
    envs = vector.AsyncVectorEnv([make_env(xml_path) for _ in range(NUM_ENVS)])
    eval_env = CurriculumEnv(xml_path)

    obs_dim = eval_env.obs_dim
    global_state_dim = 32
    num_agents = 6
    n_seekers = 2
    n_hiders = 4
    
    cfg = SmallConfig()
    
    seeker_actor = ActorGRU(obs_dim, 5, cfg).to(device)
    hider_actor = ActorGRU(obs_dim, 7, cfg).to(device)
    central_critic = CentralCriticGRU(global_state_dim, num_agents, cfg).to(device)
    
    opt_s = optim.Adam(seeker_actor.parameters(), lr=0.0005)
    opt_h = optim.Adam(hider_actor.parameters(), lr=0.0005)
    opt_critic = optim.Adam(central_critic.parameters(), lr=0.0005)
    
    EPISODES = 300
    BATCH_STEPS = 500
    
    def zeros_hidden(batch_size):
        return torch.zeros(1, batch_size, cfg.hidden_dim, device=device)

    for episode in range(EPISODES):
        
        # --- VIDEO RECORDING BLOCK ---
        if episode % 50 == 0:
            print(f"🎥 RECORDING EPISODE {episode}...")
            obs, info = eval_env.reset()
            rec_hidden_s = torch.zeros(1, n_seekers, cfg.hidden_dim, device=device)
            rec_hidden_h = torch.zeros(1, n_hiders, cfg.hidden_dim, device=device)
            frames = []
            for t in range(400):
                try:
                    frame = eval_env.render()
                    frames.append(frame)
                except: pass
                obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
                s_obs = obs_t[:n_seekers, :].unsqueeze(0).reshape(-1, obs_dim)
                with torch.no_grad():
                    logits_s, rec_hidden_s = seeker_actor(s_obs, rec_hidden_s)
                    a_s = Categorical(logits=logits_s.squeeze(1)).sample().cpu().numpy()
                h_obs = obs_t[n_seekers:, :].unsqueeze(0).reshape(-1, obs_dim)
                with torch.no_grad():
                    logits_h, rec_hidden_h = hider_actor(h_obs, rec_hidden_h)
                    a_h = Categorical(logits=logits_h.squeeze(1)).sample().cpu().numpy()
                actions = np.concatenate([a_s, a_h])
                obs, _, _, _, _ = eval_env.step(actions)
            filename = f'video_ep{episode}.mp4'
            imageio.mimsave(filename, frames, fps=30)
            print(f"💾 Saved {filename}")
            gc.collect()

        # Training Reset
        obs_batch, infos = envs.reset()
        seeker_hidden = zeros_hidden(NUM_ENVS * n_seekers)
        hider_hidden = zeros_hidden(NUM_ENVS * n_hiders)
        critic_hidden = zeros_hidden(NUM_ENVS)
        
        log_probs_s, vals_s, rewards_s = [], [], []
        log_probs_h, vals_h, rewards_h = [], [], []

        for step in range(BATCH_STEPS):
            obs_t = torch.tensor(obs_batch, dtype=torch.float32).to(device)
            
            # 1. Seekers
            s_obs_stack = obs_t[:, :n_seekers, :].reshape(-1, obs_dim)
            logits_s, new_seeker_hidden = seeker_actor(s_obs_stack, seeker_hidden)
            dist_s = Categorical(logits=logits_s.squeeze(1))
            action_s = dist_s.sample()
            lp_s = dist_s.log_prob(action_s)
            
            # 2. Hiders
            h_obs_stack = obs_t[:, n_seekers:, :].reshape(-1, obs_dim)
            logits_h, new_hider_hidden = hider_actor(h_obs_stack, hider_hidden)
            dist_h = Categorical(logits=logits_h.squeeze(1))
            action_h = dist_h.sample()
            lp_h = dist_h.log_prob(action_h)
            
            # Combine
            act_s_np = action_s.cpu().numpy().reshape(NUM_ENVS, n_seekers)
            act_h_np = action_h.cpu().numpy().reshape(NUM_ENVS, n_hiders)
            env_actions = np.concatenate([act_s_np, act_h_np], axis=1).astype(int)
            
            # Critic
            if isinstance(infos, dict) and "global_state" in infos:
                global_st = torch.tensor(np.stack(infos["global_state"]), dtype=torch.float32).to(device)
            else:
                global_st = torch.zeros(NUM_ENVS, global_state_dim, device=device)

            val_all, new_critic_hidden = central_critic(global_st, critic_hidden)
            val_all = val_all.squeeze(1)
            
            # Step
            next_obs_batch, rew_batch, _, _, infos = envs.step(env_actions)
            
            log_probs_s.append(lp_s)
            vals_s.append(val_all[:, :n_seekers].reshape(-1))
            rewards_s.append(rew_batch[:, :n_seekers].reshape(-1))
            
            log_probs_h.append(lp_h)
            vals_h.append(val_all[:, n_seekers:].reshape(-1))
            rewards_h.append(rew_batch[:, n_seekers:].reshape(-1))
            
            obs_batch = next_obs_batch
            seeker_hidden = new_seeker_hidden.detach()
            hider_hidden = new_hider_hidden.detach()
            critic_hidden = new_critic_hidden.detach()
            
        
        # 1. Calculate Returns
        rew_s_stack = np.stack(rewards_s)
        returns_s = compute_returns_from_rewards(rew_s_stack)
        returns_s_t = torch.tensor(returns_s, dtype=torch.float32, device=device)
        
        rew_h_stack = np.stack(rewards_h)
        returns_h = compute_returns_from_rewards(rew_h_stack)
        returns_h_t = torch.tensor(returns_h, dtype=torch.float32, device=device)

       
        vals_s_t = torch.stack(vals_s) 
        vals_h_t = torch.stack(vals_h)
        
        all_vals = torch.cat([vals_s_t.reshape(-1), vals_h_t.reshape(-1)])
        all_returns = torch.cat([returns_s_t.reshape(-1), returns_h_t.reshape(-1)])
        
        loss_critic = nn.MSELoss()(all_vals, all_returns)
        opt_critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        
        vals_s_det = vals_s_t.detach()
        adv_s = returns_s_t - vals_s_det
        adv_s = (adv_s - adv_s.mean()) / (adv_s.std() + 1e-8)
        
        logp_s_t = torch.stack(log_probs_s)
        loss_s = - (logp_s_t * adv_s).mean()
        
        opt_s.zero_grad()
        loss_s.backward()
        opt_s.step()
        
        # Hiders
        vals_h_det = vals_h_t.detach()
        adv_h = returns_h_t - vals_h_det
        adv_h = (adv_h - adv_h.mean()) / (adv_h.std() + 1e-8)
        
        logp_h_t = torch.stack(log_probs_h)
        loss_h = - (logp_h_t * adv_h).mean()
        
        opt_h.zero_grad()
        loss_h.backward()
        opt_h.step()
        
        if episode % 10 == 0:
            print(f" Ep {episode} | S_Rew: {returns_s_t.mean().item():.2f} | H_Rew: {returns_h_t.mean().item():.2f}")
            
    envs.close()
    eval_env.close()
    print("DONE!")