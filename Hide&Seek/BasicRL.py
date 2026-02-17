import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import imageio
import gc

# DESIGNING CURRICULUM ARENA
def generate_curriculum_xml():
    print("Generating Curriculum Arena (Walls +/- 10, Blue Seekers, Red Hider)...")
    xml_content = f"""
<mujoco model="HideAndSeek_Curriculum">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    <option timestep="0.005" gravity="0 0 -9.81" density="1.2" viscosity="0.0"/>

    <visual>
        <global offwidth="1920" offheight="1080" fovy="45"/>
        <quality shadowsize="4096"/>
        <map fogstart="40" fogend="80"/>
        <rgba fog="0.95 0.96 0.98 1"/>
    </visual>

    <default>
        <joint armature="0" damping="2.0" limited="true"/>
        <geom conaffinity="1" condim="3" density="5.0" friction="0 0 0" margin="0.01"/>
    </default>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.9 0.92 0.95" rgb2="1.0 1.0 1.0" width="512" height="512"/>
        <texture name="tex_floor" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="1024" height="1024" mark="edge" markrgb="0.1 0.1 0.1"/>
        <material name="mat_floor" texture="tex_floor" reflectance="0.8" shininess="1.0" specular="1.0" texrepeat="8 8"/>
        <material name="mat_wall" rgba="0.95 0.95 0.95 1" reflectance="0.1" shininess="0.2"/>
        <material name="mat_seeker" rgba="0.2 0.6 1.0 1" reflectance="0.5" shininess="1.0" specular="1.0"/>
        <material name="mat_hider" rgba="0.9 0.1 0.1 1" reflectance="0.5" shininess="1.0" specular="1.0"/>
        <material name="mat_box" rgba="0.6 0.4 0.2 1" reflectance="0.3" shininess="0.1"/>
    </asset>

    <worldbody>
        <light cutoff="90" diffuse="0.8 0.8 0.85" dir="-0.3 0.3 -1" directional="true" pos="0 0 20" specular="0.6 0.6 0.6" castshadow="true"/>
        <light cutoff="90" diffuse="0.3 0.3 0.35" dir="0.3 -0.3 -1" directional="false" pos="0 0 20" specular="0.2 0.2 0.2"/>

        <camera name="closeup" pos="0 -26 24" xyaxes="1 0 0 0 0.6 0.8"/>
        <geom material="mat_floor" name="floor" pos="0 0 0" size="15 15 0.1" type="plane"/>

        <body name="walls_static" pos="0 0 0">
            <geom name="wall_top" material="mat_wall" pos="0 10.0 0.5" size="10.5 0.5 0.8" type="box"/>
            <geom name="wall_btm" material="mat_wall" pos="0 -10.0 0.5" size="10.5 0.5 0.8" type="box"/>
            <geom name="wall_lft" material="mat_wall" pos="-10.0 0 0.5" size="0.5 10.5 0.8" type="box"/>
            <geom name="wall_rgt" material="mat_wall" pos="10.0 0 0.5" size="0.5 10.5 0.8" type="box"/>
        </body>

        <!-- Barriers (randomly enabled/positioned every 200 episodes) -->
        <body name="barrier0" pos="0 0 0.5">
            <geom name="geom_barrier0" material="mat_wall" size="3.0 0.2 0.8" type="box"/>
        </body>
        <body name="barrier1" pos="0 0 0.5">
            <geom name="geom_barrier1" material="mat_wall" size="3.0 0.2 0.8" type="box"/>
        </body>
        <body name="barrier2" pos="0 0 0.5">
            <geom name="geom_barrier2" material="mat_wall" size="3.0 0.2 0.8" type="box"/>
        </body>
        <body name="barrier3" pos="0 0 0.5">
            <geom name="geom_barrier3" material="mat_wall" size="3.0 0.2 0.8" type="box"/>
        </body>
        <body name="barrier4" pos="0 0 0.5">
            <geom name="geom_barrier4" material="mat_wall" size="3.0 0.2 0.8" type="box"/>
        </body>

        <!-- Multiple boxes for obstacle scaling -->
        <body name="moveable_box0" pos="5 0 0.5">
            <joint name="box0_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="box0_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_box0" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
        </body>
        <body name="moveable_box1" pos="6 2 0.5">
            <joint name="box1_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="box1_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_box1" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
        </body>
        <body name="moveable_box2" pos="-6 -2 0.5">
            <joint name="box2_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="box2_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_box2" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
        </body>
        <body name="moveable_box3" pos="-4 4 0.5">
            <joint name="box3_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="box3_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_box3" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
        </body>
        <body name="moveable_box4" pos="4 -4 0.5">
            <joint name="box4_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="box4_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_box4" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
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
        <body name="hider" pos="0 0 0.5">
            <joint name="h_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="h_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_hider" material="mat_hider" type="sphere" size="0.5" mass="1.0"/>
        </body>
    </worldbody>
    <actuator>
        <motor name="thrust_s1_x" joint="s1_x" gear="20000"/>
        <motor name="thrust_s1_y" joint="s1_y" gear="20000"/>
        <motor name="thrust_s2_x" joint="s2_x" gear="20000"/>
        <motor name="thrust_s2_y" joint="s2_y" gear="20000"/>
        <motor name="thrust_h_x" joint="h_x" gear="20000"/>
        <motor name="thrust_h_y" joint="h_y" gear="20000"/>
    </actuator>
</mujoco>
"""
    return xml_content

with open("mappo_curriculum.xml", "w") as f:
    f.write(generate_curriculum_xml())

# CREATING ENVIRONMENT
class CurriculumEnv(gym.Env):
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=1080, width=1920)

        self.level = 1
        self.episode_idx = 0
        self.n_rays = 8
        self.max_sight = 15.0

        self.action_space = spaces.MultiDiscrete([5, 5, 5])
        self.obs_dim = 2 + (self.n_rays * 3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3, self.obs_dim), dtype=np.float32)

        self.id_hider_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_hider")
        self.id_seeker1_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_seeker1")
        self.id_seeker2_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_seeker2")
        self.id_wall = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "wall_top")

        self.barrier_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier0"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier1"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier2"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier3"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier4"),
        ]

        self.box_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box0"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box1"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box2"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box3"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box4"),
        ]

    def set_level(self, level):
        self.level = level
        print(f"LEVEL UP! Advanced to Level {level}")

    def _seeker_hider_contact(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 == self.id_hider_geom and g2 in [self.id_seeker1_geom, self.id_seeker2_geom]) or \
               (g2 == self.id_hider_geom and g1 in [self.id_seeker1_geom, self.id_seeker2_geom]):
                return True
        return False

    def _hider_visible(self):
        h_pos = self.data.qpos[4:6]
        for seeker_idx in [0, 2]:
            s_pos = self.data.qpos[seeker_idx:seeker_idx+2]
            vec = np.array([h_pos[0] - s_pos[0], h_pos[1] - s_pos[1], 0.0], dtype=np.float64)
            dist = np.linalg.norm(vec[:2])
            if dist < 1e-6:
                return True
            vec = vec / (dist + 1e-9)
            start_pos = np.array([s_pos[0], s_pos[1], 0.5], dtype=np.float64)
            geomid = np.zeros(1, dtype=np.int32)
            d = mujoco.mj_ray(self.model, self.data, start_pos, vec, None, 1, -1, geomid)
            if d > 0 and d < self.max_sight and geomid[0] == self.id_hider_geom:
                return True
        return False

    def set_barriers(self):
        count = min(len(self.barrier_ids), self.episode_idx // 200 + 1)
        for i, body_id in enumerate(self.barrier_ids):
            if i < count:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                self.model.body_pos[body_id] = np.array([x, y, 0.5])
            else:
                self.model.body_pos[body_id] = np.array([0.0, 0.0, -100.0])

    def set_obstacles(self):
        obstacle_count = min(len(self.box_ids), self.episode_idx // 100 + 1)
        for i, body_id in enumerate(self.box_ids):
            if i < obstacle_count:
                self.model.body_pos[body_id][2] = 0.5
            else:
                self.model.body_pos[body_id][2] = -100.0

    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)

        self.set_barriers()
        self.set_obstacles()

        self.data.qpos[0] = np.random.uniform(-8, 8); self.data.qpos[1] = np.random.uniform(-8, -2)
        self.data.qpos[2] = np.random.uniform(-8, 8); self.data.qpos[3] = np.random.uniform(-8, -2)
        self.data.qpos[4] = np.random.uniform(-8, 8); self.data.qpos[5] = np.random.uniform(2, 8)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, actions):
        self.data.ctrl[:] = 0
        for i in range(3):
            base = i * 2
            if actions[i] == 1:
                self.data.ctrl[base] = 1.0
            elif actions[i] == 2:
                self.data.ctrl[base] = -1.0
            elif actions[i] == 3:
                self.data.ctrl[base + 1] = 1.0
            elif actions[i] == 4:
                self.data.ctrl[base + 1] = -1.0

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        s1_pos = self.data.qpos[0:2]; s2_pos = self.data.qpos[2:4]; h_pos = self.data.qpos[4:6]
        d1 = np.linalg.norm(s1_pos - h_pos); d2 = np.linalg.norm(s2_pos - h_pos)
        min_dist = min(d1, d2)

        r_seek = -min_dist * 0.1
        if min_dist < 1.0:
            r_seek += 1.0
        if min_dist < 0.6:
            r_seek += 10.0

        r_hide = min_dist * 0.1
        if min_dist < 0.6:
            r_hide -= 10.0

        contact_1 = (min_dist < 1.0)
        contact_06 = (min_dist < 0.6)
        collision = self._seeker_hider_contact()

        if collision:
            r_seek += 5.0
            r_hide -= 2.0

        if abs(h_pos[0]) > 9.5 or abs(h_pos[1]) > 9.5:
            r_hide -= 5.0

        if not self._hider_visible():
            r_hide += 1.0

        return obs, [r_seek, r_hide], False, False, {
            "collision": collision,
            "contact_1": contact_1,
            "contact_06": contact_06
        }

    def _get_obs(self):
        observations = []
        for i in range(3):
            base_idx = i * 2
            pos = self.data.qpos[base_idx:base_idx + 2]
            vel = self.data.qvel[base_idx:base_idx + 2]

            lidar_data = []
            for r in range(self.n_rays):
                angle = (r / self.n_rays) * 2 * np.pi
                vec = np.array([np.cos(angle), np.sin(angle), 0], dtype=np.float64)
                start_pos = np.array([pos[0], pos[1], 0.5], dtype=np.float64)

                geomid = np.zeros(1, dtype=np.int32)
                d = mujoco.mj_ray(self.model, self.data, start_pos, vec, None, 1, -1, geomid)

                g = geomid[0]
                is_hider = 0.0; is_wall = 0.0
                if d == -1 or d > self.max_sight:
                    d = self.max_sight
                else:
                    if g == self.id_hider_geom:
                        is_hider = 1.0
                    elif g == self.id_wall:
                        is_wall = 1.0

                d_norm = d / self.max_sight
                lidar_data.extend([d_norm, is_hider, is_wall])

            observations.append(np.concatenate([vel, np.array(lidar_data)]))
        return np.stack(observations).astype(np.float32)

    def render(self):
        self.renderer.update_scene(self.data, camera="closeup")
        return self.renderer.render()

    def close(self):
        if hasattr(self, 'renderer'):
            self.renderer.close()

# ICM 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ICM(nn.Module):
    def __init__(self, obs_dim, act_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.forward_model = nn.Sequential(nn.Linear(64 + act_dim, 128), nn.ReLU(), nn.Linear(128, 64))
        self.inverse_model = nn.Sequential(nn.Linear(64 + 64, 128), nn.ReLU(), nn.Linear(128, act_dim))

    def forward(self, state, next_state, action_idx):
        act_onehot = torch.zeros(state.size(0), 5, device=state.device)
        act_onehot.scatter_(1, action_idx.unsqueeze(1), 1.0)

        curr_feat = self.encoder(state)
        next_feat = self.encoder(next_state).detach()

        pred_next = self.forward_model(torch.cat([curr_feat, act_onehot], dim=1))
        forward_loss = (pred_next - next_feat).pow(2).mean(dim=1)

        pred_action = self.inverse_model(torch.cat([curr_feat, next_feat], dim=1))
        inverse_loss = nn.functional.cross_entropy(pred_action, action_idx, reduction="none")

        intrinsic_reward = forward_loss.detach()
        icm_loss = forward_loss.mean() + inverse_loss.mean()

        return intrinsic_reward, icm_loss
# MAPPO MODELLING
class MAPPO(nn.Module):
    def __init__(self, obs_dim, n_agents=3, act_dim=5):
        super().__init__()
        self.actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, act_dim)
            ) for _ in range(n_agents)
        ])
        self.critic = nn.Sequential(
            nn.Linear(obs_dim * n_agents, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_agents)
        )

    def act(self, obs, agent_idx):
        logits = self.actors[agent_idx](obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def value(self, global_obs):
        return self.critic(global_obs)

# --- 5. PPO UTILS ---
def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        next_value = values[t]
    return torch.tensor(advantages, device=device, dtype=torch.float32)

def ppo_update(model, optimizer, obs_batch, global_obs_batch, act_batch, logp_old, adv, returns, agent_idx, clip_eps=0.2):
    logits = model.actors[agent_idx](obs_batch)
    dist = Categorical(logits=logits)
    logp = dist.log_prob(act_batch)
    ratio = torch.exp(logp - logp_old)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    actor_loss = -torch.min(surr1, surr2).mean()

    values = model.value(global_obs_batch)[:, agent_idx]
    critic_loss = nn.functional.mse_loss(values, returns)

    entropy = dist.entropy().mean()
    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

# TRAIN
gc.collect()
env = CurriculumEnv("mappo_curriculum.xml")

model = MAPPO(env.obs_dim).to(device)
icm = ICM(env.obs_dim).to(device)

opt = optim.Adam(model.parameters(), lr=0.0003)
opt_icm = optim.Adam(icm.parameters(), lr=0.001)

EPISODES = 600
frames = []
level = 1

for episode in range(EPISODES):
    env.episode_idx = episode
    obs, _ = env.reset()

    rollout = {
        "obs": [],
        "global_obs": [],
        "acts": [],
        "logp": [],
        "rews": [],
        "dones": [],
        "values": [],
        "collision": False,
        "contact_1": False,
        "contact_06": False
    }

    record = (episode % 200 == 199)

    for step in range(150):
        if record:
            frames.append(env.render())

        obs_t = torch.tensor(obs, device=device, dtype=torch.float32)

        actions = []
        logps = []
        for i in range(3):
            a, lp, _ = model.act(obs_t[i], i)
            actions.append(a.item())
            logps.append(lp)

        next_obs, rewards, _, _, info = env.step(actions)
        if info.get("contact_1"):
            rollout["contact_1"] = True
        if info.get("contact_06"):
            rollout["contact_06"] = True
        if info.get("collision"):
            rollout["collision"] = True

        next_obs_t = torch.tensor(next_obs, device=device, dtype=torch.float32)

        a_tensors = torch.tensor(actions, device=device, dtype=torch.long)
        int_rew, icm_loss = icm(obs_t, next_obs_t, a_tensors)

        opt_icm.zero_grad()
        icm_loss.backward()
        opt_icm.step()

        total_s = rewards[0] + 0.3 * (int_rew[0].item() + int_rew[1].item())
        total_h = rewards[1] + 0.3 * int_rew[2].item()

        global_obs = obs_t.flatten()
        values = model.value(global_obs.unsqueeze(0)).squeeze(0).detach().cpu().numpy()

        rollout["obs"].append(obs_t.cpu().float())
        rollout["global_obs"].append(global_obs.cpu().float())
        rollout["acts"].append(torch.tensor(actions, dtype=torch.long))
        rollout["logp"].append(torch.stack(logps).detach().cpu())
        rollout["rews"].append([total_s, total_s, total_h])
        rollout["dones"].append(0)
        rollout["values"].append(values)

        obs = next_obs

    if rollout["contact_1"] or rollout["contact_06"] or rollout["collision"]:
        level += 1
        env.set_level(level)

    obs_batch = torch.stack(rollout["obs"]).to(device, dtype=torch.float32)
    global_obs_batch = torch.stack(rollout["global_obs"]).to(device, dtype=torch.float32)
    act_batch = torch.stack(rollout["acts"]).to(device, dtype=torch.long)
    logp_old = torch.stack(rollout["logp"]).to(device)
    rewards = torch.tensor(rollout["rews"], device=device, dtype=torch.float32)
    values = torch.tensor(rollout["values"], device=device, dtype=torch.float32)

    for agent_idx in range(3):
        adv = compute_gae(rewards[:, agent_idx], values[:, agent_idx], 0, rollout["dones"])
        returns = adv + values[:, agent_idx]

        ppo_update(
            model, opt,
            obs_batch[:, agent_idx, :],
            global_obs_batch,
            act_batch[:, agent_idx],
            logp_old[:, agent_idx],
            adv,
            returns,
            agent_idx
        )

    if episode % 50 == 0:
        print(f"Ep {episode} | Lvl {env.level} | Reward: {rewards.sum().item():.1f}")

env.close()
imageio.mimsave('curriculum_icm_merged.mp4', frames, fps=40)
print("DONE! Video Saved.")



