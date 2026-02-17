
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

# --- 1. XML: CURRICULUM ARENA (Physics Preserved) ---
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

        <body name="barrier" pos="0 0 0.5">
            <geom name="geom_barrier" material="mat_wall" size="3.0 0.2 0.8" type="box"/>
        </body>

        <body name="moveable_box" pos="5 0 0.5">
            <joint name="box_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="box_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_box" material="mat_box" size="1.0 1.0 0.5" type="box" mass="2.0"/>
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

# --- 2. ENVIRONMENT (Curriculum Logic) ---
class CurriculumEnv(gym.Env):
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=1080, width=1920)
        
        self.level = 1 
        self.n_rays = 8  
        self.max_sight = 15.0
        
        self.action_space = spaces.MultiDiscrete([5, 5, 5])
        self.obs_dim = 2 + (self.n_rays * 3) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3, self.obs_dim), dtype=np.float32)

        self.id_hider = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_hider")
        self.id_wall = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "wall_top")
        self.id_barrier = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "barrier")
        self.id_box = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box")

    def set_level(self, level):
        self.level = level
        print(f"LEVEL UP! Advanced to Level {level}")

    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        
        # Teleport objects based on Level
        barrier_z = -100.0 if self.level == 1 else 0.5
        box_z = 0.5 if self.level >= 3 else -100.0

        self.model.body_pos[self.id_barrier][2] = barrier_z
        self.model.body_pos[self.id_box][2] = box_z

        # Random Spawns (Preserved)
        self.data.qpos[0] = np.random.uniform(-8, 8); self.data.qpos[1] = np.random.uniform(-8, -2) 
        self.data.qpos[2] = np.random.uniform(-8, 8); self.data.qpos[3] = np.random.uniform(-8, -2) 
        self.data.qpos[4] = np.random.uniform(-8, 8); self.data.qpos[5] = np.random.uniform(2, 8)
        
        if self.level >= 3:
            box_idx = 6 
            self.data.qpos[box_idx] = np.random.uniform(-5, 5)
            self.data.qpos[box_idx+1] = np.random.uniform(-2, 2)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, actions):
        self.data.ctrl[:] = 0 
        for i in range(3):
            base = i * 2
            if actions[i] == 1: self.data.ctrl[base] = 1.0
            elif actions[i] == 2: self.data.ctrl[base] = -1.0
            elif actions[i] == 3: self.data.ctrl[base+1] = 1.0
            elif actions[i] == 4: self.data.ctrl[base+1] = -1.0

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        
        # Extrinsic Rewards (Distance)
        s1_pos = self.data.qpos[0:2]; s2_pos = self.data.qpos[2:4]; h_pos = self.data.qpos[4:6]
        d1 = np.linalg.norm(s1_pos - h_pos); d2 = np.linalg.norm(s2_pos - h_pos)
        min_dist = min(d1, d2)
        
        r_seek = -min_dist * 0.1
        if min_dist < 1.0: r_seek += 1.0
        if min_dist < 0.6: r_seek += 10.0
        
        r_hide = min_dist * 0.1
        if min_dist < 0.6: r_hide -= 10.0
        
        return obs, [r_seek, r_hide], False, False, {}

    def _get_obs(self):
        observations = []
        for i in range(3):
            base_idx = i * 2
            pos = self.data.qpos[base_idx:base_idx+2]
            vel = self.data.qvel[base_idx:base_idx+2]
            
            lidar_data = []
            for r in range(self.n_rays):
                angle = (r / self.n_rays) * 2 * np.pi
                vec = np.array([np.cos(angle), np.sin(angle), 0], dtype=np.float64)
                start_pos = np.array([pos[0], pos[1], 0.5], dtype=np.float64)
                
                geomid = np.zeros(1, dtype=np.int32)
                d = mujoco.mj_ray(self.model, self.data, start_pos, vec, None, 1, -1, geomid)
                
                g = geomid[0]
                is_hider = 0.0; is_wall = 0.0
                if d == -1 or d > self.max_sight: d = self.max_sight
                else:
                    if g == self.id_hider: is_hider = 1.0
                    elif g >= self.id_wall: is_wall = 1.0
                
                d_norm = d / self.max_sight
                lidar_data.extend([d_norm, is_hider, is_wall])
            
            observations.append(np.concatenate([vel, np.array(lidar_data)]))
        return np.stack(observations).astype(np.float32)

    def render(self):
        self.renderer.update_scene(self.data, camera="closeup")
        return self.renderer.render()
    
    def close(self):
        if hasattr(self, 'renderer'): self.renderer.close()

# --- 3. ICM (INTRINSIC CURIOSITY MODULE) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ICM(nn.Module):
    def __init__(self, obs_dim, act_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.forward_model = nn.Sequential(nn.Linear(64 + act_dim, 128), nn.ReLU(), nn.Linear(128, 64))
        self.inverse_model = nn.Sequential(nn.Linear(64 + 64, 128), nn.ReLU(), nn.Linear(128, act_dim))
        
    def get_reward(self, state, next_state, action_idx):
        # One-hot action
        act_onehot = torch.zeros(state.size(0), 5).to(device)
        act_onehot.scatter_(1, action_idx.unsqueeze(1), 1.0)
        
        curr_feat = self.encoder(state)
        next_feat = self.encoder(next_state)
        
        # Predict Next Feature
        pred_next = self.forward_model(torch.cat([curr_feat, act_onehot], dim=1))
        
        # Reward = Prediction Error
        error = (pred_next - next_feat).pow(2).mean(dim=1)
        return error * 10.0 # Scale up so agent notices it

class Agent(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 5), nn.Softmax(dim=-1)
        )
    def get_action(self, x):
        probs = self.net(x)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)

# --- 4. TRAINING LOOP ---
gc.collect()
env = CurriculumEnv("mappo_curriculum.xml")

seeker_agent = Agent(env.obs_dim).to(device)
hider_agent = Agent(env.obs_dim).to(device)
icm = ICM(env.obs_dim).to(device)

opt_s = optim.Adam(seeker_agent.parameters(), lr=0.001)
opt_h = optim.Adam(hider_agent.parameters(), lr=0.001)
opt_icm = optim.Adam(icm.parameters(), lr=0.001)

print(f"🚀 TRAINING: Curriculum + ICM (Lidar 8-Rays)...")

EPISODES = 600
frames = []

for episode in range(EPISODES):
    # Set Level based on episode count
    if episode == 0: env.set_level(1)
    elif episode == 200: env.set_level(2)
    elif episode == 400: env.set_level(3)
    
    obs, _ = env.reset()
    
    log_s, log_h = [], []
    rew_s, rew_h = [], []
    
    # Capture video at end of each level
    record = (episode % 200 == 199) 
    
    for step in range(150):
        if record: frames.append(env.render())
        
        obs_t = torch.tensor(obs).to(device)
        
        a_s1, ls1 = seeker_agent.get_action(obs_t[0])
        a_s2, ls2 = seeker_agent.get_action(obs_t[1])
        a_h,  lh  = hider_agent.get_action(obs_t[2])
        
        acts = [a_s1.item(), a_s2.item(), a_h.item()]
        next_obs, rewards, _, _, _ = env.step(acts)
        next_obs_t = torch.tensor(next_obs).to(device)
        
        # --- ICM REWARD CALCULATION ---
        # Calculate curiosity for each agent
        a_tensors = torch.tensor([a_s1, a_s2, a_h]).to(device)
        int_rew = icm.get_reward(obs_t, next_obs_t, a_tensors)
        
        # Update ICM (Train it to predict better)
        icm_loss = int_rew.mean() # Minimize surprise over time
        opt_icm.zero_grad()
        icm_loss.backward()
        opt_icm.step()
        
        # Add Intrinsic Reward to Extrinsic
        # We give high weight to curiosity
        r_int = int_rew.detach().cpu().numpy()
        
        # Seeker Total = Game Reward + Curiosity
        total_s = rewards[0] + r_int[0] + r_int[1] 
        # Hider Total = Game Reward + Curiosity
        total_h = rewards[1] + r_int[2]
        
        # Store
        log_s.extend([ls1, ls2])
        rew_s.extend([total_s, total_s])
        log_h.append(lh)
        rew_h.append(total_h)
        
        obs = next_obs
        
    # Update Agents
    def update(opt, logs, rews):
        R = 0; returns = []
        for r in reversed(rews):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        loss = []
        for l, r in zip(logs, returns):
            loss.append(-l * r)
        loss = torch.stack(loss).mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    update(opt_s, log_s, rew_s)
    update(opt_h, log_h, rew_h)
    
    if episode % 50 == 0:
        print(f"Ep {episode} | Lvl {env.level} | Reward: {sum(rew_s):.1f}")

env.close()
imageio.mimsave('curriculum_icm_merged.mp4', frames, fps=40)
print("DONE! Video Saved.")
