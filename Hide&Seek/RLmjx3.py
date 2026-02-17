import os
import time
import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import mujoco
from mujoco import mjx
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import imageio

# --- CONFIGURATION ---
N_ENVS = 2048          # Massive parallelization
N_STEPS = 128
TOTAL_UPDATES = 1000
LR = 3e-4
SEED = 42

# --- 1. XML GENERATION (FIXED: Added Camera) ---
def generate_curriculum_xml():
    xml_content = """
<mujoco model="HideAndSeek_Curriculum">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    <option timestep="0.005" gravity="0 0 -9.81" density="1.2" viscosity="0.0"/>
    <visual>
        <global offwidth="1920" offheight="1080" fovy="45"/>
        <map fogstart="40" fogend="80"/>
    </visual>
    <default>
        <joint armature="0" damping="2.0" limited="true"/>
        <geom conaffinity="1" condim="3" density="5.0" friction="0.1 0.1 0.1" margin="0.01"/>
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
        
        <!-- FIXED: Added Camera Back -->
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
            <geom name="geom_box" material="mat_box" size="1.0 1.0 0.5" type="box" mass="0.1"/>
        </body>
        <body name="seeker1" pos="0 0 0.5">
            <joint name="s1_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="s1_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_seeker1" material="mat_seeker" type="sphere" size="0.5" mass="0.1"/>
        </body>
        <body name="seeker2" pos="0 0 0.5">
            <joint name="s2_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="s2_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_seeker2" material="mat_seeker" type="sphere" size="0.5" mass="0.1"/>
        </body>
        <body name="hider" pos="0 0 0.5">
            <joint name="h_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="h_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_hider" material="mat_hider" type="sphere" size="0.5" mass="0.1"/>
        </body>
    </worldbody>
    <actuator>
        <motor name="thrust_s1_x" joint="s1_x" gear="15000"/>
        <motor name="thrust_s1_y" joint="s1_y" gear="10000"/>
        <motor name="thrust_s2_x" joint="s2_x" gear="10000"/>
        <motor name="thrust_s2_y" joint="s2_y" gear="10000"/>
        <motor name="thrust_h_x" joint="h_x" gear="10000"/>
        <motor name="thrust_h_y" joint="h_y" gear="10000"/>
    </actuator>
</mujoco>
"""
    return xml_content

with open("mappo_mjx.xml", "w") as f:
    f.write(generate_curriculum_xml())

# --- 2. JAX ENVIRONMENT ---
class MJXEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("mappo_mjx.xml")
        self.mx = mjx.put_model(self.model)
        self.obs_dim = 5 # Optimized for speed (Vel + RelPos + Dist)

    def reset(self, rng):
        rng_qpos, rng_barrier = random.split(rng)
        
        qpos = jnp.array(self.model.qpos0)
        qpos = qpos.at[0:6].set(random.uniform(rng_qpos, (6,), minval=-8, maxval=8))
        qvel = jnp.zeros(self.model.nv)
        
        data = mjx.make_data(self.mx)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mx, data)
        return data, self.get_obs(data)

    def step(self, data, actions):
        def get_ctrl(act_idx):
            x = jnp.where(act_idx == 1, 1.0, jnp.where(act_idx == 2, -1.0, 0.0))
            y = jnp.where(act_idx == 3, 1.0, jnp.where(act_idx == 4, -1.0, 0.0))
            return jnp.array([x, y])

        ctrl_s1 = get_ctrl(actions[0])
        ctrl_s2 = get_ctrl(actions[1])
        ctrl_h  = get_ctrl(actions[2])
        
        ctrl = jnp.concatenate([ctrl_s1, ctrl_s2, ctrl_h])
        data = data.replace(ctrl=ctrl)
        
        # Physics Step
        data = mjx.step(self.mx, data)
        
        obs = self.get_obs(data)
        
        # Extrinsic Rewards
        s1_pos = data.qpos[0:2]; s2_pos = data.qpos[2:4]; h_pos = data.qpos[4:6]
        d1 = jnp.linalg.norm(s1_pos - h_pos)
        d2 = jnp.linalg.norm(s2_pos - h_pos)
        min_dist = jnp.minimum(d1, d2)
        
        r_seek = -min_dist * 0.1 + jnp.where(min_dist < 1.0, 1.0, 0.0) + jnp.where(min_dist < 0.6, 10.0, 0.0)
        r_hide = min_dist * 0.1 + jnp.where(min_dist < 0.6, -10.0, 0.0)
        
        rewards = jnp.array([r_seek, r_seek, r_hide])
        
        return data, obs, rewards

    def get_obs(self, data):
        # Optimized vectorized observation
        pos_all = data.qpos[:6].reshape(3, 2)
        vel_all = data.qvel[:6].reshape(3, 2)
        rel_hider = pos_all[2] - pos_all
        dist_hider = jnp.linalg.norm(rel_hider, axis=1, keepdims=True)
        return jnp.concatenate([vel_all, rel_hider, dist_hider], axis=1)

# --- 3. MODELS (Agent + ICM) ---

# A. The Agent
class ActorCritic(nn.Module):
    act_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.act_dim)(x)
        value = nn.Dense(1)(x)
        return logits, value

# B. The Curiosity Module (ICM) - Ported to Flax
class ICM(nn.Module):
    act_dim: int = 5
    
    @nn.compact
    def __call__(self, obs, next_obs, action_onehot):
        # 1. Encoder
        encoder = nn.Sequential([nn.Dense(64), nn.relu, nn.Dense(64)])
        curr_feat = encoder(obs)
        next_feat = encoder(next_obs)
        
        # 2. Forward Model (Predict next feature)
        pred_input = jnp.concatenate([curr_feat, action_onehot], axis=-1)
        pred_next_feat = nn.Sequential([
            nn.Dense(128), nn.relu, nn.Dense(64)
        ])(pred_input)
        
        # 3. Inverse Model (Predict action)
        inv_input = jnp.concatenate([curr_feat, next_feat], axis=-1)
        pred_action_logits = nn.Sequential([
            nn.Dense(128), nn.relu, nn.Dense(self.act_dim)
        ])(inv_input)
        
        return curr_feat, next_feat, pred_next_feat, pred_action_logits

# --- 4. TRAINING STATE ---
class TrainStateICM(TrainState):
    icm_params: dict
    icm_opt_state: optax.OptState

env = MJXEnv()
obs_template = jnp.zeros((3, env.obs_dim))

# Initialize Models
key = random.PRNGKey(SEED)
key, subkey1, subkey2 = random.split(key, 3)

agent_model = ActorCritic(act_dim=5)
agent_params = agent_model.init(subkey1, obs_template)
tx_agent = optax.adam(LR)

icm_model = ICM(act_dim=5)
icm_params = icm_model.init(subkey2, obs_template, obs_template, jnp.zeros((3, 5)))
tx_icm = optax.adam(LR)

state = TrainStateICM.create(
    apply_fn=agent_model.apply,
    params=agent_params,
    tx=tx_agent,
    icm_params=icm_params,
    icm_opt_state=tx_icm.init(icm_params)
)

# --- 5. FUNCTIONS ---

def env_step_vmap(data, actions):
    return jax.vmap(env.step)(data, actions)

def env_reset_vmap(rng):
    rngs = random.split(rng, N_ENVS)
    return jax.vmap(env.reset)(rngs)

@jax.jit
def get_action_vmap(params, obs, rng):
    def single_env_act(o, r):
        logits, val = agent_model.apply(params, o)
        r_agents = random.split(r, 3)
        actions = jax.vmap(lambda l, r: random.categorical(r, l))(logits, r_agents)
        return actions, logits, val
    rngs = random.split(rng, N_ENVS)
    return jax.vmap(single_env_act)(obs, rngs)

# --- 6. CORE TRAINING STEP (With ICM) ---
@jax.jit
def train_step(state, batch_obs, batch_act, batch_next_obs, batch_rew):
    # Flatten batches
    # (N, 3, Obs) -> (N*3, Obs)
    B_obs = batch_obs.reshape(-1, env.obs_dim)
    B_next_obs = batch_next_obs.reshape(-1, env.obs_dim)
    B_act = batch_act.reshape(-1)
    B_act_onehot = jax.nn.one_hot(B_act, 5)
    B_ext_rew = batch_rew.reshape(-1)

    # --- 1. Update ICM ---
    def icm_loss_fn(p_icm):
        _, next_feat, pred_next, pred_act = icm_model.apply(p_icm, B_obs, B_next_obs, B_act_onehot)
        
        # Forward Loss (Curiosity Error)
        forward_loss = jnp.mean(jnp.square(pred_next - next_feat))
        
        # Inverse Loss (Predict Action)
        inv_loss = optax.softmax_cross_entropy(pred_act, B_act_onehot).mean()
        
        return forward_loss + inv_loss, forward_loss

    (icm_loss, fwd_err), icm_grads = jax.value_and_grad(icm_loss_fn, has_aux=True)(state.icm_params)
    icm_updates, new_icm_opt_state = tx_icm.update(icm_grads, state.icm_opt_state)
    new_icm_params = optax.apply_updates(state.icm_params, icm_updates)
    
    # Calculate Intrinsic Reward (exact logic: error * 10, clamp 5)
    # We re-run apply or just use fwd_err (simplified: re-use fwd_err for batch mean, 
    # but for per-agent reward we need unreduced error. Let's recalculate unreduced).
    _, next_feat, pred_next, _ = icm_model.apply(state.icm_params, B_obs, B_next_obs, B_act_onehot)
    mse_per_sample = jnp.mean(jnp.square(pred_next - next_feat), axis=-1)
    
    intrinsic_reward = mse_per_sample * 10.0
    intrinsic_reward = jnp.clip(intrinsic_reward, 0.0, 5.0)
    
    # Combined Reward
    total_reward = B_ext_rew + intrinsic_reward

    # --- 2. Update Agent ---
    def agent_loss_fn(p_agent):
        logits, values = agent_model.apply(p_agent, B_obs)
        log_probs = jnp.sum(jax.nn.log_softmax(logits) * B_act_onehot, axis=-1)
        
        # Simple Advantage (using returns of total reward)
        # normalizing returns
        returns = (total_reward - total_reward.mean()) / (total_reward.std() + 1e-8)
        
        actor_loss = -jnp.mean(log_probs * returns)
        critic_loss = jnp.mean(jnp.square(returns - values.squeeze()))
        entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1))
        
        return actor_loss + 0.5 * critic_loss - 0.01 * entropy

    agent_grads = jax.grad(agent_loss_fn)(state.params)
    new_state = state.apply_gradients(grads=agent_grads)
    
    # Update State with new ICM params
    new_state = new_state.replace(icm_params=new_icm_params, icm_opt_state=new_icm_opt_state)
    
    return new_state, jnp.mean(intrinsic_reward)

# --- 7. MAIN LOOP ---
print(f"🚀 MJX + ICM Training: {N_ENVS} Envs on GPU")

key, subkey = random.split(key)
env_data, current_obs = env_reset_vmap(subkey)

start_time = time.time()

for update in range(TOTAL_UPDATES):
    # Collect Rollout
    key, subkey = random.split(key)
    actions, _, _ = get_action_vmap(state.params, current_obs, subkey)
    env_data, next_obs, rewards = env_step_vmap(env_data, actions)
    
    state, avg_curiosity = train_step(state, current_obs, actions, next_obs, rewards)
    
    current_obs = next_obs

    if update % 50 == 0:
        elapsed = time.time() - start_time
        avg_rew = jnp.mean(rewards)
        print(f"Update {update} | Reward: {avg_rew:.3f} | Curiosity: {avg_curiosity:.3f} | Time: {elapsed:.1f}s")
        start_time = time.time()

print("Training Complete")

# --- 8. RENDER ---
print("Rendering...")
renderer = mujoco.Renderer(env.model, height=720, width=1280)
mj_data = mujoco.MjData(env.model)
mujoco.mj_resetData(env.model, mj_data)
params = state.params

frames = []
for _ in range(300):
    # CPU Obs
    pos = mj_data.qpos[:6].reshape(3,2)
    vel = mj_data.qvel[:6].reshape(3,2)
    rel = pos[2] - pos
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    obs_cpu = np.concatenate([vel, rel, dist], axis=1)
    
    logits, _ = agent_model.apply(params, obs_cpu)
    acts = np.argmax(logits, axis=-1)
    
    ctrl = np.zeros(env.model.nu)
    for i in range(3):
        if acts[i]==1: ctrl[i*2]=1
        elif acts[i]==2: ctrl[i*2]=-1
        elif acts[i]==3: ctrl[i*2+1]=1
        elif acts[i]==4: ctrl[i*2+1]=-1
    
    mj_data.ctrl[:] = ctrl
    mujoco.mj_step(env.model, mj_data)
    try:
        renderer.update_scene(mj_data, camera="closeup")
    except Exception:
        renderer.update_scene(mj_data) # Fallback to default camera

    frames.append(renderer.render())

# Save with better compression
imageio.mimsave("mjx_icm_gpu.mp4", frames, fps=30, macro_block_size=None)
print("Video Saved.")