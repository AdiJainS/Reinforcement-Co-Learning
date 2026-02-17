
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("Hamara RL\\ball.xml")
data = mujoco.MjData(model)

ball_body_id = model.body("ball").id
ball_jnt_adr = model.body_jntadr[ball_body_id]  # joint index for the body
qpos_adr = model.jnt_qposadr[ball_jnt_adr]       # qpos start index

def resetandom():
    mujoco.mj_resetData(model, data)

    ball_body_id = model.body("ball").id
    ball_jnt_adr = model.body_jntadr[ball_body_id]
    qpos_adr = model.jnt_qposadr[ball_jnt_adr]

    random_pos = np.array([
        np.random.uniform(-0.2, 0.2),
        np.random.uniform(-0.2, 0.2),
        0.05
    ])

    data.qpos[qpos_adr:qpos_adr+3] = random_pos
    data.qpos[qpos_adr+3:qpos_adr+7] = np.array([1, 0, 0, 0])

    mujoco.mj_forward(model, data)

resetandom()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.xfrc_applied[ball_body_id][:3] = [5.0, 0.0, 0.0]
        mujoco.mj_step(model, data) 
        viewer.sync()

import mujoco
import mujoco.viewer
import numpy as np
import time


model = mujoco.MjModel.from_xml_path("ball.xml")
data = mujoco.MjData(model)

jx_id = model.joint("jx").id
jy_id = model.joint("jy").id
jz_id = model.joint("jz").id

jx_qpos = model.jnt_qposadr[jx_id]
jy_qpos = model.jnt_qposadr[jy_id]
jz_qpos = model.jnt_qposadr[jz_id]


box_body_id = model.body("fixed_box").id
box_jnt_adr = model.body_jntadr[box_body_id]
box_qpos_adr = model.jnt_qposadr[box_jnt_adr]

def ball_box():
    mujoco.mj_resetData(model, data)

  
    data.qpos[jx_qpos] = np.random.uniform(-1.5, 1.5)
    data.qpos[jy_qpos] = np.random.uniform(-1.5, 1.5)
    data.qpos[jz_qpos] = np.random.uniform(0.2, 1.0)

    box_pos = np.array([
        np.random.uniform(-1.5, 1.5),
        np.random.uniform(-1.5, 1.5),
        0.1
    ])
    data.qpos[box_qpos_adr:box_qpos_adr+3] = box_pos
    data.qpos[box_qpos_adr+3:box_qpos_adr+7] = np.array([1, 0, 0, 0])

    mujoco.mj_forward(model, data)

ball_box()

jx_act_id = 0
jz_act_id = 1

t0 = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = time.time() - t0

        data.ctrl[jx_act_id] = 1.0 
        data.ctrl[jz_act_id] = 0.8  # speed of ball

        mujoco.mj_step(model, data)
        viewer.sync()
# import mujoco
# import mujoco.viewer
# import numpy as np
# import time

# model = mujoco.MjModel.from_xml_path("bots.xml")
# data = mujoco.MjData(model)

# # Joint IDs for hider
# hx_id = model.joint("slide_x").id
# hy_id = model.joint("slide_y").id
# hz_id = model.joint("rot").id

# # Joint IDs for seeker
# sx_id = model.joint("seeker_slide_x").id
# sy_id = model.joint("seeker_slide_y").id
# sz_id = model.joint("seeker_rot").id

# # Qpos addresses
# hx_qpos = model.jnt_qposadr[hx_id]
# hy_qpos = model.jnt_qposadr[hy_id]
# hz_qpos = model.jnt_qposadr[hz_id]

# sx_qpos = model.jnt_qposadr[sx_id]
# sy_qpos = model.jnt_qposadr[sy_id]
# sz_qpos = model.jnt_qposadr[sz_id]

# # Actuator IDs (now all named)
# h_x = model.actuator("x").id
# h_y = model.actuator("y").id
# h_rot = model.actuator("rot_motor").id

# s_x = model.actuator("s_x").id
# s_y = model.actuator("s_y").id
# s_rot = model.actuator("s_rot").id

# def reset_bots():
#     mujoco.mj_resetData(model, data)

#     # Random start positions (within bounds)
#     data.qpos[hx_qpos] = np.random.uniform(-1.5, 1.5)
#     data.qpos[hy_qpos] = np.random.uniform(-1.5, 1.5)
#     data.qpos[hz_qpos] = np.random.uniform(-np.pi, np.pi)

#     data.qpos[sx_qpos] = np.random.uniform(-1.5, 1.5)
#     data.qpos[sy_qpos] = np.random.uniform(-1.5, 1.5)
#     data.qpos[sz_qpos] = np.random.uniform(-np.pi, np.pi)

#     mujoco.mj_forward(model, data)

# reset_bots()

# t0 = time.time()

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         t = time.time() - t0

#         # Constant controls
#         data.ctrl[h_x] = 1.0
#         data.ctrl[h_y] = 0.8
#         data.ctrl[h_rot] = 0.3

#         data.ctrl[s_x] = 0.3
#         data.ctrl[s_y] = 1.0
#         data.ctrl[s_rot] = 0.8

#         mujoco.mj_step(model, data)
#         viewer.sync()
