# Reinforcement-Co-Learning
(trying) Implementing OpenAI's hide and seek model

#MAPPO :
MAPPO, or Multi-Agent Proximal Policy Optimization. It follows CTDE framework: Centralized Training, Decentralized Execution.
Centralized Training: During the training phase, we assume access to more information than will be available at execution time. This might include the observations and actions of all agents, or even the underlying global state of the environment. These agents have global info which will help them to coordinate .

Decentralized Execution : Once training is complete, the centralized component is discarded. Each agent deploys its learned policies which selects actions based only on its own local observation history. It does not need a central mind thing.

MAPPO adapts the standard PPO architecture using two main components:
The Actor (The Policy) - It uses the same "clipping" trick as PPO to ensure the policy doesn't change too drastically in a single update, which keeps learning stable.

The Critic (The Value Function)- MAPPO uses a centralized citic. Instead of looking at one agent, the critic looks at the global state.the critic can more accurately judge if a specific agent's action was actually good or just lucky.

# Intrinsic Curiosity Module :
The Intrinsic Curiosity Module (ICM) is a fascinating concept in RL designed to help agents explore their environments effectively, especially when external rewards are rare or non-existent.
It solves the problem of sparse rewards i.e. when the rewards are very tough to find, the agents starts doing random things . ICM helps this by giving the agent an intrinsic reward (internal motivation) based on curiosity.

When the agent do some task , and if its prediction is wrong, it means it has encountered something new and it receives a high intrinsic reward.

The ICM notable uses 2 models : the IDM and FDM 

IDM : The work of IDM is pretty simple , to ignore the background or what is happening in the background and focus on what task it has to do . It focuses on current the the upcoming tasks and the actions which will take it to these tasks. IT MAINLY FOCUSES ON THE AGENT
FDM : This model now helps in GENERATING CURIOSITY . When the work with the agent is done , it takes the current state and the action of the agent , generates curiosity and then focuses on the next state.
The error done by this model gives it intrinsic reward . 

# Gymnasium / Petting zoo -
It is a toolkit used to train RL . Before Gymnasium, every researcher had their own way of building simulations. This made it impossible to compare results. Gymnasium provides a standardized API. The difference between pettingzoo and gymnasium is pettingzoo focuses on Multi-Agent .






