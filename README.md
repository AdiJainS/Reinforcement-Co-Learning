# Reinforcement-Co-Learning
(trying) Implementing OpenAI's hide and seek model


# Intrinsic Curiosity Module :
The Intrinsic Curiosity Module (ICM) is a fascinating concept in RL designed to help agents explore their environments effectively, especially when external rewards are rare or non-existent.
It solves the problem of sparse rewards i.e. when the rewards are very tough to find, the agents starts doing random things . ICM helps this by giving the agent an intrinsic reward (internal motivation) based on curiosity.

When the agent do some task , and if its prediction is wrong, it means it has encountered something new and it receives a high intrinsic reward.

The ICM notable uses 2 models : the IDM and FDM 

IDM : The work of IDM is pretty simple , to ignore the background or what is happening in the background and focus on what task it has to do . It focuses on current the the upcoming tasks and the actions which will take it to these tasks. IT MAINLY FOCUSES ON THE AGENT
FDM : This model now helps in GENERATING CURIOSITY . When the work with the agent is done , it takes the current state and the action of the agent , generates curiosity and then focuses on the next state.
The error done by this model gives it intrinsic reward . 



