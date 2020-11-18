# Project - Continuos Control 

## Introduction

The Environment consists of double-jointed arms (the agents), which can move to a target location (the green ball). The Goal is to reach the target location and to stay on target; for each step, where the Agent's hand is at the goal location, he receives an award of 0.1.  In contrast to the Unity-Version (26 variables), the present Version has 33 Variables corresponding to position, rotation, velocity and angular velocity (of both arm-parts). There are no visual observations. Obviously the action-space is of continuous nature, equivalent to the torque variables mentioned above. Actions typically should be in an range between -1 and +1. 

To successfully complete this environment, an average of at least 30 points must be achieved in 100 consecutive episodes as a mean of all 20 agents.  

There are different Versions, a version containing a single Agent and a Version where 20 Agents are used. I have used the 20 Agents Version. 

# Getting Started 



### Running the Agent

1. Follow the Python and Pytorch requirements mentioned in the [Udacity github](https://github.com/udacity/deep-reinforcement-learning#dependencies).  

2. Download the environment from one of the links below. You need only select the environment that matches your operating system:

   1. Version 1: One (1) Agent
      - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
      - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
      - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
      - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

   2. Version 2: Twenty (20) Agents
      - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
      - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
      - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
      - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

3. Download the Repo and the Environment and place all Files in the same Folder, and unzip (or decompress) the files if necessary. 

### Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent! Caution: It might take a while (in my Case about 5 Hours) to finish Training. The Notebook is only configured to run the 20-Agent-Version. Change to the desired Environment at the beginning. 

# Results 



The Environment was solved in 102 Episodes with a mean of 30.08! For Further informations on my experiences, please read the Report.pdf

# Dependencies 

1. Pytorch
2.  UnityEnvironment
3. collections 
4. itertools 
5. time 
6. numpy 
7. pandas 
8. copy 
9. random 
10. matplotlib 