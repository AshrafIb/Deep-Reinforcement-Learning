# Project 1: Navigation 

## Introduction 

In this project i am going to train a agent to manoeuvre through an virtual environment and collect yellow bananas, while avoiding blue ones. 

The environment consists of 37 dimensions and allows the agent to perform 4 discrete actions: moving forward, moving backward, turning left and turning right. For each collect yellow banana, the agent gets a reward of +1 and for each collect blue one, he gets a reward of -1.

The environment is of episodic nature and the goal set by udacity is, to let the agent achieve a minimum score of 13 points in 100 consecutive episodes. 

In this Project i trained a Deep Q-Network as well as a Double-Deep-Q-Network and compared both results. The DDQN did achive a bit better results, but both alternatives have room for improvements. For a detailed discussion, please read the Report.pdf in this repository, where i explain my thoughts.

## Installing and Using Environment 

### Installation Guidelines

If you want to display the simulation of the agent, than you will have to install the Unity Machine Learning Agents Toolkit, which can be downloaded under the following link https://github.com/Unity-Technologies/ml-agents.  

To get the Banana Environment, please follow the instructions below.  

1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

   (*For Windows users*) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

   (*For AWS*) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file.

For Further information, please follow the given Instruction in Udacity's own github https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md.

### Usage Guidelines 

If you want to *train the agent by yourself*, please download the dqn_agent.py, the ddqn_agent.py, the model.py and the Navigation.ipynb and place all files in the same folder as the banana environment. 

If you want to load and play the agent, download both weights for the trained agents, the ddqn.pth and the dqn.pth, to the same directory. 

Either way, follow the instruction in the Navigation.ipynb - Notebook. 

# Dependencies 

To run the notebook, you need the above mentioned files, as well as the banana environment and the following libraries. 

1. PyTorch 
2. Numpy
3. Pandas 
4. Matplotlib
5. Seaborn 
6. Time 
7. Unity Machine Learning Agents Toolkit
8. OpenAI Gym
9. collections

