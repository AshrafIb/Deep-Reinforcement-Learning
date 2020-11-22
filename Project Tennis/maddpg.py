# Modules
# This MADDPG is based on the variant that was discussed during the course.
from ddpg import Agent, ReplayBuffer, device
import numpy as np
import random
from actorcritic import Actor, Critic

# Hyperparams 
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
LEARN_EVERY = 1        # Initialize Learning every 1. Game
LEARN_TIMES = 5        # update 5 times if larger than batch_size
GAMMA = 0.99            # discount factor




# Created a Class, which runs the ddpg for two Agents. 
class maddpg:
    def __init__(self,state_size, action_size,random_seed):
        super(maddpg,self).__init__()
        
        self.state_size=state_size
        self.action_size=action_size
        self.seed = random.seed(random_seed)
        # As it was clear the the amount of Agent is limited to two, i
        # 'hardcoded' both into the class
        self.maddpg_agent=[Agent(state_size,action_size, random_seed),
                           Agent(state_size,action_size, random_seed)]
        
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
     
    def act (self, states):
        # corpus for Agent Actions
        actions=[]
        for Agent, state in zip(self.maddpg_agent, states):
            action=Agent.act(state)
            actions.append(action)
        return actions
    
    # Copied from my DDPG as it has to run via my maddpg 

    def step(self, states, actions, rewards, next_states, dones, times):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            # Save experience / reward
            self.memory.add(state, action, reward, next_state, done)
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and times % LEARN_EVERY==0:
            # add agent iterator
            for agent in self.maddpg_agent:
                for _ in range(LEARN_TIMES):
                    experiences = self.memory.sample()
                    agent.learn(experiences, GAMMA)
                    
    # Resting Agents  after all episodes.                 
    def reset(self):
        for Agent in self.maddpg_agent:
            Agent.reset()

    