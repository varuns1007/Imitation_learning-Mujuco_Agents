import itertools
import torch
import random
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
import torch.distributions as distributions
import os
from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu
from policies.experts import load_expert_policy



class ImitationAgent(BaseAgent):
    '''
    Please implement an Imitation Learning agent. Read train_agent.py to see how the class is used. 
    
    
    Note: 1) You may explore the files in utils to see what helper functions are available for you.
          2)You can add extra functions or modify existing functions. Dont modify the function signature of __init__ and train_iteration.  
          3) The hyperparameters dictionary contains all the parameters you have set for your agent. You can find the details of parameters in config.py.  
          4) You may use the util functions like utils/pytorch_util/build_mlp to construct your NN. You are also free to write a NN of your own. 
    
    Usage of Expert policy:
        Use self.expert_policy.get_action(observation:torch.Tensor) to get expert action for any given observation. 
        Expert policy expects a CPU tensors. If your input observations are in GPU, then 
        You can explore policies/experts.py to see how this function is implemented.
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.replay_buffer = ReplayBuffer(self.hyperparameters['buffer_size']) #you can set the max size of replay buffer if you want
        

        #initialize your model and optimizer and other variables you may need
        self.learner_policy = ptu.build_mlp(self.observation_dim, self.action_dim, self.hyperparameters['n_layers'], self.hyperparameters['hidden_size'], self.hyperparameters['activation'],'tanh')
        self.optimizer = optim.Adam(self.learner_policy.parameters(), lr=self.hyperparameters['learning_rate'])
        self.loss_fn = nn.MSELoss(reduction='sum')
        self.beta = 1
        self.save = self.hyperparameters['save']
        self.best_avg_reward = -np.inf
        

    def forward(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        action = self.learner_policy(observation) #change this to your action
        return action


    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        
        action = self.learner_policy(observation)
        return action 

    
    
    def update(self, observations, actions):
        #*********YOUR CODE HERE******************
        loss_list = []
        for i in range(len(observations)):
            learner_policy_observation = ptu.from_numpy(observations[i])
            learner_policy_action = ptu.from_numpy(actions[i])
            
            if random.random() < self.beta:
                given_action = ptu.from_numpy(self.expert_policy.get_action(learner_policy_observation)) #expert action
            else:
                given_action = learner_policy_action #self explored action

            # given_action = ptu.from_numpy(self.expert_policy.get_action(learner_policy_observation)) #expert action

            predicted_action = self.forward(learner_policy_observation)

            loss = self.loss_fn(predicted_action, given_action)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())

        return np.mean(loss_list)
      
    


    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        if not hasattr(self, "expert_policy"):
            self.expert_policy, initial_expert_data = load_expert_policy(env, self.args.env_name)
            self.replay_buffer.add_rollouts(initial_expert_data)
            
            #to sample from replay buffer use self.replay_buffer.sample_batch(batch_size, required = <list of required keys>)
            # for example: sample = self.replay_buffer.sample_batch(32)
        
        #*********YOUR CODE HERE******************

        # collect new data based on learner policy
        # and add it to the replay buffer
        max_episode_len = env.spec.max_episode_steps
        new_unexplored_trajectories = utils.sample_n_trajectories(env,self.get_action,10,max_episode_len)
        
        steps = 0
        for trajectory in new_unexplored_trajectories:
            steps+= utils.get_traj_length(trajectory)

        self.replay_buffer.add_rollouts(new_unexplored_trajectories)

        # sample from the replay buffer and update the learner policy
        sample = self.replay_buffer.sample_batch(self.hyperparameters['batch_size'], required=['obs', 'acs'])
        episode_loss = self.update(sample['obs'], sample['acs'])

        self.beta = 0.9999*self.beta

        evaluation_trajectories, _ = utils.sample_trajectories(env,self.get_action,10*max_episode_len,max_episode_len)
        evaluation_rewards = [trajectory["reward"].sum() for trajectory in evaluation_trajectories]
        avg_reward = np.mean(evaluation_rewards)

        if avg_reward > self.best_avg_reward and self.save:
            model_path = os.path.join(os.getcwd(), 'best_models')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(self.learner_policy.state_dict(),os.path.join(model_path, self.args.env_name + ".pth"))

            print("model saved")
            self.best_avg_reward = avg_reward



        
        return {'episode_loss': episode_loss, 'trajectories': new_unexplored_trajectories, 'current_train_envsteps': steps} #you can return more metadata if you want to


