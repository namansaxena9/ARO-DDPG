import numpy as np
import torch
import torch.nn as nn
from dm_control import suite
from ddpg_model import Agent, Buffer, OUNoise
from ddpg_model import return_state_dim, process_state
import sys
from logger import save_parameters
from config import config
import dmc2gym

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

env = dmc2gym.make(config['domain'],config['task'], episode_length = config['epi_len'])
env_eval = dmc2gym.make(config['domain'],config['task'], episode_length = config['epi_len_eval'])


config['state_dim'] = env.observation_space.shape[0]
config['action_dim'] = env.action_space.shape[0]

config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

buffer = Buffer(config['buffer_size'])
agent = Agent(env_eval, config)
total_env_steps = config['total_env_steps']
batch_size = config['batch_size']
learning_start = config['warmup_samples']
eval_freq = config['eval_freq']
update_freq = config['update_freq']

noise = OUNoise(env.action_spec())
epi_len = config['epi_len']


n_steps = 0
count = 0
history_rho = 0
print("self actor-critic training...")

while(count < total_env_steps//eval_freq):
    state = env.reset()
    done = False
    for epi_steps in range(1,epi_len+1):
        n_steps += 1
        
        if(epi_steps == epi_len):
             done  = True  

        if(len(buffer)<learning_start):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action.astype(np.float32))
            buffer.insert(state,reward,action,done,next_state)
        else:
            action = agent.get_action(torch.tensor(state).float())
            action = noise.get_action(action.detach().numpy())
            next_state, reward, done, _ = env.step(action.astype(np.float32))
                        
            buffer.insert(state,reward,action,done,next_state)
                     
        state = next_state        
        
        if(n_steps % eval_freq == 0):
          count+=1    
          temp = agent.evaluate(epi_len = config['epi_len_eval'], n_iter = 2)
          
          agent.logger.add_scalar('Reward',temp[0])
          agent.logger.add_scalar('rho_eval', temp[1])
          
          if( history_rho - temp[1] > 0.2):
              save_parameters(agent,'anomaly')
              print("Fault occured",count,flush= True)
          else:
              save_parameters(agent,'log')
          
          print("rho", temp[1], flush = True)
          
          history_rho = temp[1]
    
        if(len(buffer)>=batch_size and n_steps % update_freq == 0):
            agent.update(buffer, batch_size = batch_size, gradient_steps = update_freq, critic_freq = config['critic_freq'], actor_freq = config['actor_freq'], rho_freq = 10)

agent.logger.flush()        
