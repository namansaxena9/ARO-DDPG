import numpy as np
import torch
import torch.nn as nn
from dm_control import suite
from ddpg_model import Agent, Buffer, OUNoise
from ddpg_model import return_state_dim, process_state
import sys
from logger import load_parameters
from config import config
import dmc2gym
import cv2

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

env_eval = dmc2gym.make(config['domain'],config['task'], episode_length = config['epi_len_eval'])


config['state_dim'] = env_eval.observation_space.shape[0]
config['action_dim'] = env_eval.action_space.shape[0]

config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

buffer = Buffer(config['buffer_size'])
agent = Agent(env_eval, config)

load_parameters(agent,'log')

epi_len = config['epi_len_eval']



width = 480
height = 480
camera_id = 0
n_iter = 100

#state_collection = []
#physics_collection = []

for epi in range(n_iter):
    total_reward = 0
    state = env_eval.reset()
    done = False
    video = np.zeros((epi_len,width,height,3),dtype = np.uint8)
    steps = 0    
    while not done:
        video[steps] = env_eval.render(mode = 'rgb_array',width = width, height= height,camera_id = 0)
        #state_collection.append(state)
        #physics_collection.append(env.physics.get_state())
        action = agent.actor.get_action(torch.tensor(state).float())
        next_state, reward, done, _ = env_eval.step(action.detach().numpy())
        total_reward += reward
        state = next_state
        steps +=1
    print("Reward",total_reward)
    print("steps",steps)

    out = cv2.VideoWriter('./video/project_'+str(epi)+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 40, (width,height))
    for i in range(len(video)):
       out.write(video[i])
    out.release()           
    #np.save('./store/state_collection_'+str(epi),np.array(state_collection))
    #np.save('./store/physics_collection_'+str(epi),np.array(physics_collection))    
    #np.save('./store/image_collection_'+str(epi),video)
    #state_collection = []
    #physics_collection = []


                        
    
