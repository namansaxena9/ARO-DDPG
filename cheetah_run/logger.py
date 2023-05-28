#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import os
import numpy as np
import torch
import pickle


def save_parameters(agent,path):
    torch.save(agent.actor.state_dict(),'./'+path+'/actor.pt')
    torch.save(agent.actor_target.state_dict(),'./'+path+'/actor_target.pt')
    torch.save(agent.critic.state_dict(),'./'+path+'/critic.pt')
    torch.save(agent.critic_target.state_dict(),'./'+path+'/critic_target.pt')
    torch.save(agent.actor.optimizer.state_dict(),'./'+path+'/actor_optim.pt')
    torch.save(agent.critic.optimizer.state_dict(),'./'+path+'/critic_optim.pt')

def load_parameters(agent,path):
    agent.actor.load_state_dict(torch.load('./'+path+'/actor.pt'))
    agent.actor_target.load_state_dict(torch.load('./'+path+'/actor_target.pt'))
    agent.critic.load_state_dict(torch.load('./'+path+'/critic.pt'))
    agent.critic_target.load_state_dict(torch.load('./'+path+'/critic_target.pt'))
    agent.actor.optimizer.load_state_dict(torch.load('./'+path+'/actor_optim.pt'))
    agent.critic.optimizer.load_state_dict(torch.load('./'+path+'/critic_optim.pt'))


class Logger(object):
    def __init__(self, log_dir = None, log_freq = 10000):
        self.data = {}
        if(log_dir is None):
           self.log_dir = './log'
        else:
           self.log_dir = log_dir
        
        if(not os.path.isdir(self.log_dir)):
            os.mkdir(self.log_dir)
        
        self.pointer = 0
        self.log_freq = log_freq

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def add_scalar(self, tag, value, step=None):
        value = self.to_numpy(value)
        if(tag in self.data):
            self.data[tag].append(value)
        else:
            self.data[tag] = []
            self.data[tag].append(value)

        self.pointer +=1
        if(self.pointer % self.log_freq == 0):
            pointer = 0
            self.flush()

    def flush(self):
        file = open(self.log_dir + '/logfile.pkl','wb')
        pickle.dump(self.data, file)
        file.close()


