import torch.nn as nn

config = {}

config['seed'] = 10
config['domain'] = 'cheetah'
config['task'] = 'run'
config['buffer_size'] = int(1e5)
config['total_env_steps'] = int(1e6)
config['batch_size'] = 256
config['warmup_samples'] = 5000
config['eval_freq'] = 5000
config['epi_len'] = 1000
config['epi_len_eval'] = 10000
config['act_fn'] = nn.ReLU
config['tau'] = 0.995
config['log'] = True
config['update_freq'] = 1000
config['critic_freq'] = 1
config['actor_freq'] = 2


config['lr_critic'] = 3e-4
config['lr_actor'] = 3e-4
config['lr_rho'] = 3e-4
config['critic_hidden'] = 128
config['actor_hidden'] = 128