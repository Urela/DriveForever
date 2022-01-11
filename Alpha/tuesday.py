import numpy as np
import cv2
import mss
import time
import utils
import torch
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.distributions import Categorical

monitor = {"top": 0, "left": 0, "width": 1366, "height": 768 }
class ActorCritic(nn.Module):
  def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

  def __init__(self, in_space, out_space, lr=2.5e-4):
    super(ActorCritic, self).__init__()

    self.conv = nn.Sequential(
      self.layer_init(nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=8, stride=4)), nn.ReLU(),
      self.layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)), nn.ReLU(),
      self.layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)), nn.ReLU(),
      nn.Flatten()
    )

    self.fc_critic = nn.Sequential(
      self.layer_init(nn.Linear(in_features=7*7*64, out_features=512)), nn.ReLU(),
      self.layer_init(nn.Linear(in_features=512,    out_features=1), std=0.01),
    )

    self.fc_actor = nn.Sequential(
      self.layer_init(nn.Linear(in_features=7*7*64, out_features=512)), nn.ReLU(),
      self.layer_init(nn.Linear(in_features=512,    out_features=out_space), std=0.01),
    )

    self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)
    self.to('cpu')

  def actor(self, state):
    state = self.conv(state)
    dist = self.fc_actor(state)
    dist = Categorical(logits=dist)
    return dist

  def critic(self, state):
    state = self.conv(state)
    value = self.fc_critic(state)
    return value

memory = []
class PPO:  
  def __init__(self, in_space, out_space):
    self.lr    = 2.5e-4   # 0.0003
    self.gamma = 0.99 
    self.lamda = 0.95
    self.epoch = 4
    self.eps_clip = 0.2
    self.bsize = 5 # batch_size
     
    self.AC = ActorCritic(None, 9)

  def selectAction(self, obs):
    obs = torch.tensor([obs], dtype=torch.float32, device='cpu') 
    value  = self.AC.critic(obs)
    dist   = self.AC.actor(obs)
    action = dist.sample()

    probs  = torch.squeeze(dist.log_prob(action)).item()
    action = torch.squeeze(action).item()
    value  = torch.squeeze(value).item()
    return action, probs, value

  def train(self):
    for _ in range( self.epoch):
      states  = np.array([x[0] for x in memory])
      actions = np.array([x[1] for x in memory])
      rewards = np.array([x[2] for x in memory])
      probs   = np.array([x[3] for x in memory])
      values  = np.array([x[4] for x in memory])
      dones   = np.array([1-int(x[5]) for x in memory])

      ####### Advantage using gamma returns
      nvalues = np.concatenate([values[1:] ,[values[-1]]])
      delta = rewards + self.gamma*nvalues*dones - values
      advantage, adv = [], 0
      for d in delta[::-1]:
        adv = self.gamma * self.lamda * adv + d
        advantage.append(adv)
      advantage.reverse()

      advantage = torch.tensor(advantage).to('cpu')
      values    = torch.tensor(values).to('cpu')
      
      # create mini batches
      num = len( states ) 
      batch_start = np.arange(0, num, self.bsize)

      indices = np.arange( num, dtype=np.int64 )
      np.random.shuffle( indices )
      batches = [indices[i:i+self.bsize] for i in batch_start]

      for batch in batches:
        _states  = torch.tensor(states[batch], dtype=torch.float).to('cpu')
        _probs   = torch.tensor(probs[batch], dtype=torch.float).to('cpu')
        _actions = torch.tensor(actions[batch], dtype=torch.float).to('cpu')

        dist   = self.AC.actor(_states)
        nvalue = self.AC.critic(_states)
        nvalue = torch.squeeze(nvalue)

        new_probs = dist.log_prob(_actions)
        ratio = new_probs.exp() / _probs.exp()

        surr1 = ratio * advantage[batch]
        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage[batch]
        actor_loss = -torch.min(surr1, surr2).mean() 

        returns = advantage[batch] + values[batch]
        critic_loss = (returns-nvalue)**2
        critic_loss = critic_loss.mean()

        total_loss = actor_loss + 0.5*critic_loss
        self.AC.optimizer.zero_grad()
        total_loss.backward()
        self.AC.optimizer.step()

    del memory [:]

agent = PPO(0, 8)

# Environment Hyperparameters
max_ep_len = 128                    # max timesteps in one episode
max_training_timesteps = int(1e4)   # break training loop if timeteps > max_training_timesteps
update_timestep = max_ep_len * 4    # update policy every n timesteps
print_freq = max_ep_len * 4         # print avg reward in the interval (in num timesteps)

print_running_reward   = 0
print_running_episodes = 0

time_step = 0
i_episode = 0

while time_step <= max_training_timesteps:
  for t in range(1, max_ep_len+1):

    # frames are stacked to give our observation space
    obs, reward = [], 0
    for _ in range(4):
      _frame, _speed = utils.grab_img_and_speed(monitor, sct=mss.mss(), resize_width=84, resize_height=84)
      obs.append( _frame )
      reward += _speed
      time_step +=1
    obs = np.array(obs)
    reward /= 4
    action, probs, value = agent.selectAction(obs)
    done = False
    memory.append((obs, action, reward, probs, value, done))

    #print( utils.gamecontroller(action), reward) 
    print( time_step)

    # render frame
    cv2.imshow("test", np.array(obs[-1]))
    if cv2.waitKey(25) & 0xFF == ord("q"):
      cv2.destroyAllWindows()
      break
  agent.train()

# Pickling
with open("test", "wb") as fp:   
  pickle.dump(memory, fp)
