# -*- coding: utf-8 -*-


############# libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import time

############################# GPU
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
start_time= time.time()
########################### Parameters
env = ESS_EV_PF_Env()
egreedy_final =0.01
egreedy = 0.9
egreedy_decay = 96*10000

nb_input=16
nb_output =8
nb_hidden_layer=128
learning_rate = 0.001

gamma=0.999

size_of_memory = 50000
batch_size =64
nb_epsiode = 50000

seed = 23
torch.manual_seed(seed)
random.seed(seed)

clip_error = True
update_target_frequency = 500

double_dqn = True

############################ epsilion
def calcuate_epsilion(nb_step):
  epsilion= egreedy_final + (egreedy - egreedy_final)* math.exp(-1.*(nb_step/egreedy_decay))
  return epsilion

########################## Experiment
class ReplayMemory():
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory=[]
    self.position =0

  def push(self,state,action,new_state,reward,done):
    transition=(state,action,new_state,reward,done)
    if self.capacity >= len(self.memory):
     self.memory.append(transition)
    else:
      self.memory[self.position] = transition
    self.position = (self.position +1) % self.capacity

  def sample(self,batch_size):
    return zip(*random.sample(self.memory, batch_size))

  def __len__(self):
    return len(self.memory)

#################################
class DQNnn(nn.Module):
  def __init__(self):
    super(DQNnn,self).__init__()
    self.first=nn.Linear(nb_input, nb_hidden_layer)
    self.second= nn.Linear(nb_hidden_layer, nb_hidden_layer)
    self.thid= nn.Linear(nb_hidden_layer, nb_hidden_layer)
    self.four= nn.Linear(nb_hidden_layer, nb_hidden_layer)
    self.five= nn.Linear(nb_hidden_layer, nb_hidden_layer)
    self.six= nn.Linear(nb_hidden_layer, nb_output)
    self.activation = nn.Tanh()

  def forward(self,x):
    x1=self.first(x)
    x1=self.activation(x1)
    x1=self.second(x1)
    x1=self.activation(x1)
    x1=self.thrid(x1)
    x1=self.activation(x1)
    x1=self.four(x1)
    x1=self.activation(x1)
    x1=self.five(x1)
    x1=self.activation(x1)
    x1=self.six(x1)
    return x1

############################# DQN_agent
class DQN_agent():
  def __init__(self):
    self.nn=DQNnn().to(device)
    self.nn_target=DQNnn().to(device)
    self.loss_function=nn.MSELoss()
    self.optimization_function=optim.Adam(params=self.nn.parameters(), lr=learning_rate)

    self.update_target_counter = 0

  def select_action(self,state,epsilion):
    egreedep= torch.rand(1).item()

    if egreedep > epsilion:
      with torch.no_grad():
        state = torch.Tensor(state).to(device)
        Q_value_current= self.nn(state)
        action = torch.max(Q_value_current,0)[1].item()
    else:
      action = np.random.randint(0, 8)
    return action

  def optimizer(self):
    if (len(memory) < batch_size):
      return
    state,action,new_state, reward,done= memory.sample(batch_size)
    state= torch.Tensor(state).to(device)
    new_state= torch.Tensor(new_state).to(device)
    action= torch.LongTensor(action).to(device)
    reward =torch.Tensor(reward).to(device)
    done = torch.Tensor(done).to(device)

    if double_dqn:
      new_state_indexes = self.nn(new_state).detach()
      location_of_max_Q_value = torch.max(new_state_indexes, 1)[1]

      new_state_values = self.nn_target(new_state).detach()
      max_new_state_values = new_state_values.gather(1, location_of_max_Q_value.unsqueeze(1)).squeeze(1)

    else:
      new_state_values = self.target_nn(new_state).detach()
      max_new_state_values = torch.max(new_state_values, 1)[0]


    y_target = reward + (1-done)* gamma * max_new_state_values

    y_predict = self.nn(state).gather(1,action.unsqueeze(1)).squeeze(1)

    loss=self.loss_function(y_predict, y_target)
    self.optimization_function.zero_grad()
    loss.backward()

    #if clip_error:

    for param in self.nn.parameters():
        param.grad.data.clamp_(-1,1)

    self.optimization_function.step()
    if self.update_target_counter % update_target_frequency == 0:
      self.nn_target.load_state_dict(self.nn.state_dict())

    self.update_target_counter += 1

######################## env
DQN_model = DQN_agent()
memory = ReplayMemory(size_of_memory)
total_frame =0
step_ =[]
step_best =[]

rewards_ev =2
for nb_epoch in range (nb_epsiode):
  state= env.reset()
  step =0
  total_reward =0
  action_total=[]
  while True:
    step +=1
    total_frame+=1
    epsilion=calcuate_epsilion(total_frame)
    action= DQN_model.select_action(state,epsilion )
    action_total.append(action)
    new_state,reward,done=env.step(action)
    total_reward += reward
    memory.push(state,action, new_state,reward,done)
    DQN_model.optimizer()
    state = new_state
    if done:
      break
  step_.append(total_reward)
  print(' total reward: ', total_reward, ' epsilion: ', epsilion, 'number of steps: ', step)
  torch.save(DQN_model.nn.state_dict(), 'DDQN_clippng_1.pth')
