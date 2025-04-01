

from operator import truediv
import os
import pickle
import gym
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from torch.nn.modules import loss
import pandas as pd
from copy import deepcopy

class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        other_dim = 1 + 1 + self.action_dim
        self.buf_other = torch.empty(size=(max_len, other_dim), dtype=self.data_type, device=self.device)

        if isinstance(state_dim, int):
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(state_dim, tuple):
            self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('state_dim')

    def extend_buffer(self, state, other):
        size = len(other)
        next_idx = self.next_idx + size

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx




os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim,layer_norm=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim),)


        self.a_logstd = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        if layer_norm:
            self.layer_norm(self.net)
    @staticmethod
    def layer_norm(layer,std=1.0,bias_const=0.0):
        for l in layer:
            if hasattr(l,'weight'):
                torch.nn.init.orthogonal_(l.weight,std)
                torch.nn.init.constant_(l.bias,bias_const)


    def forward(self, state):
        return self.net(state).tanh()

    def get_action(self, state):
        a_avg = self.net(state)

        a_std = self.a_logstd.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_logprob_entropy(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5

        logprob = -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)

        dist_entropy = (logprob.exp() * logprob).mean()

        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):

        delta = noise.pow(2) * 0.5
        return -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)


class CriticAdv(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim,layer_norm=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))
        if layer_norm:
            self.layer_norm(self.net,std=1.0)
    @staticmethod
    def layer_norm(layer,std=1.0,bias_const=0.0):
        for l in layer:
            if hasattr(l,'weight'):
                torch.nn.init.orthogonal_(l.weight,std)
                torch.nn.init.constant_(l.bias,bias_const)
    def forward(self, state):
        return self.net(state)


class PPO_AC_ES_with_help:
    def __init__(self):
        super().__init__()
        self.state = None
        self.device = None
        self.action_dim = None
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

        '''init modify'''
        self.ClassCri = CriticAdv
        self.ClassAct = ActorPPO

        self.ratio_clip = 0.2
        self.lambda_entropy = 0.02
        self.lambda_gae_adv = 0.98

        self.get_reward_sum = None

        self.trajectory_list = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, gpu_id=0):
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.trajectory_list = list()
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw# choose whether to use gae or not

        self.cri = self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
        del self.ClassCri, self.ClassAct


    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(states)
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()

    def explore_env(self, env, target_step):
        trajectory_temp = list()

        state = self.state
        last_done = 0
        for i in range(target_step):#
            action, noise = self.select_action(state)
            state,next_state, reward, done,= env.step(np.tanh(action))

            trajectory_temp.append((state, reward, done, action, noise))
            if done:
                state = env.reset()
                last_done = i
            else:
                state = next_state
        self.state = state

        '''splice list'''
        trajectory_list = self.trajectory_list + trajectory_temp[:last_done + 1]

        self.trajectory_list = trajectory_temp[last_done:]
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        '''put data extract and update network together'''
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_action, buf_noise, buf_reward, buf_mask = [ten.to(self.device) for ten in buffer]


            '''get buf_r_sum, buf_logprob'''
            bs = 4096
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]#

            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)


            buf_r_sum, buf_advantage = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)
            # normalize advantage
            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
            del buf_noise, buffer[:]

        '''PPO: Surrogate objective of Trust Region'''

        obj_critic = obj_actor = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(state).squeeze(1)
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            obj_critic=self.criterion(value,r_sum)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None


        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1))
        return obj_critic.item(), obj_actor.item(), a_std_log.mean().item()

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)


        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value[:, 0])
        return buf_r_sum, buf_advantage

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)


        pre_r_sum = 0
        pre_advantage = 0

        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_advantage[i] = ten_reward[i] + ten_mask[i] * (pre_advantage - ten_value[i])

            pre_advantage = ten_value[i] + buf_advantage[i] * self.lambda_gae_adv
        return buf_r_sum, buf_advantage

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1.0 - tau))

class Arguments:

    def __init__(self, agent=None, env=None):
        self.agent = agent
        self.env = env

        self.cwd = None
        self.if_remove = False

        self.visible_gpu = '0'
        self.worker_num = 2
        self.num_threads = 8

        '''Arguments for training'''
        self.num_episode=5000
        self.gamma = 0.995
        self.learning_rate = 2e-4
        self.soft_update_tau = 2 ** -8

        self.net_dim = 128
        self.batch_size = 1024
        self.repeat_times = 2 ** 3
        self.target_step = 1024
        self.max_memo = self.target_step
        self.if_per_or_gae = False


        '''Arguments for evaluate'''
        self.random_seed_list = [1234, 2234, 3234, 4234, 5234]
        self.train=True
        self.save_network=True
        self.test_network=True
        self.save_test_data=True
        self.compare_with_pyomo=True
        self.plot_on=True

    def init_before_training(self, if_main):
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}'

        if if_main:
            import shutil

            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)

def update_buffer(_trajectory):
    _trajectory = list(map(list, zip(*_trajectory)))

    ten_state = torch.as_tensor(_trajectory[0])

    ten_reward=torch.as_tensor(_trajectory[1], dtype=torch.float32)
    ten_mask = (1.0 - torch.as_tensor(_trajectory[2], dtype=torch.float32)) * gamma

    ten_action = torch.as_tensor(_trajectory[3])
    ten_noise = torch.as_tensor(_trajectory[4], dtype=torch.float32)

    buffer[:] = (ten_state, ten_action, ten_noise, ten_reward, ten_mask)

    _steps = ten_reward.shape[0]
    _r_exp = ten_reward.mean()

    return _steps, _r_exp


start_time = time.time()
if __name__=='__main__':
    args=Arguments()
    reward_record={'episode':[],'steps':[],'mean_episode_reward':[]}
    loss_record={'episode':[],'steps':[],'critic_loss':[],'actor_loss':[],'entropy_loss':[]}
    args.visible_gpu = '0'
    for seed in args.random_seed_list:
        args.random_seed=seed
        args.agent=PPO_AC_ES_with_help()

        agent_name=f'{args.agent.__class__.__name__}'
        args.agent.cri_target=True
        args.env=ESS_EV_PF_Env()
        args.init_before_training(if_main=True)
        '''init agent and environment'''
        agent=args.agent
        env=args.env
        agent.init(args.net_dim,env.state_space.shape[0],env.action_space.shape[0],args.learning_rate,args.if_per_or_gae)


        cwd=args.cwd
        gamma=args.gamma
        batch_size=args.batch_size

        target_step=args.target_step

        repeat_times=args.repeat_times

        soft_update_tau = args.soft_update_tau
        agent.state = env.reset()
        '''init buffer'''
        buffer = list()
        '''init training parameters'''
        num_episode=args.num_episode

        # args.train=False
        # args.save_network=False
        args.test_network=False
        args.save_test_data=False
        # args.compare_with_pyomo=False
        if args.train:
            for i_episode in range(num_episode):
                with torch.no_grad():
                    trajectory_list=agent.explore_env(env,target_step)
                    steps,r_exp=update_buffer(trajectory_list)
                critic_loss,actor_loss,entropy_loss = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
                loss_record['critic_loss'].append(critic_loss)
                loss_record['actor_loss'].append(actor_loss)
                loss_record['entropy_loss'].append(entropy_loss)

                with torch.no_grad():
                    episode_reward,episode_EV, episode_temp=get_episode_return(env,agent.act,agent.device)
                    reward_record['mean_episode_reward'].append(episode_reward)




    act_save_path = f'{args.cwd}/actor.pth'
    loss_record_path=f'{args.cwd}/loss_data.pkl'
    reward_record_path=f'{args.cwd}/reward_data.pkl'

    with open (loss_record_path,'wb') as tf:
        pickle.dump(loss_record,tf)
    with open (reward_record_path,'wb') as tf:
        pickle.dump(reward_record,tf)


    if args.save_network:
        torch.save(agent.act.state_dict(),act_save_path)
        print('actor parameters have been saved')


    if args.test_network:
        args.cwd=agent_name
        agent.act.load_state_dict(torch.load(act_save_path))
        print('parameters have been reload and test')
        record=test_one_episode(env,agent.act,agent.device)

    if args.save_test_data:
        test_data_save_path=f'{args.cwd}/test_data.pkl'
        with open(test_data_save_path,'wb') as tf:
            pickle.dump(record,tf)
