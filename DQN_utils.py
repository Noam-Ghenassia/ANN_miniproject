import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

from tic_env import TictactoeEnv, OptimalPlayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Q_net(nn.Module):
  
    def __init__(self, structure=(128, 128)):
        super(Q_net, self).__init__()
        self.structure = structure

        self.layer_list = torch.nn.ModuleList()

        self.layer_list.append(nn.Sequential(nn.Linear(2, 128, bias=False)))

        for ii in range(len(self.structure)):
            self.layer_list.append(
                self.hidden_layer(self.structure[ii] , self.structure[ii], use_batch_norm=False)
            )
          
        self.layer_list.append(nn.Sequential(nn.Linear(128, 9, bias=False)))


    def hidden_layer(self,input, output, use_batch_norm=False):
        linear = nn.Linear(input, output, bias=True)
        relu = nn.ReLU()
        bn = nn.BatchNorm1d(output)
        if use_batch_norm:
            return(nn.Sequential(linear, relu, bn))
        else:
            return(nn.Sequential(linear, relu))


    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x



class DQ_learner():

    def __init__(self, gamma=.99, eps_min=.1, eps_max=.8,
                    buffer_size=10000,
                    target_network_update=500,
                    batch_size=500, lr=5e-4,
                    seed=random.random()):

        self.gamma=gamma
        self.epsilon=np.max(eps_min, eps_max)
        self.eps_min, self.eps_max = eps_min, eps_max
        self.target_network_update=target_network_update
        self.batch_size=batch_size
        self.lr=lr

        random.seed(seed)
        self.memory = ReplayMemory(buffer_size)
        self.policy_net = Q_net()
        self.target_net = Q_net()
        #self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
    
    def update_epsilon(self, eps_min, eps_max, episode):
        self.epsilon = np.max(eps_min, eps_max*(1-(1/episode)))
    
    def get_state(self, env):
        return torch.tensor(env.observe[0])
    
    def epsilon_greedy(self, action, env):
        if random.random() > self.epsilon:
            return action
        else:
            available_actions = (torch.flatten(env.observe[0])==0).nonzero().tolist()
            return random.choice(available_actions)

    def is_action_available(self, env, action):
        return action in (torch.flatten(torch.sum(env.observe[0]))==0).nonzero()
    
    def update_nets(self, episode):
        if len(self.memory) < self.batch_size:
            return
        else:
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            batch_state = torch.tensor(batch.state)
            batch_action = torch.tensor(batch.action)
            batch_reward = torch.tensor(batch.reward)

            # non final next states :
            batch_next_state = torch.tensor([ns for ns in batch.next_state if ns is not None])

            # indices of final states :
            final_indices = torch.tensor([ind for ind in range(len(batch.next_state)) if batch.next_state[ind] == None])

            # final and non-final states masks :
            final_states_mask = torch.tensor(tuple(map(lambda s: s is None,
                                                    batch.next_state)), dtype=torch.bool)
            non_final_states_mask = ~final_states_mask

            # initialize targets tensor :
            targets = torch.zeros_like(final_states_mask).type(torch.LongTensor)

            # for final states, the target is simply the reward :
            targets[final_states_mask] = batch_reward[final_indices]

            # for non-final states, the target is given by the Bellman equation :
            preds = torch.max(self.target_net(batch_state[non_final_states_mask]))      # use torch.amax to take max along output dim
            # batch reward added only for completeness, since in non final states there are noe rewards.
            targets[non_final_states_mask] = self.gamma*preds + batch_reward

            # predictions of the policy network :
            state_action_values = self.policy_net(batch_state).gather(1, batch_action)

            # optimization step of policy network :
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, targets.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()  


            # Update the target network, copying all weights and biases in DQN
            if episode % self.target_network_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())



    def play_and_learn(self, env, player, actions_taken):

        # use policy network to predict action, and apply (with epsilon-greedy policy).
        # collect transition and add to replay_buffer
        """state = torch.flatten(torch.sum(env.observe[0]))"""     # easier than 3*3*2 tensor...
        state = torch.flatten(env.observe[0])
        predicted_action = self.policy_net(state)               # torch.max ?
        action = self.epsilon_greedy(predicted_action, env)
        if self.is_action_available(env, action):
            env.step(action)
            if env.observe[1]:
                next_state = None
            else:
                next_state = torch.flatten(env.observe[0])
            reward = env.reward(player=player)
        else:
            env.reset()
            reward = -1
            next_state = None
        #transition = Transition(state, action, next_state, reward)
        self.memory.push(state, action, next_state, reward)

        # update policy and target networks according to replay buffer
        self.update_nets(actions_taken)

    def train(self, number_games, opponent_epsilon):

        env = TictactoeEnv()
        players = ['X', 'O']
        #actions_taken = 0

        for game in range(number_games):
            env.reset()
            player = players[game % 2]
            opponent = OptimalPlayer(epsilon=opponent_epsilon, player=players[(game + 1) % 2])

            while not env.end:
                if env.current_player == player:
                    self.update_epsilon(self.eps_min, self.eps_max, game)
                    self.play_and_learn(env, player, game)
                    #actions_taken += 1
                else:
                    opponent_action = opponent.act(env.grid)
                    env.step(opponent_action)
    

    """def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()"""