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
  
    def __init__(self, structure=(128, 128), use_batch_norm=False):
        super(Q_net, self).__init__()
        self.structure = structure

        self.layer_list = torch.nn.ModuleList()

        self.layer_list.append(nn.Sequential(nn.Linear(9, 128, bias=False)))

        for ii in range(len(self.structure)):
            self.layer_list.append(
                self.hidden_layer(self.structure[ii] , self.structure[ii], use_batch_norm)
            )
          
        self.layer_list.append(nn.Sequential(nn.Linear(128, 9, bias=False)))


    def hidden_layer(self,input, output, use_batch_norm):
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
                    seed=random.random(),
                    use_adam=False):

        self.gamma=gamma
        self.epsilon=np.maximum(eps_min, eps_max)
        self.eps_min, self.eps_max = eps_min, eps_max
        self.target_network_update=target_network_update
        self.batch_size=batch_size
        self.lr=lr

        random.seed(seed)
        self.memory = ReplayMemory(buffer_size)
        self.policy_net = Q_net().double()
        self.target_net = Q_net().double()
        if use_adam:
            self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.RMSprop(self.policy_net.parameters())
    
    def update_epsilon(self, eps_min, eps_max, episode):
        self.epsilon = np.maximum(eps_min, eps_max*(1-(1/(episode+1))))

    
    def get_state(self, env):
        return torch.tensor(env.observe()[0])

    
    def epsilon_greedy(self, action, env):
        """This method implements the epsilon greedy policy : it returns either the action
        chosen by the agent (wirh probability epsilon), or an action chosen randomly."""
        if random.random() > self.epsilon:
            return torch.tensor([action])
        else:
            available_actions = self.available_actions(env)
            return torch.tensor(random.choice(available_actions))

    
    def available_actions(self, env):
        """This function returns the list of currrently available actions given the current
        state of the environment."""
        return (torch.flatten(torch.tensor(env.observe()[0]))==0).nonzero().tolist()

    
    def update_nets(self, episode):
        """This function updates both the policy network and (when needed) the target network.
        the target network is copied from the policy network with a frequency determined by
        the class attribute target_network_update.
        The optimizer is either RMSProp or adam (depending on the corresponding class attribute),
        and optimizes the error derived from the Bellman equation for Q-learning. For the final
        state of an episode, the error is zero when the prediction matches the reward."""
        if len(self.memory) < self.batch_size:
            return
        else:
            transitions = self.memory.sample(self.batch_size)
            # transpose batch of transitions -> transition of batche arrays
            batch = Transition(*zip(*transitions))
            batch_state = tuple_of_tensors_to_tensor(batch.state)
            batch_action = tuple_of_tensors_to_tensor(batch.action)
            batch_reward = torch.tensor(batch.reward)

            # non final next states :
            batch_next_state = torch.stack([ns for ns in batch.next_state if ns is not None], dim=1)

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

            # for non-final states, the target is given by the Bellman equation,
            # where predictions are mmade with the target network to stabilize targets :
            preds = torch.max(self.target_net(torch.t(batch_next_state).double()))      # use torch.amax to take max along output dim

            # batch reward ommited, since in non final states rewards are zero :
            with torch.no_grad():
                targets[non_final_states_mask] = self.gamma*preds

            # predictions of the policy network :
            state_action_values = self.policy_net(batch_state).gather(1, batch_action)

            # optimization step of policy network :
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, targets.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            #for param in self.policy_net.parameters():
            #    param.grad.data.clamp_(-1, 1)              # WHY SHOULD WE CLAMP THE GRADIENTS ?? ####################3
            self.optimizer.step()


            # Update the target network by making it the same as the policy network :
            if episode % self.target_network_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            return loss



    def play_and_learn(self, env, player, episode):

        # use policy network to predict action, and apply (with epsilon-greedy policy).
        # collect transition and add to replay_buffer
        state = torch.flatten(torch.from_numpy(env.observe()[0]).double())                   # easier than 3*3*2 tensor...
        predicted_action = torch.argmax(self.policy_net(state))
        action = self.epsilon_greedy(predicted_action, env)
        if [action] in self.available_actions(env):
            env.step(action.item())
            if env.observe()[1]:
                next_state = None
            else:
                next_state = torch.flatten(torch.from_numpy(env.observe()[0]))
            reward = env.reward(player=player)
        else:
            env.reset()
            reward = -1
            next_state = None
        #transition = Transition(state, action, next_state, reward)
        self.memory.push((state, action, next_state, reward))

        # update policy and target networks according to replay buffer
        #self.update_nets(actions_taken)
        loss = self.update_nets(episode=episode)
        return loss

    def train(self, number_games, opponent_epsilon):

        env = TictactoeEnv()
        players = ['X', 'O']
        #actions_taken = 0
        losses = []
        for game in range(number_games):
            env.reset()
            player = players[game % 2]
            opponent = OptimalPlayer(epsilon=opponent_epsilon, player=players[(game + 1) % 2])

            while not env.end:
                if env.current_player == player:
                    self.update_epsilon(self.eps_min, self.eps_max, game)
                    loss = self.play_and_learn(env, player, game)
                    if game % 10 == 0:
                        losses.append(loss)
                else:
                    opponent_action = opponent.act(env.grid)
                    env.step(opponent_action)
        
        return losses

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)
