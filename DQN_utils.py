from hmac import new
from matplotlib.style import available
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import matplotlib.pyplot as plt

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
        #self.layer_list.append(nn.Sequential(nn.Linear(18, 128, bias=False)))

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

    def __init__(self, gamma=.99, eps_min=.1, eps_max=.8, decaying_time=1, 
                    buffer_size=10000,
                    target_network_update=500,
                    batch_size=64, lr=5e-4,
                    seed=random.random(),
                    use_adam=False,
                    remove_buffer=False):
        """This class provides all the necessary tools to train a deep Q network.

        Args :
            gamma (float): the discount factor.
            eps_min (float): the minimum value of epsilon applicable in the epsilon-greedy policy.
            eps_max (float): the maximum value of epsilon applicable in the epsilon-greedy policy.
            decaying_time (float): the factor that controls the decaying speed of epsilon.
            buffer_size (int): the number of moves the replay buffer can hold.
            target_network_update (int): the tnumber of games between updates of the target network.
            use_adam (bool): this parameter allows to choose between the adam and RMSProp optimizers.
                Defaults to False.
            remove_buffer (bool): this parameter allows to train the etwork with a batch size of 1.
                Defaults to False.

        
        """
        self.gamma=gamma
        self.epsilon=np.maximum(eps_min, eps_max)
        self.decaying_time=decaying_time
        self.eps_min, self.eps_max = eps_min, eps_max
        self.target_network_update=target_network_update
        self.remove_buffer=remove_buffer
        if remove_buffer:
            self.batch_size=1
        else:
            self.batch_size=batch_size
        self.lr=lr

        random.seed(seed)
        if remove_buffer:
            self.memory = ReplayMemory(1)
        else:
            self.memory = ReplayMemory(buffer_size)
        self.policy_net = Q_net().double().to(device)
        self.target_net = Q_net().double().to(device)
        if use_adam:
            self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.RMSprop(self.policy_net.parameters())
    
    def update_epsilon(self, episode):
        self.epsilon = np.maximum(self.eps_min, self.eps_max*(1-(episode/self.decaying_time)))
    
    def get_state(self, env):
        return env.grid.copy()
        # if player == 'X':
        #     return env.grid.copy()
        # else:
        #     return (-1)*env.grid.copy()
        # grid = env.grid.copy()
        # if player == 'X':
        #   return np.stack([(grid - np.ones_like(grid)==0)*1, (grid + np.ones_like(grid)==0)*1]).flatten()
        # else:
        #   return np.stack([(grid + np.ones_like(grid)==0)*1, (grid - np.ones_like(grid)==0)*1]).flatten()
    
    def epsilon_greedy(self, action, env):
        """This method implements the epsilon greedy policy : it returns either the action
        chosen by the agent (with probability 1-epsilon), or an action chosen randomly."""
        if random.random() > self.epsilon:
            return torch.tensor([action])
        else:
            available_actions = self.available_actions(env)
            return torch.tensor(random.choice(available_actions))       

    def available_actions(self, env):
        """This function returns the list of available actions given the current
        state of the environment."""
        return (torch.tensor(env.grid.copy().flatten())==0).nonzero().tolist()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_nets(self, episode):
        """This function updates both the policy network and (when needed) the target network.
        the target network is copied from the policy network with a frequency determined by
        the class attribute target_network_update.
        The optimizer is either RMSProp or adam (depending on the corresponding class attribute),
        and optimizes the error derived from the Bellman equation for Q-learning. For the final
        state of an episode, the error is zero when the prediction matches the reward."""

        if self.remove_buffer:
            last_transition = Transition(*zip(*(self.memory.sample(1))))
            if last_transition.next_state[0] is None:
                target=torch.tensor(last_transition.reward[0]).double().to(device)

            else:
                target= self.gamma*torch.max(self.target_net(torch.t(last_transition.next_state[0]).double().to(device)))

            state_action_values = self.policy_net(last_transition.state[0].to(device))
            state_action_values = state_action_values.gather(0, torch.tensor(last_transition.action[0]).to(device))
            criterion = nn.HuberLoss()
            loss = criterion(state_action_values, target)
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1., 1.)
            self.optimizer.step()

        else:
            if len(self.memory) < self.batch_size:
                return
            else:
                transitions = self.memory.sample(self.batch_size)
                # transpose batch of transitions -> transition of batche arrays
                batch = Transition(*zip(*transitions))
                batch_state = tuple_of_tensors_to_tensor(batch.state).to(device)
                batch_action = tuple_of_tensors_to_tensor(batch.action).to(device)
                batch_reward = torch.tensor(batch.reward).type(torch.double).to(device)

                # non final next states :
                #print('batch_next_state : ', [ns for ns in batch.next_state if ns is not None])
                batch_next_state = torch.stack([ns for ns in batch.next_state if ns is not None], dim=1).to(device)

                # indices of final states :
                final_indices = torch.tensor([ind for ind in range(len(batch.next_state)) if batch.next_state[ind] is None]).long()

                # final and non-final states masks :
                final_states_mask = torch.tensor(tuple(map(lambda s: s is None,
                                                        batch.next_state)), dtype=torch.bool)
                non_final_states_mask = ~final_states_mask

                # initialize targets tensor :
                targets = torch.zeros_like(final_states_mask).type(torch.double).to(device)

                # for final states, the target is simply the reward :
                targets[final_states_mask] = batch_reward[final_indices]

                # for non-final states, the target is given by the Bellman equation,
                # where predictions are mmade with the target network to stabilize targets :
                preds, indices = torch.max(self.target_net(torch.t(batch_next_state).double()), 1)

                # batch reward ommited, since in non final states rewards are zero :
                with torch.no_grad():
                    targets[non_final_states_mask] = self.gamma*preds

                # predictions of the policy network :
                #print('batch_state : ', batch_state.shape, ' , batch action : ', batch_action.shape)
                state_action_values = self.policy_net(batch_state).gather(1, batch_action).to(device)

                # optimization step of policy network :
                #criterion = nn.SmoothL1Loss()
                criterion = nn.HuberLoss()   
                loss = criterion(state_action_values, targets.unsqueeze(1))

                self.optimizer.zero_grad()
                loss.backward()
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1., 1.)
                self.optimizer.step()

            # Update the target network by making it the same as the policy network :
            if episode % self.target_network_update == 0:
                self.update_target_network()
                #self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss

    def play_and_learn(self, env, player, opponent, episode):
        """This function allows to perform a single learning step : the agent plays
        according to the epsilon-greedy policy, and updates the policy and target networks."""

        # use policy network to predict action, and apply (with epsilon-greedy policy).
        # collect transition and add to replay_buffer
        state = torch.flatten(torch.from_numpy(self.get_state(env)).double().to(device))
        predicted_action = torch.argmax(self.policy_net(state))
        action = self.epsilon_greedy(predicted_action, env)

        took_available_action = True
        if [action] in self.available_actions(env):
            env.step(action.item())
            if env.end:    # game is over
                next_state = None
            else:
                # if game has not ended then opponent plays to get to the learner turn
                opponent_action = opponent.act(env.grid)
                env.step(opponent_action)
                next_state = None if env.end else torch.flatten(torch.from_numpy(self.get_state(env)))

            reward = env.reward(player=player)

        else:
            #env.reset()
            reward = -1
            next_state = None
            took_available_action = False

        self.memory.push((state, action, next_state, reward))

        # update policy and target networks according to replay buffer
        loss = self.update_nets(episode=episode)

        return loss, took_available_action

    def train(self, number_games, opponent_epsilon, env, evaluate=False):
        """This function defines the training loop. It iteratively makes
        the agent and the opponent play, while the agent is also updated
        at each move.
        
        Args:
            number_games (int): the number of games (or episodes) in the training loop.
            opponnent_epsilon (float): the probability for the opponent to play according
            to the optimal policy.
            
        Return:
            losses (list): A list containing loss values regularly recorded."""

        players = ['X', 'O']
        losses = []
        rewards = []
        Mopt = []
        Mrand = []
        for game in range(number_games):
            env.reset()
            player = players[game % 2]
            opponent = OptimalPlayer(epsilon=opponent_epsilon, player=players[(game + 1) % 2])

            #### HAVE THE OPPONENT PLAY FIRST IF IT IS HIS TURN TO START ####
            if env.current_player != player:  # Opponent starts
                opponent_action = opponent.act(env.grid)
                env.step(opponent_action)
            #################################################################

            took_available_action = True
            while not env.end and took_available_action:
                self.update_epsilon(game)
                loss, took_available_action = self.play_and_learn(env, player, opponent, game)
                #if actions_taken % 10 == 0:
                losses.append(loss)

            rewards.append(-1 if not took_available_action else env.reward(player))

            if game % 250 == 0 and evaluate:
                Mopt.append(M_opt(self, env))
                Mrand.append(M_rand(self, env))
        
        if evaluate:
            return losses, rewards, Mopt, Mrand
        else:
            return losses, rewards

    def act(self, env):
        """This function is used to play against the agent, without having it train while playing.
        It uses the greedy policy, meaning that the action taken is always the one chosen by the
        policy network. It assumes that it is the agent's turn to play whenever it is called."""

        state = torch.flatten(torch.from_numpy(self.get_state(env)).double().to(device))
        predicted_action = torch.argmax(self.policy_net(state))
        return predicted_action

    def showQValues(self, state):
        """This function allows to visualize the predicted Q values for a given state.
        
        Args :
            state (np.array): the state that is passed to the agent."""

        state = torch.flatten(torch.from_numpy(state).double().to(device))
        Q =  self.policy_net(state).cpu().detach().numpy().reshape(3, 3)

        figure = plt.figure()
        axes = figure.add_subplot(111)
        matrix_axes = axes.matshow(Q)
        figure.colorbar(matrix_axes)



##########################################################################################
##########################################################################################

class SelfLearner(DQ_learner):
    """This class inherits from the DQ_learner class. It allows the agent to learn by self practice."""

    def switch_current_player(self, player):
        return 'X' if player=='O' else 'O'

    def train(self, number_games, env, evaluate=False):
        """This function allows the network to ttrain by self practice. The main difference with the
        train method of the parent class is that during a game, the same agent 'changes teams' after
        each move. Essentially, the function learnt by the network is one that takes a state, and
        outputs the best move AFTER FIGURING OUT WHOSE TURN IT IS TO PLAY.
        
        Args:
            number_games (int): the size of the training loop.
            env : the environment in which the agent plays.
            evaluate (bool): whether to return M_opt and M_rand. Defaults to False."""
            
        Mopt = []
        Mrand = []
        losses = []
        rewards_X = []
        rewards_O = []

        for game in range(number_games):
            env.reset()
            current_player = 'X'

            last_state = None
            last_action = None

            while not env.end:
                self.update_epsilon(game)

                state = torch.flatten(torch.from_numpy(self.get_state(env)).double().to(device))
                action = self.epsilon_greedy(self.act(env), env)

                if action.item() not in np.where(np.abs(env.grid).flatten()<.5)[0]:     # action is not available
                    self.memory.push((state, action, None, -1))
                    break
                else:
                    env.step(action.item())
                    new_state = torch.flatten(torch.from_numpy(self.get_state(env)).double().to(device))
                    if env.end:
                        #self.memory.push((last_state, last_action, new_state, env.reward(current_player)))
                        #self.memory.push((state, action, None, env.reward(self.switch_current_player(current_player))))
                        self.memory.push((last_state, last_action, None, env.reward(self.switch_current_player(current_player))))
                        self.memory.push((state, action, None, env.reward(current_player)))
                        break
                    
                    if last_state is not None:
                        self.memory.push((last_state, last_action, new_state, 0))

                last_state = state
                last_action = action
                current_player = self.switch_current_player(current_player)

                losses.append(self.update_nets(game))
            
            if env.end:
                rewards_X.append(env.reward(player='x'))
                rewards_O.append(env.reward(player='O'))

            if game % 250 == 0 and evaluate:
                Mopt.append(M_opt(self, env))
                Mrand.append(M_rand(self, env))
        
        if evaluate:
            return losses, rewards_X, rewards_O, Mopt, Mrand
        else:
            return losses, rewards_X, rewards_O

                



######################################################################################################
######################################################################################################


def M(learner, opponent_epsilon, env):

    M = 0.

    # wins1 = 0.
    # looses1 = 0.
    # wins2 = 0.
    # looses2 = 0.
    # game = 0

    opponent = OptimalPlayer(epsilon = opponent_epsilon, player='O')
    for _ in range(250):
        #game+=1
        env.reset()
        while not env.end:
            agent_action = learner.act(env)
            if agent_action.item() not in np.where(np.abs(env.grid).flatten()<.5)[0]:
                M -= 1
                #looses1 = looses1 + 1.
                break
            else:
                env.step(agent_action.item())
                if env.end:
                    r = env.reward(player='X')
                    M += r
                    # if r==1:
                    #     wins1 = wins1 + 1
                    # elif r ==-1:
                    #     looses1 = looses1 + 1
                    break
                else:
                    opponent_action = opponent.act(env.grid)
                    env.step(opponent_action)
                    r = env.reward(player='X')
                    M += r
                    # if r==1:
                    #     wins1 = wins1 + 1
                    # elif r ==-1:
                    #     looses1 = looses1 + 1

    opponent = OptimalPlayer(epsilon = opponent_epsilon, player='X')
    for _ in range(250):
        #game+=1
        env.reset()
        while not env.end:
            opponent_action = opponent.act(env.grid)
            env.step(opponent_action)
            if env.end:
                r = env.reward(player='O')
                M += r
                # if r==1:
                #     wins2 = wins2 + 1.
                # elif r ==-1:
                #     looses2 = looses2 +1.
            else:
                agent_action = learner.act(env)
                if agent_action.item() not in np.where(np.abs(env.grid).flatten()<.5)[0]:
                    M -= 1
                    #looses2 = looses2 + 1.
                    break
                else:
                    env.step(agent_action.item())
                    r = env.reward(player='O')
                    M += r
                    # if r==1:
                    #     wins2 = wins2 +1.
                    # elif r ==-1:
                    #     looses2 = looses2 + 1.

    #print('w1 : ', wins1, ', l1 : ', looses1, ', w2 : ', wins2, ', l2 : ', looses2)
    return M/500

def M_opt(learner, env):
    return M(learner, 0., env)

def M_rand(learner, env):
    return M(learner, 1., env)


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.stack(list(tuple_of_tensors), dim=0)

