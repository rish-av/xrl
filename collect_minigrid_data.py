from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch
import torch.nn as nn
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
    
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from stable_baselines3 import PPO


from argparse import ArgumentParser

def parse_args():
    args = ArgumentParser()
    args.add_argument('--env', type=str, default='MiniGrid-Empty-8x8-v0')
    args.add_argument('--algo', type=str, default='ppo')
    args.add_argument('--policy', type=str, default='MlpPolicy')
    args.add_argument('--total_timesteps', type=int, default=1e6)
    args.add_argument('--g1_threshold', type=int, default=200, help='Number of episodes to collect for goal 1')
    args.add_argument('--g2_threshold', type=int, default=200, help='Number of episodes to collect for goal 2')
    args.add_argument('--l_threshold', type=int, default=40, help='Number of episodes to collect for lava')
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--save_path', type=str, default='data.pkl')
    return args.parse_args()

class FlatFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.linear = nn.Sequential(
            nn.Linear(7*7*3, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b, c, h, w = observations.shape

        observations = observations.reshape((b, c * h * w))
        return self.linear(observations)


policy_kwargs = dict(
    features_extractor_class=FlatFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)



class MiniGridTwoGoalsLava(MiniGridEnv):
    def __init__(self, size=9, render_mode='human'):
        super().__init__(grid_size=size, render_mode=render_mode, mission_space = MissionSpace(mission_func=self._gen_mission))

        # wall co-ordinates, spread well across the grid of size 9 x 9
        self.wall_coords = [(2, 2), (1, 4), (4, 2), (7, 3), (4, 7)]


        #start position of the agent, pick from 4 random positions of size 9x9, should not goal or wall or lava
        self.possible_start_positions = [(1, 1), (2, 5), (5, 2), (3, 4)]
    
    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def place_agent(self):
        # Place the agent at a random position
        self.agent_pos = self._rand_elem(self.possible_start_positions)
        self.agent_dir = self._rand_int(0, 4)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(), width - 2, height - 2)
        self.put_obj(Goal(), width - 2, 1)
        self.put_obj(Lava(), width//2, height//2)
        
        for i in range(5):
            self.put_obj(Wall(), *self.wall_coords[i])
        
        

        self.place_agent()
        self.mission = "get to the green goal square"

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        if self.grid.get(*self.agent_pos) is not None and self.grid.get(*self.agent_pos).type == 'lava':
            reward = 0.25
            done = True
        
        if self.grid.get(*self.agent_pos) is not None and self.grid.get(*self.agent_pos).type == 'goal':
            reward = 0.5
            done = True
        
        return obs, reward, done, trunc, info


args = parse_args()
env = ImgObsWrapper(MiniGridTwoGoalsLava(size=9, render_mode='rgb_array'))
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
steps = args.total_timesteps
model.learn(steps)


g1_threshold = args.g1_threshold
g2_threshold = args.g2_threshold
l_threshold = args.l_threshold

#function to collect the data after training in d4rl format, colelct agent_pos and agent_dir
def collect_data(env, model):
    dataset = {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'agent_pos': [], 'agent_dir': []}

    g1_data = {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'agent_pos': [], 'agent_dir': []}
    g2_data = {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'agent_pos': [], 'agent_dir': []}

    l_data = {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'agent_pos': [], 'agent_dir': []}

    g1_count = 0
    g2_count = 0
    l_count = 0

    while True:
        obs, _ = env.reset()
        done = False
        episode = {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'agent_pos': [], 'agent_dir': []}
        while not done:
            action, _ = model.predict(obs)
            next_obs, reward, done, trunc,  info = env.step(action)
            episode['observations'].append(obs)
            episode['next_observations'].append(next_obs)
            episode['actions'].append(action)
            episode['rewards'].append(reward)
            episode['terminals'].append(done)
            episode['agent_pos'].append(env.unwrapped.agent_pos)
            episode['agent_dir'].append(env.unwrapped.agent_dir)


            if env.unwrapped.agent_pos == (7, 7) and g1_count <= g1_threshold and len(episode['observations']) <= 40:
                #add episode to g1_data
                print("ep len g1: ", len(episode['observations']))
                g1_data['observations'].extend(episode['observations'])
                g1_data['next_observations'].extend(episode['next_observations'])
                g1_data['actions'].extend(episode['actions'])
                g1_data['rewards'].extend(episode['rewards'])
                g1_data['terminals'].extend(episode['terminals'])
                g1_data['agent_pos'].extend(episode['agent_pos'])
                g1_data['agent_dir'].extend(episode['agent_dir'])
                g1_count += 1
            
            elif env.unwrapped.agent_pos == (7, 1) and g2_count <= g2_threshold and len(episode['observations']) <= 40:
                #add episode to g2_data
                print("ep len g2: ", len(episode['observations']))
                g2_data['observations'].extend(episode['observations'])
                g2_data['next_observations'].extend(episode['next_observations'])
                g2_data['actions'].extend(episode['actions'])
                g2_data['rewards'].extend(episode['rewards'])
                g2_data['terminals'].extend(episode['terminals'])
                g2_data['agent_pos'].extend(episode['agent_pos'])
                g2_data['agent_dir'].extend(episode['agent_dir'])
                g2_count += 1
            
            elif env.unwrapped.agent_pos == (4, 4) and l_count <= l_threshold and len(episode['observations']) <= 40:
                #add episode to l_data
                print("ep len l: ", len(episode['observations']))
                l_data['observations'].extend(episode['observations'])
                l_data['next_observations'].extend(episode['next_observations'])
                l_data['actions'].extend(episode['actions'])
                l_data['rewards'].extend(episode['rewards'])
                l_data['terminals'].extend(episode['terminals'])
                l_data['agent_pos'].extend(episode['agent_pos'])
                l_data['agent_dir'].extend(episode['agent_dir'])
                l_count += 1
            obs = next_obs

        print(f"g1_count: {g1_count}, g2_count: {g2_count}, l_count: {l_count}")
        if g1_count >= g1_threshold and g2_count >= g2_threshold and l_count >= l_threshold:
            break
    

    #combine all
    dataset['observations'] = g1_data['observations'] + g2_data['observations'] + l_data['observations']
    dataset['next_observations'] = g1_data['next_observations'] + g2_data['next_observations'] + l_data['next_observations']
    dataset['actions'] = g1_data['actions'] + g2_data['actions'] + l_data['actions']
    dataset['rewards'] = g1_data['rewards'] + g2_data['rewards'] + l_data['rewards']
    dataset['terminals'] = g1_data['terminals'] + g2_data['terminals'] + l_data['terminals']
    dataset['agent_pos'] = g1_data['agent_pos'] + g2_data['agent_pos'] + l_data['agent_pos']
    dataset['agent_dir'] = g1_data['agent_dir'] + g2_data['agent_dir'] + l_data['agent_dir']
    return dataset


data = collect_data(env, model)
file_name = args.save_path
import pickle
with open(file_name, 'wb') as f:
    pickle.dump(data, f)