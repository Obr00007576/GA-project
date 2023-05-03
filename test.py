import torch
from GAmodel import GAmodel, device
import gym

model = GAmodel()
model.load_state_dict(torch.load('./checkpoint'))

env = gym.make("LunarLander-v2", render_mode='human')

observation, info = env.reset(seed=42) 
n = 0
reward_sum = 0
while n < 5:
    action = model.get_action(observation)
    new_observation, reward, terminated, truncated, info = env.step(action)
    reward_sum += reward

    observation = new_observation

    if terminated or truncated:
        n+=1
        print(f'epoch {n}: {reward_sum}')
        reward_sum=0
        observation, info = env.reset(seed=42)
env.close()