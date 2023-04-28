import gym
import copy
import torch
import random
import os
from GAmodel import GAmodel, device
 
def main():
    model = GAmodel()
    env = gym.make("LunarLander-v2", render_mode='human')
    observation, info = env.reset(seed=42) 
    n = 0
    reward_sum = 0
    while n < 10000:
        action = model.get_action(observation)
        new_observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward

        #print(f'y: {y}, pred: {torch.max(model(torch.FloatTensor(observation)))}')
        observation = new_observation

        if terminated or truncated:
            n+=1
            print(f'epoch {n}: {reward_sum}')
            reward_sum=0
            observation, info = env.reset()
    env.close()

if __name__=='__main__':
    main()